"""Pruebas del cargador DQN con tensores y deserialización reales."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from config import TrainConfig
from dqn_agent import DQNAgent, QNetwork


FORCE = "TORCH_FORCE_WEIGHTS_ONLY_LOAD"
NO_FORCE = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"


class CheckpointLoadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="dqn_load_")
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "modelo.pt"
        flags = patch.dict(os.environ, {FORCE: "0", NO_FORCE: "0"})
        flags.start()
        self.addCleanup(flags.stop)

    def agent(self, **config):
        return DQNAgent(9, 9, TrainConfig(**config), torch.device("cpu"))

    def checkpoint(self, source=None):
        source = self.agent(hidden_size=32, seed=11) if source is None else source
        return {
            "policy_state_dict": source.policy_net.state_dict(),
            "target_state_dict": source.target_net.state_dict(),
            "optimizer_state_dict": source.optimizer.state_dict(),
            "total_steps": source.total_steps,
            "config": vars(source.config).copy(),
        }

    def test_policy_only_without_config_infers_width_with_one_read_and_load(self):
        payload = self.checkpoint()
        torch.save({"policy_state_dict": payload["policy_state_dict"]}, self.path)
        expected_sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        agent = self.agent()
        original_read = Path.read_bytes
        with (patch.object(torch, "load", wraps=torch.load) as deserialize,
              patch.object(Path, "read_bytes", autospec=True, side_effect=original_read) as read):
            identity = agent.load(self.path, mode="inference")
        read.assert_called_once_with(self.path)
        deserialize.assert_called_once()
        self.assertEqual(deserialize.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        self.assertEqual(identity.configuration, "Base32")
        self.assertIsNone(identity.training_seed)
        self.assertEqual(identity.training_config, {})
        self.assertEqual(identity.sha256, expected_sha)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), expected_sha)

    def test_inference_is_greedy_and_ignores_training_components(self):
        source = self.agent(hidden_size=32, seed=11)
        source.total_steps = 29
        payload = self.checkpoint(source)
        payload.update(target_state_dict=None, optimizer_state_dict=None)
        torch.save(payload, self.path)
        agent = self.agent()
        identity = agent.load(self.path, mode="inference")
        self.assertEqual(identity.training_seed, 11)
        self.assertEqual(identity.training_config["hidden_size"], 32)
        self.assertIsNone(agent.optimizer)
        self.assertIsNone(agent.target_net)
        self.assertIsNone(agent.replay)
        self.assertEqual(agent.total_steps, 0)
        self.assertFalse(agent.policy_net.training)
        observation = np.linspace(-1, 1, 9, dtype=np.float32)
        with patch("dqn_agent.random.random", side_effect=AssertionError("exploración")):
            self.assertEqual(agent.select_action(observation, training=False),
                             source.select_action(observation, training=False))
        with self.assertRaisesRegex(RuntimeError, "resume"):
            agent.train_step()

    def test_unsupported_pytorch_cannot_deserialize(self):
        agent = self.agent()
        for version in ("2.9.1", "2.14.1", "2.15.0rc1"):
            with (self.subTest(version=version), patch.object(torch, "__version__", version),
                  patch.object(torch, "load") as deserialize):
                with self.assertRaisesRegex(RuntimeError, "PyTorch estable"):
                    agent.load(self.path, mode="inference")
                deserialize.assert_not_called()

    def test_inherited_torch_flags_keep_explicit_restricted_load(self):
        torch.save(self.checkpoint(), self.path)
        agent = self.agent()
        for enabled in (FORCE, NO_FORCE):
            flags = {FORCE: "0", NO_FORCE: "0", enabled: "true"}
            with (self.subTest(enabled=enabled), patch.dict(os.environ, flags),
                  patch.object(torch, "load", wraps=torch.load) as deserialize):
                agent.load(self.path, mode="inference")
                deserialize.assert_called_once()
                self.assertIs(deserialize.call_args.kwargs["weights_only"], True)
                self.assertEqual({key: os.environ[key] for key in flags}, flags)

    def test_conflicting_torch_flags_fail_before_deserialization(self):
        agent = self.agent()
        flags = {FORCE: "yes", NO_FORCE: "1"}
        with patch.dict(os.environ, flags), patch.object(torch, "load") as deserialize:
            with self.assertRaisesRegex(RuntimeError, "TORCH_FORCE_"):
                agent.load(self.path, mode="inference")
            deserialize.assert_not_called()
            self.assertEqual({key: os.environ[key] for key in flags}, flags)

    def test_malformed_checkpoint_has_clear_error_and_no_unsafe_retry(self):
        self.path.write_bytes(b"invalido")
        error = IndexError("pop from empty list")
        agent = self.agent()
        with patch.object(torch, "load", side_effect=error) as deserialize:
            with self.assertRaisesRegex(ValueError, "weights_only=True") as result:
                agent.load(self.path, mode="inference")
            self.assertIs(result.exception.__cause__, error)
            deserialize.assert_called_once()
            self.assertIs(deserialize.call_args.kwargs["weights_only"], True)

    def test_rejects_conflicting_metadata_and_invalid_network_shapes(self):
        agent = self.agent()
        original_policy = agent.policy_net
        for key, invalid, message in (
            ("hidden_size", 64, "config.hidden_size"),
            ("seed", None, "config.seed"),
        ):
            with self.subTest(key=key):
                payload = self.checkpoint()
                payload["config"][key] = invalid
                torch.save(payload, self.path)
                with self.assertRaisesRegex(ValueError, message):
                    agent.load(self.path, mode="inference")
                self.assertIs(agent.policy_net, original_policy)
        for key, shape in (("layers.0.weight", (32, 8)), ("layers.4.bias", (8,))):
            with self.subTest(key=key):
                payload = self.checkpoint()
                payload["policy_state_dict"][key] = torch.zeros(shape)
                torch.save(payload, self.path)
                with self.assertRaisesRegex(ValueError, key):
                    agent.load(self.path, mode="inference")
                self.assertIs(agent.policy_net, original_policy)

    def test_default_resume_restores_target_optimizer_steps_and_config(self):
        source = self.agent(hidden_size=32, seed=11, gamma=0.93, learning_rate=0.002)
        source.total_steps = 83
        with torch.no_grad():
            next(source.target_net.parameters()).add_(0.25)
        for parameter in source.policy_net.parameters():
            parameter.grad = torch.ones_like(parameter)
        source.optimizer.step()
        source.save(str(self.path))
        agent = self.agent()
        observation = np.zeros(9, dtype=np.float32)
        agent.remember(observation, 0, 0.0, observation, False)
        identity = agent.load(str(self.path))
        self.assertEqual(identity.configuration, "Base32")
        self.assertEqual(agent.config, source.config)
        self.assertEqual(agent.total_steps, 83)
        self.assertAlmostEqual(agent.epsilon, source.epsilon)
        self.assertEqual(len(agent.replay), 0)
        self.assertFalse(agent.target_net.training)
        torch.testing.assert_close(agent.policy_net.state_dict(), source.policy_net.state_dict())
        torch.testing.assert_close(agent.target_net.state_dict(), source.target_net.state_dict())
        torch.testing.assert_close(agent.optimizer.state_dict(), source.optimizer.state_dict())
        first = next(agent.policy_net.parameters())
        self.assertEqual(agent.optimizer.state[first]["step"].item(), 1)
        first.grad = torch.ones_like(first)
        agent.optimizer.step()
        self.assertEqual(agent.optimizer.state[first]["step"].item(), 2)

    def test_resume_rejects_missing_optimizer_and_mismatched_target_without_mutation(self):
        agent = self.agent()
        original_policy, original_target, original_optimizer = (
            agent.policy_net, agent.target_net, agent.optimizer,
        )
        for key, value, message in (
            ("optimizer_state_dict", None, "optimizador"),
            ("target_state_dict", QNetwork(9, 9, 64).state_dict(), "anchos distintos"),
            ("total_steps", -1, "total_steps"),
        ):
            with self.subTest(key=key):
                payload = self.checkpoint()
                payload[key] = value
                torch.save(payload, self.path)
                with self.assertRaisesRegex(ValueError, message):
                    agent.load(self.path, mode="resume")
                self.assertIs(agent.policy_net, original_policy)
                self.assertIs(agent.target_net, original_target)
                self.assertIs(agent.optimizer, original_optimizer)
                self.assertEqual(agent.config.hidden_size, 64)


if __name__ == "__main__":
    unittest.main()
