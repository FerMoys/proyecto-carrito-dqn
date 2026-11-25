"""Regresiones de reserva de salidas para el entrenamiento DQN."""

from contextlib import chdir, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import train


class TrainingOutputTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="dqn_train_output_")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)

    def run_training(self, *arguments):
        with patch("sys.argv", ["train.py", *map(str, arguments)]):
            with redirect_stdout(io.StringIO()):
                train.main()

    def test_rejects_used_directory_before_training_and_keeps_files(self):
        destination = self.root / "previo"
        destination.mkdir()
        artifacts = {
            "best_model.pt": b"best",
            "final_model.pt": b"final",
            "training_log.csv": b"episode,reward\n1,5\n",
        }
        for name, payload in artifacts.items():
            (destination / name).write_bytes(payload)

        with patch("train.CarTrackEnv") as env:
            with self.assertRaisesRegex(SystemExit, "ya existe.*--output"):
                self.run_training("--episodes", 1, "--max-steps", 2, "--output", destination)
            env.assert_not_called()
        self.assertEqual({file.name: file.read_bytes() for file in destination.iterdir()}, artifacts)

    def test_rejects_existing_empty_directory_and_existing_file(self):
        existing_dir = self.root / "vacio"
        existing_dir.mkdir()
        existing_file = self.root / "archivo"
        existing_file.write_bytes(b"contenido original")

        for destination in (existing_dir, existing_file):
            with self.subTest(destination=destination.name), patch("train.CarTrackEnv") as env:
                with self.assertRaisesRegex(SystemExit, "ya existe.*--output"):
                    self.run_training("--output", destination)
                env.assert_not_called()
        self.assertEqual(list(existing_dir.iterdir()), [])
        self.assertEqual(existing_file.read_bytes(), b"contenido original")

    def test_reports_os_error_before_starting_the_environment(self):
        destination = self.root / "sin-permiso"
        denied = PermissionError(13, "permiso denegado", str(destination))
        with (patch("train.Path.mkdir", side_effect=denied),
              patch("train.CarTrackEnv") as env):
            with self.assertRaisesRegex(SystemExit, "No se pudo.*permiso denegado"):
                self.run_training("--output", destination)
            env.assert_not_called()
        self.assertFalse(destination.exists())

    def test_default_reserves_distinct_directories_under_results(self):
        with chdir(self.root):
            Path("results").mkdir()
            historical = Path("results/best_model.pt")
            historical.write_bytes(b"checkpoint anterior")
            first = train.reserve_output(None)
            (first / "training_log.csv").write_bytes(b"primera ejecucion")
            second = train.reserve_output(None)
            self.assertEqual(first.parent, Path("results").resolve())
            self.assertEqual(second.parent, first.parent)
            self.assertNotEqual(first, second)
            self.assertEqual(list(second.iterdir()), [])
            self.assertEqual((first / "training_log.csv").read_bytes(), b"primera ejecucion")
            self.assertEqual(historical.read_bytes(), b"checkpoint anterior")

    def test_best_checkpoint_can_be_updated_twice_in_same_run(self):
        destination = self.root / "grupo" / "nuevo"
        best = destination / "best_model.pt"
        final = destination / "final_model.pt"
        observation = [0.0] * 9
        transitions = [(observation, float(reward), False, True, {"steps": 1})
                       for reward in range(31)]

        def save_fixture(path):
            path = Path(path)
            previous = int(path.read_text(encoding="utf-8")) if path.exists() else 0
            path.write_text(str(previous + 1), encoding="utf-8")

        with (patch("train.CarTrackEnv") as make_env,
              patch("train.DQNAgent") as make_agent,
              patch("torch.utils.tensorboard.SummaryWriter"),
              patch("train.save_curves"),
              patch("train.evaluate_without_render", return_value={"evaluation_success_rate": 0.0})):
            env = make_env.return_value
            env.reset.return_value = observation, {}
            env.action_space.n = 9
            env.step.side_effect = transitions
            agent = make_agent.return_value
            agent.select_action.return_value = 0
            agent.train_step.return_value = {}
            agent.epsilon = 0.5
            agent.save.side_effect = save_fixture

            self.run_training("--episodes", 31, "--max-steps", 1, "--output", destination)

        self.assertEqual(agent.save.call_args_list, [call(str(best)), call(str(best)), call(str(final))])
        self.assertEqual(best.read_text(encoding="utf-8"), "2")
        self.assertTrue((destination / "training_log.csv").is_file())


if __name__ == "__main__":
    unittest.main()
