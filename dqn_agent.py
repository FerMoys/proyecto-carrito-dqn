"""Red neuronal DQN, memoria de repetición y actualización de Bellman."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import hashlib
import io
import math
import os
from pathlib import Path
import pickle
import random
import re
from typing import Deque, Dict, Literal, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from config import TrainConfig


@dataclass(frozen=True)
class CheckpointIdentity:
    """Identidad del archivo y configuración registrada en el checkpoint."""

    configuration: str
    training_seed: int | None
    sha256: str
    training_config: dict[str, int | float]


def _deserialize_checkpoint(path: str | Path) -> tuple[Mapping, str]:
    version = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+[\w.-]+)?", torch.__version__)
    release = tuple(map(int, version.groups())) if version else (0, 0, 0)
    if release < (2, 10, 0) or (2, 14, 0) <= release < (2, 15, 0):
        raise RuntimeError("Usa PyTorch estable 2.10–2.13 o 2.15+ para cargar checkpoints.")
    variables = ("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    if all(os.getenv(name, "").lower() in {"1", "y", "yes", "true"} for name in variables):
        raise RuntimeError("Las dos variables TORCH_FORCE_* están activadas. "
                           "Desactiva al menos una antes de cargar el checkpoint.")
    data = Path(path).read_bytes()
    try:
        checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except (EOFError, IndexError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError("No se pudo leer el checkpoint con weights_only=True.") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Formato de checkpoint DQN no reconocido.")
    return checkpoint, hashlib.sha256(data).hexdigest()


def _validate_network(state, observation_size: int, action_size: int, name: str) -> int:
    first = state.get("layers.0.weight") if isinstance(state, Mapping) else None
    if not isinstance(first, torch.Tensor) or first.ndim != 2 or first.shape[0] <= 0:
        raise ValueError(f"No se puede verificar la arquitectura de {name}.")
    width = first.shape[0]
    dimensions = {
        "layers.0.weight": (width, observation_size), "layers.0.bias": (width,),
        "layers.2.weight": (width, width), "layers.2.bias": (width,),
        "layers.4.weight": (action_size, width), "layers.4.bias": (action_size,),
    }
    if set(state) != dimensions.keys():
        raise ValueError(f"Las claves de {name} no corresponden a una red DQN.")
    for key, shape in dimensions.items():
        tensor = state[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                or tensor.layout != torch.strided or not tensor.is_floating_point()):
            raise ValueError(f"Tensor incompatible en {name}: {key}. Se esperaba {shape}.")
    return width


class ReplayBuffer:
    """Guarda transiciones y devuelve minilotes aleatorios."""

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.data: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.data.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int, device: torch.device):
        batch = self.rng.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
            torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )


class QNetwork(nn.Module):
    """Red pequeña: nueve entradas, dos capas ocultas y nueve valores Q."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class DQNAgent:
    """Agente con política epsilon-greedy, red objetivo y pérdida Huber."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.action_size = action_size
        self.device = device
        self.policy_net = QNetwork(
            observation_size, action_size, config.hidden_size
        ).to(device)
        self.target_net = QNetwork(
            observation_size, action_size, config.hidden_size
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=config.learning_rate, amsgrad=True
        )
        self.replay = ReplayBuffer(config.replay_capacity, config.seed)
        self.total_steps = 0

    def _require_training(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Agente cargado para inferencia. Usa mode='resume' para entrenar.")

    @property
    def epsilon(self) -> float:
        fraction = math.exp(-self.total_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_end + (
            self.config.epsilon_start - self.config.epsilon_end
        ) * fraction

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        if training:
            self._require_training()
        if training and random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = int(self.policy_net(state).argmax(dim=1).item())
        if training:
            self.total_steps += 1
        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._require_training()
        self.replay.add(state, action, reward, next_state, done)

    def train_step(self) -> Dict[str, float]:
        self._require_training()
        if len(self.replay) < max(
            self.config.batch_size, self.config.learning_starts
        ):
            return {}

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size, self.device
        )
        predicted_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.soft_update()

        return {
            "loss": float(loss.item()),
            "q_mean": float(predicted_q.mean().item()),
            "target_mean": float(target_q.mean().item()),
        }

    def soft_update(self) -> None:
        self._require_training()
        tau = self.config.tau
        with torch.no_grad():
            for target_parameter, policy_parameter in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_parameter.mul_(1.0 - tau)
                target_parameter.add_(tau * policy_parameter)

    def save(self, path: str) -> None:
        self._require_training()
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "config": self.config.__dict__,
            },
            path,
        )

    def load(
        self, path: str | Path, *, mode: Literal["inference", "resume"] = "resume",
    ) -> CheckpointIdentity:
        """Carga una vez y devuelve la identidad de los mismos bytes leídos.

        inference conserva solo la política. resume restaura los componentes
        guardados para entrenar e inicia un búfer de repetición vacío.
        """
        if mode not in ("inference", "resume"):
            raise ValueError("mode debe ser 'inference' o 'resume'.")
        checkpoint, sha256 = _deserialize_checkpoint(path)
        observation_size = self.policy_net.layers[0].in_features
        policy_state = checkpoint.get("policy_state_dict")
        width = _validate_network(policy_state, observation_size, self.action_size, "la política")
        saved_config = checkpoint.get("config", {})
        if not isinstance(saved_config, Mapping):
            raise ValueError("El campo config del checkpoint es incompatible.")
        if "hidden_size" in saved_config and (
            type(saved_config["hidden_size"]) is not int or saved_config["hidden_size"] != width
        ):
            raise ValueError("config.hidden_size contradice la arquitectura de los tensores.")
        seed = saved_config.get("seed")
        if "seed" in saved_config and (type(seed) is not int or seed < 0):
            raise ValueError("config.seed del checkpoint es incompatible.")

        training_config = {}
        for field in fields(self.config):
            if field.name not in saved_config:
                continue
            value = saved_config[field.name]
            default = field.default
            valid = (type(value) is int or (type(default) is float
                                          and type(value) is float and math.isfinite(value)))
            if valid:
                training_config[field.name] = value
            elif mode == "resume":
                raise ValueError(f"config.{field.name} del checkpoint es incompatible.")

        updates = dict(training_config) if mode == "resume" else {}
        updates["hidden_size"] = width
        config = replace(self.config, **updates)
        target_state = optimizer_state = None
        steps = 0
        if mode == "resume":
            target_state = checkpoint.get("target_state_dict")
            if _validate_network(target_state, observation_size, self.action_size, "la red objetivo") != width:
                raise ValueError("La red objetivo y la política tienen anchos distintos.")
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("El estado del optimizador es necesario para reanudar.")
            steps = checkpoint.get("total_steps", 0)
            if type(steps) is not int or steps < 0:
                raise ValueError("total_steps debe ser un entero no negativo.")

        # Construir en variables locales conserva el agente si alguna carga falla.
        policy = QNetwork(observation_size, self.action_size, width).to(self.device)
        policy.load_state_dict(policy_state)
        target = optimizer = replay = None
        if mode == "resume":
            target = QNetwork(observation_size, self.action_size, width).to(self.device)
            target.load_state_dict(target_state)
            target.eval()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate, amsgrad=True)
            try:
                optimizer.load_state_dict(optimizer_state)
            except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
                raise ValueError("Estado del optimizador incompatible con la política.") from error
            for parameter, state in optimizer.state.items():
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and (
                        not isinstance(state[key], torch.Tensor) or state[key].shape != parameter.shape
                    ):
                        raise ValueError(f"Tensor del optimizador incompatible: {key}.")
            replay = ReplayBuffer(config.replay_capacity, config.seed)
        else:
            policy.eval()

        self.config = config
        self.policy_net, self.target_net = policy, target
        self.optimizer, self.replay = optimizer, replay
        self.total_steps = steps
        return CheckpointIdentity(f"Base{width}", seed, sha256, training_config)
