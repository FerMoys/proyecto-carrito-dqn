"""Carga un modelo entrenado y observa su conducción sin exploración aleatoria."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from car_env import CarTrackEnv
from config import TrainConfig
from dqn_agent import DQNAgent
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación visual del carrito DQN")
    parser.add_argument("--model", type=Path, default=Path("results/best_model.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
    "--csv-output",
    type=Path,
    default=None,
    help="Archivo CSV donde guardar los resultados"
    )
    parser.add_argument(
        "--configuracion",
        type=str,
        default="Base64"
    )
    parser.add_argument(
        "--semilla-entrenamiento",
        type=int,
        default=42
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Evaluar con posiciones iniciales aleatorias"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"No se encontró {args.model}. Ejecuta primero: python train.py"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    render_mode = None if args.no_render else "human"
    env = CarTrackEnv(render_mode=render_mode)
    observation, _ = env.reset(seed=1000, options={"random_start": False})
    agent = DQNAgent(
        observation_size=len(observation),
        action_size=env.action_space.n,
        config=TrainConfig(),
        device=device,
    )
    agent.load(str(args.model))

    results = []
    for episode in range(args.episodes):
        observation, _ = env.reset(
            seed=1000 + episode, 
            options={"random_start": args.random_start}
        )
        total_reward = 0.0
        done = False
        info = {}
        while not done and not env.closed:
            action = agent.select_action(observation, training=False)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        results.append(
            {
                "reward": total_reward,
                "success": info.get("success", 0.0),
                "collision": info.get("collision", 0.0),
                "steps": info.get("steps", 0.0),
            }
        )
        if env.closed:
            break
        print(f"Episodio {episode + 1}: {results[-1]}")

    env.close()

    if results:
        print(
            "Resumen:",
            {
                "reward_mean": float(np.mean([row["reward"] for row in results])),
                "success_rate": float(np.mean([row["success"] for row in results])),
                "collision_rate": float(np.mean([row["collision"] for row in results])),
            },
        )

        if args.csv_output:
            args.csv_output.parent.mkdir(parents=True, exist_ok=True)

            file_exists = (
                args.csv_output.exists()
                and args.csv_output.stat().st_size > 0
            )

            with args.csv_output.open(
                "a", newline="", encoding="utf-8"
            ) as file:
                fieldnames = [
                    "configuracion",
                    "semilla",
                    "condicion",
                    "episodio",
                    "reward",
                    "success",
                    "collision",
                    "steps",
                ]

                writer = csv.DictWriter(file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                for index, result in enumerate(results, start=1):
                    writer.writerow({
                        "configuracion": args.configuracion,
                        "semilla": args.semilla_entrenamiento,
                        "condicion": (
                            "aleatoria" if args.random_start else "fija"
                        ),
                        "episodio": index,
                        "reward": result["reward"],
                        "success": result["success"],
                        "collision": result["collision"],
                        "steps": result["steps"],
                    })

            print(f"Resultados guardados en: {args.csv_output}")


if __name__ == "__main__":
    main()

