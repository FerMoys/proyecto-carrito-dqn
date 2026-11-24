"""Evalúa un modelo DQN sin exploración y exporta resultados reproducibles."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Sequence, TextIO

CSV_COLUMNS = (
    "model", "episode", "seed", "random_start", "reward", "steps",
    "success", "collision", "terminated", "truncated",
)
EXPERIMENTAL_CSV_COLUMNS = (
    "configuracion", "semilla", "condicion", "episodio",
    "reward", "success", "collision", "steps",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("debe ser un entero mayor que cero")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("debe ser un entero mayor o igual que cero")
    return number


def nonempty_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("debe ser una etiqueta no vacía")
    return text


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación reproducible del carrito DQN")
    parser.add_argument("--model", type=Path, default=Path("results/best_model.pt"))
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=nonnegative_int, default=1000,
                        help="semilla del primer episodio (por defecto: 1000)")
    parser.add_argument("--random-start", action="store_true",
                        help="habilitar variaciones iniciales reproducibles")
    parser.add_argument("--output", type=Path,
                        help="CSV nuevo; no se sobrescriben archivos existentes")
    parser.add_argument("--csv-output", type=Path,
                        help="CSV experimental nuevo de ocho columnas")
    parser.add_argument("--configuracion", type=nonempty_text, default="Base64",
                        help="etiqueta del experimento; no cambia la arquitectura")
    parser.add_argument("--semilla-entrenamiento", type=nonnegative_int, default=42,
                        help="semilla con la que se entrenó el modelo; distinta de --seed")
    args = parser.parse_args(argv)
    if (args.output is not None and args.csv_output is not None
            and args.output.resolve() == args.csv_output.resolve()):
        parser.error("--output y --csv-output deben usar rutas diferentes")
    return args


def evaluate_episodes(
    env,
    agent,
    model: str | Path,
    *,
    episodes: int = 5,
    seed: int = 1000,
    random_start: bool = False,
    on_episode: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Registra transiciones confirmadas; el llamador debe cerrar el entorno.

    Un episodio iniciado sin terminated/truncated es interrumpido. Los episodios
    no iniciados no producen filas. Ctrl+C durante reset no confirma un inicio.
    """
    if episodes <= 0 or seed < 0:
        raise ValueError("episodes debe ser positivo y seed no negativo")
    results = []
    for episode in range(episodes):
        if env.closed:
            break
        episode_seed = seed + episode
        try:
            observation, info = env.reset(
                seed=episode_seed, options={"random_start": random_start}
            )
        except KeyboardInterrupt:
            break
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        try:
            while not (terminated or truncated) and not env.closed:
                action = agent.select_action(observation, training=False)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
        except KeyboardInterrupt:
            # No se inventa la transición que no llegó a devolver env.step().
            pass
        row = {
            "model": str(model),
            "episode": episode + 1,
            "seed": episode_seed,
            "random_start": int(random_start),
            "reward": total_reward,
            "steps": steps,
            "success": int(bool(info.get("success", 0))),
            "collision": int(bool(info.get("collision", 0))),
            "terminated": int(terminated),
            "truncated": int(truncated),
        }
        results.append(row)
        if on_episode is not None:
            on_episode(row)
        if not (terminated or truncated) or env.closed:
            break
    return results


def calculate_metrics(results: Sequence[dict]) -> dict:
    """Las métricas de desempeño excluyen episodios interrumpidos."""
    completed = [row for row in results if row["terminated"] or row["truncated"]]
    rewards = [row["reward"] for row in completed]
    success_steps = [row["steps"] for row in completed if row["success"]]
    return {
        "reward_mean": fmean(rewards) if rewards else None,
        "reward_std": pstdev(rewards) if rewards else None,
        "success_rate": fmean(row["success"] for row in completed) if completed else None,
        "collision_rate": fmean(row["collision"] for row in completed) if completed else None,
        "steps_success_mean": fmean(success_steps) if success_steps else None,
        "episodes_completed": len(completed),
        "episodes_interrupted": len(results) - len(completed),
    }


def experimental_record(
    row: dict, *, configuracion: str, semilla_entrenamiento: int,
) -> dict | None:
    """Proyecta un episodio completo al contrato del notebook experimental.

    El formato de ocho columnas no puede identificar una interrupción. Omitir
    esas filas evita tratarlas como episodios completos en los agregados.
    """
    if not (row["terminated"] or row["truncated"]):
        return None
    return {
        "configuracion": configuracion,
        "semilla": semilla_entrenamiento,
        "condicion": "aleatoria" if row["random_start"] else "fija",
        "episodio": row["episode"],
        "reward": row["reward"],
        "success": row["success"],
        "collision": row["collision"],
        "steps": row["steps"],
    }


def open_csv_output(
    stack: ExitStack, path: Path | None, columns: Sequence[str],
) -> tuple[csv.DictWriter, TextIO] | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    # La creación exclusiva protege también frente a procesos concurrentes.
    output_file = stack.enter_context(path.open("x", encoding="utf-8", newline=""))
    writer = csv.DictWriter(output_file, fieldnames=columns)
    writer.writeheader()
    output_file.flush()
    return writer, output_file


def print_summary(metrics: dict) -> None:
    print("Resumen (métricas sobre episodios completos):")
    for key, label, fmt in (
        ("reward_mean", "Retorno promedio", ".6f"),
        ("reward_std", "Desviación estándar del retorno (poblacional)", ".6f"),
        ("success_rate", "Tasa de éxito", ".2%"),
        ("collision_rate", "Tasa de colisión", ".2%"),
        ("steps_success_mean", "Pasos promedio en episodios exitosos", ".2f"),
        ("episodes_completed", "Episodios completados", "d"),
        ("episodes_interrupted", "Episodios interrumpidos", "d"),
    ):
        value = metrics[key]
        print(f"  {label}: {'no disponible' if value is None else format(value, fmt)}")


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            f"No se encontró {args.model}. Ejecuta primero: python train.py"
        )
    for output in (args.output, args.csv_output):
        if output is not None and (output.exists() or output.is_symlink()):
            raise FileExistsError(f"El CSV ya existe: {output}. Elige otra ruta.")

    # Permite validar argumentos y probar el módulo sin instalar el simulador.
    import torch

    from car_env import CarTrackEnv
    from config import TrainConfig
    from dqn_agent import DQNAgent

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    render_mode = None if args.no_render else "human"
    env = CarTrackEnv(render_mode=render_mode)
    try:
        observation, _ = env.reset(
            seed=args.seed, options={"random_start": args.random_start}
        )
        agent = DQNAgent(
            observation_size=len(observation),
            action_size=env.action_space.n,
            config=TrainConfig(),
            device=device,
        )
        agent.load(str(args.model))
        agent.policy_net.eval()

        with ExitStack() as stack:
            detailed_output = open_csv_output(stack, args.output, CSV_COLUMNS)
            experimental_output = open_csv_output(
                stack, args.csv_output, EXPERIMENTAL_CSV_COLUMNS,
            )

            def record_episode(row: dict) -> None:
                if detailed_output is not None:
                    writer, output_file = detailed_output
                    writer.writerow(row)
                    output_file.flush()
                if experimental_output is not None:
                    experimental = experimental_record(
                        row, configuracion=args.configuracion,
                        semilla_entrenamiento=args.semilla_entrenamiento,
                    )
                    if experimental is not None:
                        writer, output_file = experimental_output
                        writer.writerow(experimental)
                        output_file.flush()
                print(f"Episodio {row['episode']}: {row}")

            results = evaluate_episodes(
                env, agent, args.model, episodes=args.episodes, seed=args.seed,
                random_start=args.random_start, on_episode=record_episode,
            )
        metrics = calculate_metrics(results)
        print_summary(metrics)
        not_started = args.episodes - len(results)
        if not_started:
            print(f"Episodios no iniciados: {not_started}")
        if args.output is not None:
            print(f"CSV guardado en: {args.output}")
        if args.csv_output is not None:
            print(f"CSV experimental guardado en: {args.csv_output}")
            interrupted = metrics["episodes_interrupted"]
            if interrupted:
                print(f"Episodios parciales omitidos del CSV experimental: {interrupted}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

