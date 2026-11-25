"""Evalúa un modelo DQN sin exploración y exporta resultados reproducibles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields, is_dataclass
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


def configuration_label(value: str) -> str:
    if not re.fullmatch(r"Base[1-9][0-9]*", value, re.IGNORECASE | re.ASCII):
        raise argparse.ArgumentTypeError("usa Base seguido del ancho, por ejemplo Base64")
    return "Base" + value[4:]


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
    parser.add_argument("--manifest", type=Path,
                        help="manifiesto JSONL nuevo; con CSV se crea uno por defecto")
    parser.add_argument("--configuracion", type=configuration_label,
                        help="confirma el ancho del checkpoint; se infiere si se omite")
    parser.add_argument("--semilla-entrenamiento", type=nonnegative_int,
                        help="confirma la semilla guardada; se infiere si se omite")
    args = parser.parse_args(argv)
    if (args.output is not None and args.csv_output is not None
            and args.output.resolve() == args.csv_output.resolve()):
        parser.error("--output y --csv-output deben usar rutas diferentes")
    return args


@dataclass(frozen=True)
class CheckpointIdentity:
    configuration: str
    training_seed: int | None
    sha256: str


def read_checkpoint(torch, data: bytes) -> tuple[CheckpointIdentity, Mapping]:
    # Versiones estables con las correcciones de los avisos de deserialización.
    version = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+[\w.-]+)?", torch.__version__)
    if version is None or (release := tuple(map(int, version.groups()))) < (2, 10, 0) or (
        (2, 14, 0) <= release < (2, 15, 0)
    ):
        raise RuntimeError("Usa PyTorch estable 2.10–2.13 o 2.15+ para cargar checkpoints.")
    force_variables = ("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    if all(os.getenv(name, "").lower() in {"1", "y", "yes", "true"} for name in force_variables):
        raise RuntimeError("Las dos variables TORCH_FORCE_* están activadas. "
                           "Desactiva al menos una antes de ejecutar evaluate.py.")
    try:
        checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except (EOFError, IndexError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError("No se pudo leer el checkpoint con weights_only=True.") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Formato de checkpoint DQN no reconocido.")
    policy = checkpoint.get("policy_state_dict")
    if not isinstance(policy, Mapping):
        raise ValueError("Checkpoint sin policy_state_dict verificable.")
    shapes = [
        tuple(getattr(policy.get(f"layers.{index}.weight"), "shape", ()))
        for index in (0, 2, 4)
    ]
    if any(len(shape) != 2 or min(shape) < 1 for shape in shapes):
        raise ValueError("No se puede verificar la arquitectura con los tensores guardados.")
    first, middle, last = shapes
    width = first[0]
    if middle != (width, width) or last[1] != width:
        raise ValueError("Las dimensiones de las capas son incompatibles entre sí.")
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("El campo config del checkpoint es incompatible.")
    if "hidden_size" in config and (
        type(config["hidden_size"]) is not int or config["hidden_size"] != width
    ):
        raise ValueError("config.hidden_size contradice la arquitectura de los tensores.")
    seed = config.get("seed")
    if "seed" in config and (type(seed) is not int or seed < 0):
        raise ValueError("config.seed del checkpoint es incompatible.")
    identity = CheckpointIdentity(f"Base{width}", seed, hashlib.sha256(data).hexdigest())
    return identity, checkpoint


def validate_checkpoint(args: argparse.Namespace, identity: CheckpointIdentity) -> None:
    if args.configuracion is not None and args.configuracion != identity.configuration:
        raise ValueError(
            f"--configuracion {args.configuracion} no coincide con {identity.configuration}."
        )
    if identity.training_seed is None:
        if args.csv_output is not None or args.semilla_entrenamiento is not None:
            raise ValueError("Checkpoint sin config.seed verificable. Usa solo el CSV detallado.")
    elif (args.semilla_entrenamiento is not None
          and args.semilla_entrenamiento != identity.training_seed):
        raise ValueError("--semilla-entrenamiento no coincide con config.seed del checkpoint.")


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


def check_new_outputs(*paths: Path | None) -> None:
    destinations = [path for path in paths if path is not None]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("Las salidas requieren rutas diferentes.")
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"La salida ya existe: {path}. Elige otra ruta.")


def prepare_outputs(
    stack: ExitStack, detailed: Path | None, experimental: Path | None,
    manifest: Path | None = None, provenance: dict | None = None,
) -> tuple[tuple[csv.DictWriter, TextIO] | None, tuple[csv.DictWriter, TextIO] | None,
           TextIO | None]:
    created = []
    try:
        with ExitStack() as preparation:
            def open_new(path: Path | None) -> TextIO | None:
                if path is None:
                    return None
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = preparation.enter_context(path.open("x", encoding="utf-8", newline=""))
                created.append((path, os.fstat(handle.fileno())))
                return handle

            details_handle = open_new(detailed)
            experiment_handle = open_new(experimental)
            journal = open_new(manifest)
            outputs = []
            for handle, columns in ((details_handle, CSV_COLUMNS),
                                    (experiment_handle, EXPERIMENTAL_CSV_COLUMNS)):
                if handle is None:
                    outputs.append(None)
                else:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    handle.flush()
                    outputs.append((writer, handle))
            if journal is not None:
                append_manifest(journal, {"kind": "evaluation", "schema": 1, **(provenance or {})})
            stack.enter_context(preparation.pop_all())
    except BaseException as error:
        # Aquí aún no se ha iniciado ningún episodio. Retirar solo archivos
        # creados por esta preparación y que sigan siendo los mismos.
        for path, original in reversed(created):
            try:
                if not path.is_symlink() and os.path.samestat(original, path.stat()):
                    path.unlink()
            except FileNotFoundError:
                continue
            except OSError as cleanup_error:
                error.add_note(f"No se pudo retirar una salida nueva: {cleanup_error}")
        error.add_note("Falló la preparación de las salidas. No se iniciaron episodios.")
        raise
    return outputs[0], outputs[1], journal


def append_manifest(handle: TextIO, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def evaluation_provenance(args, identity, checkpoint, torch, device, env, train_config) -> dict:
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    training_config = {
        field.name: value
        for field in (fields(train_config) if is_dataclass(train_config) else ())
        if type(value := checkpoint.get("config", {}).get(field.name)) in (int, float, bool)
    }
    environment_config = {}
    for name in ("track_config", "car_config"):
        config = getattr(env, name, None)
        if is_dataclass(config):
            environment_config[name] = asdict(config)
    return {
        "model": args.model.name,
        "checkpoint": asdict(identity),
        "training_config": training_config,
        "evaluation": {
            "initial_seed": args.seed, "episodes_requested": args.episodes,
            "seed_rule": "initial_seed + episode - 1",
            "random_start": args.random_start, "render": not args.no_render,
            "environment_config": environment_config,
        },
        "runtime": {"python": platform.python_version(), "pytorch": str(torch.__version__),
                    "device": str(device)},
        "code": {"git_head": commit, "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("evaluate.py", "dqn_agent.py", "config.py", "car_env.py", "simulator_core.py")
        }},
        "outputs": {"detailed": args.output.name if args.output else None,
                    "experimental": args.csv_output.name if args.csv_output else None},
        "outputs_next_to_manifest": {
            key: path.parent.resolve() == args.manifest.parent.resolve()
            for key, path in (("detailed", args.output), ("experimental", args.csv_output))
            if path is not None
        },
    }


def read_manifest(path: Path) -> tuple[dict, list[dict], str]:
    rows = []
    with path.open("rb") as handle:
        first = handle.readline()
        header = json.loads(first) if first.endswith(b"\n") else {}
        if (not isinstance(header, dict) or header.get("kind") != "evaluation"
                or header.get("schema") != 1):
            raise ValueError("Cabecera del manifiesto incompleta o incompatible.")
        status = "sin cierre"
        for line in handle:
            if status != "sin cierre":
                raise ValueError("El manifiesto contiene datos después del cierre.")
            if not line.endswith(b"\n"):
                status = "último registro incompleto"
                break
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Registro inválido en el manifiesto.")
            if record.get("kind") == "finish" and record.get("status") in ("completed", "interrupted"):
                status = record["status"]
                if status == "completed" and (
                    len(rows) != header["evaluation"]["episodes_requested"]
                    or any(not (row["terminated"] or row["truncated"]) for row in rows)
                ):
                    raise ValueError("El cierre completo no coincide con los episodios registrados.")
                continue
            row = record.get("row")
            number = len(rows) + 1
            evaluation = header["evaluation"]
            if (record.get("kind") != "episode" or not isinstance(row, dict)
                    or set(row) != set(CSV_COLUMNS) or row["episode"] != number
                    or number > evaluation["episodes_requested"]
                    or row["seed"] != evaluation["initial_seed"] + number - 1
                    or row["model"] != header["model"]
                    or row["random_start"] != int(evaluation["random_start"])):
                raise ValueError(f"Registro inválido en el episodio {number} del manifiesto.")
            rows.append(row)
    return header, rows, status


def csv_matches(path: Path, columns: Sequence[str], expected: Sequence[dict]) -> bool:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            return reader.fieldnames == list(columns) and list(reader) == [
                {key: str(value) for key, value in row.items()} for row in expected
            ]
    except (OSError, UnicodeError, csv.Error):
        return False


def project_experimental(header: dict, rows: Sequence[dict]) -> list[dict]:
    identity = header["checkpoint"]
    return [
        row for original in rows if (row := experimental_record(
            original, configuracion=identity["configuration"],
            semilla_entrenamiento=identity["training_seed"],
        )) is not None
    ]


def verify_exports(manifest: Path, detailed: Path | None, experimental: Path | None) -> bool:
    header, rows, status = read_manifest(manifest)
    print(f"Manifiesto: {status}. Episodios registrados: {len(rows)}.")
    consistent = True
    projections = project_experimental(header, rows)
    for key, explicit, columns, expected in (
        ("detailed", detailed, CSV_COLUMNS, rows),
        ("experimental", experimental, EXPERIMENTAL_CSV_COLUMNS, projections),
    ):
        name = header["outputs"][key]
        path = explicit
        if path is None and name:
            if not header["outputs_next_to_manifest"][key]:
                option = "--output" if key == "detailed" else "--csv-output"
                print(f"{name}: indica la ruta original con {option}.")
                consistent = False
                continue
            path = manifest.parent / Path(name).name
        if path is not None:
            matches = csv_matches(path, columns, expected)
            print(f"{path.name}: {'alineado' if matches else 'desalineado, ausente o ilegible'}.")
            consistent &= matches
    return consistent


def manifest_exports(argv: Sequence[str]) -> int:
    mode, *arguments = argv
    parser = argparse.ArgumentParser(prog=f"evaluate.py {mode}")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, help="ruta del CSV detallado")
    parser.add_argument("--csv-output", type=Path, help="ruta del CSV experimental")
    args = parser.parse_args(arguments)
    if mode == "verify-exports":
        return 0 if verify_exports(args.manifest, args.output, args.csv_output) else 1
    if args.output is None and args.csv_output is None:
        parser.error("indica al menos una ruta nueva con --output o --csv-output")
    header, rows, status = read_manifest(args.manifest)
    print(f"Manifiesto: {status}. Episodios registrados: {len(rows)}.")
    if args.csv_output is not None and header["checkpoint"]["training_seed"] is None:
        parser.error("el manifiesto no contiene una semilla de entrenamiento verificable")
    check_new_outputs(args.output, args.csv_output)
    projections = project_experimental(header, rows)
    with ExitStack() as stack:
        details, experiment, _ = prepare_outputs(stack, args.output, args.csv_output)
        for output, expected in ((details, rows), (experiment, projections)):
            if output is not None:
                writer, handle = output
                writer.writerows(expected)
                handle.flush()
    print("Recuperación escrita en archivos nuevos. Conserva el manifiesto original.")
    return 0


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


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in ("verify-exports", "recover-exports"):
        return manifest_exports(argv)
    args = parse_args(argv)
    if not args.model.is_file():
        raise FileNotFoundError(
            f"No se encontró {args.model}. Ejecuta primero: python train.py"
        )
    first_output = args.output or args.csv_output
    if args.manifest is None and first_output is not None:
        args.manifest = first_output.with_name(first_output.name + ".manifest.jsonl")
    check_new_outputs(args.output, args.csv_output, args.manifest)

    # Permite validar argumentos y probar el módulo sin instalar el simulador.
    import torch

    from car_env import CarTrackEnv
    from config import TrainConfig
    from dqn_agent import DQNAgent

    identity, checkpoint = read_checkpoint(torch, args.model.read_bytes())
    validate_checkpoint(args, identity)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    render_mode = None if args.no_render else "human"
    env = CarTrackEnv(render_mode=render_mode)
    prepared = False
    try:
        observation, _ = env.reset(
            seed=args.seed, options={"random_start": args.random_start}
        )
        hidden_size = int(identity.configuration.removeprefix("Base"))
        train_config = TrainConfig(hidden_size=hidden_size)
        agent = DQNAgent(
            observation_size=len(observation),
            action_size=env.action_space.n,
            config=train_config,
            device=device,
        )
        # Solo inferencia hasta integrar DQNAgent.load() con carga explícita.
        # Aquí se usa la política que ya construye el agente de esta rama.
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.policy_net.eval()

        provenance = None
        if args.manifest is not None:
            provenance = evaluation_provenance(args, identity, checkpoint, torch, device,
                                               env, train_config)
        with ExitStack() as stack:
            detailed_output, experimental_output, journal = prepare_outputs(
                stack, args.output, args.csv_output, args.manifest, provenance,
            )
            prepared = True
            print(f"Checkpoint: {args.model.name} | SHA-256: {identity.sha256} | "
                  f"{identity.configuration} | semilla de entrenamiento: "
                  f"{identity.training_seed if identity.training_seed is not None else 'ausente'}")

            def record_episode(row: dict) -> None:
                if journal is not None:
                    append_manifest(journal, {"kind": "episode", "row": row})
                if detailed_output is not None:
                    writer, output_file = detailed_output
                    writer.writerow(row)
                    output_file.flush()
                if experimental_output is not None:
                    experimental = experimental_record(
                        row, configuracion=identity.configuration,
                        semilla_entrenamiento=identity.training_seed,
                    )
                    if experimental is not None:
                        writer, output_file = experimental_output
                        writer.writerow(experimental)
                        output_file.flush()
                print(f"Episodio {row['episode']}: {row}")

            results = evaluate_episodes(
                env, agent, args.model.name, episodes=args.episodes, seed=args.seed,
                random_start=args.random_start, on_episode=record_episode,
            )
            metrics = calculate_metrics(results)
            if journal is not None:
                status = ("completed" if metrics["episodes_completed"] == args.episodes
                          else "interrupted")
                append_manifest(journal, {"kind": "finish", "status": status})
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
        if args.manifest is not None:
            print(f"Manifiesto guardado en: {args.manifest}")
            return 0 if verify_exports(args.manifest, args.output, args.csv_output) else 1
        return 0
    except BaseException as error:
        if prepared and args.manifest is not None:
            try:
                verify_exports(args.manifest, args.output, args.csv_output)
            except (OSError, ValueError) as audit_error:
                error.add_note(f"No se pudo verificar el manifiesto: {audit_error}")
            error.add_note(f"Manifiesto: {args.manifest}. "
                           "Usa evaluate.py verify-exports y recover-exports con rutas nuevas.")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())

