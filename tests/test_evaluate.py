"""Pruebas sin checkpoints entrenados; los modelos y agentes son simulados."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
import importlib.util
import io
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import evaluate


def make_env():
    env = Mock()
    env.closed = False
    env.action_space.n = 9
    env.reset.return_value = ([0.0] * 9, {"success": 0, "collision": 0})
    env.step.return_value = (
        [0.0] * 9, 2.5, True, False,
        {"success": 1, "collision": 0, "steps": 999},
    )
    return env


def make_agent():
    agent = Mock()
    agent.select_action.return_value = 1
    return agent


@contextmanager
def mock_runtime(env, agent):
    """Sustituye solo durante la prueba los imports pesados de main()."""
    env_class = Mock(return_value=env)
    agent_class = Mock(return_value=agent)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=lambda name: name,
    )
    modules = {
        "torch": torch,
        "car_env": SimpleNamespace(CarTrackEnv=env_class),
        "dqn_agent": SimpleNamespace(DQNAgent=agent_class),
        "config": SimpleNamespace(TrainConfig=Mock(return_value=object())),
    }
    with patch.dict(sys.modules, modules):
        yield env_class, agent_class


class ArgumentTests(unittest.TestCase):
    def test_original_defaults(self):
        args = evaluate.parse_args([])
        self.assertEqual(args.model, Path("results/best_model.pt"))
        self.assertEqual(args.episodes, 5)
        self.assertEqual(args.seed, 1000)
        self.assertFalse(args.random_start)
        self.assertFalse(args.no_render)
        self.assertIsNone(args.output)

    def test_original_command(self):
        args = evaluate.parse_args([
            "--model", "results/best_model.pt", "--episodes", "10", "--no-render",
        ])
        self.assertEqual(args.episodes, 10)
        self.assertTrue(args.no_render)
        self.assertEqual(args.seed, 1000)
        self.assertFalse(args.random_start)

    def test_new_command(self):
        args = evaluate.parse_args([
            "--model", "results/best_model.pt", "--episodes", "10", "--seed", "42",
            "--random-start", "--no-render", "--output", "results/evaluacion_detallada.csv",
        ])
        self.assertEqual(args.seed, 42)
        self.assertTrue(args.random_start)
        self.assertEqual(args.output, Path("results/evaluacion_detallada.csv"))

    def test_invalid_arguments(self):
        for option, value in (
            ("--episodes", "0"), ("--episodes", "-1"),
            ("--episodes", "1.5"), ("--episodes", "texto"),
            ("--seed", "-1"), ("--seed", "1.5"), ("--seed", "texto"),
        ):
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    evaluate.parse_args([option, value])
                self.assertEqual(error.exception.code, 2)

    def test_zero_seed_is_valid(self):
        self.assertEqual(evaluate.parse_args(["--seed", "0"]).seed, 0)

    def test_experimental_defaults_and_arguments(self):
        defaults = evaluate.parse_args([])
        self.assertIsNone(defaults.csv_output)
        self.assertEqual(defaults.configuracion, "Base64")
        self.assertEqual(defaults.semilla_entrenamiento, 42)
        args = evaluate.parse_args([
            "--csv-output", "csv_results/nuevo.csv", "--configuracion", "Base32",
            "--semilla-entrenamiento", "7", "--seed", "900", "--random-start",
        ])
        self.assertEqual(args.csv_output, Path("csv_results/nuevo.csv"))
        self.assertEqual(args.configuracion, "Base32")
        self.assertEqual(args.semilla_entrenamiento, 7)
        self.assertEqual(args.seed, 900)
        self.assertIsNone(args.output)

    def test_invalid_experimental_metadata(self):
        for arguments in (
            ["--semilla-entrenamiento", "-1"],
            ["--semilla-entrenamiento", "texto"],
            ["--configuracion", "   "],
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    evaluate.parse_args(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_output_formats_must_use_distinct_paths(self):
        for other in ("resultados.csv", "subcarpeta/../resultados.csv"):
            with self.subTest(other=other), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    evaluate.parse_args([
                        "--output", "resultados.csv", "--csv-output", other,
                    ])
                self.assertEqual(error.exception.code, 2)


class EvaluationTests(unittest.TestCase):
    def test_fixed_start_seeds_steps_and_no_exploration(self):
        env, agent = make_env(), make_agent()
        records = []
        rows = evaluate.evaluate_episodes(
            env, agent, "modelo.pt", episodes=3, on_episode=records.append,
        )
        self.assertEqual(records, rows)
        self.assertEqual([row["episode"] for row in rows], [1, 2, 3])
        self.assertEqual([row["seed"] for row in rows], [1000, 1001, 1002])
        self.assertTrue(all(row["random_start"] == 0 for row in rows))
        self.assertTrue(all(row["steps"] == 1 for row in rows))
        self.assertEqual(env.reset.call_args_list, [
            call(seed=1000 + i, options={"random_start": False}) for i in range(3)
        ])
        self.assertEqual(agent.select_action.call_count, 3)
        self.assertTrue(all(c.kwargs == {"training": False}
                            for c in agent.select_action.call_args_list))
        agent.train_step.assert_not_called()
        agent.remember.assert_not_called()

    def test_accumulates_reward_and_counts_actual_steps(self):
        env = make_env()
        env.step.side_effect = [
            ([0], 1.25, False, False, {}),
            ([0], -3.5, True, False, {"collision": 1, "steps": 100}),
        ]
        row = evaluate.evaluate_episodes(env, make_agent(), "m.pt", episodes=1)[0]
        self.assertEqual(row["reward"], -2.25)
        self.assertEqual(row["steps"], 2)
        self.assertEqual(row["collision"], 1)
        self.assertEqual(row["success"], 0)

    def test_truncated_episode_is_complete(self):
        env = make_env()
        env.step.return_value = ([0], 1.0, False, True, {})
        rows = evaluate.evaluate_episodes(env, make_agent(), "m.pt", episodes=1)
        self.assertEqual(rows[0]["terminated"], 0)
        self.assertEqual(rows[0]["truncated"], 1)
        self.assertEqual(evaluate.calculate_metrics(rows)["episodes_completed"], 1)

    def test_window_close_records_only_confirmed_transition(self):
        env = make_env()

        def close_window(action):
            env.closed = True
            return [0], 1.5, False, False, {}

        env.step.side_effect = close_window
        rows = evaluate.evaluate_episodes(env, make_agent(), "m.pt", episodes=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reward"], 1.5)
        self.assertEqual(rows[0]["steps"], 1)
        self.assertEqual(evaluate.calculate_metrics(rows)["episodes_interrupted"], 1)

    def test_close_on_terminal_transition_still_counts_complete(self):
        env = make_env()

        def finish_and_close(action):
            env.closed = True
            return [0], 2.0, True, False, {"success": 1}

        env.step.side_effect = finish_and_close
        rows = evaluate.evaluate_episodes(env, make_agent(), "m.pt", episodes=2)
        self.assertEqual(evaluate.calculate_metrics(rows)["episodes_completed"], 1)
        self.assertEqual(evaluate.calculate_metrics(rows)["episodes_interrupted"], 0)

    def test_keyboard_interrupt_preserves_partial_episode(self):
        env = make_env()
        env.step.side_effect = [([0], 4.0, False, False, {}), KeyboardInterrupt()]
        rows = evaluate.evaluate_episodes(env, make_agent(), "m.pt", episodes=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["steps"], 1)
        self.assertEqual(rows[0]["reward"], 4.0)
        self.assertEqual(rows[0]["terminated"], 0)
        self.assertEqual(rows[0]["truncated"], 0)

    def test_interrupt_during_action_selection_records_zero_steps(self):
        agent = make_agent()
        agent.select_action.side_effect = KeyboardInterrupt()
        rows = evaluate.evaluate_episodes(make_env(), agent, "m.pt", episodes=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["steps"], 0)
        self.assertEqual(rows[0]["reward"], 0.0)

    def test_interrupt_during_reset_does_not_invent_episode(self):
        env = make_env()
        env.reset.side_effect = [([0], {}), KeyboardInterrupt()]
        rows = evaluate.evaluate_episodes(env, make_agent(), "m.pt", episodes=3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["terminated"], 1)

    def test_closed_before_start_produces_no_rows(self):
        env = make_env()
        env.closed = True
        self.assertEqual(evaluate.evaluate_episodes(env, make_agent(), "m.pt"), [])
        env.reset.assert_not_called()

    def test_closed_during_reset_records_started_episode(self):
        env = make_env()

        def reset_and_close(**kwargs):
            env.closed = True
            return [0], {}

        env.reset.side_effect = reset_and_close
        rows = evaluate.evaluate_episodes(env, make_agent(), "m.pt")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["steps"], 0)
        env.step.assert_not_called()

    def test_programmatic_validation(self):
        for kwargs in ({"episodes": 0}, {"episodes": -1}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate.evaluate_episodes(make_env(), make_agent(), "m.pt", **kwargs)

    def test_same_seeds_reproduce_rows_with_simulated_environment(self):
        class SeededEnv:
            closed = False

            def reset(self, *, seed, options):
                self.value = random.Random(seed).random() if options["random_start"] else 0.0
                return [self.value], {}

            def step(self, action):
                return [self.value], self.value + action, True, False, {}

        def run(seed):
            return evaluate.evaluate_episodes(
                SeededEnv(), make_agent(), "simulado.pt", episodes=3,
                seed=seed, random_start=True,
            )

        self.assertEqual(run(42), run(42))
        self.assertNotEqual([r["reward"] for r in run(42)], [r["reward"] for r in run(43)])
        self.assertEqual([r["seed"] for r in run(42)], [42, 43, 44])
        self.assertTrue(all(r["random_start"] == 1 for r in run(42)))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy no está disponible")
    def test_reproducibility_real_simulator_with_mock_agent(self):
        from simulator_core import CarSimulator

        class SimulatorEnv:
            closed = False

            def __init__(self):
                self.simulator = CarSimulator()

            def reset(self, *, seed, options):
                self.simulator.seed(seed)
                return self.simulator.reset(random_start=options["random_start"]), {}

            def step(self, action):
                return self.simulator.step(action)

        def run(seed):
            return evaluate.evaluate_episodes(
                SimulatorEnv(), make_agent(), "agente_simulado.pt", episodes=3,
                seed=seed, random_start=True,
            )

        first = run(42)
        self.assertEqual(first, run(42))
        self.assertNotEqual([r["reward"] for r in first], [r["reward"] for r in run(43)])


class MetricTests(unittest.TestCase):
    def test_exact_metrics_exclude_interrupted_episode(self):
        rows = [
            dict(reward=10.0, steps=3, success=1, collision=0, terminated=1, truncated=0),
            dict(reward=30.0, steps=9, success=0, collision=1, terminated=1, truncated=0),
            dict(reward=1000.0, steps=100, success=0, collision=0, terminated=0, truncated=0),
        ]
        self.assertEqual(evaluate.calculate_metrics(rows), {
            "reward_mean": 20.0, "reward_std": 10.0,
            "success_rate": 0.5, "collision_rate": 0.5,
            "steps_success_mean": 3.0, "episodes_completed": 2, "episodes_interrupted": 1,
        })

    def test_no_completed_episodes_has_no_performance_metrics(self):
        for rows in ([], [dict(terminated=0, truncated=0)]):
            with self.subTest(rows=rows):
                metrics = evaluate.calculate_metrics(rows)
                self.assertEqual(metrics["episodes_completed"], 0)
                self.assertEqual(metrics["episodes_interrupted"], len(rows))
                for key in ("reward_mean", "reward_std", "success_rate",
                            "collision_rate", "steps_success_mean"):
                    self.assertIsNone(metrics[key])

    def test_one_episode_std_zero_and_no_success_steps(self):
        row = dict(reward=3.0, steps=5, success=0, collision=0, terminated=0, truncated=1)
        metrics = evaluate.calculate_metrics([row])
        self.assertEqual(metrics["reward_std"], 0.0)
        self.assertIsNone(metrics["steps_success_mean"])

    def test_summary_handles_no_data(self):
        output = io.StringIO()
        with redirect_stdout(output):
            evaluate.print_summary(evaluate.calculate_metrics([]))
        self.assertIn("no disponible", output.getvalue())
        self.assertIn("Episodios completados: 0", output.getvalue())


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test_evaluate_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model = self.directory / "modelo_simulado.pt"
        self.model.touch()  # Fixture vacía; agent.load se simula, no es un checkpoint.
        self.env, self.agent = make_env(), make_agent()

    def run_main(self, *arguments):
        output = io.StringIO()
        with mock_runtime(self.env, self.agent) as classes:
            with patch.object(sys, "argv", ["evaluate.py", "--model", str(self.model), *arguments]):
                with redirect_stdout(output):
                    evaluate.main()
        return output.getvalue(), classes

    def test_original_command_with_mock_runtime(self):
        output, (env_class, _) = self.run_main("--episodes", "10", "--no-render")
        env_class.assert_called_once_with(render_mode=None)
        self.agent.load.assert_called_once_with(str(self.model))
        self.agent.policy_net.eval.assert_called_once_with()
        self.env.close.assert_called_once_with()
        self.assertEqual(self.agent.select_action.call_count, 10)
        self.assertIn("Episodios completados: 10", output)

    def test_render_remains_enabled_by_default(self):
        _, (env_class, _) = self.run_main("--episodes", "1")
        env_class.assert_called_once_with(render_mode="human")

    def test_new_command_exports_ten_rows_and_exact_columns(self):
        destination = self.directory / "nueva" / "evaluación.csv"
        self.run_main(
            "--episodes", "10", "--seed", "42", "--random-start", "--no-render",
            "--output", str(destination),
        )
        with destination.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, [
            "model", "episode", "seed", "random_start", "reward", "steps",
            "success", "collision", "terminated", "truncated",
        ])
        self.assertEqual(len(rows), 10)
        self.assertEqual([int(r["seed"]) for r in rows], list(range(42, 52)))
        self.assertEqual([int(r["episode"]) for r in rows], list(range(1, 11)))
        self.assertTrue(all(r["model"] == str(self.model) for r in rows))
        self.assertTrue(all(r["random_start"] == "1" for r in rows))
        self.assertTrue(all(r["reward"] == "2.5" and r["steps"] == "1" for r in rows))
        self.assertTrue(all(r["terminated"] == "1" and r["truncated"] == "0" for r in rows))
        self.assertTrue(all(r["success"] == "1" and r["collision"] == "0" for r in rows))
        self.assertTrue(all(c.kwargs["options"] == {"random_start": True}
                            for c in self.env.reset.call_args_list))
        self.env.close.assert_called_once_with()

    def test_existing_csv_is_preserved_before_model_load(self):
        destination = self.directory / "existente.csv"
        destination.write_text("evidencia original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_main("--output", str(destination))
        self.assertEqual(destination.read_text(encoding="utf-8"), "evidencia original")
        self.agent.load.assert_not_called()
        self.env.reset.assert_not_called()

    def test_model_cannot_be_used_as_output(self):
        with self.assertRaises(FileExistsError):
            self.run_main("--output", str(self.model))
        self.assertEqual(self.model.read_bytes(), b"")

    def test_missing_model_does_not_create_output(self):
        self.model = self.directory / "no_existe.pt"
        destination = self.directory / "no_creada" / "salida.csv"
        with self.assertRaises(FileNotFoundError):
            self.run_main("--output", str(destination))
        self.assertFalse(destination.parent.exists())
        self.agent.load.assert_not_called()

    def test_model_directory_is_rejected(self):
        self.model = self.directory
        with self.assertRaises(FileNotFoundError):
            self.run_main()

    def test_load_failure_closes_environment_without_csv(self):
        self.agent.load.side_effect = RuntimeError("fallo de carga simulado")
        destination = self.directory / "salida.csv"
        with self.assertRaisesRegex(RuntimeError, "fallo de carga simulado"):
            self.run_main("--output", str(destination))
        self.env.close.assert_called_once_with()
        self.assertFalse(destination.exists())

    def test_initial_reset_failure_closes_environment(self):
        self.env.reset.side_effect = RuntimeError("fallo de reset simulado")
        with self.assertRaisesRegex(RuntimeError, "fallo de reset simulado"):
            self.run_main()
        self.env.close.assert_called_once_with()

    def test_step_failure_preserves_completed_rows_and_closes(self):
        self.env.step.side_effect = [
            ([0], 2.0, True, False, {"success": 1}),
            RuntimeError("fallo de step simulado"),
        ]
        destination = self.directory / "salida.csv"
        with self.assertRaisesRegex(RuntimeError, "fallo de step simulado"):
            self.run_main("--episodes", "3", "--output", str(destination))
        self.env.close.assert_called_once_with()
        with destination.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["episode"], "1")

    def test_ctrl_c_exports_partial_row_and_reports_counts(self):
        self.env.step.side_effect = [
            ([0], 2.0, True, False, {"success": 1}),
            ([0], 3.0, False, False, {}),
            KeyboardInterrupt(),
        ]
        destination = self.directory / "parcial.csv"
        output, _ = self.run_main("--episodes", "10", "--output", str(destination))
        with destination.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["steps"], "1")
        self.assertEqual(rows[1]["reward"], "3.0")
        self.assertEqual(rows[1]["terminated"], "0")
        self.assertEqual(rows[1]["truncated"], "0")
        self.assertIn("Episodios completados: 1", output)
        self.assertIn("Episodios interrumpidos: 1", output)
        self.assertIn("Episodios no iniciados: 8", output)
        self.assertIn("Retorno promedio: 2.000000", output)
        self.env.close.assert_called_once_with()

    def test_reset_interruption_exports_header_without_invented_rows(self):
        self.env.reset.side_effect = [([0], {}), KeyboardInterrupt()]
        destination = self.directory / "sin_episodios.csv"
        output, _ = self.run_main("--episodes", "10", "--output", str(destination))
        with destination.open(encoding="utf-8", newline="") as source:
            self.assertEqual(list(csv.DictReader(source)), [])
        self.assertIn("Episodios no iniciados: 10", output)
        self.assertIn("Episodios interrumpidos: 0", output)
        self.env.close.assert_called_once_with()

    def test_exclusive_creation_protects_against_race(self):
        destination = self.directory / "ocupado.csv"
        with patch.object(Path, "open", side_effect=FileExistsError("creado por otro proceso")):
            with self.assertRaises(FileExistsError):
                self.run_main("--output", str(destination))
        self.env.close.assert_called_once_with()
        self.agent.select_action.assert_not_called()

    def test_experimental_csv_exact_contract_and_distinct_seeds(self):
        destination = self.directory / "experimentos" / "nuevo.csv"
        output, _ = self.run_main(
            "--episodes", "3", "--seed", "900", "--random-start", "--no-render",
            "--csv-output", str(destination), "--configuracion", "Base64",
            "--semilla-entrenamiento", "7",
        )
        with destination.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, [
            "configuracion", "semilla", "condicion", "episodio",
            "reward", "success", "collision", "steps",
        ])
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["episodio"] for r in rows], ["1", "2", "3"])
        self.assertTrue(all(r["semilla"] == "7" for r in rows))
        self.assertTrue(all(r["configuracion"] == "Base64" for r in rows))
        self.assertTrue(all(r["condicion"] == "aleatoria" for r in rows))
        self.assertTrue(all(r["reward"] == "2.5" and r["steps"] == "1" for r in rows))
        self.assertEqual([c.kwargs["seed"] for c in self.env.reset.call_args_list],
                         [900, 900, 901, 902])
        self.assertTrue(all(c.kwargs == {"training": False}
                            for c in self.agent.select_action.call_args_list))
        self.assertIn("Episodios completados: 3", output)
        self.env.close.assert_called_once_with()

    def test_dual_exports_preserve_rows_and_aggregate_metrics(self):
        detailed = self.directory / "detalle.csv"
        experimental = self.directory / "experimental.csv"
        self.env.step.side_effect = [
            ([0], 10.0, True, False, {"success": 1}),
            ([0], -2.0, True, False, {"collision": 1}),
            ([0], 4.0, False, True, {}),
        ]
        output, _ = self.run_main(
            "--episodes", "3", "--seed", "1000", "--no-render",
            "--output", str(detailed), "--csv-output", str(experimental),
            "--configuracion", "Base64", "--semilla-entrenamiento", "42",
        )
        with detailed.open(encoding="utf-8", newline="") as source:
            detail_reader = csv.DictReader(source)
            detail_rows = list(detail_reader)
        with experimental.open(encoding="utf-8", newline="") as source:
            experiment_rows = list(csv.DictReader(source))
        self.assertEqual(detail_reader.fieldnames, list(evaluate.CSV_COLUMNS))
        self.assertEqual(len(detail_rows), 3)
        self.assertEqual(len(experiment_rows), 3)
        for detail, experiment in zip(detail_rows, experiment_rows):
            for key in ("reward", "success", "collision", "steps"):
                self.assertEqual(detail[key], experiment[key])
            self.assertEqual(detail["episode"], experiment["episodio"])
            self.assertEqual(experiment["semilla"], "42")
            self.assertEqual(experiment["condicion"], "fija")
        self.assertEqual([r["seed"] for r in detail_rows], ["1000", "1001", "1002"])
        self.assertEqual(sum(float(r["reward"]) for r in experiment_rows) / 3, 4.0)
        self.assertEqual(sum(int(r["success"]) for r in experiment_rows), 1)
        self.assertEqual(sum(int(r["collision"]) for r in experiment_rows), 1)
        self.assertIn("Retorno promedio: 4.000000", output)
        self.assertIn("Tasa de éxito: 33.33%", output)
        self.assertIn("Tasa de colisión: 33.33%", output)
        self.assertIn("Episodios completados: 3", output)
        self.assertEqual(self.agent.select_action.call_count, 3)
        self.env.close.assert_called_once_with()

    def test_partial_episode_is_detailed_but_not_experimental(self):
        detailed = self.directory / "detalle.csv"
        experimental = self.directory / "experimental.csv"
        self.env.step.side_effect = [
            ([0], 2.0, True, False, {"success": 1}),
            ([0], 300.0, False, False, {}),
            KeyboardInterrupt(),
        ]
        output, _ = self.run_main(
            "--episodes", "10", "--output", str(detailed),
            "--csv-output", str(experimental),
        )
        with detailed.open(encoding="utf-8", newline="") as source:
            detail_rows = list(csv.DictReader(source))
        with experimental.open(encoding="utf-8", newline="") as source:
            experiment_rows = list(csv.DictReader(source))
        self.assertEqual(len(detail_rows), 2)
        self.assertEqual(len(experiment_rows), 1)
        self.assertEqual(detail_rows[1]["reward"], "300.0")
        self.assertEqual(experiment_rows[0]["reward"], "2.0")
        self.assertIn("Retorno promedio: 2.000000", output)
        self.assertIn("Episodios interrumpidos: 1", output)
        self.assertIn("Episodios parciales omitidos del CSV experimental: 1", output)

    def test_partial_only_produces_experimental_header(self):
        destination = self.directory / "experimental.csv"
        self.env.step.side_effect = KeyboardInterrupt()
        output, _ = self.run_main("--episodes", "10", "--csv-output", str(destination))
        with destination.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            self.assertEqual(list(reader), [])
            self.assertEqual(reader.fieldnames, list(evaluate.EXPERIMENTAL_CSV_COLUMNS))
        self.assertIn("Episodios completados: 0", output)
        self.assertIn("Episodios parciales omitidos del CSV experimental: 1", output)
        self.env.close.assert_called_once_with()

    def test_existing_experimental_csv_is_not_appended_or_overwritten(self):
        experimental = self.directory / "historico.csv"
        historical_bytes = b"configuracion,semilla\r\nBase64,42\r\n"
        experimental.write_bytes(historical_bytes)
        detailed = self.directory / "detalle_nuevo.csv"
        with self.assertRaises(FileExistsError):
            self.run_main("--output", str(detailed), "--csv-output", str(experimental))
        self.assertEqual(experimental.read_bytes(), historical_bytes)
        self.assertFalse(detailed.exists())
        self.agent.load.assert_not_called()

    def test_existing_detail_prevents_new_experimental_file(self):
        detailed = self.directory / "historico.csv"
        detailed.write_bytes(b"evidencia existente\r\n")
        experimental = self.directory / "experimental_nuevo.csv"
        with self.assertRaises(FileExistsError):
            self.run_main("--output", str(detailed), "--csv-output", str(experimental))
        self.assertEqual(detailed.read_bytes(), b"evidencia existente\r\n")
        self.assertFalse(experimental.exists())
        self.agent.load.assert_not_called()

    def test_experimental_output_cannot_overwrite_checkpoint(self):
        with self.assertRaises(FileExistsError):
            self.run_main("--csv-output", str(self.model))
        self.assertEqual(self.model.read_bytes(), b"")
        self.agent.load.assert_not_called()

    def test_missing_checkpoint_creates_neither_output(self):
        self.model = self.directory / "ausente.pt"
        detailed = self.directory / "detalle" / "nuevo.csv"
        experimental = self.directory / "experimento" / "nuevo.csv"
        with self.assertRaises(FileNotFoundError):
            self.run_main("--output", str(detailed), "--csv-output", str(experimental))
        self.assertFalse(detailed.parent.exists())
        self.assertFalse(experimental.parent.exists())

    def test_load_failure_creates_neither_output_and_closes(self):
        self.agent.load.side_effect = RuntimeError("carga simulada fallida")
        detailed = self.directory / "detalle.csv"
        experimental = self.directory / "experimental.csv"
        with self.assertRaisesRegex(RuntimeError, "carga simulada fallida"):
            self.run_main("--output", str(detailed), "--csv-output", str(experimental))
        self.assertFalse(detailed.exists())
        self.assertFalse(experimental.exists())
        self.env.close.assert_called_once_with()

    def test_experimental_open_race_closes_both_file_and_environment(self):
        detailed = self.directory / "detalle.csv"
        experimental = self.directory / "experimental.csv"
        original_open = Path.open
        opened = []

        def racing_open(path, *args, **kwargs):
            if path == experimental:
                raise FileExistsError("ruta ocupada por otro proceso")
            handle = original_open(path, *args, **kwargs)
            opened.append(handle)
            return handle

        with patch.object(Path, "open", racing_open), self.assertRaises(FileExistsError):
            self.run_main("--output", str(detailed), "--csv-output", str(experimental))
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].closed)
        self.env.close.assert_called_once_with()
        self.agent.select_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
