"""Regenera los derivados académicos desde los notebooks y CSV, sin entrenar."""

import argparse
import csv
import hashlib
from pathlib import Path
from statistics import fmean, pstdev

from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[2]
LOG_SHA256 = "2dd65a595cc5378e6fbad9007501ce5731656849f8cf92cb90303962ce187ab8"

HISTORICAL = '''
assert len(df) == 240 and len(por_semilla) == 24
plot_mean_std("tasa_colision", "Colisiones por política e inicio", "Colisiones (%)",
              "tasa_colision", percent=True)
por_semilla.to_csv(output_dir / "historico_por_semilla.csv", index=False, mode="x", lineterminator=chr(10))
captions = {
    "tasa_exito": ("historico_exito.png", "Éxito por política e inicio", "Episodios exitosos (%)"),
    "tasa_colision": ("historico_colision.png", "Colisiones por política e inicio", "Episodios con colisión (%)"),
    "retorno_promedio": ("historico_retorno.png", "Retorno por política e inicio", "Retorno acumulado por episodio"),
    "pasos_promedio_exitosos": ("historico_pasos_exitosos.png", "Pasos en episodios exitosos", "Pasos por episodio exitoso"),
}
for name, (file, title, ylabel) in captions.items():
    figure = figures[name]
    ax = figure.axes[0]
    ax.set(title=title, xlabel="Política", ylabel=ylabel)
    ax.set_xticks(range(4), ["Aleatorio", "Base32", "Base64", "Heurístico"])
    ax.set_xlim(-0.55, 3.55)
    ax.legend(title="Inicio", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    if name == "pasos_promedio_exitosos":
        ax.text(0, 35, "Sin éxitos", ha="center", color="#333333", fontsize=9)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.text(0.02, 0.06, "Fuente: csv_results/comparison_results_integrado.csv (240 episodios).", fontsize=9)
    figure.text(0.02, 0.02, "Media y desviación muestral entre tres semillas. Diez episodios por semilla y condición.", fontsize=8)
    with (output_dir / file).open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150, bbox_inches="tight", metadata={"Software": None})
'''

TRAINING = '''
assert len(logs) == 145 and list(logs.episode) == list(range(1, 146))
comparison.to_csv(output_dir / "entrenamiento_base64_contraste.csv", index_label="metrica", mode="x", lineterminator=chr(10))
from matplotlib.ticker import PercentFormatter

figure = figures["curvas_entrenamiento"]
figure.suptitle("Entrenamiento Base64 de demostración (semilla 42)", fontsize=16)
top, bottom = figure.axes
top.set_ylabel("Retorno acumulado por episodio")
top.legend(["Retorno por episodio", "Media móvil (30 episodios)"], loc="upper left")
bottom.set_ylabel("Éxito móvil (%)")
bottom.yaxis.set_major_formatter(PercentFormatter(1))
bottom.legend(["Media móvil (30 episodios)", "Meta de éxito (90 %)"], loc="upper left")
figure.tight_layout(rect=(0, 0.07, 1, 0.95))
figure.text(0.02, 0.02, "Fuente: result_model_prueba.zip / training_log.csv (145 episodios).", fontsize=10)
with (output_dir / "entrenamiento_base64_curvas.png").open("xb") as stream:
    figure.savefig(stream, format="png", dpi=150, bbox_inches="tight", metadata={"Software": None})
'''


def run_notebook(name, output, setup, appendix):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    configuration = "import os\n" + "\n".join(
        f"os.environ[{key!r}] = {str(value)!r}" for key, value in setup.items()
    )
    notebook.cells.insert(0, nbformat.v4.new_code_cell(configuration))
    export = next(cell for cell in notebook.cells if cell.cell_type == "code"
                  and 'for name, figure in figures.items()' in cell.source)
    export.source = f"output_dir = Path({str(output)!r})\n" + appendix
    NotebookClient(notebook, kernel_name="carrito-dqn", timeout=120,
                   resources={"metadata": {"path": str(ROOT)}}).execute()


def export_demo(output):
    summaries = []
    for condition in ("fija", "aleatoria"):
        source = ROOT / "results" / f"evaluacion_best_{condition}_py311_20261123_01.csv"
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rewards = [float(row["reward"]) for row in rows]
        successes = [row for row in rows if row["success"] == "1"]
        summaries.append({
            "condicion": condition,
            "episodios": len(rows),
            "semilla_inicial": rows[0]["seed"],
            "semilla_final": rows[-1]["seed"],
            "exito_pct": 100 * len(successes) / len(rows),
            "colision_pct": 100 * sum(int(row["collision"]) for row in rows) / len(rows),
            "retorno_medio": fmean(rewards),
            "retorno_sd_poblacional": pstdev(rewards),
            "pasos_medios_exitosos": fmean(int(row["steps"]) for row in successes),
        })
    with (output / "base64_reproducido_resumen.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True, type=Path, help="CSV Base64 de 145 episodios")
    parser.add_argument("--output", required=True, type=Path, help="Directorio nuevo de salida")
    args = parser.parse_args()
    training = args.training_log.resolve()
    if hashlib.sha256(training.read_bytes()).hexdigest() != LOG_SHA256:
        parser.error("El SHA-256 no coincide con el registro Base64 de demostración.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_notebook("analisis_experimental.ipynb", output,
                 {"CARRITO_EVALUATION_CSV": "csv_results/comparison_results_integrado.csv"},
                 HISTORICAL)
    run_notebook("analisis_entrenamiento.ipynb", output,
                 {"CARRITO_TRAINING_LOG": training}, TRAINING)
    export_demo(output)


if __name__ == "__main__":
    main()
