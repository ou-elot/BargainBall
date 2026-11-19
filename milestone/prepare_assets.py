"""Prepare the official style and a compact figure from existing saved results."""

from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parent
RUN = "20261109_204731_573737"


def main():
    style = REPORT / "neurips_2026.sty"

    if(not style.exists()):
        source = Path.home() / "Downloads" / "proposal_template" / "proposal_template" / "neurips_2026.sty"

        if(not source.exists()):
            raise FileNotFoundError("Place the official neurips_2026.sty beside main.tex.")

        shutil.copyfile(source, style)

    results = ROOT / "results" / "test" / "numpy_four_stat" / RUN
    metrics = pd.read_csv(results / "metrics.csv")
    selected = metrics[metrics["target"] == "pca_score"].set_index("split").loc[["train", "dev", "test"]]
    np.testing.assert_allclose(selected["model_mse"], [0.5841372005083281, 0.5726868197745555, 1.1839634119235531], rtol=1e-10)
    figures = REPORT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Serif", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(5.3, 2.05), layout="constrained")
    positions = np.arange(3)
    colors = ["#2563eb", "#ea580c", "#16a34a"]

    for i, metric in enumerate(["mse", "mae"]):
        values = selected["model_" + metric].to_numpy()
        baselines = selected["baseline_" + metric].to_numpy()
        axes[i].bar(positions, values, width=0.58, color=colors)
        axes[i].scatter(positions + 0.22, baselines, s=24, marker="D", color="#475569", facecolors="none", label="Persistence", zorder=3)
        axes[i].set_xticks(positions, ["Train", "Dev", "Test"])
        axes[i].set_ylabel(metric.upper())
        axes[i].set_ylim(0, max(values.max(), baselines.max()) * 1.24)
        axes[i].grid(axis="y", alpha=0.18)
        axes[i].set_axisbelow(True)

    axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    figure.savefig(figures / "composite_losses.pdf")
    figure.savefig(figures / "composite_losses.png", dpi=250)
    plt.close(figure)
    sources = [ROOT / "data" / "processed" / "dataset_summary.json", results / "metrics.csv", ROOT / "models" / "numpy_four_stat" / RUN / "manifest.json", ROOT / "models" / "numpy_four_stat" / RUN / "pca_parameters.json"]
    provenance = {}

    for path in sources:
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    (REPORT / "report_data_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print ("Prepared official style, compact loss figure, and saved-result provenance.")


if(__name__ == "__main__"):
    main()
