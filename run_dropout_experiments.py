"""Compare more dropout probabilities and a larger hidden-layer architecture.

Run: python run_dropout_experiments.py
The first round's 0% and 10% fits are reused for 32/16 and 64/32.
"""

import run_experiments
import csv

ROOT = run_experiments.ROOT

# All new fits use learning rate 0.001 to isolate width and dropout effects.
CONFIGURATIONS = [
    {"name": "reference_32_16", "hidden": [32, 16], "learning_rate": 0.001},
    {"name": "wide_64_32", "hidden": [64, 32], "learning_rate": 0.001},
    {"name": "large_128_64", "hidden": [128, 64], "learning_rate": 0.001, "dropout_candidates": [0.0, 0.1, 0.2, 0.3, 0.5]},
]
DROPOUT_CANDIDATES = [0.2, 0.3, 0.5]
REFERENCE_RUN = ROOT / "results" / "development" / "numpy_experiments" / "20261109_143702_820846"


def save_dropout_plot(result_folder):
    """Show the complete five-probability sweep at the same learning rate."""
    with (result_folder / "combined_experiment_summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    figure, axes = run_experiments.plt.subplots(1, 3, figsize=(16, 5), layout="constrained")

    for i, target in enumerate(run_experiments.TARGETS):
        for config in CONFIGURATIONS:
            points = []

            for row in rows:
                if(row["configuration"] == config["name"] and row["target"] == target):
                    points.append((float(row["dropout"]), float(row["dev_rmse"])))

            points.sort()
            probabilities = []
            errors = []

            for probability, error in points:
                probabilities.append(probability)
                errors.append(error)

            axes[i].plot(probabilities, errors, marker="o", label=str(config["hidden"]))

        axes[i].set_title(target.upper())
        axes[i].set_xlabel("Drop probability")
        axes[i].set_ylabel("Development RMSE (lower is better)")
        axes[i].set_xticks([0.0, 0.1, 0.2, 0.3, 0.5])
        axes[i].legend(title="Hidden layers")

    figure.suptitle("Network width vs dropout | learning rate 0.001 | seed 42")
    figure.savefig(result_folder / "dropout_vs_width.png", dpi=160)
    run_experiments.plt.close(figure)


if(__name__ == "__main__"):
    result_folder = run_experiments.main(configurations=CONFIGURATIONS, dropout_candidates=DROPOUT_CANDIDATES, reference_run=REFERENCE_RUN, extra_sources=["run_dropout_experiments.py"])
    save_dropout_plot(result_folder)
