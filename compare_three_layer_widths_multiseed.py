"""Compare three-hidden-layer widths across five seeds on train/dev only.

Run: python compare_three_layer_widths_multiseed.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import csv
import hashlib
import json
import platform
import numpy as np
from train_numpy import train_target

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46]
WIDTHS = {
    "narrow_16_8_4": [16, 8, 4],
    "medium_32_16_8": [32, 16, 8],
    "equal_32_32_32": [32, 32, 32],
    "wide_64_32_16": [64, 32, 16],
}
LEARNING_RATES = {"pie": 0.001, "ws_48": 0.001, "bpm": 0.003, "minutes": 0.001}
TARGETS = list(LEARNING_RATES)
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_root = ROOT / "models" / "numpy_three_layer_width_experiments" / run_id
    result_root = ROOT / "results" / "development" / "numpy_three_layer_width_experiments" / run_id
    result_root.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / "data" / "processed" / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    settings = {
        "question": "How do width schedules affect three-hidden-layer models?",
        "width_configurations": WIDTHS,
        "targets": TARGETS,
        "seeds": SEEDS,
        "learning_rates": LEARNING_RATES,
        "dropout_probability": DROPOUT,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "selection": "Best checkpoint selected by development RMSE in each run; report mean and sample SD across seeds.",
        "split": "Existing train/development split; test arrays are not accessed.",
        "model_data_sha256": data_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "interpretation_limit": "Development model-selection comparison; not an independent test estimate.",
    }
    (result_root / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    rows = []

    for target in TARGETS:
        for configuration, hidden in WIDTHS.items():
            for seed in SEEDS:
                args = SimpleNamespace(
                    dataset="rates",
                    data_dir=ROOT / "data" / "processed",
                    output=model_root / target / configuration / ("seed_" + str(seed)),
                    hidden=hidden,
                    dropout=[DROPOUT],
                    epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    learning_rate=LEARNING_RATES[target],
                    weight_decay=WEIGHT_DECAY,
                    seed=seed,
                )
                print("\nTHREE-LAYER WIDTH EXPERIMENT: " + target.upper() + " | " + configuration + " | seed " + str(seed), flush=True)
                train_target(target, args)
                model_dir = args.output / "rates" / target
                report = json.loads((model_dir / "development_results.json").read_text(encoding="utf-8"))
                selected = report["selected"]
                layer_dims = report["layer_dims"]
                parameter_count = sum(layer_dims[i] * layer_dims[i - 1] + layer_dims[i] for i in range(1, len(layer_dims)))
                rows.append({
                    "target": target,
                    "configuration": configuration,
                    "hidden_layers": len(hidden),
                    "hidden_widths": "-".join(map(str, hidden)),
                    "parameter_count": parameter_count,
                    "seed": seed,
                    "learning_rate": LEARNING_RATES[target],
                    "dropout": DROPOUT,
                    "best_epoch": selected["best_epoch"],
                    "train_rmse": selected["train_rmse"],
                    "dev_rmse": selected["dev_rmse"],
                    "dev_mae": selected["dev_mae"],
                    "baseline_dev_rmse": report["baseline_dev_rmse"],
                    "baseline_dev_mae": report["baseline_dev_mae"],
                    "checkpoint": str((model_dir / "best_model.npz").relative_to(ROOT)),
                    "model_data_sha256": report["model_data_sha256"],
                    "test_evaluated": False,
                })

    per_seed_path = result_root / "per_seed_results.csv"

    with per_seed_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    for target in TARGETS:
        for configuration in WIDTHS:
            candidate_rows = [row for row in rows if row["target"] == target and row["configuration"] == configuration]
            rmse = np.array([row["dev_rmse"] for row in candidate_rows], dtype=np.float64)
            mae = np.array([row["dev_mae"] for row in candidate_rows], dtype=np.float64)
            summary.append({
                "target": target,
                "configuration": configuration,
                "hidden_widths": candidate_rows[0]["hidden_widths"],
                "parameter_count": candidate_rows[0]["parameter_count"],
                "seed_count": len(candidate_rows),
                "dev_rmse_mean": float(rmse.mean()),
                "dev_rmse_sd": float(rmse.std(ddof=1)),
                "dev_rmse_min": float(rmse.min()),
                "dev_rmse_max": float(rmse.max()),
                "dev_mae_mean": float(mae.mean()),
                "dev_mae_sd": float(mae.std(ddof=1)),
                "baseline_dev_rmse": candidate_rows[0]["baseline_dev_rmse"],
                "baseline_dev_mae": candidate_rows[0]["baseline_dev_mae"],
            })

    summary_path = result_root / "width_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = ["# Three-layer width comparison", "", "Five random seeds; existing train/development split; no test data accessed.", "Development RMSE selects each checkpoint and compares candidates; these are model-selection results, not an independent performance estimate.", "", "| Target | Widths | Mean development RMSE | SD | Mean development MAE |", "|---|---:|---:|---:|---:|"]

    for row in summary:
        lines.append("| " + row["target"] + " | " + row["hidden_widths"] + " | " + format(row["dev_rmse_mean"], ".6f") + " | " + format(row["dev_rmse_sd"], ".6f") + " | " + format(row["dev_mae_mean"], ".6f") + " |")

    lines.extend(["", "## Lowest mean development RMSE per target", ""])

    for target in TARGETS:
        candidates = [row for row in summary if row["target"] == target]
        winner = min(candidates, key=lambda row: row["dev_rmse_mean"])
        lines.append("- " + target + ": " + winner["hidden_widths"] + " units (" + format(winner["dev_rmse_mean"], ".6f") + " ± " + format(winner["dev_rmse_sd"], ".6f") + ")")

    (result_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nRESULTS: " + str(result_root), flush=True)

    for target in TARGETS:
        candidates = [row for row in summary if row["target"] == target]
        candidates.sort(key=lambda row: row["dev_rmse_mean"])
        print(target.upper() + " | " + " | ".join(row["hidden_widths"] + ": " + format(row["dev_rmse_mean"], ".6f") + " +/- " + format(row["dev_rmse_sd"], ".6f") for row in candidates), flush=True)


if __name__ == "__main__":
    main()
