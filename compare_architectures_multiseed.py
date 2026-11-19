"""Compare selected architecture candidates across random seeds.

This is a train/development-only stability check. The test split is never read.
Run: python compare_architectures_multiseed.py
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
TARGETS = {
    "pie": {
        "learning_rate": 0.001,
        "candidates": {
            "current_32_16": [32, 16],
            "single_32": [32],
            "three_layer_32": [32, 32, 32],
        },
    },
    "ws_48": {
        "learning_rate": 0.001,
        "candidates": {
            "current_128_64": [128, 64],
            "single_16": [16],
            "single_32": [32],
        },
    },
    "bpm": {
        "learning_rate": 0.003,
        "candidates": {
            "current_32_16": [32, 16],
            "single_16": [16],
            "single_64": [64],
        },
    },
    "minutes": {
        "learning_rate": 0.001,
        "candidates": {
            "current_64_32": [64, 32],
            "single_32": [32],
            "single_64": [64],
        },
    },
}
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_root = ROOT / "models" / "numpy_multiseed_experiments" / run_id
    result_root = ROOT / "results" / "development" / "numpy_multiseed_experiments" / run_id
    result_root.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / "data" / "processed" / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    settings = {
        "question": "Are the leading architecture choices stable across random seeds?",
        "seeds": SEEDS,
        "targets_and_architectures": TARGETS,
        "dropout_probability": DROPOUT,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "selection": "Lowest development RMSE per target and candidate; report per-seed values and mean/sample SD.",
        "split": "Existing train/development split; test arrays are not accessed.",
        "model_data_sha256": data_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "warning": "Development RMSE is used for early stopping and architecture comparison; results are exploratory, not an independent test estimate.",
    }
    (result_root / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    rows = []

    for target, target_config in TARGETS.items():
        learning_rate = target_config["learning_rate"]

        for candidate, hidden in target_config["candidates"].items():
            for seed in SEEDS:
                name = target + "_" + candidate + "_seed_" + str(seed)
                args = SimpleNamespace(
                    dataset="rates",
                    data_dir=ROOT / "data" / "processed",
                    output=model_root / target / candidate / ("seed_" + str(seed)),
                    hidden=hidden,
                    dropout=[DROPOUT],
                    epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                    batch_size=BATCH_SIZE,
                    learning_rate=learning_rate,
                    weight_decay=WEIGHT_DECAY,
                    seed=seed,
                )
                print("\nMULTI-SEED EXPERIMENT: " + name + " | hidden " + str(hidden), flush=True)
                train_target(target, args)
                model_dir = args.output / "rates" / target
                report = json.loads((model_dir / "development_results.json").read_text(encoding="utf-8"))
                selected = report["selected"]
                layer_dims = report["layer_dims"]
                parameter_count = sum(layer_dims[i] * layer_dims[i - 1] + layer_dims[i] for i in range(1, len(layer_dims)))
                rows.append({
                    "target": target,
                    "candidate": candidate,
                    "hidden_layers": len(hidden),
                    "hidden_widths": "-".join(map(str, hidden)),
                    "parameter_count": parameter_count,
                    "seed": seed,
                    "learning_rate": learning_rate,
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

    run_csv = result_root / "per_seed_results.csv"

    with run_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []

    for target, target_config in TARGETS.items():
        for candidate in target_config["candidates"]:
            selected_rows = [row for row in rows if row["target"] == target and row["candidate"] == candidate]
            dev_rmse = np.array([row["dev_rmse"] for row in selected_rows], dtype=np.float64)
            dev_mae = np.array([row["dev_mae"] for row in selected_rows], dtype=np.float64)
            summary.append({
                "target": target,
                "candidate": candidate,
                "hidden_layers": selected_rows[0]["hidden_layers"],
                "hidden_widths": selected_rows[0]["hidden_widths"],
                "parameter_count": selected_rows[0]["parameter_count"],
                "seed_count": len(selected_rows),
                "dev_rmse_mean": float(dev_rmse.mean()),
                "dev_rmse_sd": float(dev_rmse.std(ddof=1)),
                "dev_rmse_min": float(dev_rmse.min()),
                "dev_rmse_max": float(dev_rmse.max()),
                "dev_mae_mean": float(dev_mae.mean()),
                "dev_mae_sd": float(dev_mae.std(ddof=1)),
                "baseline_dev_rmse": selected_rows[0]["baseline_dev_rmse"],
                "baseline_dev_mae": selected_rows[0]["baseline_dev_mae"],
            })

    summary_csv = result_root / "architecture_summary.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    report_lines = ["# Multi-seed architecture comparison", "", "All comparisons use the same existing train/development split. No test data were accessed.", "Development RMSE selects checkpoints and compares candidates; these are model-selection results, not an independent performance estimate.", "", "| Target | Candidate | Hidden widths | Mean development RMSE | SD | Mean development MAE |", "|---|---|---:|---:|---:|---:|"]

    for row in summary:
        report_lines.append("| " + row["target"] + " | " + row["candidate"] + " | " + row["hidden_widths"] + " | " + format(row["dev_rmse_mean"], ".6f") + " | " + format(row["dev_rmse_sd"], ".6f") + " | " + format(row["dev_mae_mean"], ".6f") + " |")

    report_lines.append("")

    for target in TARGETS:
        candidates = [row for row in summary if row["target"] == target]
        candidates.sort(key=lambda row: row["dev_rmse_mean"])
        winner = candidates[0]
        report_lines.append("For " + target + ", lowest mean development RMSE was " + winner["candidate"] + " (" + format(winner["dev_rmse_mean"], ".6f") + " ± " + format(winner["dev_rmse_sd"], ".6f") + ").")

    (result_root / "RESULTS.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\nRESULTS: " + str(result_root), flush=True)

    for target in TARGETS:
        candidates = [row for row in summary if row["target"] == target]
        candidates.sort(key=lambda row: row["dev_rmse_mean"])
        print(target.upper() + " | " + " | ".join(row["candidate"] + ": " + format(row["dev_rmse_mean"], ".6f") + " +/- " + format(row["dev_rmse_sd"], ".6f") for row in candidates), flush=True)


if __name__ == "__main__":
    main()
