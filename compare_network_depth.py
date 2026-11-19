"""Compare one, two, and three hidden layers on train/development data only.

Each hidden layer has 32 units. Other settings are fixed within each target;
target-specific learning rates match the selected four-stat models.
Run: python compare_network_depth.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import csv
import hashlib
import json
import platform
import numpy as np
from train_numpy import train_target

ROOT = Path(__file__).resolve().parent
TARGETS = ["pie", "ws_48", "bpm", "minutes"]
LEARNING_RATES = {"pie": 0.001, "ws_48": 0.001, "bpm": 0.003, "minutes": 0.001}
DEPTHS = {
    str(depth) + "_hidden_layers": [32] * depth
    for depth in range(1, 11)
}
DROPOUT = 0.1
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_root = ROOT / "models" / "numpy_depth_experiments" / run_id
    result_root = ROOT / "results" / "development" / "numpy_depth_experiments" / run_id
    result_root.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / "data" / "processed" / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    settings = {
        "question": "Does hidden-layer depth affect development performance?",
        "depths_and_widths": DEPTHS,
        "targets": TARGETS,
        "learning_rates": LEARNING_RATES,
        "dropout_probability": DROPOUT,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "selection": "Best checkpoint selected by development RMSE within each fixed-depth run",
        "split": "Existing train/development split; no test arrays accessed",
        "model_data_sha256": data_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "interpretation_limit": "One random seed; exploratory comparison, not a multi-seed estimate.",
    }
    (result_root / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    rows = []

    for depth_name, hidden in DEPTHS.items():
        for target in TARGETS:
            args = SimpleNamespace(
                dataset="rates",
                data_dir=ROOT / "data" / "processed",
                output=model_root / depth_name / target,
                hidden=hidden,
                dropout=[DROPOUT],
                epochs=MAX_EPOCHS,
                patience=PATIENCE,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATES[target],
                weight_decay=WEIGHT_DECAY,
                seed=SEED,
            )
            print("\nDEPTH EXPERIMENT: " + target.upper() + " | " + depth_name + " | " + str(hidden), flush=True)
            train_target(target, args)
            model_dir = args.output / "rates" / target
            report = json.loads((model_dir / "development_results.json").read_text(encoding="utf-8"))
            selected = report["selected"]
            layer_dims = report["layer_dims"]
            parameter_count = sum(layer_dims[i] * layer_dims[i - 1] + layer_dims[i] for i in range(1, len(layer_dims)))
            rows.append({
                "target": target,
                "depth": len(hidden),
                "hidden_widths": "-".join(map(str, hidden)),
                "parameter_count": parameter_count,
                "learning_rate": LEARNING_RATES[target],
                "dropout": DROPOUT,
                "seed": SEED,
                "best_epoch": selected["best_epoch"],
                "epochs_run": selected["epochs_run"],
                "train_rmse": selected["train_rmse"],
                "dev_rmse": selected["dev_rmse"],
                "dev_mae": selected["dev_mae"],
                "baseline_dev_rmse": report["baseline_dev_rmse"],
                "baseline_dev_mae": report["baseline_dev_mae"],
                "checkpoint": str((model_dir / "best_model.npz").relative_to(ROOT)),
                "model_data_sha256": report["model_data_sha256"],
                "test_evaluated": False,
            })

    summary = result_root / "depth_comparison.csv"

    with summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nRESULTS: " + str(summary), flush=True)

    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        winner = min(target_rows, key=lambda row: row["dev_rmse"])
        print(target.upper() + " | " + " | ".join(
            str(row["depth"]) + " layer(s): RMSE " + format(row["dev_rmse"], ".6f")
            for row in target_rows
        ) + " | lowest: " + str(winner["depth"]) + " layer(s)", flush=True)


if __name__ == "__main__":
    main()
