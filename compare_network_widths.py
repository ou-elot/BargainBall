"""Compare hidden-layer widths at the best depths from the depth experiment.

Uses existing train/development arrays only; no test arrays are accessed.
Run: python compare_network_widths.py
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
TARGET_CONFIGS = {
    "pie": [[16, 16, 16], [32, 32, 32], [64, 64, 64]],
    "ws_48": [[16], [32], [64], [128]],
    "bpm": [[16], [32], [64], [128]],
    "minutes": [[16], [32], [64], [128]],
}
LEARNING_RATES = {"pie": 0.001, "ws_48": 0.001, "bpm": 0.003, "minutes": 0.001}
DROPOUT = 0.1
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_root = ROOT / "models" / "numpy_width_experiments" / run_id
    result_root = ROOT / "results" / "development" / "numpy_width_experiments" / run_id
    result_root.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / "data" / "processed" / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    settings = {
        "question": "Does hidden-layer width affect development performance at the best observed depths?",
        "hidden_widths_by_target": TARGET_CONFIGS,
        "depth_basis": "Prior single-seed depth experiment: 3 layers for PIE, 1 layer for WS/48, BPM, and minutes.",
        "learning_rates": LEARNING_RATES,
        "dropout_probability": DROPOUT,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "selection": "Best checkpoint selected by development RMSE within each fixed-width run",
        "split": "Existing train/development split; no test arrays accessed",
        "model_data_sha256": data_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "interpretation_limit": "One random seed; exploratory comparison, not a multi-seed estimate.",
    }
    (result_root / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    rows = []

    for target, configs in TARGET_CONFIGS.items():
        for hidden in configs:
            name = target + "_" + "x".join(map(str, hidden))
            args = SimpleNamespace(
                dataset="rates",
                data_dir=ROOT / "data" / "processed",
                output=model_root / name,
                hidden=hidden,
                dropout=[DROPOUT],
                epochs=MAX_EPOCHS,
                patience=PATIENCE,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATES[target],
                weight_decay=WEIGHT_DECAY,
                seed=SEED,
            )
            print("\nWIDTH EXPERIMENT: " + target.upper() + " | hidden units " + str(hidden), flush=True)
            train_target(target, args)
            model_dir = args.output / "rates" / target
            report = json.loads((model_dir / "development_results.json").read_text(encoding="utf-8"))
            selected = report["selected"]
            layer_dims = report["layer_dims"]
            parameter_count = sum(layer_dims[i] * layer_dims[i - 1] + layer_dims[i] for i in range(1, len(layer_dims)))
            rows.append({
                "target": target,
                "hidden_layers": len(hidden),
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

    summary = result_root / "width_comparison.csv"

    with summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nRESULTS: " + str(summary), flush=True)

    for target, configs in TARGET_CONFIGS.items():
        target_rows = [row for row in rows if row["target"] == target]
        winner = min(target_rows, key=lambda row: row["dev_rmse"])
        summaries = [row["hidden_widths"] + " units: RMSE " + format(row["dev_rmse"], ".6f") for row in target_rows]
        print(target.upper() + " | " + " | ".join(summaries) + " | lowest: " + winner["hidden_widths"], flush=True)


if __name__ == "__main__":
    main()
