"""Run a focused NumPy hyperparameter comparison on the training/dev split.

Edit the settings below, then run: python run_experiments.py
Each invocation creates a new timestamped results and model directory.
"""

import os

# Keep small matrix operations on one CPU thread for repeatable, efficient runs.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_numpy import train_target
from numpy_network import load_model, predict

ROOT = Path(__file__).resolve().parent

# HYPERPARAMETERS: compare one change at a time against the 32/16 reference.
CONFIGURATIONS = [
    {"name": "tiny_4_2", "hidden": [4, 2], "learning_rate": 0.001},
    {"name": "small_16_8", "hidden": [16, 8], "learning_rate": 0.001},
    {"name": "reference_32_16", "hidden": [32, 16], "learning_rate": 0.001},
    {"name": "wide_64_32", "hidden": [64, 32], "learning_rate": 0.001},
    {"name": "slow_32_16", "hidden": [32, 16], "learning_rate": 0.0003},
    {"name": "fast_32_16", "hidden": [32, 16], "learning_rate": 0.003},
]
TARGETS = ["pie", "ws_48", "bpm"]
DROPOUT_CANDIDATES = [0.0, 0.1]
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001
SEED = 42


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_checkpoint(folder, report):
    """Recalculate saved development predictions after loading each model."""
    parameters, metadata = load_model(folder / "best_model.npz")

    with np.load(ROOT / "data" / "processed" / "model_data_rates.npz", allow_pickle=False) as data:
        X = data["X_dev"].astype(np.float32)

        if(report["target"] in ["ws_48", "bpm"]):
            X = X[data["rate_mask_dev"].astype(bool)]

    predictions = predict(X.T, parameters, metadata["target_mean"], metadata["target_scale"])

    with np.load(folder / "development_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(predictions, saved["predicted"], rtol=1e-6, atol=1e-7)
        rmse = float(np.sqrt(np.mean((predictions.astype(np.float64) - saved["actual"].astype(np.float64)) ** 2)))
        np.testing.assert_allclose(rmse, report["selected"]["dev_rmse"], rtol=1e-10)


def save_plots(result_folder, rows, winners):
    figure, axes = plt.subplots(1, len(TARGETS), figsize=(17, 5), layout="constrained")
    configurations = []
    dropout_candidates = []

    for row in rows:
        if(row["configuration"] not in configurations):
            configurations.append(row["configuration"])
        if(row["dropout"] not in dropout_candidates):
            dropout_candidates.append(row["dropout"])

    dropout_candidates.sort()
    width = 0.8 / len(dropout_candidates)

    for i, target in enumerate(TARGETS):
        axis = axes[i]
        target_rows = []

        for row in rows:
            if(row["target"] == target):
                target_rows.append(row)

        positions = np.arange(len(configurations))

        for j, dropout in enumerate(dropout_candidates):
            values = []

            for name in configurations:
                value = np.nan

                for row in target_rows:
                    if(row["configuration"] == name and row["dropout"] == dropout):
                        value = row["dev_rmse"]

                values.append(value)

            axis.bar(positions + (j - (len(dropout_candidates) - 1) / 2) * width, values, width=width, label="Dropout " + str(dropout))

        axis.axhline(target_rows[0]["baseline_dev_rmse"], color="black", linestyle="--", label="Previous-season baseline")
        axis.set_xticks(positions)
        axis.set_xticklabels(configurations, rotation=40, ha="right", fontsize=8)
        axis.set_title(target.upper())
        axis.set_ylabel("Development RMSE (lower is better)")
        axis.legend(fontsize=8)

    figure.savefig(result_folder / "hyperparameter_comparison.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, len(TARGETS), figsize=(17, 5), layout="constrained")

    for i, winner in enumerate(winners):
        history_path = Path(winner["checkpoint_path"]).parent / "training_history.csv"
        epochs = []
        training_mse = []
        development_mse = []

        with history_path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if(float(row["dropout"]) == winner["dropout"]):
                    epochs.append(int(row["epoch"]))
                    training_mse.append(float(row["train_mse"]))
                    development_mse.append(float(row["dev_mse"]))

        axis = axes[i]
        axis.plot(epochs, training_mse, label="Training MSE")
        axis.plot(epochs, development_mse, label="Development MSE")
        axis.axvline(winner["best_epoch"], color="black", linestyle="--", label="Selected epoch")
        axis.set_yscale("log")
        axis.set_title(winner["target"].upper() + " | " + winner["configuration"] + " | dropout " + str(winner["dropout"]))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE, original target units (log scale)")
        axis.legend(fontsize=8)

    figure.savefig(result_folder / "selected_learning_curves.png", dpi=160)
    plt.close(figure)


def load_reference_rows(reference_run):
    """Reuse completed fits instead of retraining the same seed and settings."""
    rows = []
    float_columns = ["learning_rate", "dropout", "weight_decay", "train_mse", "train_rmse", "dev_mse", "dev_rmse", "dev_mae", "baseline_dev_rmse", "rmse_improvement_percent"]
    int_columns = ["parameters", "batch_size", "seed", "best_epoch", "epochs_run", "training_examples", "development_examples"]
    expected_hash = hashlib.sha256((ROOT / "data" / "processed" / "model_data_rates.npz").read_bytes()).hexdigest()

    with (reference_run / "experiment_summary.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for key in float_columns:
                row[key] = float(row[key])

            for key in int_columns:
                row[key] = int(row[key])

            row["reached_epoch_limit"] = row["reached_epoch_limit"] == "True"

            if(row["model_data_sha256"] != expected_hash or row["seed"] != SEED or row["batch_size"] != BATCH_SIZE or row["weight_decay"] != WEIGHT_DECAY):
                raise ValueError("Reference results must use the same arrays, seed, batch size, and weight decay.")

            rows.append(row)

    return rows


def main(configurations=None, dropout_candidates=None, reference_run=None, extra_sources=None):
    if(configurations is None):
        configurations = CONFIGURATIONS
    if(dropout_candidates is None):
        dropout_candidates = DROPOUT_CANDIDATES

    reference_rows = []

    if(reference_run is not None):
        reference_rows = load_reference_rows(reference_run)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_folder = ROOT / "models" / "numpy_experiments" / run_id
    result_folder = ROOT / "results" / "development" / "numpy_experiments" / run_id
    model_folder.mkdir(parents=True, exist_ok=False)
    result_folder.mkdir(parents=True, exist_ok=False)
    settings = {"configurations": configurations, "targets": TARGETS, "dropout_candidates": dropout_candidates, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "weight_decay": WEIGHT_DECAY, "seed": SEED, "dataset": "rates", "python_version": platform.python_version(), "numpy_version": np.__version__, "cpu_threads": 1, "reference_run": str(reference_run) if reference_run is not None else None, "reused_candidates": len(reference_rows), "selection": "Lowest development RMSE per target; one seed, exploratory comparison", "loss_reporting": "Unpenalized train/dev MSE at fixed epoch-end weights, dropout disabled; checkpoint scores use the best dev epoch", "test_evaluated": False}
    (result_folder / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    snapshot_folder = result_folder / "source_snapshots"
    snapshot_folder.mkdir()
    hashes = {}

    source_files = ["numpy_network.py", "train_numpy.py", "run_experiments.py"]

    if(extra_sources is not None):
        source_files.extend(extra_sources)

    for name in source_files:
        source = (ROOT / name).read_bytes()
        (snapshot_folder / name).write_bytes(source)
        hashes[name] = hashlib.sha256(source).hexdigest()

    (result_folder / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    rows = []

    for config in configurations:
        for target in TARGETS:
            args = SimpleNamespace(dataset="rates", data_dir=ROOT / "data" / "processed", output=model_folder / config["name"], hidden=config["hidden"], dropout=config.get("dropout_candidates", dropout_candidates), epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, learning_rate=config["learning_rate"], weight_decay=WEIGHT_DECAY, seed=SEED)
            print ("\nEXPERIMENT: " + config["name"] + " | target: " + target + " | learning rate: " + str(args.learning_rate), flush=True)
            train_target(target, args)
            folder = args.output / "rates" / target
            report = json.loads((folder / "development_results.json").read_text(encoding="utf-8"))
            verify_checkpoint(folder, report)
            parameter_count = 0
            dims = report["layer_dims"]

            for l in range(1, len(dims)):
                parameter_count = parameter_count + dims[l] * dims[l - 1] + dims[l]

            for run in report["runs"]:
                checkpoint_path = ""

                if(run["dropout"] == report["selected"]["dropout"]):
                    checkpoint_path = str(folder / "best_model.npz")

                row = {"configuration": config["name"], "target": target, "hidden_layers": "-".join(map(str, config["hidden"])), "parameters": parameter_count, "learning_rate": config["learning_rate"], "dropout": run["dropout"], "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "seed": SEED, "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"], "train_mse": run["train_mse"], "train_rmse": run["train_rmse"], "dev_mse": run["dev_mse"], "dev_rmse": run["dev_rmse"], "dev_mae": run["dev_mae"], "baseline_dev_rmse": report["baseline_dev_rmse"], "rmse_improvement_percent": 100.0 * (1.0 - run["dev_rmse"] / report["baseline_dev_rmse"]), "reached_epoch_limit": run["reached_epoch_limit"], "training_examples": report["training_examples"], "development_examples": report["development_examples"], "model_data_sha256": report["model_data_sha256"], "checkpoint_path": checkpoint_path}
                rows.append(row)

            save_csv(result_folder / "experiment_summary.csv", rows)

    winners = []
    comparison_rows = reference_rows + rows

    if(reference_rows):
        save_csv(result_folder / "combined_experiment_summary.csv", comparison_rows)

    for target in TARGETS:
        candidates = []

        for row in comparison_rows:
            if(row["target"] == target):
                candidates.append(row)

        candidates.sort(key=lambda row: row["dev_rmse"])
        winners.append(candidates[0])

    save_csv(result_folder / "best_by_target.csv", winners)
    save_plots(result_folder, comparison_rows, winners)
    print ("\nEXPERIMENTS COMPLETE. Results: " + str(result_folder), flush=True)

    for winner in winners:
        print (winner["target"].upper() + " | " + winner["configuration"] + " | dropout " + str(winner["dropout"]) + " | train MSE " + str(winner["train_mse"]) + " | dev MSE " + str(winner["dev_mse"]) + " | dev RMSE " + str(winner["dev_rmse"]), flush=True)

    return result_folder


if(__name__ == "__main__"):
    main()
