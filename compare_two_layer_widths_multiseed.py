"""Compare two-hidden-layer widths across five seeds, train/dev only.

Existing seed-42..46 results for the current per-target widths are reused;
other width candidates are newly trained with identical fixed settings.
Run: python compare_two_layer_widths_multiseed.py
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
from train_numpy import train_target, load_target_arrays
from numpy_network import load_model, predict
from numpy_pipeline import FOUR_TARGETS, combine_scores, load_bundle

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46]
CURRENT_RUN = "20261118_153455_718940"
CURRENT_CANDIDATE = {"pie": "current_32_16", "ws_48": "current_128_64", "bpm": "current_32_16", "minutes": "current_64_32"}
WIDTHS = {
    "pie": {"narrow_16_8": [16, 8], "current_32_16": [32, 16], "wide_64_32": [64, 32], "large_128_64": [128, 64]},
    "ws_48": {"narrow_16_8": [16, 8], "medium_32_16": [32, 16], "wide_64_32": [64, 32], "current_128_64": [128, 64]},
    "bpm": {"narrow_16_8": [16, 8], "current_32_16": [32, 16], "wide_64_32": [64, 32], "large_128_64": [128, 64]},
    "minutes": {"narrow_16_8": [16, 8], "medium_32_16": [32, 16], "current_64_32": [64, 32], "large_128_64": [128, 64]},
}
LEARNING_RATES = {"pie": 0.001, "ws_48": 0.001, "bpm": 0.003, "minutes": 0.001}
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001
PCA_BUNDLE = ROOT / "models" / "numpy_four_stat" / "20261109_204731_573737"


def run_train(target, candidate, hidden, seed, model_root):
    args = SimpleNamespace(
        dataset="rates",
        data_dir=ROOT / "data" / "processed",
        output=model_root / target / candidate / ("seed_" + str(seed)),
        hidden=hidden,
        dropout=[DROPOUT],
        epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATES[target],
        weight_decay=WEIGHT_DECAY,
        seed=seed,
    )
    train_target(target, args)
    return args.output / "rates" / target


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_root = ROOT / "models" / "numpy_two_layer_width_experiments" / run_id
    result_root = ROOT / "results" / "development" / "numpy_two_layer_width_experiments" / run_id
    result_root.mkdir(parents=True, exist_ok=False)
    data_path = ROOT / "data" / "processed" / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    reference_manifest, unused_models, pca = load_bundle(PCA_BUNDLE)

    if(reference_manifest["model_data_sha256"] != data_hash):
        raise ValueError("PCA bundle and data arrays differ.")

    settings = {
        "question": "How do two-hidden-layer widths affect per-target and composite development RMSE across seeds?",
        "widths_by_target": WIDTHS,
        "current_candidate_reuse": CURRENT_CANDIDATE,
        "seeds": SEEDS,
        "learning_rates": LEARNING_RATES,
        "dropout_probability": DROPOUT,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "split": "Existing train/development split; test arrays are not accessed.",
        "composite": "Apply frozen four-stat PCA fitted on actual training targets only.",
        "selection": "Per-target candidate selection by mean development RMSE across seeds; report composite for that bundle and current bundle.",
        "model_data_sha256": data_hash,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    (result_root / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    rows = []
    checkpoint_map = {}
    reusable_root = ROOT / "models" / "numpy_multiseed_experiments" / CURRENT_RUN

    for target in FOUR_TARGETS:
        for candidate, hidden in WIDTHS[target].items():
            for seed in SEEDS:
                if(candidate == CURRENT_CANDIDATE[target]):
                    model_dir = reusable_root / target / candidate / ("seed_" + str(seed)) / "rates" / target
                else:
                    print("\nTWO-LAYER WIDTH EXPERIMENT: " + target.upper() + " | " + candidate + " | seed " + str(seed), flush=True)
                    model_dir = run_train(target, candidate, hidden, seed, model_root)

                report_path = model_dir / "development_results.json"
                checkpoint_path = model_dir / "best_model.npz"

                if(not report_path.is_file() or not checkpoint_path.is_file()):
                    raise FileNotFoundError("Missing reusable/current width result: " + str(model_dir))

                report = json.loads(report_path.read_text(encoding="utf-8"))
                selected = report["selected"]
                layer_dims = report["layer_dims"]
                if(report["model_data_sha256"] != data_hash or report["seed"] != seed):
                    raise ValueError("Candidate results do not match requested data or seed.")
                parameter_count = sum(layer_dims[i] * layer_dims[i - 1] + layer_dims[i] for i in range(1, len(layer_dims)))
                rows.append({
                    "target": target,
                    "candidate": candidate,
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
                    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "model_data_sha256": data_hash,
                })
                checkpoint_map[(target, candidate, seed)] = checkpoint_path

    per_seed_csv = result_root / "per_seed_results.csv"

    with per_seed_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []

    for target in FOUR_TARGETS:
        for candidate in WIDTHS[target]:
            candidate_rows = [row for row in rows if row["target"] == target and row["candidate"] == candidate]
            values = np.array([row["dev_rmse"] for row in candidate_rows], dtype=np.float64)
            summaries.append({
                "target": target,
                "candidate": candidate,
                "hidden_widths": candidate_rows[0]["hidden_widths"],
                "parameter_count": candidate_rows[0]["parameter_count"],
                "seed_count": len(candidate_rows),
                "dev_rmse_mean": float(values.mean()),
                "dev_rmse_sd": float(values.std(ddof=1)),
                "dev_rmse_min": float(values.min()),
                "dev_rmse_max": float(values.max()),
                "baseline_dev_rmse": candidate_rows[0]["baseline_dev_rmse"],
            })

    summary_csv = result_root / "width_summary.csv"

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    best_by_target = {}
    for target in FOUR_TARGETS:
        options = [row for row in summaries if row["target"] == target]
        best_by_target[target] = min(options, key=lambda row: row["dev_rmse_mean"])["candidate"]

    with np.load(data_path, allow_pickle=False) as data:
        X_dev = data["X_dev"].astype(np.float32)
        mask = data["rate_mask_dev"].astype(bool)
        actual_columns = []
        baseline_columns = []

        for target in FOUR_TARGETS:
            actual, baseline = load_target_arrays(data, target, "dev")
            actual_columns.append(actual.astype(np.float64))
            baseline_columns.append(baseline.astype(np.float64))
            mask &= np.isfinite(actual) & np.isfinite(baseline)

    actual_matrix = np.column_stack(actual_columns)[mask]
    baseline_matrix = np.column_stack(baseline_columns)[mask]
    baseline_score = combine_scores(baseline_matrix, pca)
    composite_rows = []

    for seed in SEEDS:
        for bundle_name, candidates in [("current_widths", CURRENT_CANDIDATE), ("best_mean_target_widths", best_by_target)]:
            prediction_columns = []

            for target in FOUR_TARGETS:
                checkpoint = checkpoint_map[(target, candidates[target], seed)]
                parameters, metadata = load_model(checkpoint)
                prediction = predict(X_dev.T, parameters, metadata["target_mean"], metadata["target_scale"], metadata.get("prediction_minimum"))
                prediction_columns.append(prediction.astype(np.float64))

            prediction_matrix = np.column_stack(prediction_columns)[mask]
            predicted_score = combine_scores(prediction_matrix, pca)
            composite_rows.append({
                "bundle": bundle_name,
                "seed": seed,
                "candidate_widths": json.dumps(candidates, sort_keys=True),
                "examples": int(mask.sum()),
                "composite_rmse": float(np.sqrt(np.mean((predicted_score - combine_scores(actual_matrix, pca)) ** 2))),
                "baseline_rmse": float(np.sqrt(np.mean((baseline_score - combine_scores(actual_matrix, pca)) ** 2))),
                "composite_mae": float(np.mean(np.abs(predicted_score - combine_scores(actual_matrix, pca)))),
                "baseline_mae": float(np.mean(np.abs(baseline_score - combine_scores(actual_matrix, pca)))),
                "test_evaluated": False,
            })

    composite_csv = result_root / "composite_by_seed.csv"

    with composite_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(composite_rows[0]))
        writer.writeheader()
        writer.writerows(composite_rows)

    lines = ["# Two-layer width comparison", "", "Five seeds, existing train/development split; no test data accessed. Per-target candidates are selected by mean development RMSE. Composite errors use the existing frozen training-only PCA.", "", "## Per-target results", "", "| Target | Widths | Mean dev RMSE | SD |", "|---|---:|---:|---:|"]

    for target in FOUR_TARGETS:
        candidates = [row for row in summaries if row["target"] == target]
        candidates.sort(key=lambda row: row["dev_rmse_mean"])
        for row in candidates:
            lines.append("| " + target + " | " + row["hidden_widths"] + " | " + format(row["dev_rmse_mean"], ".6f") + " | " + format(row["dev_rmse_sd"], ".6f") + " |")

    lines.extend(["", "## Composite comparison", "", "| Candidate bundle | Mean dev composite RMSE | SD |", "|---|---:|---:|"])

    for bundle_name in ["current_widths", "best_mean_target_widths"]:
        values = np.array([row["composite_rmse"] for row in composite_rows if row["bundle"] == bundle_name], dtype=np.float64)
        lines.append("| " + bundle_name + " | " + format(values.mean(), ".6f") + " | " + format(values.std(ddof=1), ".6f") + " |")
    lines.extend(["", "Best mean-development-RMSE widths by target: " + json.dumps(best_by_target, sort_keys=True) + ".", "", "These are development model-selection results; they are not an independent test estimate."])
    (result_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nRESULTS: " + str(result_root), flush=True)
    print("Best widths: " + json.dumps(best_by_target, sort_keys=True), flush=True)

    for bundle_name in ["current_widths", "best_mean_target_widths"]:
        values = np.array([row["composite_rmse"] for row in composite_rows if row["bundle"] == bundle_name], dtype=np.float64)
        print(bundle_name + " composite RMSE | mean " + format(values.mean(), ".6f") + " | SD " + format(values.std(ddof=1), ".6f"), flush=True)


if __name__ == "__main__":
    main()
