"""Evaluate a fixed four-stat composite on development rows only.

Run: python evaluate_four_stat_development.py --bundle models/numpy_four_stat/ID
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train_numpy import load_target_arrays
from numpy_network import predict
from numpy_pipeline import FOUR_TARGETS, load_bundle, predict_bundle, combine_scores
from evaluate_final_models import calculate_metrics

ROOT = Path(__file__).resolve().parent


def save_plots(folder, bundle, manifest, table):
    history = pd.read_csv(bundle / "minutes" / "training_history.csv")
    selected = manifest["models"]["minutes"]
    history = history[history["dropout"] == selected["dropout"]]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(history["epoch"], history["train_rmse"], label="Training RMSE")
    axes[0].plot(history["epoch"], history["dev_rmse"], label="Development RMSE")
    axes[0].axvline(selected["best_epoch"], color="black", linestyle="--", label="Selected epoch")
    axes[0].set_title("Total-minute model learning curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE in season minutes")
    axes[0].legend()
    actual = table["actual_minutes"].to_numpy()
    axes[1].scatter(actual, table["baseline_minutes"], s=9, alpha=0.25, label="Prior-season minutes")
    axes[1].scatter(actual, table["predicted_minutes"], s=9, alpha=0.35, label="Predicted next-season minutes")
    high = float(max(actual.max(), table["baseline_minutes"].max(), table["predicted_minutes"].max()))
    axes[1].plot([0, high], [0, high], color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Development playing-time predictions")
    axes[1].set_xlabel("Actual next-season total minutes")
    axes[1].set_ylabel("Predicted minutes")
    axes[1].legend(fontsize=8)
    figure.savefig(folder / "minutes_development.png", dpi=160)
    plt.close(figure)


def save_report(folder, manifest, pca, metrics):
    lines = ["# Four-stat PCA: development evaluation", "", "The composite combines predicted PIE, WS/48, BPM, and next-season total minutes.", "PCA and its scaler use actual training targets only. Minute predictions use", "the same 22 prior-season inputs as the other networks, including age, games,", "and minutes/game. Observed next-season minutes are a target, not an input.", "", "## Models", "", "| Target | Layer dimensions | Learning rate | Drop probability | Best epoch |", "|---|---|---:|---:|---:|"]

    for target in FOUR_TARGETS:
        model = manifest["models"][target]
        lines.append("| " + target.upper() + " | " + str(model["layer_dims"]) + " | " + str(model["learning_rate"]) + " | " + str(model["dropout"]) + " | " + str(model["best_epoch"]) + " |")

    lines.extend(["", "## Training-fitted PCA weights", "", "All four targets are standardized before PCA, so minutes cannot dominate", "merely because its raw numerical units are larger.", "", "| Standardized target | PC1 coefficient |", "|---|---:|"])

    for target, weight in zip(FOUR_TARGETS, pca["component_weights"]):
        lines.append("| " + target.upper() + " | " + str(round(weight, 6)) + " |")

    lines.extend(["", "PC1 captures " + str(round(100.0 * pca["explained_variance_ratio"], 2)) + "% of standardized training-target variance on " + str(pca["training_examples"]) + " valid training rows. This is not prediction accuracy.", "", "## Development errors", "", "| Target | Examples | Model RMSE | Baseline RMSE | Model MAE | Baseline MAE | RMSE reduction |", "|---|---:|---:|---:|---:|---:|---:|"])

    for row in metrics:
        lines.append("| " + row["target"] + " | " + str(row["examples"]) + " | " + str(round(row["model_rmse"], 6)) + " | " + str(round(row["baseline_rmse"], 6)) + " | " + str(round(row["model_mae"], 6)) + " | " + str(round(row["baseline_mae"], 6)) + " | " + str(round(row["rmse_improvement_percent"], 2)) + "% |")

    lines.extend(["", "## Interpretation", "", "- The four-stat score mixes effectiveness with season-long playing time. It is not a pure per-minute ability measure or a salary-value score.", "- Prior-season total minutes provide the new target's baseline. Both composite baselines combine the corresponding prior-season statistics with the same training-fitted transformations.", "- The three-stat comparison uses the same effectiveness networks and the same valid development rows. The composites have different target definitions, so raw RMSE differences do not prove which definition is more useful.", "- The minutes model trains on all observed minute targets, including zero minutes; undefined rate targets remain excluded from joint PCA, without imputation.", "- The linear network learns standardized minutes using MSE. Decoded minutes are floored at zero during inference and development selection; no upper cap is imposed.", "- All player and contract types remain eligible, including rookie deals. Players without any observed immediately following NBA season are absent from the dataset, so this model does not predict league exit.", "- This is a development-only follow-up. The original test split was already examined for the earlier design and is not a fresh confirmation set for this revision.", "", "Exact scores and errors are in `metrics.csv` and `development_predictions.csv`.", ""])
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_bundle(bundle, output=None):
    bundle = Path(bundle)
    manifest, models, pca = load_bundle(bundle)

    if(manifest["target_order"] != FOUR_TARGETS):
        raise ValueError("This evaluator requires PIE / WS48 / BPM / total minutes.")

    data_path = ROOT / "data" / "processed" / "model_data_rates.npz"

    if(hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["model_data_sha256"]):
        raise ValueError("Evaluation arrays differ from the fitted model arrays.")

    if(output is None):
        output = ROOT / "results" / "development" / "numpy_four_stat" / (manifest["bundle_id"] + "_evaluation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        output.mkdir(parents=True, exist_ok=False)

    metrics = []
    rows = pd.read_csv(ROOT / "data" / "processed" / "dev_rows_rates.csv")
    table = rows[["player", "player_id", "season", "target_season"]].copy()
    actual_columns = []
    baseline_columns = []

    # Intentionally load development keys only, never test keys.
    with np.load(data_path, allow_pickle=False) as data:
        X = data["X_dev"].astype(np.float32)
        common_mask = data["rate_mask_dev"].astype(bool)

        if(len(rows) != len(X) or data["feature_names"].tolist() != manifest["feature_names"]):
            raise ValueError("Development metadata or feature order differs from the model arrays.")

        predicted, predicted_score = predict_bundle(X, models, pca, manifest["feature_names"])

        for i, target in enumerate(FOUR_TARGETS):
            actual, baseline = load_target_arrays(data, target, "dev")
            actual = actual.astype(np.float64)
            baseline = baseline.astype(np.float64)
            mask = np.ones(len(X), dtype=bool)

            if(target in ["ws_48", "bpm"]):
                mask = data["rate_mask_dev"].astype(bool)

            actual_name = "next_season_" + target
            baseline_name = target

            if(target == "minutes"):
                baseline_name = "mp"

            np.testing.assert_allclose(actual[mask], rows.loc[mask, actual_name], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(baseline[mask], rows.loc[mask, baseline_name], rtol=1e-6, atol=1e-6)
            common_mask = common_mask & np.isfinite(actual) & np.isfinite(baseline)
            actual_columns.append(actual)
            baseline_columns.append(baseline)
            table["actual_" + target] = actual
            table["predicted_" + target] = predicted[:, i]
            table["baseline_" + target] = baseline
            table["error_" + target] = predicted[:, i] - actual
            row = calculate_metrics(actual[mask], predicted[mask, i], baseline[mask], "dev", target)
            metrics.append(row)
            np.testing.assert_allclose(row["model_rmse"], manifest["models"][target]["development_selection"]["dev_rmse"], rtol=1e-6, atol=1e-8)

    actual_matrix = np.column_stack(actual_columns)[common_mask]
    baseline_matrix = np.column_stack(baseline_columns)[common_mask]
    reference = json.loads((bundle / "reference_pca_parameters.json").read_text(encoding="utf-8"))

    for name, parameters, width in [("three_stat_pca", reference, 3), ("four_stat_pca", pca, 4)]:
        actual = combine_scores(actual_matrix[:, :width], parameters)
        baseline = combine_scores(baseline_matrix[:, :width], parameters)
        prediction = combine_scores(predicted[common_mask, :width], parameters)
        if(width == 4):
            np.testing.assert_allclose(prediction, predicted_score[common_mask], rtol=1e-10, atol=1e-12)
        table["actual_" + name] = np.nan
        table["predicted_" + name] = np.nan
        table["baseline_" + name] = np.nan
        table.loc[common_mask, "actual_" + name] = actual
        table.loc[common_mask, "predicted_" + name] = prediction
        table.loc[common_mask, "baseline_" + name] = baseline
        metrics.append(calculate_metrics(actual, prediction, baseline, "dev", name))

    table["four_stat_valid"] = common_mask
    table.to_csv(output / "development_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    weights = pd.DataFrame({"target": FOUR_TARGETS, "pc1_weight": pca["component_weights"], "training_mean": pca["target_means"], "training_std": pca["target_stds"]})
    weights.to_csv(output / "pca_weights.csv", index=False)
    save_plots(output, bundle, manifest, table)
    save_report(output, manifest, pca, metrics)
    minute_parameters, minute_metadata = models["minutes"]
    raw_minutes = predict(X.T, minute_parameters, minute_metadata["target_mean"], minute_metadata["target_scale"])
    audit = {"bundle": str(bundle.resolve()), "manifest_sha256": hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(), "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "split": "development", "examples": len(X), "valid_four_stat_examples": int(common_mask.sum()), "negative_raw_minute_predictions_floored": int((raw_minutes < 0).sum()), "observed_future_minutes_used_as_inputs": False, "test_evaluated": False}
    (output / "evaluation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print ("\nDevelopment results: " + str(output), flush=True)
    print (weights.to_string(index=False))

    for row in metrics:
        print (row["target"] + " | RMSE " + str(row["model_rmse"]) + " | baseline " + str(row["baseline_rmse"]) + " | improvement (%) " + str(row["rmse_improvement_percent"]), flush=True)

    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    evaluate_bundle(args.bundle)


if(__name__ == "__main__"):
    main()
