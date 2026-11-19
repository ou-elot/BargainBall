"""Evaluate an already-frozen NumPy model bundle on development and test data.

Run: python evaluate_final_models.py --bundle models/numpy_final/BUNDLE_ID
No training, model selection, or PCA fitting occurs in this script.
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
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy_pipeline import FOUR_TARGETS, combine_scores, load_bundle, predict_bundle
from train_numpy import load_target_arrays

ROOT = Path(__file__).resolve().parent


def calculate_metrics(actual, predicted, baseline, split, target):
    for values in [actual, predicted, baseline]:
        if(not np.isfinite(values).all()):
            raise ValueError("Metric inputs must be finite.")

    errors = predicted - actual
    baseline_errors = baseline - actual
    mse = float(np.mean(errors ** 2))
    baseline_mse = float(np.mean(baseline_errors ** 2))
    rmse = float(np.sqrt(mse))
    baseline_rmse = float(np.sqrt(baseline_mse))
    variance = float(np.mean((actual - actual.mean()) ** 2))
    improvement = None
    mse_improvement = None
    r_squared = None

    if(baseline_rmse > 0):
        improvement = 100.0 * (1.0 - rmse / baseline_rmse)
        mse_improvement = 100.0 * (1.0 - mse / baseline_mse)
    if(variance > 0):
        r_squared = 1.0 - mse / variance

    return {"split": split, "target": target, "examples": len(actual), "model_mse": mse, "baseline_mse": baseline_mse, "model_rmse": rmse, "baseline_rmse": baseline_rmse, "model_mae": float(np.mean(np.abs(errors))), "baseline_mae": float(np.mean(np.abs(baseline_errors))), "mean_prediction_error": float(errors.mean()), "model_r_squared": r_squared, "rmse_improvement_percent": improvement, "mse_improvement_percent": mse_improvement}


def evaluate_split(data, split, manifest, models, pca_parameters):
    X = data["X_" + split].astype(np.float32)
    rate_mask = data["rate_mask_" + split].astype(bool)
    predictions, composite_predictions = predict_bundle(X, models, pca_parameters, manifest["feature_names"])
    rows = pd.read_csv(ROOT / "data" / "processed" / (split + "_rows_rates.csv"))

    if(len(rows) != len(X) or rate_mask.shape != (len(X),)):
        raise ValueError("Metadata, mask, and model arrays must have the same row count.")

    table = rows[["player", "player_id", "season", "target_season"]].copy()
    table["rate_valid"] = rate_mask
    actual_columns = []
    baseline_columns = []
    metrics = []

    for i, target in enumerate(manifest["target_order"]):
        actual, baseline = load_target_arrays(data, target, split)
        actual = actual.astype(np.float64)
        baseline = baseline.astype(np.float64)
        mask = rate_mask

        if(target in ["pie", "minutes"]):
            mask = np.ones(len(X), dtype=bool)

        baseline_name = target

        if(target == "minutes"):
            baseline_name = "mp"

        np.testing.assert_allclose(actual[mask], rows.loc[mask, "next_season_" + target], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(baseline[mask], rows.loc[mask, baseline_name], rtol=1e-6, atol=1e-6)
        table["actual_" + target] = actual
        table["predicted_" + target] = predictions[:, i]
        table["baseline_" + target] = baseline
        table["error_" + target] = predictions[:, i] - actual
        actual_columns.append(actual[rate_mask])
        baseline_columns.append(baseline[rate_mask])
        metrics.append(calculate_metrics(actual[mask], predictions[mask, i], baseline[mask], split, target))

    actual_composite = combine_scores(np.column_stack(actual_columns), pca_parameters)
    baseline_composite = combine_scores(np.column_stack(baseline_columns), pca_parameters)
    table["actual_pca_score"] = np.nan
    table["baseline_pca_score"] = np.nan
    table["predicted_pca_score"] = composite_predictions
    table.loc[rate_mask, "actual_pca_score"] = actual_composite
    table.loc[rate_mask, "baseline_pca_score"] = baseline_composite
    table["error_pca_score"] = table["predicted_pca_score"] - table["actual_pca_score"]
    metrics.append(calculate_metrics(actual_composite, composite_predictions[rate_mask], baseline_composite, split, "pca_score"))
    return table, metrics


def save_plots(folder, table, metrics, target_order):
    plotted_targets = list(target_order) + ["pca_score"]
    figure, axes = plt.subplots(1, len(plotted_targets), figsize=(4.5 * len(plotted_targets), 4.5), layout="constrained")

    for i, target in enumerate(plotted_targets):
        actual = table["actual_" + target].to_numpy()
        predicted = table["predicted_" + target].to_numpy()
        baseline = table["baseline_" + target].to_numpy()
        mask = np.isfinite(actual) & np.isfinite(baseline)
        low = float(np.min(np.concatenate([actual[mask], predicted[mask], baseline[mask]])))
        high = float(np.max(np.concatenate([actual[mask], predicted[mask], baseline[mask]])))
        axes[i].scatter(actual[mask], baseline[mask], s=8, alpha=0.2, label="Previous-season baseline", color="#b66a2b")
        axes[i].scatter(actual[mask], predicted[mask], s=8, alpha=0.3, label="Frozen NumPy model", color="#2474a6")
        axes[i].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        axes[i].set_title(target.upper())
        axes[i].set_xlabel("Actual next-season value")
        axes[i].set_ylabel("Predicted value")
        axes[i].legend(fontsize=7)

    figure.suptitle("Test-split predictions from fixed models")
    figure.savefig(folder / "test_predictions.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    labels = []
    values = []

    for row in metrics:
        if(row["split"] == "test"):
            labels.append(row["target"].upper())
            values.append(row["model_rmse"] / row["baseline_rmse"])

    axis.bar(labels, values, color="#2474a6")
    axis.axhline(1.0, color="black", linestyle="--", label="Previous-season baseline")
    axis.set_ylabel("Model RMSE / baseline RMSE (lower is better)")
    axis.set_title("Test-split error relative to each target's baseline")
    axis.legend()
    figure.savefig(folder / "test_error_comparison.png", dpi=160)
    plt.close(figure)


def save_report(folder, manifest, metrics, pca_parameters):
    title = "# Frozen NumPy models: test-split evaluation"
    evaluation_note = "Models were selected by development RMSE and frozen before this evaluation."

    if(manifest["target_order"] == FOUR_TARGETS):
        title = "# Four-stat output loss: follow-up test-split evaluation"
        evaluation_note = "The original test split was already examined for the earlier three-stat design. This follow-up uses fixed models but is not fresh independent confirmation of the revised method."

    lines = [title, "", "Bundle: `" + manifest["bundle_id"] + "`", "", evaluation_note, "The scaler and PCA were fitted on actual training targets only.", "", "## Final PCA output loss", "", "MSE is the mean squared difference between predicted and observed PCA scores.", "These are unpenalized inference errors with dropout disabled, not optimizer", "losses from changing minibatch weights. Baseline scores repeat prior-season statistics.", "", "| Split | Examples | Model MSE | Baseline MSE | Model RMSE | Model MAE |", "|---|---:|---:|---:|---:|---:|"]

    for row in metrics:
        if(row["target"] == "pca_score"):
            lines.append("| " + row["split"] + " | " + str(row["examples"]) + " | " + str(round(row["model_mse"], 8)) + " | " + str(round(row["baseline_mse"], 8)) + " | " + str(round(row["model_rmse"], 8)) + " | " + str(round(row["model_mae"], 8)) + " |")

    lines.extend(["", "## Frozen models", "", "| Target | Layer dimensions | Learning rate | Drop probability | Selected epoch |", "|---|---|---:|---:|---:|"])

    for target in manifest["target_order"]:
        model = manifest["models"][target]
        lines.append("| " + target.upper() + " | " + str(model["layer_dims"]) + " | " + str(model["learning_rate"]) + " | " + str(model["dropout"]) + " | " + str(model["best_epoch"]) + " |")

    lines.extend(["", "## Test-split performance", "", "| Target | Examples | Model MSE | Model RMSE | Baseline RMSE | Model MAE | RMSE reduction |", "|---|---:|---:|---:|---:|---:|---:|"])

    for row in metrics:
        if(row["split"] == "test"):
            lines.append("| " + row["target"].upper() + " | " + str(row["examples"]) + " | " + str(round(row["model_mse"], 8)) + " | " + str(round(row["model_rmse"], 8)) + " | " + str(round(row["baseline_rmse"], 8)) + " | " + str(round(row["model_mae"], 8)) + " | " + str(round(row["rmse_improvement_percent"], 2)) + "% |")

    lines.extend(["", "## Development versus test", "", "| Target | Development RMSE | Test RMSE |", "|---|---:|---:|"])

    for target in manifest["target_order"] + ["pca_score"]:
        errors = {}

        for row in metrics:
            if(row["target"] == target):
                errors[row["split"]] = row["model_rmse"]

        lines.append("| " + target.upper() + " | " + str(round(errors["dev"], 8)) + " | " + str(round(errors["test"], 8)) + " |")

    lines.extend(["", "## Interpretation", "", "- The baseline predicts that each statistic repeats its previous-season value.", "- PCA uses the same fixed transformation for predicted, actual, and baseline statistics.", "- PCA was fitted on " + str(pca_parameters["training_examples"]) + " training rows and captures " + str(round(100.0 * pca_parameters["explained_variance_ratio"], 2)) + "% of standardized training-target variance. This is not prediction accuracy.", "- The composite is measured in training-composite standard deviations; it is not a dollar-value or salary-adjusted score.", "- This random split evaluates held-out historical season pairs, not future seasons. A player may appear in multiple splits in different seasons.", "- Results describe players with an observed immediately following NBA season; they do not measure the probability of remaining in the league.", "- All weights, preprocessing, PCA coefficients, and inference policies were fixed for this run. No training or test-based parameter updates occurred.", "", "Exact metrics are in `metrics.csv`; row-level predictions are in `training_predictions.csv`, `development_predictions.csv`, and `test_predictions.csv`.", ""])
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    manifest, models, pca_parameters = load_bundle(args.bundle)

    data_path = ROOT / "data" / "processed" / "model_data_rates.npz"

    if(hashlib.sha256(data_path.read_bytes()).hexdigest() != manifest["model_data_sha256"]):
        raise ValueError("The evaluation arrays differ from the frozen training arrays.")

    family = "numpy_final"
    evaluation_kind = "fixed_model_test_evaluation"

    if(manifest["target_order"] == FOUR_TARGETS):
        family = "numpy_four_stat"
        evaluation_kind = "follow_up_on_previously_examined_test_split"

    output = ROOT / "results" / "test" / family / manifest["bundle_id"]
    output.mkdir(parents=True, exist_ok=False)
    audit = {"bundle": str(args.bundle.resolve()), "bundle_id": manifest["bundle_id"], "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(), "evaluation_started_at_utc": started, "frozen_at_utc": manifest["frozen_at_utc"], "model_data_sha256": manifest["model_data_sha256"], "test_evaluated": False, "selection_changed": False}
    audit["evaluation_kind"] = evaluation_kind
    audit["source_sha256"] = {}
    source_folder = output / "evaluation_sources"
    source_folder.mkdir()

    for name in ["evaluate_final_models.py", "numpy_pipeline.py", "numpy_network.py", "train_numpy.py"]:
        shutil.copyfile(ROOT / name, source_folder / name)
        audit["source_sha256"][name] = hashlib.sha256((source_folder / name).read_bytes()).hexdigest()
    (output / "evaluation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    metrics = []

    # Only this final-evaluation script accesses the held-out test arrays.
    with np.load(data_path, allow_pickle=False) as data:
        if(data["feature_names"].tolist() != manifest["feature_names"]):
            raise ValueError("Evaluation feature order differs from the frozen models.")

        for split in ["train", "dev", "test"]:
            table, split_metrics = evaluate_split(data, split, manifest, models, pca_parameters)
            filename = {"train": "training_predictions.csv", "dev": "development_predictions.csv", "test": "test_predictions.csv"}[split]

            if(split == "test"):
                test_table = table

            # Check saved selection scores before progressing to test evaluation.
            if(split in ["train", "dev"]):
                for row in split_metrics:
                    if(row["target"] in manifest["target_order"]):
                        expected = manifest["models"][row["target"]]["development_selection"][split + "_mse"]
                        np.testing.assert_allclose(row["model_mse"], expected, rtol=1e-6, atol=1e-10)

            table.to_csv(output / filename, index=False)
            metrics.extend(split_metrics)

    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(metrics).query("target == 'pca_score'").to_csv(output / "output_loss.csv", index=False)
    save_plots(output, test_table, metrics, manifest["target_order"])
    save_report(output, manifest, metrics, pca_parameters)
    audit["test_evaluated"] = True
    audit["evaluation_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "evaluation_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print ("Final evaluation saved to: " + str(output))

    for row in metrics:
        if(row["split"] == "test"):
            print (row["target"].upper() + " | test MSE " + str(row["model_mse"]) + " | test RMSE " + str(row["model_rmse"]) + " | baseline RMSE " + str(row["baseline_rmse"]) + " | RMSE improvement (%) " + str(row["rmse_improvement_percent"]))

    print ("\nFinal PCA output loss:")
    print (pd.DataFrame(metrics).query("target == 'pca_score'")[["split", "examples", "model_mse", "baseline_mse", "model_rmse", "model_mae"]].to_string(index=False))


if(__name__ == "__main__"):
    main()
