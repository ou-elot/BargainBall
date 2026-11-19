"""Plot saved train/dev/test errors and learning curves without retraining.

Test errors exist only at the selected checkpoints, not at every epoch.
Run: python plot_loss_diagnostics.py
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy_pipeline import load_bundle

ROOT = Path(__file__).resolve().parent
BUNDLE_ID = "20261109_204731_573737"
SPLITS = ["train", "dev", "test"]
SPLIT_LABELS = ["Train", "Dev", "Test"]
COLORS = ["#2563eb", "#ea580c", "#16a34a"]
TITLES = {"pie": "PIE", "ws_48": "WS/48", "bpm": "BPM", "minutes": "Total minutes", "pca_score": "Four-stat composite"}


def metric_rows(metrics, target):
    rows = []

    for split in SPLITS:
        selected = metrics[(metrics["target"] == target) & (metrics["split"] == split)]

        if(len(selected) != 1):
            raise ValueError("Expected exactly one saved metric row per target and split.")

        rows.append(selected.iloc[0])

    return rows


def plot_split_bars(axis, rows, metric, title):
    values = []
    baselines = []

    for row in rows:
        values.append(float(row["model_" + metric]))
        baselines.append(float(row["baseline_" + metric]))

    bars = axis.bar(SPLIT_LABELS, values, color=COLORS, width=0.58)
    axis.bar_label(bars, fmt="%.3g", padding=4, fontsize=9)
    axis.scatter(np.arange(3), baselines, color="#475569", marker="D", s=36, facecolors="none", label="Previous-season baseline", zorder=3)
    axis.set_ylim(0, max(values + baselines) * 1.22)
    axis.set_title(title)
    axis.set_ylabel(metric.upper())
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def save_split_comparisons(output, metrics, targets, concentration):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
    rows = metric_rows(metrics, "pca_score")

    for i, metric in enumerate(["mse", "rmse", "mae"]):
        plot_split_bars(axes[i], rows, metric, metric.upper())

    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle("Final PCA output: frozen-checkpoint train / dev / test error")
    figure.supxlabel("All examples retained. Largest test row contributes " + str(round(100.0 * concentration, 1)) + "% of test squared error.", fontsize=10)
    figure.savefig(output / "composite_loss_comparison.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(2, len(targets), figsize=(4.2 * len(targets), 7.2), layout="constrained")

    for i, target in enumerate(targets):
        rows = metric_rows(metrics, target)
        plot_split_bars(axes[0, i], rows, "mse", TITLES[target] + " - MSE")
        plot_split_bars(axes[1, i], rows, "mae", TITLES[target] + " - MAE")

    axes[0, 0].legend(fontsize=7)
    figure.suptitle("Individual outputs: selected checkpoints, dropout disabled")
    figure.supxlabel("Each panel uses its own target units; error magnitudes are not comparable across different targets.", fontsize=10)
    figure.savefig(output / "network_split_losses.png", dpi=180)
    plt.close(figure)


def save_learning_curves(output, bundle, manifest, metrics):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    summaries = []

    for i, target in enumerate(manifest["target_order"]):
        model = manifest["models"][target]
        history = pd.read_csv(bundle / target / "training_history.csv")
        history = history[history["dropout"] == model["dropout"]].sort_values("epoch")
        selected = history[history["epoch"] == model["best_epoch"]]

        if(len(selected) != 1):
            raise ValueError("Selected epoch must appear exactly once in the selected-dropout history.")

        selected = selected.iloc[0]
        last = history.iloc[-1]
        rows = metric_rows(metrics, target)
        np.testing.assert_allclose(selected["train_mse"], rows[0]["model_mse"], rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(selected["dev_mse"], rows[1]["model_mse"], rtol=1e-6, atol=1e-10)
        axis = axes.flat[i]
        axis.plot(history["epoch"], history["train_mse"], label="Train MSE", color=COLORS[0])
        axis.plot(history["epoch"], history["dev_mse"], label="Dev MSE", color=COLORS[1])
        axis.axvline(model["best_epoch"], color="#64748b", linestyle="--", linewidth=1, label="Selected epoch")
        axis.scatter([model["best_epoch"]], [rows[2]["model_mse"]], marker="*", s=140, color=COLORS[2], edgecolors="black", linewidth=0.4, zorder=5, label="Test MSE: selected checkpoint only")
        axis.set_yscale("log")
        axis.set_title(TITLES[target] + " | hidden " + str(model["layer_dims"][1:-1]) + " | dropout " + str(model["dropout"]))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Unpenalized MSE (log scale)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        summaries.append({"target": target, "selected_epoch": model["best_epoch"], "last_epoch": int(last["epoch"]), "selected_train_mse": float(rows[0]["model_mse"]), "selected_dev_mse": float(rows[1]["model_mse"]), "selected_test_mse": float(rows[2]["model_mse"]), "last_epoch_train_mse": float(last["train_mse"]), "last_epoch_dev_mse": float(last["dev_mse"]), "last_vs_selected_dev_mse_percent": 100.0 * (float(last["dev_mse"]) / float(selected["dev_mse"]) - 1.0), "dev_train_mse_ratio": float(rows[1]["model_mse"] / rows[0]["model_mse"]), "test_train_mse_ratio": float(rows[2]["model_mse"] / rows[0]["model_mse"]), "train_mean_error": float(rows[0]["mean_prediction_error"]), "dev_mean_error": float(rows[1]["mean_prediction_error"]), "test_mean_error": float(rows[2]["mean_prediction_error"])})

    figure.suptitle("Selected-run learning curves with one test point per model")
    figure.supxlabel("Train/dev curves use fixed epoch-end weights with dropout off. No test curve or composite epoch history was recorded.", fontsize=9)
    figure.savefig(output / "network_learning_curves.png", dpi=180)
    plt.close(figure)
    pd.DataFrame(summaries).to_csv(output / "learning_curve_summary.csv", index=False)
    return summaries


def save_calibration(output, predictions):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    calibration_rows = []

    for i, target in enumerate(["pca_score", "minutes"]):
        training_predictions = predictions["train"]["predicted_" + target].to_numpy()
        internal = np.unique(np.quantile(training_predictions, np.arange(0.1, 1.0, 0.1)))
        edges = np.concatenate([[-np.inf], internal, [np.inf]])
        extrema = []

        for j, split in enumerate(SPLITS):
            table = predictions[split]
            actual = table["actual_" + target].to_numpy()
            predicted = table["predicted_" + target].to_numpy()
            valid = np.isfinite(actual) & np.isfinite(predicted)
            assignments = pd.cut(predicted, bins=edges, labels=False, include_lowest=True)
            xs = []
            ys = []

            for bucket in range(len(edges) - 1):
                mask = valid & (assignments == bucket)

                if(mask.any()):
                    mean_predicted = float(predicted[mask].mean())
                    mean_actual = float(actual[mask].mean())
                    xs.append(mean_predicted)
                    ys.append(mean_actual)
                    extrema.extend([mean_predicted, mean_actual])
                    calibration_rows.append({"target": target, "split": split, "prediction_bin": bucket + 1, "examples": int(mask.sum()), "mean_prediction": mean_predicted, "mean_actual": mean_actual, "mean_prediction_error": mean_predicted - mean_actual})

            axes[i].plot(xs, ys, marker="o", markersize=4, color=COLORS[j], label=SPLIT_LABELS[j])

        low = min(extrema)
        high = max(extrema)
        axes[i].plot([low, high], [low, high], linestyle="--", color="#475569", linewidth=1, label="Ideal calibration")
        axes[i].set_title(TITLES[target])
        axes[i].set_xlabel("Average prediction within a prediction bin")
        axes[i].set_ylabel("Average observed outcome within that bin")
        axes[i].grid(alpha=0.2)
        axes[i].legend(fontsize=8)

    figure.suptitle("Calibration diagnostic: prediction bins defined on training data")
    figure.supxlabel("Above diagonal = underprediction; below = overprediction. Descriptive bin means, without confidence intervals.", fontsize=9)
    figure.savefig(output / "calibration_diagnostics.png", dpi=180)
    plt.close(figure)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_bins.csv", index=False)


def save_report(output, metrics, summaries, concentration):
    rows = metric_rows(metrics, "pca_score")
    ratios = []

    for row in summaries:
        ratios.append(row["dev_train_mse_ratio"])

    lines = ["# Train / development / test loss diagnostics", "", "All plots use saved losses and predictions from the frozen four-stat bundle.", "No training, PCA fitting, or model selection occurs in this analysis.", "", "## Final composite", "", "| Split | MSE | MAE | Mean prediction error |", "|---|---:|---:|---:|"]

    for split, row in zip(SPLIT_LABELS, rows):
        lines.append("| " + split + " | " + str(round(row["model_mse"], 6)) + " | " + str(round(row["model_mae"], 6)) + " | " + str(round(row["mean_prediction_error"], 6)) + " |")

    lines.extend(["", "## Reading the plots", "", "- `composite_loss_comparison.png`: final-score MSE, RMSE, and MAE for the frozen checkpoints. Hollow diamonds show each split's previous-season baseline.", "- `network_split_losses.png`: MSE and MAE for the four individual outputs. Each panel has different units.", "- `network_learning_curves.png`: selected-dropout train/dev curves, plus a single test star at the saved checkpoint. No per-epoch test history or joint composite learning curve exists.", "- `calibration_diagnostics.png`: observed versus predicted bin means for the composite and minutes; training-derived prediction bins are shared across splits.", "", "## Bias / variance interpretation", "", "The selected models do not exhibit a large training-versus-development MSE", "gap: development/training ratios range from " + str(round(min(ratios), 3)) + " to " + str(round(max(ratios), 3)) + ". This is not the classic pattern of near-zero training loss and very high development loss.", "", "Some networks continue lowering training loss after development loss stops", "improving. Early stopping keeps the earlier checkpoint; see the dashed lines", "and `learning_curve_summary.csv` for selected versus final epoch values.", "", "The test composite MSE gap is strongly affected by a single player-season,", "which contributes " + str(round(100.0 * concentration, 2)) + "% of all test squared error. All rows remain in the headline results. Composite MAE changes much less between development and test, so the MSE spike alone is not sufficient evidence of broad model overfitting.", "", "High bias/underfitting cannot be diagnosed from nonzero training MSE alone.", "We do not know the irreducible error from injuries, future opportunity, and", "unobserved team context. Learning curves by training-set size, simpler-model", "benchmarks, and repeated-seed experiments would provide stronger evidence.", "A small overall mean error also does not rule out subgroup/calibration bias.", "", "The test split was examined before this method revision; these are descriptive", "follow-up diagnostics, not a fresh independent model-selection exercise.", ""])
    lines.extend(["## Calibration interpretation", "", "Prediction-bin means are reasonably close to the ideal line overall for", "both minutes and the composite. This suggests reasonable aggregate", "calibration, but accurate bin averages do not imply accurate individual", "player forecasts or rule out bias within particular player groups.", "These are generalization/calibration diagnostics, not a formal statistical", "bias-variance decomposition from repeated fitted models.", ""])
    (output / "DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "models" / "numpy_four_stat" / BUNDLE_ID)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "test" / "numpy_four_stat" / BUNDLE_ID)
    args = parser.parse_args()
    manifest, models, pca = load_bundle(args.bundle)
    metrics_path = args.results / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    predictions = {}
    prediction_hashes = {}

    for split, filename in [("train", "training_predictions.csv"), ("dev", "development_predictions.csv"), ("test", "test_predictions.csv")]:
        path = args.results / filename
        table = pd.read_csv(path)
        predictions[split] = table
        prediction_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()

        for target in manifest["target_order"] + ["pca_score"]:
            actual = table["actual_" + target].to_numpy()
            predicted = table["predicted_" + target].to_numpy()
            valid = np.isfinite(actual) & np.isfinite(predicted)

            if(target in ["ws_48", "bpm"]):
                valid = valid & table["rate_valid"].to_numpy(dtype=bool)

            saved = metrics[(metrics["split"] == split) & (metrics["target"] == target)].iloc[0]
            np.testing.assert_allclose(float(np.mean((predicted[valid] - actual[valid]) ** 2)), saved["model_mse"], rtol=1e-9, atol=1e-12)

    errors = predictions["test"]["error_pca_score"].to_numpy()
    squared = errors[np.isfinite(errors)] ** 2
    concentration = float(squared.max() / squared.sum())
    output = args.results / "loss_diagnostics"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    save_split_comparisons(output, metrics, manifest["target_order"], concentration)
    summaries = save_learning_curves(output, args.bundle, manifest, metrics)
    save_calibration(output, predictions)
    save_report(output, metrics, summaries, concentration)
    audit = {"bundle_id": manifest["bundle_id"], "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(), "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(), "predictions_sha256": prediction_hashes, "test_history": "One observation at the selected checkpoint per model; no epochwise test losses", "models_retrained": False, "rows_removed_from_headline_metrics": 0, "test_squared_error_largest_row_fraction": concentration}
    (output / "plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print ("Saved loss plots and diagnostics to: " + str(output))
    print (pd.DataFrame(summaries)[["target", "selected_epoch", "dev_train_mse_ratio", "test_train_mse_ratio", "last_vs_selected_dev_mse_percent"]].to_string(index=False))


if(__name__ == "__main__"):
    main()
