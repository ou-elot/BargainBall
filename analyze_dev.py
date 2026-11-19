"""Plot development predictions and training curves for all three targets."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import create_model

ROOT = Path(__file__).resolve().parent
TARGETS = ["pie", "ws", "vorp"]


def save_table(table, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    torch.set_num_threads(1)
    processed = ROOT / "data" / "processed"
    output = ROOT / "results" / "development"
    output.mkdir(parents=True, exist_ok=True)
    data_path = processed / "model_data.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    rows = pd.read_csv(processed / "dev_rows.csv")
    columns = ["player", "player_id", "season", "target_season", "age", "g", "mp_per_game", "pos"]
    predictions_table = rows[columns].copy()
    summary = []
    curve_figure, curve_axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    prediction_figure, prediction_axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    error_figure, error_axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")

    # Load only development arrays. No test predictions or test metrics are calculated.
    with np.load(data_path, allow_pickle=False) as data:
        inputs = torch.from_numpy(data["X_dev"].astype(np.float32))
        feature_names = data["feature_names"].tolist()

        if len(inputs) != len(rows):
            raise ValueError("Development metadata and input arrays have different lengths.")

        for index, target in enumerate(TARGETS):
            model_folder = ROOT / "models" / target
            checkpoint = torch.load(model_folder / "best_model.pt", map_location="cpu", weights_only=True)

            if checkpoint["model_data_sha256"] != data_hash:
                raise ValueError("The " + target + " model was trained on different arrays. Run train.py again.")

            if checkpoint["target"] != target or checkpoint["feature_names"] != feature_names:
                raise ValueError("Checkpoint target or feature columns do not match.")

            actual = data["y_" + target + "_dev"].astype(np.float64)
            baseline = data["baseline_" + target + "_dev"].astype(np.float64)
            np.testing.assert_allclose(actual, rows["next_season_" + target], rtol=0.000001, atol=0.000001)
            np.testing.assert_allclose(baseline, rows[target], rtol=0.000001, atol=0.000001)
            model = create_model(checkpoint["input_size"], checkpoint["dropout"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            with torch.no_grad():
                predicted = model(inputs).squeeze(1) * checkpoint["target_scale"] + checkpoint["target_mean"]
                predicted = predicted.numpy().astype(np.float64)

            if not np.isfinite(predicted).all():
                raise ValueError("Nonfinite predictions for " + target)

            errors = predicted - actual
            baseline_errors = baseline - actual
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))
            improvement = None

            if baseline_rmse > 0:
                improvement = 100.0 * (1.0 - rmse / baseline_rmse)

            summary.append({"target": target, "examples": len(actual), "model_rmse": rmse, "baseline_rmse": baseline_rmse, "model_mae": float(np.mean(np.abs(errors))), "baseline_mae": float(np.mean(np.abs(baseline_errors))), "mean_prediction_error": float(np.mean(errors)), "rmse_improvement_percent": improvement})
            predictions_table["actual_" + target] = actual
            predictions_table["predicted_" + target] = predicted
            predictions_table["baseline_" + target] = baseline
            predictions_table["error_" + target] = errors
            predictions_table["absolute_error_" + target] = np.abs(errors)
            target_table = rows[columns].copy()
            target_table["actual"] = actual
            target_table["predicted"] = predicted
            target_table["baseline"] = baseline
            target_table["error"] = errors
            target_table["absolute_error"] = np.abs(errors)
            target_table = target_table.sort_values("absolute_error", ascending=False)
            save_table(target_table.head(20), output / (target + "_largest_errors.csv"))

            history = pd.read_csv(model_folder / "training_history.csv")
            report = json.loads((model_folder / "development_results.json").read_text(encoding="utf-8"))
            axis = curve_axes[index]

            for dropout, group in history.groupby("dropout"):
                axis.plot(group["epoch"], group["dev_rmse"], label="Dropout " + str(dropout))

            axis.axhline(baseline_rmse, color="gray", linestyle="--", label="Previous-season baseline")
            selected = report["selected"]
            axis.scatter([selected["best_epoch"]], [selected["dev_rmse"]], color="black", marker="*", s=90, zorder=5, label="Selected checkpoint")
            axis.set(title=target.upper(), xlabel="Epoch", ylabel="Development RMSE")
            axis.legend(fontsize=7)

            axis = prediction_axes[index]
            axis.scatter(actual, baseline, s=9, alpha=0.25, color="#b66a2b", label="Previous-season baseline")
            axis.scatter(actual, predicted, s=9, alpha=0.35, color="#2474a6", label="Neural network")
            limits = np.concatenate([actual, baseline, predicted])
            low = float(limits.min())
            high = float(limits.max())
            axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1, label="Perfect prediction")
            axis.set(title=target.upper(), xlabel="Actual next-season value", ylabel="Predicted next-season value")
            axis.legend(fontsize=7)

            axis = error_axes[index]
            bins = np.histogram_bin_edges(np.concatenate([errors, baseline_errors]), bins=50)
            axis.hist(baseline_errors, bins=bins, histtype="step", color="#b66a2b", linewidth=1.5, label="Previous-season baseline")
            axis.hist(errors, bins=bins, histtype="step", color="#2474a6", linewidth=1.5, label="Neural network")
            axis.axvline(0, color="black", linestyle="--", linewidth=1)
            axis.set(title=target.upper(), xlabel="Prediction minus actual value", ylabel="Examples")
            axis.legend(fontsize=7)
            print(target.upper() + " | model RMSE " + str(round(rmse, 6)) + " | baseline RMSE " + str(round(baseline_rmse, 6)))

    curve_figure.suptitle("Development RMSE during training")
    prediction_figure.suptitle("Development predictions: closer to the diagonal is better")
    error_figure.suptitle("Development errors: negative means underprediction")
    curve_figure.savefig(output / "training_curves.png", dpi=160)
    prediction_figure.savefig(output / "predictions.png", dpi=160)
    error_figure.savefig(output / "errors.png", dpi=160)
    plt.close("all")
    save_table(predictions_table, output / "development_predictions.csv")
    save_table(pd.DataFrame(summary), output / "development_metrics.csv")
    print("\nSaved plots, predictions, and largest-error tables to: " + str(output))


if __name__ == "__main__":
    main()
