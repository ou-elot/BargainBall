"""Combine the three networks' predictions using training-only PCA.

Run after train_rates.py: python pca_rates.py
PCA fits actual training targets, not predicted or development values.
The score is centered at zero and has unit standard deviation among actual
training targets. It summarizes PIE, WS/48, and BPM; it is not a salary-value score.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from train import create_model

ROOT = Path(__file__).resolve().parent
TARGETS = ["pie", "ws_48", "bpm"]


def combine(values, parameters):
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 2 or values.shape[1] != len(parameters["target_order"]):
        raise ValueError("Expected columns in the saved target_order.")

    if not np.isfinite(values).all():
        raise ValueError("PCA inputs must be finite.")

    standardized = (values - np.array(parameters["target_means"])) / np.array(parameters["target_stds"])
    centered = standardized - np.array(parameters["pca_center"])
    return (centered @ np.array(parameters["component_weights"]) - parameters["score_mean"]) / parameters["score_std"]


def main():
    torch.set_num_threads(1)
    processed = ROOT / "data" / "processed"
    output = ROOT / "results" / "development" / "rates"
    model_folder = ROOT / "models" / "rates" / "pca"
    output.mkdir(parents=True, exist_ok=True)
    model_folder.mkdir(parents=True, exist_ok=True)
    data_path = processed / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    training_columns = []
    actual_columns = []
    baseline_columns = []
    predicted_columns = []

    # No test arrays are loaded or evaluated.
    with np.load(data_path, allow_pickle=False) as data:
        inputs = torch.from_numpy(data["X_dev"].astype(np.float32))
        feature_names = data["feature_names"].tolist()
        train_mask = data["rate_mask_train"].astype(bool)
        dev_mask = data["rate_mask_dev"].astype(bool)
        inputs = inputs[dev_mask]
        base_hash = str(data["base_model_data_sha256"].item())

        for target in TARGETS:
            required = "y_" + target + "_train"

            if required not in data:
                raise ValueError("Run prepare_rates.py, preprocess_rates.py, and train_rates.py first.")

            training_columns.append(data[required][train_mask].astype(np.float64))
            actual_columns.append(data["y_" + target + "_dev"][dev_mask].astype(np.float64))
            baseline_columns.append(data["baseline_" + target + "_dev"][dev_mask].astype(np.float64))
            checkpoint_path = ROOT / "models" / "rates" / target / "best_model.pt"
            expected_hash = data_hash

            if target == "pie":
                checkpoint_path = ROOT / "models" / "pie" / "best_model.pt"
                expected_hash = base_hash

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

            if checkpoint["model_data_sha256"] != expected_hash:
                raise ValueError(target.upper() + " was trained on different arrays. Check the original PIE model or rerun train_rates.py for the rate models.")

            if checkpoint["target"] != target or checkpoint["feature_names"] != feature_names:
                raise ValueError("Checkpoint target or feature order does not match.")

            model = create_model(checkpoint["input_size"], checkpoint["dropout"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            with torch.no_grad():
                predictions = model(inputs).squeeze(1) * checkpoint["target_scale"] + checkpoint["target_mean"]
                predicted_columns.append(predictions.numpy().astype(np.float64))

    training = np.column_stack(training_columns)
    actual = np.column_stack(actual_columns)
    baseline = np.column_stack(baseline_columns)
    predicted = np.column_stack(predicted_columns)

    if not np.isfinite(training).all() or (training.std(axis=0) == 0).any():
        raise ValueError("Training targets must be finite and have nonzero variation.")

    # Fit both the scaler and PCA on actual TRAINING targets only.
    scaler = StandardScaler()
    standardized = scaler.fit_transform(training)
    pca = PCA(n_components=3, svd_solver="full")
    pca.fit(standardized)
    weights = pca.components_[0].copy()

    # PCA sign is arbitrary. Orient the score toward higher combined statistics.
    if weights.sum() < 0:
        weights = -weights

    training_scores = (standardized - pca.mean_) @ weights
    score_mean = float(training_scores.mean())
    score_std = float(training_scores.std(ddof=0))
    parameters = {"target_order": TARGETS, "fit_split": "train", "fit_values": "actual next-season targets", "training_examples": len(training), "cohort": "defined rate targets and prior baselines", "target_means": scaler.mean_.tolist(), "target_stds": scaler.scale_.tolist(), "pca_center": pca.mean_.tolist(), "component_weights": weights.tolist(), "score_mean": score_mean, "score_std": score_std, "explained_variance_ratio": float(pca.explained_variance_ratio_[0]), "all_components_explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "model_data_sha256": data_hash, "interpretation": "Zero is the mean actual training composite; one unit is one training-composite standard deviation. This summarizes performance statistics, not dollar value."}
    actual_score = combine(actual, parameters)
    predicted_score = combine(predicted, parameters)
    baseline_score = combine(baseline, parameters)
    errors = predicted_score - actual_score
    baseline_errors = baseline_score - actual_score
    model_rmse = float(np.sqrt(np.mean(errors ** 2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))
    improvement = None

    if baseline_rmse > 0:
        improvement = 100.0 * (1.0 - model_rmse / baseline_rmse)

    metrics = {"split": "development", "examples": len(actual_score), "model_rmse": model_rmse, "baseline_rmse": baseline_rmse, "model_mae": float(np.mean(np.abs(errors))), "baseline_mae": float(np.mean(np.abs(baseline_errors))), "rmse_improvement_percent": improvement, "explained_variance_ratio": parameters["explained_variance_ratio"], "test_evaluated": False, "model_data_sha256": data_hash}
    rows = pd.read_csv(processed / "dev_rows_rates.csv").loc[dev_mask].reset_index(drop=True)

    if len(rows) != len(actual_score):
        raise ValueError("Development rows and arrays differ in length.")

    scores = rows[["player", "player_id", "season", "target_season", "mp", "next_season_minutes"]].copy()

    for index, target in enumerate(TARGETS):
        np.testing.assert_allclose(actual[:, index], rows["next_season_" + target], rtol=0.000001, atol=0.000001)
        np.testing.assert_allclose(baseline[:, index], rows[target], rtol=0.000001, atol=0.000001)
        scores["predicted_" + target] = predicted[:, index]

    scores["actual_pca_score"] = actual_score
    scores["predicted_pca_score"] = predicted_score
    scores["baseline_pca_score"] = baseline_score
    scores["error"] = errors
    scores["absolute_error"] = np.abs(errors)
    temporary = output / "pca_scores.csv.tmp"
    scores.to_csv(temporary, index=False)
    temporary.replace(output / "pca_scores.csv")
    (model_folder / "pca_parameters.json").write_text(json.dumps(parameters, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "pca_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PCA fitted on " + str(len(training)) + " actual training examples.")
    print("Variance captured by one component (%): " + str(round(100.0 * parameters["explained_variance_ratio"], 2)))

    for index, target in enumerate(TARGETS):
        print(target.upper() + " weight on standardized statistic: " + str(round(float(weights[index]), 6)))

    if (weights < 0).any():
        print("Mixed weight signs: inspect the score before interpreting higher as better overall performance.")

    print("Development composite model RMSE: " + str(model_rmse))
    print("Development composite baseline RMSE: " + str(baseline_rmse))
    print("Development composite model MAE: " + str(metrics["model_mae"]))

    if improvement is not None:
        print("RMSE improvement over composite baseline (%): " + str(round(improvement, 2)))

    print("Saved parameters to: " + str(model_folder / "pca_parameters.json"))
    print("Saved player scores to: " + str(output / "pca_scores.csv"))
    print("Score 0 is the training average; the score is not a salary-value estimate.")


if __name__ == "__main__":
    main()
