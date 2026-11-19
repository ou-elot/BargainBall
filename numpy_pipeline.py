"""Load the final NumPy networks and combine their predictions with fixed PCA."""

from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy_network import load_model, predict

TARGETS = ["pie", "ws_48", "bpm"]
FOUR_TARGETS = TARGETS + ["minutes"]


def fit_pca(training_targets, data_hash, target_order=None):
    """Fit on actual training targets ONLY; optionally add total minutes fourth."""
    values = np.asarray(training_targets, dtype=np.float64)

    if(target_order is None):
        target_order = TARGETS

    if(target_order not in [TARGETS, FOUR_TARGETS] or values.ndim != 2 or values.shape[1] != len(target_order) or len(values) < len(target_order)):
        raise ValueError("Expected training columns in the specified three- or four-target order.")
    if(not np.isfinite(values).all() or (values.std(axis=0) == 0).any()):
        raise ValueError("PCA training targets must be finite with nonzero variation.")

    scaler = StandardScaler()
    standardized = scaler.fit_transform(values)
    pca = PCA(n_components=len(target_order), svd_solver="full")
    pca.fit(standardized)
    weights = pca.components_[0].copy()

    if(weights.sum() < 0):
        weights = -weights

    scores = np.dot(standardized - pca.mean_, weights)
    parameters = {"target_order": list(target_order), "fit_split": "train", "fit_values": "actual next-season targets", "training_examples": len(values), "cohort": "defined rate targets and prior baselines", "target_means": scaler.mean_.tolist(), "target_stds": scaler.scale_.tolist(), "pca_center": pca.mean_.tolist(), "component_weights": weights.tolist(), "score_mean": float(scores.mean()), "score_std": float(scores.std(ddof=0)), "explained_variance_ratio": float(pca.explained_variance_ratio_[0]), "all_components_explained_variance_ratio": pca.explained_variance_ratio_.tolist(), "model_data_sha256": data_hash, "interpretation": "Zero is the actual training-composite mean; one unit is one training-composite standard deviation. This is a performance score, not a salary-value score."}

    if(target_order == FOUR_TARGETS):
        parameters["interpretation"] = "A hybrid of PIE, WS48, BPM effectiveness and season-long playing time, in training-composite standard deviations; not a salary-value or causal impact estimate."

    return parameters


def combine_scores(values, parameters):
    """Apply the fixed training scaler, first PCA component, and score scale."""
    values = np.asarray(values, dtype=np.float64)

    if(parameters["target_order"] not in [TARGETS, FOUR_TARGETS] or values.ndim != 2 or values.shape[1] != len(parameters["target_order"])):
        raise ValueError("PCA columns must follow the saved three- or four-target order.")
    if(not np.isfinite(values).all()):
        raise ValueError("Composite inputs must be finite.")

    standardized = (values - np.array(parameters["target_means"])) / np.array(parameters["target_stds"])
    centered = standardized - np.array(parameters["pca_center"])
    score = np.dot(centered, np.array(parameters["component_weights"]))
    return (score - parameters["score_mean"]) / parameters["score_std"]


def load_bundle(folder):
    """Verify the frozen artifacts and return manifest, networks, PCA parameters."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))

    if(manifest["target_order"] not in [TARGETS, FOUR_TARGETS] or not manifest["selection_frozen"]):
        raise ValueError("Expected a frozen three- or four-target bundle.")

    for name, expected_hash in manifest["artifact_sha256"].items():
        if(hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected_hash):
            raise ValueError("Frozen artifact was changed: " + name)

    models = {}

    for target in manifest["target_order"]:
        parameters, metadata = load_model(folder / manifest["models"][target]["checkpoint"])

        if(metadata["target"] != target or metadata["feature_names"] != manifest["feature_names"] or metadata["model_data_sha256"] != manifest["model_data_sha256"]):
            raise ValueError("Checkpoint does not match the frozen bundle: " + target)

        models[target] = (parameters, metadata)

    pca_parameters = json.loads((folder / "pca_parameters.json").read_text(encoding="utf-8"))

    if(pca_parameters["model_data_sha256"] != manifest["model_data_sha256"] or pca_parameters["target_order"] != manifest["target_order"]):
        raise ValueError("PCA and networks were built on different arrays.")

    return manifest, models, pca_parameters


def predict_bundle(X, models, pca_parameters, feature_names):
    """Predict from preprocessed ROWS: (examples, features), in feature_names order.

    Returns columns in the saved PCA target_order and the final composite vector.
    Raw player statistics must first use the saved input preprocessing.
    """
    X = np.asarray(X, dtype=np.float32)

    if(X.ndim != 2 or X.shape[1] != len(feature_names) or not np.isfinite(X).all()):
        raise ValueError("Expected finite preprocessed rows in the saved feature order.")

    columns = []

    for target in pca_parameters["target_order"]:
        parameters, metadata = models[target]

        if(metadata["feature_names"] != list(feature_names)):
            raise ValueError("Feature order differs from the checkpoint.")

        columns.append(predict(X.T, parameters, metadata["target_mean"], metadata["target_scale"], metadata.get("prediction_minimum")))

    predictions = np.column_stack(columns).astype(np.float64)
    scores = combine_scores(predictions, pca_parameters)
    return predictions, scores
