"""Inference helpers for the multi-output forecast-profile model."""

from pathlib import Path
import json

import numpy as np

from multitask_network import predict
from profile_targets import PCA_TARGETS, PROFILE_TARGETS


def load_bundle(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("target_order") != PROFILE_TARGETS:
        raise ValueError("Bundle target order does not match profile_targets.py.")
    with np.load(folder / "best_model.npz", allow_pickle=False) as saved:
        if json.loads(str(saved["metadata"].item())) != metadata:
            raise ValueError("Checkpoint metadata differs from metadata.json.")
        parameters = {}
        for layer in range(1, len(metadata["layer_dims"])):
            for prefix in ("W", "b"):
                name = prefix + str(layer)
                parameters[name] = saved[name].copy()
    pca = json.loads((folder / "profile_pca.json").read_text(encoding="utf-8"))
    if pca.get("skill_target_order") != PCA_TARGETS:
        raise ValueError("PCA skill target order does not match profile_targets.py.")
    return metadata, parameters, pca


def predict_profile(inputs, feature_names, metadata, parameters, pca):
    """Predict all target stats and return retained PCA skill-profile coordinates."""
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != len(metadata["feature_names"]):
        raise ValueError("Inputs must be rows in the trained feature dimension.")
    if list(feature_names) != metadata["feature_names"]:
        raise ValueError("Input feature order differs from the trained model.")
    if not np.isfinite(inputs).all():
        raise ValueError("Input matrix contains nonfinite values.")
    predictions = predict(
        inputs.T,
        parameters,
        np.asarray(metadata["target_means"]),
        np.asarray(metadata["target_scales"]),
        minute_index=PROFILE_TARGETS.index("minutes"),
    )
    skill_indices = [PROFILE_TARGETS.index(target) for target in PCA_TARGETS]
    skills = predictions[:, skill_indices]
    standardized = (
        skills - np.asarray(pca["skill_means"])
    ) / np.asarray(pca["skill_scales"])
    coordinates = (
        standardized - np.asarray(pca["pca_mean"])
    ) @ np.asarray(pca["components"]).T
    return predictions, coordinates
