"""Inference helpers for the shared multi-output four-stat model."""

from pathlib import Path
import json

import numpy as np

from multitask_network import predict
from numpy_pipeline import FOUR_TARGETS, combine_scores


def load_bundle(folder):
    """Load one four-output checkpoint and its fixed, training-fitted PCA."""
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    pca_parameters = json.loads(
        (folder / "pca_parameters.json").read_text(encoding="utf-8")
    )
    if metadata.get("target_order") != FOUR_TARGETS:
        raise ValueError("Expected the PIE / WS48 / BPM / minutes multi-output model.")
    if pca_parameters.get("target_order") != FOUR_TARGETS:
        raise ValueError("Saved PCA target order does not match the model.")
    parameters = {}
    with np.load(folder / "best_model.npz", allow_pickle=False) as checkpoint:
        saved_metadata = json.loads(str(checkpoint["metadata"].item()))
        if saved_metadata != metadata:
            raise ValueError("Checkpoint metadata does not match metadata.json.")
        for layer in range(1, len(metadata["layer_dims"])):
            for prefix in ("W", "b"):
                name = prefix + str(layer)
                parameters[name] = checkpoint[name].copy()
    return metadata, parameters, pca_parameters


def predict_bundle(inputs, metadata, parameters, pca_parameters, feature_names):
    """Predict all four statistics and their fixed PCA composite.

    `inputs` must already use the saved preprocessing and feature order.
    """
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != len(metadata["feature_names"]):
        raise ValueError("Expected rows in the saved feature order.")
    if list(feature_names) != metadata["feature_names"]:
        raise ValueError("Input feature names/order differ from the trained model.")
    if not np.isfinite(inputs).all():
        raise ValueError("Input matrix contains nonfinite values.")
    statistics = predict(
        inputs.T,
        parameters,
        np.asarray(metadata["target_means"]),
        np.asarray(metadata["target_scales"]),
        minute_index=3,
    )
    composite = combine_scores(statistics, pca_parameters)
    return statistics, composite
