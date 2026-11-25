"""Shared NumPy MLP primitives for simultaneous multi-target regression."""

import numpy as np


def initialize_parameters(layer_dims, seed=42):
    """Initialize a ReLU MLP with a multi-output linear final layer."""
    if len(layer_dims) < 2 or any(size < 1 for size in layer_dims):
        raise ValueError("Layer dimensions must be positive and include input/output sizes.")
    rng = np.random.default_rng(seed)
    parameters = {}
    for layer in range(1, len(layer_dims)):
        scale = np.sqrt(2.0 / layer_dims[layer - 1])
        parameters[f"W{layer}"] = (
            rng.standard_normal((layer_dims[layer], layer_dims[layer - 1])) * scale
        ).astype(np.float32)
        parameters[f"b{layer}"] = np.zeros((layer_dims[layer], 1), dtype=np.float32)
    return parameters


def forward(inputs, parameters, keep_prob=1.0, training=False, rng=None):
    """Forward pass; input shape is features-by-examples."""
    if not 0 < keep_prob <= 1:
        raise ValueError("keep_prob must be in (0, 1].")
    if training and keep_prob < 1 and rng is None:
        raise ValueError("A random generator is required when dropout is enabled.")
    activations = inputs
    hidden_caches = []
    n_layers = len(parameters) // 2
    for layer in range(1, n_layers):
        previous = activations
        z = parameters[f"W{layer}"] @ previous + parameters[f"b{layer}"]
        activations = np.maximum(z, 0)
        dropout_mask = None
        if training and keep_prob < 1:
            dropout_mask = (
                (rng.random(activations.shape) < keep_prob).astype(np.float32)
                / keep_prob
            )
            activations = activations * dropout_mask
        hidden_caches.append((previous, z, dropout_mask))
    output_layer = n_layers
    output_input = activations
    predictions = (
        parameters[f"W{output_layer}"] @ output_input
        + parameters[f"b{output_layer}"]
    )
    return predictions, (hidden_caches, output_input)


def masked_mse(predictions, targets, mask):
    """Mean of per-target MSEs, each averaged over its observed labels."""
    mask = np.asarray(mask, dtype=bool)
    counts = mask.sum(axis=1)
    valid_targets = counts > 0
    if not valid_targets.any():
        raise ValueError("Batch has no observed targets.")
    per_target = np.zeros(predictions.shape[0], dtype=np.float64)
    for target in np.flatnonzero(valid_targets):
        errors = predictions[target, mask[target]] - targets[target, mask[target]]
        per_target[target] = np.mean(errors.astype(np.float64) ** 2)
    return float(per_target[valid_targets].mean()), per_target, counts


def backward(predictions, targets, mask, caches, parameters, weight_decay=0.0):
    """Backpropagate a task-balanced masked MSE and coupled L2 penalty."""
    hidden_caches, output_input = caches
    n_layers = len(parameters) // 2
    counts = mask.sum(axis=1).astype(np.float32)
    n_valid_targets = max(int(np.count_nonzero(counts)), 1)
    safe_counts = np.maximum(counts, 1.0)
    d_z = (
        2.0
        * (predictions - targets)
        * mask
        / safe_counts[:, None]
        / n_valid_targets
    )
    gradients = {}
    gradients[f"dW{n_layers}"] = (
        d_z @ output_input.T + weight_decay * parameters[f"W{n_layers}"]
    )
    gradients[f"db{n_layers}"] = (
        d_z.sum(axis=1, keepdims=True) + weight_decay * parameters[f"b{n_layers}"]
    )
    d_a = parameters[f"W{n_layers}"].T @ d_z

    for layer in reversed(range(1, n_layers)):
        previous, z, dropout_mask = hidden_caches[layer - 1]
        if dropout_mask is not None:
            d_a = d_a * dropout_mask
        d_z = d_a * (z > 0)
        gradients[f"dW{layer}"] = (
            d_z @ previous.T + weight_decay * parameters[f"W{layer}"]
        )
        gradients[f"db{layer}"] = (
            d_z.sum(axis=1, keepdims=True) + weight_decay * parameters[f"b{layer}"]
        )
        d_a = parameters[f"W{layer}"].T @ d_z
    return gradients


def adam_initialize(parameters):
    first = {key: np.zeros_like(value) for key, value in parameters.items()}
    second = {key: np.zeros_like(value) for key, value in parameters.items()}
    return first, second


def adam_update(parameters, gradients, first, second, step, learning_rate,
                beta1=0.9, beta2=0.999, epsilon=1e-8):
    for key in parameters:
        first[key] = beta1 * first[key] + (1.0 - beta1) * gradients["d" + key]
        second[key] = beta2 * second[key] + (1.0 - beta2) * gradients["d" + key] ** 2
        first_hat = first[key] / (1.0 - beta1**step)
        second_hat = second[key] / (1.0 - beta2**step)
        parameters[key] -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
    return parameters, first, second


def predict(inputs, parameters, target_means, target_scales, minute_index=None):
    predictions, _ = forward(inputs, parameters)
    decoded = predictions.astype(np.float64).T * target_scales + target_means
    if minute_index is not None:
        decoded[:, minute_index] = np.maximum(decoded[:, minute_index], 0.0)
    return decoded
