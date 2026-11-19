"""A NumPy regression network organized like the CS230 course notebooks.

Read in this order: initialize_parameters_deep, linear_forward,
linear_activation_forward, L_model_forward, compute_cost, linear_backward,
linear_activation_backward, L_model_backward, update_parameters_with_adam.

X has shape (features, examples); Y has shape (1, examples).
Hidden layers use ReLU. The final layer is LINEAR because NBA statistics are
continuous targets. The loss is MSE rather than classification cross-entropy.
"""

import numpy as np


def initialize_parameters_deep(layer_dims, seed=42, dtype=np.float32):
    """Create W1, b1, ..., WL, bL for dimensions such as [22, 32, 16, 1].

    Wl: (neurons in layer l, neurons in layer l-1).
    bl: (neurons in layer l, 1), broadcast across the examples.
    He initialization uses sqrt(2 / incoming neurons) for the ReLU network.
    """
    if(len(layer_dims) < 2 or layer_dims[-1] != 1):
        raise ValueError("Use input size, hidden sizes, and one regression output.")
    rng = np.random.default_rng(seed)
    parameters = {}

    L = len(layer_dims)

    for l in range(1, L):
        if(layer_dims[l - 1] < 1 or layer_dims[l] < 1):
            raise ValueError("Layer sizes must be positive.")
        scale = np.sqrt(2.0 / layer_dims[l - 1])
        parameters["W" + str(l)] = (rng.standard_normal((layer_dims[l], layer_dims[l - 1])) * scale).astype(dtype)
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1), dtype=dtype)

    return parameters


def linear_forward(A, W, b):
    """Calculate Z = W A + b and save the inputs needed for backpropagation."""
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def relu(Z):
    """ReLU keeps positive values and replaces negative values with zero."""
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """The derivative of ReLU is 1 for Z > 0 and 0 for Z <= 0."""
    Z = cache
    dZ = dA * (Z > 0)
    return dZ


def linear_activation_forward(A_prev, W, b, activation):
    """One LINEAR -> RELU or LINEAR -> LINEAR layer, as in the notebook."""
    Z, linear_cache = linear_forward(A_prev, W, b)

    if(activation == "relu"):
        A, activation_cache = relu(Z)
    elif(activation == "linear"):
        A = Z
        activation_cache = Z
    else:
        raise ValueError("Activation must be relu or linear.")

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters, keep_prob=1.0, training=False, rng=None):
    """Run [LINEAR -> RELU -> optional DROPOUT] * (L-1) -> LINEAR.

    keep_prob is the probability of KEEPING an activation: 0.9 means 10%
    dropout. Each cache contains (layer_cache, D), where D is that hidden
    layer's dropout mask, or None. Prediction always disables dropout.
    Pass a persistent random generator when training with dropout; resetting
    the seed inside this function would repeat the same masks every batch.
    """
    if(not np.isfinite(keep_prob) or keep_prob <= 0 or keep_prob > 1):
        raise ValueError("keep_prob must be greater than zero and at most one.")
    if(training and keep_prob < 1 and rng is None):
        raise ValueError("Supply a random generator for training with dropout.")

    A = X
    caches = []
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, layer_cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        D = None

        if(training and keep_prob < 1):
            # The same four dropout steps as Regularization.ipynb.
            D = rng.random(A.shape)
            D = (D < keep_prob).astype(A.dtype)
            A = A * D
            A = A / keep_prob

        caches.append((layer_cache, D))

    AL, layer_cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
    caches.append((layer_cache, None))
    return AL, caches


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5, rng=None):
    """Notebook-style training entry point; dropout applies to hidden layers."""
    return L_model_forward(X, parameters, keep_prob, training=True, rng=rng)


def compute_cost(AL, Y, parameters=None, weight_decay=0.0):
    """MSE = sum((AL - Y)^2) / m, plus optional coupled L2.

    Coupled L2 applies to weights AND biases, matching the existing experiment.
    It is not divided by m; weight_decay is the coefficient of this penalty.
    """
    if(AL.shape != Y.shape or AL.ndim != 2 or AL.shape[0] != 1):
        raise ValueError("Predictions and targets must both have shape (1, examples).")
    if(weight_decay != 0 and parameters is None):
        raise ValueError("Parameters are required to calculate the L2 penalty.")

    m = Y.shape[1]
    cost = float(np.sum((AL - Y) ** 2) / m)

    if(parameters is not None):
        for value in parameters.values():
            cost = cost + 0.5 * weight_decay * float(np.sum(value ** 2))

    return cost


def linear_backward(dZ, cache):
    """Backpropagate through Z = W A_prev + b.

    As in the notebook, average dW and db over the m examples here. dA_prev
    remains unaveraged so the preceding layer does not average it twice.
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1.0 / m) * np.dot(dZ, A_prev.T)
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """Apply the activation derivative, then backpropagate the linear part."""
    linear_cache, activation_cache = cache

    if(activation == "relu"):
        dZ = relu_backward(dA, activation_cache)
    elif(activation == "linear"):
        dZ = dA
    else:
        raise ValueError("Activation must be relu or linear.")

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, keep_prob=1.0, weight_decay=0.0):
    """Return notebook-style grads: dW1, db1, ..., dWL, dbL, plus dA values.

    MSE starts with 2 * (AL - Y). linear_backward supplies the factor 1/m.
    Use the SAME keep_prob and cached masks as in the forward pass.
    Stored dA values use the notebook's unaveraged-per-example convention.
    """
    if(AL.shape != Y.shape or AL.ndim != 2 or AL.shape[0] != 1):
        raise ValueError("Predictions and targets must both have shape (1, examples).")
    if(not np.isfinite(keep_prob) or keep_prob <= 0 or keep_prob > 1):
        raise ValueError("keep_prob must be greater than zero and at most one.")

    grads = {}
    L = len(caches)
    dAL = 2.0 * (AL - Y)

    layer_cache, D = caches[L - 1]
    dA_prev, dW, db = linear_activation_backward(dAL, layer_cache, "linear")
    linear_cache, activation_cache = layer_cache
    A_prev, W, b = linear_cache
    grads["dW" + str(L)] = dW + weight_decay * W
    grads["db" + str(L)] = db + weight_decay * b
    grads["dA" + str(L - 1)] = dA_prev

    for l in reversed(range(1, L)):
        layer_cache, D = caches[l - 1]
        dA = grads["dA" + str(l)]

        if(D is not None):
            # Reuse the forward mask and apply the same 1/keep_prob scaling.
            dA = dA * D
            dA = dA / keep_prob

        dA_prev, dW, db = linear_activation_backward(dA, layer_cache, "relu")
        linear_cache, activation_cache = layer_cache
        A_prev, W, b = linear_cache
        grads["dW" + str(l)] = dW + weight_decay * W
        grads["db" + str(l)] = db + weight_decay * b
        grads["dA" + str(l - 1)] = dA_prev

    return grads


def backward_propagation_with_dropout(AL, Y, caches, keep_prob, weight_decay=0.0):
    """Notebook-style dropout backward pass, generalized to any hidden sizes."""
    return L_model_backward(AL, Y, caches, keep_prob, weight_decay)


def initialize_adam(parameters):
    """v stores moving-average gradients; s stores moving-average squares."""
    v = {}
    s = {}

    for key, value in parameters.items():
        v["d" + key] = np.zeros_like(value)
        s["d" + key] = np.zeros_like(value)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Update parameters in place, returning the five values from the notebook.

    t counts MINI-BATCH UPDATES, not epochs. Bias correction depends on t.
    """
    v_corrected = {}
    s_corrected = {}

    for key in parameters:
        gradient_key = "d" + key
        v[gradient_key] = beta1 * v[gradient_key] + (1.0 - beta1) * grads[gradient_key]
        s[gradient_key] = beta2 * s[gradient_key] + (1.0 - beta2) * grads[gradient_key] ** 2
        v_corrected[gradient_key] = v[gradient_key] / (1.0 - beta1 ** t)
        s_corrected[gradient_key] = s[gradient_key] / (1.0 - beta2 ** t)
        parameters[key] -= learning_rate * v_corrected[gradient_key] / (np.sqrt(s_corrected[gradient_key]) + epsilon)

    return parameters, v, s, v_corrected, s_corrected


def update_parameters(parameters, grads, learning_rate):
    """The simpler gradient-descent update from the step-by-step notebook."""
    for key in parameters:
        parameters[key] -= learning_rate * grads["d" + key]
    return parameters


def copy_parameters(parameters):
    copied = {}

    for key, value in parameters.items():
        copied[key] = value.copy()

    return copied


def predict(inputs, parameters, target_mean=0.0, target_scale=1.0, prediction_minimum=None):
    predictions, caches = L_model_forward(inputs, parameters)
    predictions = predictions.reshape(-1) * target_scale + target_mean

    if(prediction_minimum is not None):
        predictions = np.maximum(predictions, prediction_minimum)

    return predictions


def load_model(path):
    """Return parameters and metadata; inference inputs have shape (features, examples)."""
    import json
    parameters = {}

    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"].item()))

        for layer in range(1, len(metadata["layer_dims"])):
            for prefix in ["W", "b"]:
                key = prefix + str(layer)
                parameters[key] = checkpoint[key].copy()

    return parameters, metadata
