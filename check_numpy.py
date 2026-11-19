"""Check manual gradients, dropout, Adam, and regression learning. NumPy only."""

import numpy as np
from numpy_network import initialize_parameters_deep, L_model_forward, compute_cost, L_model_backward, initialize_adam, update_parameters_with_adam, predict, forward_propagation_with_dropout, backward_propagation_with_dropout


def check_gradients(dropout):
    rng = np.random.default_rng(5)
    inputs = rng.normal(size=(3, 6))
    targets = rng.normal(size=(1, 6))
    parameters = initialize_parameters_deep([3, 4, 2, 1], 7, np.float64)
    # Nonzero biases avoid ReLU kinks when dropout removes all incoming activations.
    for key in parameters:
        if(key.startswith("b")):
            parameters[key][:] = 0.17

    keep_prob = 1.0 - dropout
    predictions, caches = L_model_forward(inputs, parameters, keep_prob, True, np.random.default_rng(9))
    gradients = L_model_backward(predictions, targets, caches, keep_prob, 0.03)
    numerical = []
    analytical = []
    epsilon = 1e-6

    for key, values in parameters.items():
        for index in np.ndindex(values.shape):
            original = values[index]
            values[index] = original + epsilon
            plus, unused = L_model_forward(inputs, parameters, keep_prob, True, np.random.default_rng(9))
            plus_loss = compute_cost(plus, targets, parameters, 0.03)
            values[index] = original - epsilon
            minus, unused = L_model_forward(inputs, parameters, keep_prob, True, np.random.default_rng(9))
            minus_loss = compute_cost(minus, targets, parameters, 0.03)
            values[index] = original
            numerical.append((plus_loss - minus_loss) / (2.0 * epsilon))
            analytical.append(gradients["d" + key][index])

    numerical = np.array(numerical)
    analytical = np.array(analytical)
    relative_error = np.linalg.norm(numerical - analytical) / max(np.linalg.norm(numerical) + np.linalg.norm(analytical), 1e-12)
    assert relative_error < 1e-7, "Gradient check failed: " + str(relative_error)
    print ("Gradient check | dropout " + str(dropout) + " | relative error " + str(relative_error))
    expected, unused = L_model_forward(inputs, parameters)
    actual, unused = L_model_forward(inputs, parameters, keep_prob, False)
    np.testing.assert_array_equal(expected, actual)


def check_adam():
    parameters = {"W1": np.array([[0.5, -0.2]], dtype=np.float64)}
    first, second = initialize_adam(parameters)
    expected = parameters["W1"].copy()
    history = []

    for step in range(1, 5):
        gradient = np.array([[0.3 * step, -0.1 / step]])
        history.append(gradient)
        update_parameters_with_adam(parameters, {"dW1": gradient}, first, second, step)
        first_corrected = np.zeros_like(gradient)
        second_corrected = np.zeros_like(gradient)

        for index, old_gradient in enumerate(history):
            first_corrected += 0.1 * 0.9 ** (step - index - 1) * old_gradient
            second_corrected += 0.001 * 0.999 ** (step - index - 1) * old_gradient ** 2

        first_corrected /= 1.0 - 0.9 ** step
        second_corrected /= 1.0 - 0.999 ** step
        expected -= 0.001 * first_corrected / (np.sqrt(second_corrected) + 1e-8)
        np.testing.assert_allclose(parameters["W1"], expected, rtol=1e-12, atol=1e-12)

    print ("Adam bias correction and multiple updates: passed")


def check_learning():
    rng = np.random.default_rng(12)
    inputs = rng.normal(size=(3, 128))
    targets = 2.0 * inputs[0:1] - 0.7 * inputs[1:2] + 0.3
    parameters = initialize_parameters_deep([3, 8, 1], 12, np.float64)
    first, second = initialize_adam(parameters)

    for step in range(1, 1501):
        predictions, caches = L_model_forward(inputs, parameters)
        gradients = L_model_backward(predictions, targets, caches)
        update_parameters_with_adam(parameters, gradients, first, second, step, 0.01)

    rmse = float(np.sqrt(np.mean((predict(inputs, parameters) - targets.reshape(-1)) ** 2)))
    assert rmse < 0.03, "Synthetic regression failed."
    print ("Synthetic regression RMSE: " + str(rmse))


def check_dropout():
    """Check the actual four dropout steps, not only gradient consistency."""
    inputs = np.ones((3, 64))
    parameters = initialize_parameters_deep([3, 8, 4, 1], 3, np.float64)
    keep_prob = 0.75
    predictions, caches = forward_propagation_with_dropout(inputs, parameters, keep_prob, np.random.default_rng(4))
    found_dropped = False
    found_kept = False

    for l in range(len(caches) - 1):
        layer_cache, D = caches[l]
        linear_cache, Z = layer_cache
        next_layer_cache, next_D = caches[l + 1]
        next_linear_cache, next_Z = next_layer_cache
        A, next_W, next_b = next_linear_cache
        np.testing.assert_allclose(A, np.maximum(Z, 0) * D / keep_prob)
        assert np.isin(D, [0, 1]).all()
        found_dropped = found_dropped or bool((D == 0).any())
        found_kept = found_kept or bool((D == 1).any())

    assert found_dropped and found_kept
    assert caches[-1][1] is None, "The output layer must not use dropout."
    grads = backward_propagation_with_dropout(predictions, np.zeros_like(predictions), caches, keep_prob)
    assert grads["dW1"].shape == parameters["W1"].shape
    print ("Dropout masks, inverted scaling, and linear output: passed")


if(__name__ == "__main__"):
    check_gradients(0.0)
    check_gradients(0.25)
    check_dropout()
    check_adam()
    check_learning()
    print ("All checks passed.")
