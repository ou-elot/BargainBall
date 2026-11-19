"""Walk through one real training example using the CS230 notebook functions.

Run: python inspect_layers.py
Show dropout: python inspect_layers.py --keep-prob 0.75
Inspect a trained network: python inspect_layers.py --model PATH_TO_BEST_MODEL.npz
"""

from pathlib import Path
import argparse
import csv
import hashlib
import numpy as np
from numpy_network import initialize_parameters_deep, L_model_forward, L_model_backward, compute_cost, initialize_adam, update_parameters_with_adam, copy_parameters, load_model
from train_numpy import load_target_arrays

ROOT = Path(__file__).resolve().parent


def show_layers(X, parameters, feature_names, caches):
    print ("\nINPUT X: " + str(X.shape) + " = (features, examples)")
    print ("Numeric features are already standardized; position features are one-hot.")

    for i in range(len(feature_names)):
        print ("  " + feature_names[i] + " = " + str(round(float(X[i, 0]), 6)))

    for l in range(1, len(caches) + 1):
        layer_cache, D = caches[l - 1]
        linear_cache, Z = layer_cache
        A_prev, W, b = linear_cache
        print ("\nLAYER " + str(l) + " | W" + str(l) + " " + str(W.shape) + " | b" + str(l) + " " + str(b.shape))
        print ("  Z" + str(l) + " = W" + str(l) + " @ A_prev + b" + str(l))
        print ("  First neuron's weighted input terms:")

        for i in range(W.shape[1]):
            label = "previous neuron " + str(i + 1)

            if(l == 1):
                label = feature_names[i]

            term = float(W[0, i] * A_prev[i, 0])
            print ("    " + label + ": " + str(round(float(W[0, i]), 6)) + " * " + str(round(float(A_prev[i, 0]), 6)) + " = " + str(round(term, 6)))

        print ("    bias = " + str(round(float(b[0, 0]), 6)))
        print ("    sum + bias = " + str(round(float(Z[0, 0]), 6)))
        print ("  All Z values: " + np.array2string(Z[:, 0], precision=6))

        if(l < len(caches)):
            print ("  ReLU(Z): " + np.array2string(np.maximum(Z[:, 0], 0), precision=6))
            next_layer_cache, next_D = caches[l]
            next_linear_cache, next_Z = next_layer_cache
            A, next_W, next_b = next_linear_cache

            if(D is not None):
                print ("  Dropout mask D: " + np.array2string(D[:, 0]))
                print ("  After A = A * D / keep_prob: " + np.array2string(A[:, 0], precision=6))

            print ("  A sent to the next layer: " + np.array2string(A[:, 0], precision=6))
        else:
            print ("  Linear output: AL = Z (no ReLU or sigmoid here).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--row", type=int, default=0, help="Zero-based row in the original training split.")
    parser.add_argument("--target", choices=["pie", "ws_48", "bpm", "minutes"], default="pie")
    parser.add_argument("--keep-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    if(args.row < 0 or min(args.hidden) < 1 or args.seed < 0):
        parser.error("Use a nonnegative row and seed, and positive hidden sizes.")
    if(not np.isfinite(args.keep_prob) or args.keep_prob <= 0 or args.keep_prob > 1):
        parser.error("keep-prob must be greater than zero and at most one.")

    target = args.target
    dataset = "rates"
    parameters = None
    metadata = None

    if(args.model is not None):
        parameters, metadata = load_model(args.model)
        target = metadata["target"]
        dataset = metadata["dataset"]

        if(dataset not in ["original", "rates"]):
            parser.error("Checkpoint dataset must be original or rates.")

    processed = ROOT / "data" / "processed"
    suffix = ""

    if(dataset == "rates"):
        suffix = "_rates"

    data_path = processed / ("model_data" + suffix + ".npz")

    with np.load(data_path, allow_pickle=False) as data:
        if(args.row >= len(data["X_train"])):
            parser.error("row is outside the training split.")
        if(target in ["ws_48", "bpm"] and not data["rate_mask_train"][args.row]):
            parser.error("This row has an undefined rate target; choose another row.")

        X = data["X_train"][args.row:args.row + 1].astype(np.float32).T
        targets, unused = load_target_arrays(data, target, "train")
        actual = float(targets[args.row])
        feature_names = data["feature_names"].tolist()

    if(metadata is None):
        layer_dims = [X.shape[0]] + args.hidden + [1]
        parameters = initialize_parameters_deep(layer_dims, args.seed)
        target_mean = 0.0
        target_scale = 1.0
        print ("UNTRAINED, randomly initialized network. Its prediction is not a fitted estimate.")
    else:
        if(metadata["feature_names"] != feature_names):
            parser.error("Checkpoint feature order differs from the dataset.")
        if(metadata["model_data_sha256"] != hashlib.sha256(data_path.read_bytes()).hexdigest()):
            parser.error("Checkpoint was trained on different arrays.")
        layer_dims = metadata["layer_dims"]
        target_mean = metadata["target_mean"]
        target_scale = metadata["target_scale"]
        print ("TRAINED checkpoint: " + str(args.model))

    with (processed / ("train_rows" + suffix + ".csv")).open(newline="", encoding="utf-8") as file:
        for i, row in enumerate(csv.DictReader(file)):
            if(i == args.row):
                print ("Player: " + row["player"] + " | input season ending " + row["season"] + " | target season ending " + row["target_season"])
                break

    print ("Layer dimensions: " + str(layer_dims) + " | target: " + target.upper())
    print ("keep_prob = " + str(args.keep_prob) + " | drop probability = " + str(1.0 - args.keep_prob))
    AL, caches = L_model_forward(X, parameters, args.keep_prob, training=True, rng=np.random.default_rng(args.seed + 2))
    show_layers(X, parameters, feature_names, caches)
    prediction = float(AL[0, 0]) * target_scale + target_mean

    if(target == "minutes"):
        prediction = max(prediction, 0.0)
        print ("Decoded total-minute predictions are floored at zero.")

    print ("\nPrediction in original target units: " + str(prediction))
    print ("Observed next-season target: " + str(actual))

    Y = np.array([[(actual - target_mean) / target_scale]], dtype=np.float32)
    grads = L_model_backward(AL, Y, caches, args.keep_prob)
    print ("\nBACKPROPAGATION: dAL = 2 * (AL - Y), then work backward through the layers.")

    for l in range(len(layer_dims) - 1, 0, -1):
        print ("  dW" + str(l) + " shape " + str(grads["dW" + str(l)].shape) + " | db" + str(l) + " shape " + str(grads["db" + str(l)].shape))

    print ("  dW1, first neuron: " + np.array2string(grads["dW1"][0], precision=6))
    updated = copy_parameters(parameters)
    v, s = initialize_adam(updated)
    update_parameters_with_adam(updated, grads, v, s, t=1)
    after, unused = L_model_forward(X, updated, args.keep_prob, training=True, rng=np.random.default_rng(args.seed + 2))
    print ("\nONE DEMONSTRATION ADAM UPDATE on this example (fresh optimizer state):")
    print ("  W1[0, 0]: " + str(float(parameters["W1"][0, 0])) + " -> " + str(float(updated["W1"][0, 0])))
    print ("  MSE before: " + str(compute_cost(AL, Y)) + " | after: " + str(compute_cost(after, Y)))
    print ("  Same dropout masks used for this before/after comparison. No files are saved.")
    print ("  This single-example update illustrates the mechanism; it is not model evaluation.")


if(__name__ == "__main__"):
    main()
