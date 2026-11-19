"""Train manual NumPy regressors using existing preprocessed train/dev arrays."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from numpy_network import initialize_parameters_deep, L_model_forward, L_model_backward, initialize_adam, update_parameters_with_adam, copy_parameters, predict

ROOT = Path(__file__).resolve().parent


def metrics(predictions, targets):
    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    return float(np.sqrt(np.mean(errors ** 2))), float(np.mean(np.abs(errors)))


def load_target_arrays(data, target, split):
    """Minutes are an observed NEXT-season target, never a future input feature."""
    if(target == "minutes"):
        actual = data["next_season_minutes_" + split]
        baseline = data["prior_minutes_" + split]

        if(not np.isfinite(actual).all() or not np.isfinite(baseline).all() or (actual < 0).any() or (baseline < 0).any()):
            raise ValueError("Minutes and prior-minute baselines must be finite and nonnegative.")
    else:
        actual = data["y_" + target + "_" + split]
        baseline = data["baseline_" + target + "_" + split]

    return actual.copy(), baseline.copy()


def train_target(target, args):
    filename = "model_data.npz"

    if(args.dataset == "rates"):
        filename = "model_data_rates.npz"

    data_path = args.data_dir / filename
    output = args.output / args.dataset / target
    output.mkdir(parents=True, exist_ok=True)

    # Never access test arrays during training or model selection.
    with np.load(data_path, allow_pickle=False) as data:
        x_train = data["X_train"].astype(np.float32)
        x_dev = data["X_dev"].astype(np.float32)
        y_train, unused = load_target_arrays(data, target, "train")
        y_dev, baseline = load_target_arrays(data, target, "dev")
        y_train = y_train.astype(np.float32)
        y_dev = y_dev.astype(np.float32)
        baseline = baseline.astype(np.float64)
        feature_names = data["feature_names"].tolist()

        # PIE retains its original training rows. Rate targets exclude undefined rates.
        if(target in ["ws_48", "bpm"]):
            train_mask = data["rate_mask_train"].astype(bool)
            dev_mask = data["rate_mask_dev"].astype(bool)
            x_train = x_train[train_mask]
            y_train = y_train[train_mask]
            x_dev = x_dev[dev_mask]
            y_dev = y_dev[dev_mask]
            baseline = baseline[dev_mask]

    if(x_train.ndim != 2 or x_dev.ndim != 2 or x_train.shape[1] != x_dev.shape[1]):
        raise ValueError("Training and development inputs must have matching feature counts.")
    if(y_train.shape != (len(x_train),) or y_dev.shape != (len(x_dev),) or baseline.shape != y_dev.shape):
        raise ValueError("Unexpected target or baseline shapes.")
    if(len(x_train) == 0 or len(x_dev) == 0 or len(feature_names) != x_train.shape[1]):
        raise ValueError("Empty split or inconsistent feature names.")

    for values in [x_train, x_dev, y_train, y_dev, baseline]:
        if(not np.isfinite(values).all()):
            raise ValueError("Inputs, targets, and baseline must be finite.")

    # Preserve the previous training convention: PIE unscaled, other targets standardized.
    target_mean = 0.0
    target_scale = 1.0
    prediction_minimum = None

    if(target == "minutes"):
        # The linear network learns standardized minutes with MSE. At inference,
        # floor the decoded minute prediction at zero, including dev selection.
        prediction_minimum = 0.0

    if(target != "pie"):
        target_mean = float(np.mean(y_train))
        target_scale = float(np.std(y_train))

        if(target_scale <= 0):
            raise ValueError("Training target has no variation.")

    training_targets = ((y_train - target_mean) / target_scale).reshape(1, -1)
    x_train = x_train.T
    x_dev = x_dev.T
    layer_dims = [x_train.shape[0]] + args.hidden + [1]
    baseline_rmse, baseline_mae = metrics(baseline, y_dev)
    best_overall = float("inf")
    history = []
    runs = []
    started = time.perf_counter()
    print ("\nTraining target: " + target.upper() + " | layer dimensions: " + str(layer_dims), flush=True)
    print ("Each column is one example. Hidden layers use ReLU; the output is linear.")

    for l in range(1, len(layer_dims)):
        print ("Layer " + str(l) + ": W" + str(l) + " shape " + str((layer_dims[l], layer_dims[l - 1])) + " | b" + str(l) + " shape " + str((layer_dims[l], 1)))

    for dropout in args.dropout:
        # CLI dropout is a DROP probability; the notebook uses a KEEP probability.
        keep_prob = 1.0 - dropout
        parameters = initialize_parameters_deep(layer_dims, args.seed)
        v, s = initialize_adam(parameters)
        shuffle_rng = np.random.default_rng(args.seed + 1)
        dropout_rng = np.random.default_rng(args.seed + 2)
        best_rmse = float("inf")
        stale_epochs = 0
        step = 0

        for epoch in range(1, args.epochs + 1):
            order = shuffle_rng.permutation(x_train.shape[1])
            squared_error = 0.0

            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                X = x_train[:, indices]
                Y = training_targets[:, indices]

                # 1. Forward propagation: inputs -> hidden activations -> prediction.
                AL, caches = L_model_forward(X, parameters, keep_prob, True, dropout_rng)

                # 2. Backpropagation: prediction errors -> dW and db for each layer.
                grads = L_model_backward(AL, Y, caches, keep_prob, args.weight_decay)

                # 3. Adam uses those gradients to update every W and b.
                step = step + 1
                parameters, v, s, v_corrected, s_corrected = update_parameters_with_adam(parameters, grads, v, s, step, args.learning_rate)
                squared_error = squared_error + float(np.sum((AL - Y) ** 2))

            train_mse = squared_error / x_train.shape[1]
            # Compare both splits at the SAME end-of-epoch weights, dropout off.
            train_predictions = predict(x_train, parameters, target_mean, target_scale, prediction_minimum)
            train_rmse, train_mae = metrics(train_predictions, y_train)
            dev_predictions = predict(x_dev, parameters, target_mean, target_scale, prediction_minimum)
            dev_rmse, dev_mae = metrics(dev_predictions, y_dev)

            if(not np.isfinite(train_mse) or not np.isfinite(train_rmse) or not np.isfinite(dev_rmse)):
                raise ValueError("Training produced a nonfinite loss.")

            history.append({"dropout": dropout, "epoch": epoch, "train_mse_scaled": train_mse, "train_eval_mse_scaled": (train_rmse / target_scale) ** 2, "train_mse": train_rmse ** 2, "train_rmse": train_rmse, "train_mae": train_mae, "dev_mse": dev_rmse ** 2, "dev_rmse": dev_rmse, "dev_mae": dev_mae})

            if(dev_rmse < best_rmse):
                best_rmse = dev_rmse
                best_epoch = epoch
                best_state = copy_parameters(parameters)
                stale_epochs = 0
            else:
                stale_epochs = stale_epochs + 1

            if(epoch == 1 or epoch % 25 == 0):
                print ("Dropout " + str(dropout) + " | epoch " + str(epoch) + " | train RMSE " + str(round(train_rmse, 6)) + " | dev RMSE " + str(round(dev_rmse, 6)), flush=True)

            if(stale_epochs >= args.patience):
                break

        predictions = predict(x_dev, best_state, target_mean, target_scale, prediction_minimum)
        dev_rmse, dev_mae = metrics(predictions, y_dev)
        train_predictions = predict(x_train, best_state, target_mean, target_scale, prediction_minimum)
        train_rmse, train_mae = metrics(train_predictions, y_train)
        result = {"dropout": dropout, "best_epoch": best_epoch, "epochs_run": epoch, "train_mse": train_rmse ** 2, "train_eval_mse_scaled": (train_rmse / target_scale) ** 2, "train_rmse": train_rmse, "train_mae": train_mae, "dev_mse": dev_rmse ** 2, "dev_rmse": dev_rmse, "dev_mae": dev_mae, "reached_epoch_limit": epoch == args.epochs}
        runs.append(result)

        if(dev_rmse < best_overall):
            best_overall = dev_rmse
            selected = result.copy()
            selected_state = copy_parameters(best_state)
            selected_predictions = predictions.copy()

    metadata = {"target": target, "dataset": args.dataset, "layer_dims": layer_dims, "feature_names": feature_names, "target_mean": target_mean, "target_scale": target_scale, "selected": selected, "runs": runs, "baseline_dev_rmse": baseline_rmse, "baseline_dev_mae": baseline_mae, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "regularization": "coupled L2 on weights and biases", "batch_size": args.batch_size, "patience": args.patience, "maximum_epochs": args.epochs, "seed": args.seed, "training_examples": x_train.shape[1], "development_examples": len(y_dev), "model_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(), "test_evaluated": False, "elapsed_seconds": time.perf_counter() - started}
    metadata["prediction_minimum"] = prediction_minimum

    if(target == "minutes"):
        metadata["target_definition"] = "Observed next-season regular-season total minutes; conditional on appearing in the following season."
        metadata["prediction_policy"] = "Linear standardized-target MSE training; decoded predictions floored at zero for inference and development selection."
    checkpoint = copy_parameters(selected_state)
    checkpoint["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    np.savez(output / "best_model.npz", **checkpoint)
    np.savez(output / "development_predictions.npz", predicted=selected_predictions, actual=y_dev, baseline=baseline)
    (output / "development_results.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print ("Selected dropout: " + str(selected["dropout"]))
    print ("Best epoch: " + str(selected["best_epoch"]))
    print ("Selected checkpoint training MSE: " + str(selected["train_mse"]))
    print ("Selected checkpoint development MSE: " + str(selected["dev_mse"]))
    print ("Development model RMSE: " + str(selected["dev_rmse"]))
    print ("Development baseline RMSE: " + str(baseline_rmse))
    print ("Development model MAE: " + str(selected["dev_mae"]))
    print ("Saved model to: " + str(output / "best_model.npz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["original", "rates"], default="rates")
    parser.add_argument("--target", choices=["all", "pie", "ws", "vorp", "ws_48", "bpm", "minutes"], default="all")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.1])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "numpy")
    args = parser.parse_args()

    if(min(args.epochs, args.patience, args.batch_size) < 1 or min(args.hidden) < 1 or args.seed < 0):
        parser.error("Epochs, patience, batch size and hidden sizes must be positive; seed must be nonnegative.")
    if(not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Use a positive finite learning rate and nonnegative finite weight decay.")

    for dropout in args.dropout:
        if(not np.isfinite(dropout) or dropout < 0 or dropout >= 1):
            parser.error("Dropout must be between zero and one, excluding one.")

    allowed = ["pie", "ws", "vorp"]

    if(args.dataset == "rates"):
        allowed = ["pie", "ws_48", "bpm", "minutes"]

    targets = [args.target]

    if(args.target == "all"):
        targets = allowed
    elif(args.target not in allowed):
        parser.error("Target does not belong to the selected dataset: " + args.target)

    # Validate every requested target before starting a training run.
    filename = "model_data.npz"

    if(args.dataset == "rates"):
        filename = "model_data_rates.npz"

    with np.load(args.data_dir / filename, allow_pickle=False) as data:
        for target in targets:
            for split in ["train", "dev"]:
                load_target_arrays(data, target, split)

    for target in targets:
        train_target(target, args)


if(__name__ == "__main__"):
    main()
