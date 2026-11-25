"""Train one shared NumPy network for PIE, WS/48, BPM, and total minutes.

The shared MLP has one four-value linear output layer. Rate-target labels use a
mask so examples with undefined WS/48 or BPM can still train PIE and minutes.
Model selection uses training/development only; test arrays are never loaded.
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import time

import numpy as np
import pandas as pd

from multitask_network import (
    adam_initialize,
    adam_update,
    backward,
    forward,
    initialize_parameters,
    masked_mse,
    predict,
)
from numpy_pipeline import (
    FOUR_TARGETS,
    combine_scores,
    fit_pca,
    load_bundle as load_independent_bundle,
    predict_bundle as predict_independent_bundle,
)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DATA_PATH = PROCESSED / "model_data_rates.npz"
TRAIN_ROWS_PATH = PROCESSED / "train_rows_rates.csv"
DEV_ROWS_PATH = PROCESSED / "dev_rows_rates.csv"
INDEPENDENT_BUNDLE = ROOT / "models" / "numpy_four_stat" / "20261109_204731_573737"
HIDDEN = [64, 32]
DROPOUTS = [0.0, 0.1]
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
SEED = 42


def target_arrays(data, split):
    actual = np.column_stack(
        [
            data["y_pie_" + split],
            data["y_ws_48_" + split],
            data["y_bpm_" + split],
            data["next_season_minutes_" + split],
        ]
    ).astype(np.float64)
    baseline = np.column_stack(
        [
            data["baseline_pie_" + split],
            data["baseline_ws_48_" + split],
            data["baseline_bpm_" + split],
            data["prior_minutes_" + split],
        ]
    ).astype(np.float64)
    mask = np.ones(actual.shape, dtype=bool)
    mask[:, 1:3] = data["rate_mask_" + split].astype(bool)[:, None]
    mask &= np.isfinite(actual)
    return actual, baseline, mask


def target_statistics(values, mask):
    means = np.zeros(values.shape[1], dtype=np.float64)
    scales = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        selected = values[mask[:, column], column]
        if not len(selected):
            raise ValueError("Training target has no observed labels: " + FOUR_TARGETS[column])
        means[column] = selected.mean()
        scales[column] = selected.std()
        if not np.isfinite(scales[column]) or scales[column] <= 0:
            raise ValueError("Training target has no finite variation: " + FOUR_TARGETS[column])
    return means, scales


def standardized(values, mask, means, scales):
    result = np.zeros(values.shape, dtype=np.float32)
    for column in range(values.shape[1]):
        valid = mask[:, column]
        result[valid, column] = (
            (values[valid, column] - means[column]) / scales[column]
        ).astype(np.float32)
    return result


def raw_metrics(predictions, actual, mask):
    result = {}
    for column, target in enumerate(FOUR_TARGETS):
        valid = mask[:, column]
        errors = predictions[valid, column] - actual[valid, column]
        result[target] = {
            "rows": int(valid.sum()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
        }
    return result


def main():
    started = time.perf_counter()
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    with np.load(DATA_PATH, allow_pickle=False) as data:
        x_train = data["X_train"].astype(np.float32)
        x_dev = data["X_dev"].astype(np.float32)
        y_train, baseline_train, mask_train = target_arrays(data, "train")
        y_dev, baseline_dev, mask_dev = target_arrays(data, "dev")
        feature_names = data["feature_names"].astype(str).tolist()
        base_hash = str(data["base_model_data_sha256"].item())

    for values in [x_train, x_dev]:
        if not np.isfinite(values).all():
            raise ValueError("Model inputs must be finite.")
    if x_train.shape[1] != len(feature_names) or x_dev.shape[1] != len(feature_names):
        raise ValueError("Input feature names do not match model arrays.")
    if len(x_train) != len(y_train) or len(x_dev) != len(y_dev):
        raise ValueError("Input and target arrays do not align.")
    if not mask_train[:, 0].all() or not mask_train[:, 3].all():
        raise ValueError("PIE and total-minute labels must be available for all training rows.")

    target_means, target_scales = target_statistics(y_train, mask_train)
    y_train_scaled = standardized(y_train, mask_train, target_means, target_scales)
    y_dev_scaled = standardized(y_dev, mask_dev, target_means, target_scales)
    input_train = x_train.T
    input_dev = x_dev.T
    layer_dims = [x_train.shape[1], *HIDDEN, len(FOUR_TARGETS)]
    parameter_candidates = []
    training_history = []
    best = None

    print("One shared multi-output network: " + str(layer_dims), flush=True)
    for dropout_index, dropout in enumerate(DROPOUTS):
        parameters = initialize_parameters(layer_dims, SEED)
        first, second = adam_initialize(parameters)
        shuffle_rng = np.random.default_rng(SEED + 10 + dropout_index)
        dropout_rng = np.random.default_rng(SEED + 20 + dropout_index)
        best_loss = float("inf")
        stale = 0
        step = 0
        best_state = None

        for epoch in range(1, MAX_EPOCHS + 1):
            order = shuffle_rng.permutation(len(x_train))
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start : start + BATCH_SIZE]
                batch_mask = mask_train[indices].T.astype(np.float32)
                batch_targets = y_train_scaled[indices].T
                predictions, caches = forward(
                    input_train[:, indices],
                    parameters,
                    keep_prob=1.0 - dropout,
                    training=True,
                    rng=dropout_rng,
                )
                gradients = backward(
                    predictions,
                    batch_targets,
                    batch_mask,
                    caches,
                    parameters,
                    WEIGHT_DECAY,
                )
                step += 1
                parameters, first, second = adam_update(
                    parameters, gradients, first, second, step, LEARNING_RATE
                )

            train_scaled_predictions = forward(input_train, parameters)[0].T
            dev_scaled_predictions = forward(input_dev, parameters)[0].T
            train_loss = masked_mse(
                train_scaled_predictions.T,
                y_train_scaled.T,
                mask_train.T,
            )[0]
            dev_loss = masked_mse(
                dev_scaled_predictions.T,
                y_dev_scaled.T,
                mask_dev.T,
            )[0]
            if not np.isfinite(train_loss) or not np.isfinite(dev_loss):
                raise ValueError("Nonfinite multi-output training loss.")
            training_history.append(
                {
                    "dropout": dropout,
                    "epoch": epoch,
                    "train_task_balanced_scaled_mse": train_loss,
                    "dev_task_balanced_scaled_mse": dev_loss,
                }
            )
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch
                best_state = {key: value.copy() for key, value in parameters.items()}
                stale = 0
            else:
                stale += 1
            if epoch == 1 or epoch % 25 == 0:
                print(
                    f"dropout={dropout:.1f} epoch={epoch} "
                    f"train masked MSE={train_loss:.5f} "
                    f"dev masked MSE={dev_loss:.5f}",
                    flush=True,
                )
            if stale >= PATIENCE:
                break

        run = {
            "dropout": dropout,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "best_dev_task_balanced_scaled_mse": best_loss,
        }
        parameter_candidates.append(run)
        if best is None or best_loss < best["loss"]:
            best = {
                "loss": best_loss,
                "state": best_state,
                "dropout": dropout,
                "best_epoch": best_epoch,
                "epochs_run": epoch,
            }

    selected = best["state"]
    train_predictions = predict(
        input_train, selected, target_means, target_scales, minute_index=3
    )
    dev_predictions = predict(
        input_dev, selected, target_means, target_scales, minute_index=3
    )
    train_metrics = raw_metrics(train_predictions, y_train, mask_train)
    dev_metrics = raw_metrics(dev_predictions, y_dev, mask_dev)
    dev_baseline_metrics = raw_metrics(baseline_dev, y_dev, mask_dev)

    # PCA remains downstream of the four predicted statistics and is fitted
    # only to actual, jointly observed training targets.
    pca_train_mask = mask_train.all(axis=1) & np.isfinite(baseline_train).all(axis=1)
    pca_dev_mask = mask_dev.all(axis=1) & np.isfinite(baseline_dev).all(axis=1)
    pca_parameters = fit_pca(y_train[pca_train_mask], data_hash, FOUR_TARGETS)
    pca_train_actual = combine_scores(y_train[pca_train_mask], pca_parameters)
    pca_dev_actual = combine_scores(y_dev[pca_dev_mask], pca_parameters)
    pca_dev_predicted = combine_scores(dev_predictions[pca_dev_mask], pca_parameters)
    pca_dev_baseline = combine_scores(baseline_dev[pca_dev_mask], pca_parameters)
    composite_metrics = {
        "train_rows_used_to_fit_pca": int(pca_train_mask.sum()),
        "development_rows": int(pca_dev_mask.sum()),
        "predicted_composite_rmse": float(
            np.sqrt(np.mean((pca_dev_predicted - pca_dev_actual) ** 2))
        ),
        "predicted_composite_mae": float(
            np.mean(np.abs(pca_dev_predicted - pca_dev_actual))
        ),
        "persistence_composite_rmse": float(
            np.sqrt(np.mean((pca_dev_baseline - pca_dev_actual) ** 2))
        ),
        "persistence_composite_mae": float(
            np.mean(np.abs(pca_dev_baseline - pca_dev_actual))
        ),
    }
    independent_manifest, independent_models, independent_pca = load_independent_bundle(
        INDEPENDENT_BUNDLE
    )
    if independent_manifest["model_data_sha256"] != data_hash:
        raise ValueError("Independent reference bundle was trained on different rate arrays.")
    independent_predictions, unused_independent_composite = predict_independent_bundle(
        x_dev,
        independent_models,
        independent_pca,
        feature_names,
    )
    comparison_rows = []
    for model_name, predictions_matrix in [
        ("shared_multi_output", dev_predictions),
        ("separate_target_networks", independent_predictions),
        ("persistence", baseline_dev),
    ]:
        for column, target in enumerate(FOUR_TARGETS):
            valid = mask_dev[:, column]
            errors = predictions_matrix[valid, column] - y_dev[valid, column]
            comparison_rows.append(
                {
                    "model": model_name,
                    "target": target,
                    "rows": int(valid.sum()),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "mae": float(np.mean(np.abs(errors))),
                }
            )
        errors = (
            combine_scores(predictions_matrix[pca_dev_mask], pca_parameters)
            - pca_dev_actual
        )
        comparison_rows.append(
            {
                "model": model_name,
                "target": "four_stat_pca_composite",
                "rows": int(pca_dev_mask.sum()),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
            }
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_dir = ROOT / "models" / "numpy_multitask_four_stat" / run_id
    result_dir = ROOT / "results" / "development" / "numpy_multitask_four_stat" / run_id
    model_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "architecture": "single shared NumPy MLP with four linear outputs",
        "layer_dims": layer_dims,
        "target_order": FOUR_TARGETS,
        "target_means": target_means.tolist(),
        "target_scales": target_scales.tolist(),
        "feature_names": feature_names,
        "selected_dropout": best["dropout"],
        "best_epoch": best["best_epoch"],
        "epochs_run": best["epochs_run"],
        "dropout_candidates": DROPOUTS,
        "candidate_runs": parameter_candidates,
        "training_examples": int(len(x_train)),
        "development_examples": int(len(x_dev)),
        "training_valid_labels_by_target": {
            target: int(mask_train[:, i].sum()) for i, target in enumerate(FOUR_TARGETS)
        },
        "development_valid_labels_by_target": {
            target: int(mask_dev[:, i].sum()) for i, target in enumerate(FOUR_TARGETS)
        },
        "missing_target_handling": "Per-target masked MSE; each target loss is averaged over its observed labels, then target losses are averaged equally.",
        "minute_prediction_policy": "Decode to raw minutes and floor predictions at zero; no upper clipping.",
        "pca_after_prediction": True,
        "pca_training_rows": int(pca_train_mask.sum()),
        "pca_parameters": pca_parameters,
        "test_data_loaded": False,
        "development_informed_prior_project_choices": True,
        "model_data_rates_sha256": data_hash,
        "base_model_data_sha256": base_hash,
        "training_seconds": time.perf_counter() - started,
    }
    checkpoint = {key: value for key, value in selected.items()}
    checkpoint["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    np.savez(model_dir / "best_model.npz", **checkpoint)
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (model_dir / "pca_parameters.json").write_text(
        json.dumps(pca_parameters, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(training_history[0]))
        writer.writeheader()
        writer.writerows(training_history)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(result_dir / "development_model_comparison.csv", index=False)

    dev_rows = pd.read_csv(DEV_ROWS_PATH)
    predictions = dev_rows[["player", "player_id", "season", "target_season"]].copy()
    for index, target in enumerate(FOUR_TARGETS):
        predictions["actual_" + target] = y_dev[:, index]
        predictions["predicted_" + target] = dev_predictions[:, index]
        predictions["baseline_" + target] = baseline_dev[:, index]
        predictions["target_observed_" + target] = mask_dev[:, index]
    predictions["pca_cohort"] = pca_dev_mask
    predictions["actual_pca_score"] = np.nan
    predictions["predicted_pca_score"] = np.nan
    predictions["baseline_pca_score"] = np.nan
    predictions.loc[pca_dev_mask, "actual_pca_score"] = pca_dev_actual
    predictions.loc[pca_dev_mask, "predicted_pca_score"] = pca_dev_predicted
    predictions.loc[pca_dev_mask, "baseline_pca_score"] = pca_dev_baseline
    predictions.to_csv(result_dir / "development_predictions.csv", index=False)
    report = {
        "architecture": metadata["architecture"],
        "layer_dims": layer_dims,
        "selected_dropout": best["dropout"],
        "best_epoch": best["best_epoch"],
        "train_metrics_by_target": train_metrics,
        "development_metrics_by_target": dev_metrics,
        "development_persistence_metrics_by_target": dev_baseline_metrics,
        "four_stat_pca_composite": composite_metrics,
        "comparison_with_independent_target_networks": comparison_rows,
        "development_selection_loss": best["loss"],
        "test_evaluated": False,
    }
    (result_dir / "development_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("\nSelected dropout / epoch: " + str(best["dropout"]) + " / " + str(best["best_epoch"]))
    print("Development metrics (one shared model):")
    for target in FOUR_TARGETS:
        model = dev_metrics[target]
        baseline = dev_baseline_metrics[target]
        print(
            f"  {target}: RMSE {model['rmse']:.5f}, MAE {model['mae']:.5f}; "
            f"persistence RMSE {baseline['rmse']:.5f}, MAE {baseline['mae']:.5f}"
        )
    print(
        "  PCA composite RMSE "
        f"{composite_metrics['predicted_composite_rmse']:.5f}; "
        f"persistence {composite_metrics['persistence_composite_rmse']:.5f}"
    )
    print("Development comparison (shared / separate / persistence):")
    print(comparison.to_string(index=False))
    print("Model: " + str(model_dir))
    print("Results: " + str(result_dir))
    print("Test split not loaded.")


if __name__ == "__main__":
    main()
