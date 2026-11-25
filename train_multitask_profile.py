"""Train one network to forecast a broad skill profile and find player comps.

One shared NumPy MLP predicts PIE, WS/48, BPM, efficiency/role rates, and
minutes. The skill/efficiency outputs (not minutes or salary) define a
training-fitted PCA space for next-season production-profile neighbors.
"""

from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from multitask_network import (
    adam_initialize,
    adam_update,
    backward,
    forward,
    initialize_parameters,
    masked_mse,
    predict,
)
from profile_targets import (
    BASELINE_COLUMNS,
    NEXT_SEASON_COLUMNS,
    PCA_TARGETS,
    PROFILE_TARGETS,
)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DATA_PATH = PROCESSED / "model_data_rates.npz"
SALARY_PATH = PROCESSED / "salary_cap_normalized_pairs.csv"
RESULTS_ROOT = ROOT / "results" / "development" / "multitask_profile"
MODELS_ROOT = ROOT / "models" / "numpy_multitask_profile"
HIDDEN = [64, 32]
DROPOUTS = [0.0, 0.1]
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
SEED = 42
PCA_VARIANCE = 0.95
N_COMPS = 10
BENCH_MAX_MPG = 24.0
STARTER_MIN_MPG = 28.0
MIN_ROLE_GAMES = 20
MIN_PRIOR_MINUTES = 300.0


def masked_target_arrays(values, masks, means=None, scales=None):
    result = np.zeros(values.shape, dtype=np.float32)
    if means is None or scales is None:
        means = np.zeros(values.shape[1], dtype=np.float64)
        scales = np.ones(values.shape[1], dtype=np.float64)
        for j in range(values.shape[1]):
            observed = values[masks[:, j], j]
            if not len(observed):
                raise ValueError("No training labels for " + PROFILE_TARGETS[j])
            means[j] = observed.mean()
            scales[j] = observed.std()
            if not np.isfinite(scales[j]) or scales[j] <= 0:
                raise ValueError("No training variation for " + PROFILE_TARGETS[j])
    for j in range(values.shape[1]):
        valid = masks[:, j]
        result[valid, j] = ((values[valid, j] - means[j]) / scales[j]).astype(np.float32)
    return result, means, scales


def metrics_by_target(prediction, actual, actual_mask, baseline, baseline_mask):
    rows = []
    for j, target in enumerate(PROFILE_TARGETS):
        valid = actual_mask[:, j] & baseline_mask[:, j]
        errors = prediction[valid, j] - actual[valid, j]
        baseline_errors = baseline[valid, j] - actual[valid, j]
        rows.append(
            {
                "target": target,
                "rows": int(valid.sum()),
                "model_rmse": float(np.sqrt(np.mean(errors**2))),
                "model_mae": float(np.mean(np.abs(errors))),
                "persistence_rmse": float(np.sqrt(np.mean(baseline_errors**2))),
                "persistence_mae": float(np.mean(np.abs(baseline_errors))),
            }
        )
    return pd.DataFrame(rows)


def label_role(minutes_per_game, games):
    if pd.isna(minutes_per_game) or pd.isna(games):
        return "unknown"
    if games < MIN_ROLE_GAMES:
        return "limited_games"
    if minutes_per_game < BENCH_MAX_MPG:
        return "bench"
    if minutes_per_game >= STARTER_MIN_MPG:
        return "starter"
    return "middle"


def salary_context(frame, salary):
    salary_cols = salary[
        ["player_id", "target_season", "salary_cap_share"]
    ]
    return frame.merge(
        salary_cols,
        on=["player_id", "target_season"],
        how="left",
        validate="one_to_one",
    )


def make_neighbors(
    queries,
    query_coordinates,
    candidate_rows,
    candidate_coordinates,
    candidate_indices,
    candidate_actual,
    starter_only=False,
    bench_only=False,
):
    outputs = []
    for query_idx, query in queries.iterrows():
        if bench_only:
            if query["mp"] < MIN_PRIOR_MINUTES:
                continue
            if label_role(query["mp_per_game"], query["g"]) != "bench":
                continue

        eligible = (
            candidate_rows["target_season"].le(query["season"])
            & candidate_rows["player_id"].ne(query["player_id"])
        )
        if starter_only:
            eligible &= (
                candidate_rows["next_season_mp_per_game"].ge(STARTER_MIN_MPG)
                & candidate_rows["next_season_games"].ge(MIN_ROLE_GAMES)
            )
        local = np.flatnonzero(eligible.to_numpy())
        if not len(local):
            continue
        ranked = local[
            np.argsort(
                np.linalg.norm(
                    candidate_coordinates[local] - query_coordinates[query_idx],
                    axis=1,
                ),
                kind="stable",
            )
        ]
        chosen = []
        seen_player_ids = set()
        for local_index in ranked:
            actual_index = candidate_indices[local_index]
            player_id = candidate_rows.iloc[local_index]["player_id"]
            if player_id in seen_player_ids:
                continue
            chosen.append(local_index)
            seen_player_ids.add(player_id)
            if len(chosen) == N_COMPS:
                break

        for rank, local_index in enumerate(chosen, start=1):
            actual_index = candidate_indices[local_index]
            comp = candidate_rows.iloc[local_index]
            distance = float(
                np.linalg.norm(
                    candidate_coordinates[local_index] - query_coordinates[query_idx]
                )
            )
            row = {
                "query_player": query["player"],
                "query_player_id": query["player_id"],
                "query_prior_season": int(query["season"]),
                "query_target_season": int(query["target_season"]),
                "query_prior_minutes_per_game": float(query["mp_per_game"]),
                "comp_rank": rank,
                "profile_distance": distance,
                "historical_comp": comp["player"],
                "comp_player_id": comp["player_id"],
                "comp_prior_season": int(comp["season"]),
                "comp_outcome_season": int(comp["target_season"]),
                "comp_target_minutes_per_game": float(comp["next_season_mp_per_game"]),
                "comp_target_games": int(comp["next_season_games"]),
                "comp_target_salary_cap_share_pct": (
                    float(comp["salary_cap_share"] * 100)
                    if pd.notna(comp["salary_cap_share"]) else np.nan
                ),
                "comp_salary_review_flag": (
                    bool(comp["salary_needs_source_review"])
                    if pd.notna(comp["salary_needs_source_review"]) else True
                ),
                "comp_salary_used_in_distance": False,
                "starter_only_search": starter_only,
            }
            for j, target in enumerate(PROFILE_TARGETS):
                row["comp_next_" + target] = float(candidate_actual[actual_index, j])
            outputs.append(row)
    return outputs


def main(min_prior_games=0):
    started = time.perf_counter()
    data_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    train_rows = pd.read_csv(PROCESSED / "train_rows_rates.csv")
    if min_prior_games < 0:
        raise ValueError("Minimum prior games cannot be negative.")
    if len(train_rows) == 0:
        raise ValueError("Training metadata is empty.")
    train_keep = train_rows["g"].ge(min_prior_games).to_numpy()
    if not train_keep.any():
        raise ValueError("The training-games threshold excludes every row.")
    with np.load(DATA_PATH, allow_pickle=False) as arrays:
        x_train = arrays["X_train"].astype(np.float32)[train_keep]
        x_dev = arrays["X_dev"].astype(np.float32)
        y_train = arrays["profile_y_train"].astype(np.float64)[train_keep]
        y_dev = arrays["profile_y_dev"].astype(np.float64)
        mask_train = arrays["profile_mask_train"].astype(bool)[train_keep]
        mask_dev = arrays["profile_mask_dev"].astype(bool)
        baseline_train = arrays["profile_baseline_train"].astype(np.float64)[train_keep]
        baseline_dev = arrays["profile_baseline_dev"].astype(np.float64)
        baseline_mask_train = arrays["profile_baseline_mask_train"].astype(bool)[train_keep]
        baseline_mask_dev = arrays["profile_baseline_mask_dev"].astype(bool)
        declared_targets = arrays["profile_target_names"].astype(str).tolist()
        declared_pca_targets = arrays["pca_target_names"].astype(str).tolist()
        feature_names = arrays["feature_names"].astype(str).tolist()
        base_data_hash = str(arrays["base_model_data_sha256"].item())

    if declared_targets != PROFILE_TARGETS or declared_pca_targets != PCA_TARGETS:
        raise ValueError("Prepared target order does not match profile_targets.py.")
    if not np.isfinite(x_train).all() or not np.isfinite(x_dev).all():
        raise ValueError("Model inputs must be finite.")
    if y_train.shape != mask_train.shape or y_dev.shape != mask_dev.shape:
        raise ValueError("Target labels and masks do not align.")
    if len(x_train) != len(y_train) or len(x_dev) != len(y_dev):
        raise ValueError("Input and target arrays do not align.")
    if len(train_rows) != len(train_keep):
        raise ValueError("Training row metadata does not align with model arrays.")
    train_rows = train_rows.loc[train_keep].reset_index(drop=True)

    y_train_scaled, target_means, target_scales = masked_target_arrays(
        y_train, mask_train
    )
    y_dev_scaled, _, _ = masked_target_arrays(
        y_dev, mask_dev, target_means, target_scales
    )
    input_train = x_train.T
    input_dev = x_dev.T
    layer_dims = [x_train.shape[1], *HIDDEN, len(PROFILE_TARGETS)]
    run_candidates = []
    history = []
    best = None

    print(
        f"Shared multi-output profile model: {layer_dims}; "
        f"{len(PCA_TARGETS)} PCA skill targets + minutes",
        flush=True,
    )
    for run_index, dropout in enumerate(DROPOUTS):
        parameters = initialize_parameters(layer_dims, SEED)
        first, second = adam_initialize(parameters)
        shuffle_rng = np.random.default_rng(SEED + 10 + run_index)
        dropout_rng = np.random.default_rng(SEED + 20 + run_index)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        step = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            order = shuffle_rng.permutation(len(x_train))
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start : start + BATCH_SIZE]
                predictions, caches = forward(
                    input_train[:, indices],
                    parameters,
                    keep_prob=1.0 - dropout,
                    training=True,
                    rng=dropout_rng,
                )
                gradients = backward(
                    predictions,
                    y_train_scaled[indices].T,
                    mask_train[indices].T.astype(np.float32),
                    caches,
                    parameters,
                    WEIGHT_DECAY,
                )
                step += 1
                parameters, first, second = adam_update(
                    parameters, gradients, first, second, step, LEARNING_RATE
                )

            train_scaled_predictions = forward(input_train, parameters)[0]
            dev_scaled_predictions = forward(input_dev, parameters)[0]
            train_loss = masked_mse(
                train_scaled_predictions, y_train_scaled.T, mask_train.T
            )[0]
            dev_loss = masked_mse(
                dev_scaled_predictions, y_dev_scaled.T, mask_dev.T
            )[0]
            if not np.isfinite(train_loss) or not np.isfinite(dev_loss):
                raise ValueError("Nonfinite masked multi-output loss.")
            history.append(
                {
                    "dropout": dropout,
                    "epoch": epoch,
                    "train_task_balanced_mse": train_loss,
                    "development_task_balanced_mse": dev_loss,
                }
            )
            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch
                best_state = {key: value.copy() for key, value in parameters.items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % 25 == 0:
                print(
                    f"dropout={dropout:.1f}, epoch={epoch}, "
                    f"train={train_loss:.5f}, dev={dev_loss:.5f}",
                    flush=True,
                )
            if stale_epochs >= PATIENCE:
                break

        candidate = {
            "dropout": dropout,
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "best_development_loss": best_loss,
        }
        run_candidates.append(candidate)
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
        input_train, selected, target_means, target_scales,
        minute_index=PROFILE_TARGETS.index("minutes"),
    )
    dev_predictions = predict(
        input_dev, selected, target_means, target_scales,
        minute_index=PROFILE_TARGETS.index("minutes"),
    )
    target_metrics = metrics_by_target(
        dev_predictions, y_dev, mask_dev, baseline_dev, baseline_mask_dev
    )

    pca_indices = [PROFILE_TARGETS.index(target) for target in PCA_TARGETS]
    train_skill = y_train[:, pca_indices]
    dev_skill = y_dev[:, pca_indices]
    dev_baseline_skill = baseline_dev[:, pca_indices]
    pca_train_mask = np.isfinite(train_skill).all(axis=1)
    pca_dev_mask = np.isfinite(dev_skill).all(axis=1)
    pca_dev_baseline_mask = pca_dev_mask & np.isfinite(dev_baseline_skill).all(axis=1)
    skill_scaler = StandardScaler()
    train_standardized = skill_scaler.fit_transform(train_skill[pca_train_mask])
    profile_pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    train_coordinates_fit = profile_pca.fit_transform(train_standardized)
    train_coordinates = np.full(
        (len(train_skill), profile_pca.n_components_), np.nan, dtype=np.float64
    )
    train_coordinates[pca_train_mask] = train_coordinates_fit

    def transform_skill(skills):
        standardized = skill_scaler.transform(skills)
        return profile_pca.transform(standardized)

    dev_pred_skill = dev_predictions[:, pca_indices]
    dev_pred_coordinates = transform_skill(dev_pred_skill)
    dev_actual_coordinates = np.full_like(dev_pred_coordinates, np.nan)
    dev_actual_coordinates[pca_dev_mask] = transform_skill(dev_skill[pca_dev_mask])
    dev_baseline_coordinates = np.full_like(dev_pred_coordinates, np.nan)
    dev_baseline_coordinates[pca_dev_baseline_mask] = transform_skill(
        dev_baseline_skill[pca_dev_baseline_mask]
    )
    pca_errors = (
        dev_pred_coordinates[pca_dev_mask] - dev_actual_coordinates[pca_dev_mask]
    )
    pca_baseline_errors = (
        dev_baseline_coordinates[pca_dev_baseline_mask]
        - dev_actual_coordinates[pca_dev_baseline_mask]
    )
    profile_metrics = {
        "skill_targets": PCA_TARGETS,
        "training_complete_profiles": int(pca_train_mask.sum()),
        "development_complete_profiles": int(pca_dev_mask.sum()),
        "pca_components_retained": int(profile_pca.n_components_),
        "pca_variance_retained": float(profile_pca.explained_variance_ratio_.sum()),
        "predicted_profile_coordinate_rmse": float(
            np.sqrt(np.mean(pca_errors**2))
        ),
        "persistence_profile_coordinate_rmse": float(
            np.sqrt(np.mean(pca_baseline_errors**2))
        ),
        "interpretation": "Coordinate errors are in the training-fitted PCA scale; nearest-neighbor distance uses retained profile coordinates.",
    }

    dev_rows = pd.read_csv(PROCESSED / "dev_rows_rates.csv")
    salary = pd.read_csv(SALARY_PATH)
    train_rows = salary_context(train_rows, salary)
    dev_rows = salary_context(dev_rows, salary)
    if len(train_rows) != len(x_train) or len(dev_rows) != len(x_dev):
        raise ValueError("Player metadata does not align with the model arrays.")

    neighbor_profile = train_rows.copy()
    neighbor_profile["_array_index"] = np.arange(len(train_rows))
    neighbor_profile = neighbor_profile.loc[pca_train_mask].reset_index(drop=True)
    neighbor_coordinates = train_coordinates[pca_train_mask]
    neighbor_indices = np.flatnonzero(pca_train_mask)
    all_neighbors = make_neighbors(
        dev_rows,
        dev_pred_coordinates,
        neighbor_profile,
        neighbor_coordinates,
        neighbor_indices,
        y_train,
        starter_only=False,
        bench_only=False,
    )
    starter_neighbors = make_neighbors(
        dev_rows,
        dev_pred_coordinates,
        neighbor_profile,
        neighbor_coordinates,
        neighbor_indices,
        y_train,
        starter_only=True,
        bench_only=True,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_dir = MODELS_ROOT / run_id
    result_dir = RESULTS_ROOT / run_id
    model_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    pca_parameters = {
        "skill_target_order": PCA_TARGETS,
        "target_count": len(PCA_TARGETS),
        "fit_split": "training",
        "fit_examples": int(pca_train_mask.sum()),
        "variance_threshold": PCA_VARIANCE,
        "skill_means": skill_scaler.mean_.tolist(),
        "skill_scales": skill_scaler.scale_.tolist(),
        "pca_mean": profile_pca.mean_.tolist(),
        "components": profile_pca.components_.tolist(),
        "explained_variance_ratio": profile_pca.explained_variance_ratio_.tolist(),
        "variance_retained": float(profile_pca.explained_variance_ratio_.sum()),
        "minutes_in_similarity_profile": False,
        "salary_in_similarity_profile": False,
    }
    metadata = {
        "architecture": "single shared NumPy MLP with one linear output per target",
        "layer_dims": layer_dims,
        "target_order": PROFILE_TARGETS,
        "skill_pca_target_order": PCA_TARGETS,
        "target_means": target_means.tolist(),
        "target_scales": target_scales.tolist(),
        "feature_names": feature_names,
        "selected_dropout": best["dropout"],
        "best_epoch": best["best_epoch"],
        "epochs_run": best["epochs_run"],
        "candidate_runs": run_candidates,
        "training_examples": int(len(x_train)),
        "training_minimum_prior_games": int(min_prior_games),
        "training_rows_excluded_by_minimum_games": int((~train_keep).sum()),
        "development_examples": int(len(x_dev)),
        "training_valid_labels_by_target": {
            target: int(mask_train[:, j].sum())
            for j, target in enumerate(PROFILE_TARGETS)
        },
        "development_valid_labels_by_target": {
            target: int(mask_dev[:, j].sum())
            for j, target in enumerate(PROFILE_TARGETS)
        },
        "missing_label_policy": "Masked task-balanced MSE; each target is standardized on its observed training labels and contributes equally to the loss.",
        "similarity_definition": "Euclidean distance in PCA coordinates of predicted next-season skill/efficiency outcomes, compared with actual historical next-season profiles.",
        "similarity_excludes": ["total minutes", "games", "salary"],
        "minutes_policy": "Predicted separately by the same network and reported as opportunity context; predictions are floored at zero.",
        "pca_parameters": pca_parameters,
        "test_data_loaded": False,
        "data_hash": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "base_model_data_hash": base_data_hash,
        "development_was_previously_used": True,
        "experiment_note": (
            "Exploratory minimum-prior-games sensitivity run."
            if min_prior_games
            else "Standard unrestricted-prior-games training run."
        ),
        "training_seconds": time.perf_counter() - started,
    }
    checkpoint = {key: value for key, value in selected.items()}
    checkpoint["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    np.savez(model_dir / "best_model.npz", **checkpoint)
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (model_dir / "profile_pca.json").write_text(
        json.dumps(pca_parameters, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    target_metrics.to_csv(result_dir / "development_target_metrics.csv", index=False)
    pd.DataFrame(all_neighbors).to_csv(
        result_dir / "development_profile_neighbors.csv", index=False
    )
    pd.DataFrame(starter_neighbors).to_csv(
        result_dir / "development_bench_to_starter_neighbors.csv", index=False
    )

    prediction_rows = dev_rows[["player", "player_id", "season", "target_season"]].copy()
    for j, target in enumerate(PROFILE_TARGETS):
        prediction_rows["actual_" + target] = y_dev[:, j]
        prediction_rows["predicted_" + target] = dev_predictions[:, j]
        prediction_rows["prior_" + target] = baseline_dev[:, j]
        prediction_rows["target_observed_" + target] = mask_dev[:, j]
    prediction_rows["skill_profile_complete"] = pca_dev_mask
    prediction_rows["predicted_salary_cap_share_pct"] = (
        dev_rows["salary_cap_share"] * 100
    )
    for j in range(dev_pred_coordinates.shape[1]):
        prediction_rows[f"predicted_pc{j + 1}"] = dev_pred_coordinates[:, j]
        prediction_rows[f"actual_pc{j + 1}"] = dev_actual_coordinates[:, j]
    prediction_rows.to_csv(result_dir / "development_profile_predictions.csv", index=False)

    report = {
        "architecture": metadata["architecture"],
        "layer_dims": layer_dims,
        "target_order": PROFILE_TARGETS,
        "skill_pca_target_order": PCA_TARGETS,
        "selected_dropout": best["dropout"],
        "best_epoch": best["best_epoch"],
        "development_target_metrics": target_metrics.to_dict(orient="records"),
        "profile_pca_metrics": profile_metrics,
        "development_players_with_any_historical_profile_comp": int(
            len({row["query_player_id"] for row in all_neighbors})
        ),
        "development_bench_to_starter_comp_rows": int(len(starter_neighbors)),
        "test_evaluated": False,
        "results_directory": str(result_dir.relative_to(ROOT)),
    }
    (result_dir / "development_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Selected dropout/epoch: {best['dropout']}/{best['best_epoch']}; "
        f"PCA retains {profile_metrics['pca_components_retained']} components "
        f"({profile_metrics['pca_variance_retained']:.1%} variance)."
    )
    print(
        f"Training complete skill profiles: {int(pca_train_mask.sum())}; "
        f"development complete profiles: {int(pca_dev_mask.sum())}."
    )
    print("Development target metrics:")
    print(target_metrics.to_string(index=False))
    print("Historical profile-comparison rows: " + str(len(all_neighbors)))
    print("Bench-to-starter comparison rows: " + str(len(starter_neighbors)))
    print("Model: " + str(model_dir))
    print("Results: " + str(result_dir))
    print("Test split not loaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-prior-games",
        type=int,
        default=0,
        help="Exclude training player-seasons below this prior-season game count.",
    )
    main(min_prior_games=parser.parse_args().min_prior_games)
