"""Find historical player-season analogs and summarize their next-season jumps.

PCA and salary scaling are fit on training inputs only. Development queries use
the nearest distinct players whose observed target seasons were already complete
by the query player's prior season. The test split is never loaded.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "development" / "player_comps"
MODEL_DATA = PROCESSED / "model_data.npz"
SALARY_DATA = PROCESSED / "salary_cap_normalized_pairs.csv"
N_COMPONENTS = 10
PCA_VARIANCE = 0.90
SEED = 42
TARGETS = ["pie", "ws", "vorp"]


def attach_prior_salary(rows, salary):
    prior_salary = salary[
        [
            "player_id", "target_season", "salary_cap_share",
            "salary_available_check", "salary_needs_source_review",
        ]
    ].rename(
        columns={
            "target_season": "season",
            "salary_cap_share": "prior_salary_cap_share",
            "salary_available_check": "prior_salary_available",
            "salary_needs_source_review": "prior_salary_flagged",
        }
    )
    joined = rows.merge(
        prior_salary,
        on=["player_id", "season"],
        how="left",
        validate="one_to_one",
    )
    joined["prior_salary_cap_share"] = pd.to_numeric(
        joined["prior_salary_cap_share"], errors="coerce"
    )
    joined["prior_salary_available"] = (
        joined["prior_salary_available"].fillna(False).astype(bool)
    )
    joined["prior_salary_flagged"] = joined["prior_salary_flagged"].fillna(True).astype(bool)
    return joined


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    salary = pd.read_csv(SALARY_DATA)
    train_rows = pd.read_csv(PROCESSED / "train_rows.csv")
    dev_rows = pd.read_csv(PROCESSED / "dev_rows.csv")
    with np.load(MODEL_DATA, allow_pickle=False) as arrays:
        train_x = arrays["X_train"].astype(float)
        dev_x = arrays["X_dev"].astype(float)
        if len(train_rows) != len(train_x) or len(dev_rows) != len(dev_x):
            raise ValueError("Metadata rows do not align with saved train/development inputs.")
        train = train_rows[
            ["player", "player_id", "season", "target_season", *TARGETS]
        ].copy()
        dev = dev_rows[
            ["player", "player_id", "season", "target_season", *TARGETS,
             "next_season_pie", "next_season_ws", "next_season_vorp"]
        ].copy()
        train_outcomes = np.column_stack(
            [arrays["y_" + target + "_train"].astype(float) for target in TARGETS]
        )

    train = attach_prior_salary(train, salary)
    dev = attach_prior_salary(dev, salary)
    train["prior_salary_cap_share"] = train["prior_salary_cap_share"].where(
        train["prior_salary_cap_share"].gt(0)
    )
    dev["prior_salary_cap_share"] = dev["prior_salary_cap_share"].where(
        dev["prior_salary_cap_share"].gt(0)
    )
    train["log_prior_salary_cap_share"] = np.log(train["prior_salary_cap_share"])
    dev["log_prior_salary_cap_share"] = np.log(dev["prior_salary_cap_share"])

    train_valid = (
        train["log_prior_salary_cap_share"].notna()
        & np.isfinite(train_x).all(axis=1)
        & np.isfinite(train_outcomes).all(axis=1)
    )
    dev_valid = (
        dev["log_prior_salary_cap_share"].notna()
        & np.isfinite(dev_x).all(axis=1)
        & np.isfinite(dev[["next_season_" + target for target in TARGETS]].to_numpy(float)).all(axis=1)
    )
    train_fit = train.loc[train_valid].reset_index(drop=True)
    dev_fit = dev.loc[dev_valid].reset_index(drop=True)
    train_x = train_x[train_valid]
    dev_x = dev_x[dev_valid]
    train_outcomes = train_outcomes[train_valid]
    if len(train_fit) <= N_COMPONENTS:
        raise ValueError("Too few training rows with prior salary for PCA.")

    salary_scaler = StandardScaler()
    train_salary = salary_scaler.fit_transform(
        train_fit[["log_prior_salary_cap_share"]].to_numpy(float)
    )
    dev_salary = salary_scaler.transform(
        dev_fit[["log_prior_salary_cap_share"]].to_numpy(float)
    )
    train_features = np.column_stack([train_x, train_salary])
    dev_features = np.column_stack([dev_x, dev_salary])
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)
    train_pc = pca.fit_transform(train_features)
    dev_pc = pca.transform(dev_features)

    deltas = train_outcomes - train_fit[TARGETS].to_numpy(float)
    train_fit[[f"next_season_delta_{target}" for target in TARGETS]] = deltas

    neighbor_records = []
    prediction_records = []
    for query_index, query in dev_fit.iterrows():
        # Historical analog outcomes must have been observed by the query's
        # prior season; the query player is not allowed to match themself.
        eligible = (
            train_fit["target_season"].le(query["season"])
            & train_fit["player_id"].ne(query["player_id"])
        ).to_numpy()
        candidate_indices = np.flatnonzero(eligible)
        if not len(candidate_indices):
            continue
        distances = np.linalg.norm(train_pc[candidate_indices] - dev_pc[query_index], axis=1)
        ranked = candidate_indices[np.argsort(distances, kind="stable")]
        # Avoid several seasons from one analog player dominating the forecast.
        chosen = []
        seen_players = set()
        for idx in ranked:
            player_id = train_fit.iloc[idx]["player_id"]
            if player_id in seen_players:
                continue
            chosen.append(idx)
            seen_players.add(player_id)
            if len(chosen) == N_COMPONENTS:
                break
        if not chosen:
            continue

        chosen_distances = np.linalg.norm(train_pc[chosen] - dev_pc[query_index], axis=1)
        chosen_deltas = deltas[chosen]
        median_delta = np.median(chosen_deltas, axis=0)
        prior = query[TARGETS].to_numpy(float)
        forecast = prior + median_delta
        pred = {
            "player": query["player"],
            "player_id": query["player_id"],
            "season": int(query["season"]),
            "target_season": int(query["target_season"]),
            "analogs_used": int(len(chosen)),
            "median_analog_distance": float(np.median(chosen_distances)),
            "prior_salary_cap_share": float(query["prior_salary_cap_share"]),
            "prior_salary_flagged": bool(query["prior_salary_flagged"]),
        }
        for j, target in enumerate(TARGETS):
            pred[f"prior_{target}"] = float(prior[j])
            pred[f"predicted_delta_{target}"] = float(median_delta[j])
            pred[f"predicted_next_{target}"] = float(forecast[j])
            pred[f"actual_next_{target}"] = float(query["next_season_" + target])
            pred[f"actual_delta_{target}"] = float(query["next_season_" + target] - prior[j])
        prediction_records.append(pred)

        for rank, (idx, distance) in enumerate(zip(chosen, chosen_distances), start=1):
            analog = train_fit.iloc[idx]
            record = {
                "query_player": query["player"],
                "query_player_id": query["player_id"],
                "query_season": int(query["season"]),
                "query_target_season": int(query["target_season"]),
                "analog_rank": rank,
                "distance_in_retained_pca_space": float(distance),
                "analog_player": analog["player"],
                "analog_player_id": analog["player_id"],
                "analog_prior_season": int(analog["season"]),
                "analog_outcome_season": int(analog["target_season"]),
                "analog_prior_salary_cap_share": float(analog["prior_salary_cap_share"]),
                "analog_salary_flagged": bool(analog["prior_salary_flagged"]),
            }
            for j, target in enumerate(TARGETS):
                record[f"analog_prior_{target}"] = float(analog[target])
                record[f"analog_next_{target}"] = float(train_outcomes[idx, j])
                record[f"analog_delta_{target}"] = float(deltas[idx, j])
            neighbor_records.append(record)

    predictions = pd.DataFrame(prediction_records)
    neighbors = pd.DataFrame(neighbor_records)
    predictions.to_csv(RESULTS / "development_comp_predictions.csv", index=False)
    neighbors.to_csv(RESULTS / "development_comp_neighbors.csv", index=False)

    metric_rows = []
    for target in TARGETS:
        actual = predictions[f"actual_next_{target}"].to_numpy(float)
        analog = predictions[f"predicted_next_{target}"].to_numpy(float)
        persistence = predictions[f"prior_{target}"].to_numpy(float)
        metric_rows.append(
            {
                "target": target,
                "development_rows": int(len(predictions)),
                "analog_median_delta_rmse": float(np.sqrt(np.mean((analog - actual) ** 2))),
                "analog_median_delta_mae": float(np.mean(np.abs(analog - actual))),
                "persistence_rmse": float(np.sqrt(np.mean((persistence - actual) ** 2))),
                "persistence_mae": float(np.mean(np.abs(persistence - actual))),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "development_comp_metrics.csv", index=False)
    metadata = {
        "method": "PCA-reduced Euclidean nearest neighbors; median observed next-season change among up to 10 distinct-player analogs",
        "pca_variance_retained": PCA_VARIANCE,
        "pca_components_retained": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "feature_inputs": "Existing 22 standardized prior-season player features plus standardized log prior-season salary-cap share",
        "salary_timing": "Prior-season salary only, joined where salary target_season equals the model row's prior season; flagged available values included, missing values excluded",
        "analog_time_rule": "Analog target season must be no later than the query player's prior season",
        "distinct_analog_players": True,
        "max_analogs_per_query": N_COMPONENTS,
        "training_rows_total": int(len(train)),
        "training_rows_used": int(len(train_fit)),
        "development_rows_total": int(len(dev)),
        "development_rows_with_prior_salary": int(len(dev_fit)),
        "development_rows_with_at_least_one_historical_analog": int(len(predictions)),
        "development_rows_without_historical_analog": int(len(dev_fit) - len(predictions)),
        "training_salary_flagged_rows_included": int(
            train_fit["prior_salary_flagged"].sum()
        ),
        "development_salary_flagged_rows_used": int(
            predictions["prior_salary_flagged"].sum()
        ) if len(predictions) else 0,
        "test_data_loaded": False,
        "model_arrays_sha256": hashlib.sha256(MODEL_DATA.read_bytes()).hexdigest(),
        "salary_data_sha256": hashlib.sha256(SALARY_DATA.read_bytes()).hexdigest(),
        "interpretation": "Exploratory player-comparison forecasts; development results are not independent because development informed earlier project choices.",
    }
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Retained PCA components: {pca.n_components_} ({pca.explained_variance_ratio_.sum():.1%} variance)")
    print(f"Training analog pool: {len(train_fit)} rows")
    print(f"Development queries with historical analogs: {len(predictions)} / {len(dev)}")
    print(f"Outputs: {RESULTS}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
