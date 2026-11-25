"""Compare bench-player skill profiles with historical starter profiles.

Similarity is based on age, position, per-36 box-score rates, efficiency, usage,
and rate-based impact metrics. Playing time and salary are deliberately excluded
from the distance; playing time defines bench/starter roles, while salary and a
training-only salary-value residual are reported as context.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
SALARY_VALUE = ROOT / "results" / "development" / "salary_value"
RESULTS = ROOT / "results" / "development" / "role_crossing_comps"
SALARY_DATA = PROCESSED / "salary_cap_normalized_pairs.csv"
N_COMPS = 10
PCA_VARIANCE = 0.90
BENCH_MAX_MPG = 24.0
STARTER_MIN_MPG = 28.0
MIN_SEASON_MINUTES = 300.0
TARGETS = ["pie", "ws", "vorp"]
COUNTING_PER_GAME = [
    "pts_per_game", "trb_per_game", "ast_per_game",
    "stl_per_game", "blk_per_game", "tov_per_game",
]
RATE_FEATURES = [
    "ts_percent", "x3p_percent", "x3p_ar", "f_tr", "usg_percent",
    "per", "ws_48", "bpm", "pie",
]


def prepare_features(frame, position_columns):
    """Create skill/style inputs without minutes, per-game volume, or salary."""
    out = pd.DataFrame(index=frame.index)
    out["age"] = pd.to_numeric(frame["age"], errors="coerce")
    minutes_per_game = pd.to_numeric(frame["mp_per_game"], errors="coerce")
    total_minutes = pd.to_numeric(frame["mp"], errors="coerce")
    for column in COUNTING_PER_GAME:
        per_game = pd.to_numeric(frame[column], errors="coerce")
        per36 = per_game.div(minutes_per_game.where(minutes_per_game.gt(0))).mul(36.0)
        out[column.replace("_per_game", "_per36")] = per36
    for column in RATE_FEATURES:
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    positions = frame["pos"].fillna("UNK").astype(str).str.upper()
    for position in position_columns:
        out["pos_" + position] = positions.str.contains(position, regex=False).astype(float)
    return out


def attach_salary_context(frame, salary):
    fields = [
        "player_id", "target_season", "salary_cap_share",
        "salary_available_check", "salary_needs_source_review",
    ]
    salary_fields = salary[fields].rename(
        columns={
            "salary_available_check": "normalized_salary_available",
            "salary_needs_source_review": "normalized_salary_review_flag",
        }
    )
    return frame.merge(
        salary_fields,
        on=["player_id", "target_season"],
        how="left",
        validate="one_to_one",
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PROCESSED / "train_rows.csv")
    dev = pd.read_csv(PROCESSED / "dev_rows.csv")
    salary = pd.read_csv(SALARY_DATA)
    value_train = pd.read_csv(SALARY_VALUE / "training_salary_benchmark_rows.csv")
    salary_keys = ["player_id", "target_season"]
    train = attach_salary_context(train, salary)
    dev = attach_salary_context(dev, salary)
    train = train.merge(
        value_train[salary_keys + ["observed_value_residual"]].rename(
            columns={"observed_value_residual": "salary_value_residual"}
        ),
        on=salary_keys,
        how="left",
        validate="one_to_one",
    )

    positions = sorted(
        set(train["pos"].dropna().astype(str).str.upper())
        | set(dev["pos"].dropna().astype(str).str.upper())
    )
    train_features = prepare_features(train, positions)
    dev_features = prepare_features(dev, positions)
    train_reliability = (
        pd.to_numeric(train["mp"], errors="coerce").clip(lower=0)
        / (pd.to_numeric(train["mp"], errors="coerce").clip(lower=0) + 600.0)
    ).fillna(0).to_numpy()
    dev_reliability = (
        pd.to_numeric(dev["mp"], errors="coerce").clip(lower=0)
        / (pd.to_numeric(dev["mp"], errors="coerce").clip(lower=0) + 600.0)
    ).fillna(0).to_numpy()
    for column in [name.replace("_per_game", "_per36") for name in COUNTING_PER_GAME]:
        league_mean = float(train_features[column].mean())
        train_features[column] = (
            train_features[column] * train_reliability
            + league_mean * (1 - train_reliability)
        )
        dev_features[column] = (
            dev_features[column] * dev_reliability
            + league_mean * (1 - dev_reliability)
        )
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_features)
    dev_imputed = imputer.transform(dev_features)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    dev_scaled = scaler.transform(dev_imputed)

    train_usable = (
        train["mp"].ge(MIN_SEASON_MINUTES).to_numpy()
        & np.isfinite(train_scaled).all(axis=1)
        & train[["next_season_" + target for target in TARGETS]]
        .apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    )
    dev_usable = (
        dev["mp"].ge(MIN_SEASON_MINUTES).to_numpy()
        & np.isfinite(dev_scaled).all(axis=1)
        & dev[["next_season_" + target for target in TARGETS]]
        .apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    )
    train = train.loc[train_usable].reset_index(drop=True)
    dev = dev.loc[dev_usable].reset_index(drop=True)
    train_scaled = train_scaled[train_usable]
    dev_scaled = dev_scaled[dev_usable]

    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    train_pc = pca.fit_transform(train_scaled)
    dev_pc = pca.transform(dev_scaled)
    coordinate_columns = [f"pc{i + 1}" for i in range(pca.n_components_)]
    train_coordinates = train[
        ["player", "player_id", "season", "target_season", "mp_per_game"]
    ].copy()
    train_coordinates["split"] = "training"
    train_coordinates["role"] = np.where(
        train_coordinates["mp_per_game"] >= STARTER_MIN_MPG, "starter",
        np.where(train_coordinates["mp_per_game"] < BENCH_MAX_MPG, "bench", "middle")
    )
    train_coordinates[coordinate_columns] = train_pc
    dev_coordinates = dev[
        ["player", "player_id", "season", "target_season", "mp_per_game"]
    ].copy()
    dev_coordinates["split"] = "development"
    dev_coordinates["role"] = np.where(
        dev_coordinates["mp_per_game"] >= STARTER_MIN_MPG, "starter",
        np.where(dev_coordinates["mp_per_game"] < BENCH_MAX_MPG, "bench", "middle")
    )
    dev_coordinates[coordinate_columns] = dev_pc
    pd.concat([train_coordinates, dev_coordinates], ignore_index=True).to_csv(
        RESULTS / "skill_pca_coordinates.csv", index=False
    )
    target_cols = ["next_season_" + target for target in TARGETS]
    deltas = train[target_cols].to_numpy(float) - train[TARGETS].to_numpy(float)

    valid_salary = (
        train["normalized_salary_available"].fillna(False).astype(bool)
        & ~train["normalized_salary_review_flag"].fillna(True).astype(bool)
        & train["salary_cap_share"].notna()
    )
    high_salary_threshold = float(train.loc[valid_salary, "salary_cap_share"].quantile(0.75))

    neighbors = []
    summaries = []
    for query_index, query in dev.iterrows():
        if not (query["mp_per_game"] < BENCH_MAX_MPG):
            continue
        available = (
            train["mp_per_game"].ge(STARTER_MIN_MPG)
            & train["target_season"].le(query["season"])
            & train["player_id"].ne(query["player_id"])
        ).to_numpy()
        candidate_indices = np.flatnonzero(available)
        if not len(candidate_indices):
            continue
        distances = np.linalg.norm(
            train_pc[candidate_indices] - dev_pc[query_index], axis=1
        )
        ranked = candidate_indices[np.argsort(distances, kind="stable")]
        chosen = []
        seen = set()
        for index in ranked:
            player_id = train.iloc[index]["player_id"]
            if player_id in seen:
                continue
            chosen.append(index)
            seen.add(player_id)
            if len(chosen) == N_COMPS:
                break
        if not chosen:
            continue

        query_summary = {
            "bench_player": query["player"],
            "bench_player_id": query["player_id"],
            "prior_season": int(query["season"]),
            "target_season": int(query["target_season"]),
            "prior_minutes_per_game": float(query["mp_per_game"]),
            "prior_total_minutes": float(query["mp"]),
            "prior_salary_cap_share_pct": (
                float(query["salary_cap_share"] * 100)
                if pd.notna(query["salary_cap_share"]) else np.nan
            ),
            "starter_comps_found": len(chosen),
            "high_salary_threshold_training_p75_pct": high_salary_threshold * 100,
        }
        for j, target in enumerate(TARGETS):
            query_summary["bench_prior_" + target] = float(query[target])
            query_summary["bench_actual_next_" + target] = float(query[target_cols[j]])
            query_summary["median_starter_comp_delta_" + target] = float(
                np.median(deltas[chosen, j])
            )
        summaries.append(query_summary)

        for rank, index in enumerate(chosen, start=1):
            comp = train.iloc[index]
            share = comp["salary_cap_share"]
            residual = comp["salary_value_residual"]
            high_paid_under_benchmark = bool(
                pd.notna(share)
                and pd.notna(residual)
                and share >= high_salary_threshold
                and residual < 0
            )
            item = {
                "bench_player": query["player"],
                "bench_player_id": query["player_id"],
                "bench_prior_season": int(query["season"]),
                "bench_target_season": int(query["target_season"]),
                "bench_prior_minutes_per_game": float(query["mp_per_game"]),
                "comp_rank": rank,
                "skill_distance_in_pca_space": float(
                    np.linalg.norm(train_pc[index] - dev_pc[query_index])
                ),
                "starter_comp": comp["player"],
                "starter_comp_player_id": comp["player_id"],
                "comp_prior_season": int(comp["season"]),
                "comp_outcome_season": int(comp["target_season"]),
                "comp_prior_minutes_per_game": float(comp["mp_per_game"]),
                "comp_salary_cap_share_pct": (
                    float(share * 100) if pd.notna(share) else np.nan
                ),
                "comp_salary_review_flag": bool(
                    comp["normalized_salary_review_flag"]
                ) if pd.notna(comp["normalized_salary_review_flag"]) else True,
                "comp_salary_performance_residual": (
                    float(residual) if pd.notna(residual) else np.nan
                ),
                "high_paid_below_benchmark_proxy": high_paid_under_benchmark,
            }
            for j, target in enumerate(TARGETS):
                item["comp_prior_" + target] = float(comp[target])
                item["comp_next_" + target] = float(comp[target_cols[j]])
                item["comp_delta_" + target] = float(deltas[index, j])
            neighbors.append(item)

    neighbor_frame = pd.DataFrame(neighbors)
    summary_frame = pd.DataFrame(summaries)
    neighbor_frame.to_csv(RESULTS / "development_bench_to_starter_comps.csv", index=False)
    summary_frame.to_csv(RESULTS / "development_bench_comp_summary.csv", index=False)

    metric_rows = []
    for target in TARGETS:
        actual = summary_frame["bench_actual_next_" + target].to_numpy(float)
        prior = summary_frame["bench_prior_" + target].to_numpy(float)
        analog = prior + summary_frame["median_starter_comp_delta_" + target].to_numpy(float)
        metric_rows.append(
            {
                "target": target,
                "bench_development_rows": int(len(summary_frame)),
                "starter_comp_median_delta_rmse": float(np.sqrt(np.mean((analog - actual) ** 2))),
                "starter_comp_median_delta_mae": float(np.mean(np.abs(analog - actual))),
                "persistence_rmse": float(np.sqrt(np.mean((prior - actual) ** 2))),
                "persistence_mae": float(np.mean(np.abs(prior - actual))),
            }
        )
    pd.DataFrame(metric_rows).to_csv(RESULTS / "development_bench_comp_metrics.csv", index=False)

    high_value_comps = neighbor_frame["high_paid_below_benchmark_proxy"].sum() if len(neighbor_frame) else 0
    audit = {
        "method": "PCA-reduced nearest skill-profile matching from bench queries to historical starter candidates",
        "similarity_features": list(train_features.columns),
        "excluded_from_similarity": [
            "minutes and games (role filter only)",
            "per-game counting totals (converted to shrunken per-36 rates)",
            "salary (reported as context only)",
        ],
        "minutes_shrinkage": "per-36 rates shrunk toward the training feature mean with reliability total_minutes / (total_minutes + 600)",
        "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
        "pca_components_retained": int(pca.n_components_),
        "variance_by_component": pca.explained_variance_ratio_.tolist(),
        "bench_definition": f"prior-season minutes per game < {BENCH_MAX_MPG}; at least {MIN_SEASON_MINUTES} total minutes",
        "starter_definition": f"prior-season minutes per game >= {STARTER_MIN_MPG}; at least {MIN_SEASON_MINUTES} total minutes",
        "analog_time_rule": "starter comp target season must be no later than the bench player's prior season",
        "distinct_starter_players": True,
        "starter_comps_per_query_max": N_COMPS,
        "salary_note": "Salary excluded from skill distance. High-paid/below-benchmark is a proxy using training 75th percentile salary-cap share and negative salary-value residual; it is not a claim of overpayment.",
        "high_salary_cap_share_threshold": high_salary_threshold,
        "training_rows_after_minute_filter": int(len(train)),
        "development_rows_after_minute_filter": int(len(dev)),
        "development_bench_queries_with_starter_comps": int(len(summary_frame)),
        "starter_comps_flagged_high_paid_below_benchmark": int(high_value_comps),
        "salary_normalized_file_sha256": hashlib.sha256(SALARY_DATA.read_bytes()).hexdigest(),
        "test_data_loaded": False,
        "interpretation": "Exploratory role-crossing skill comps; development results are not independent because development informed earlier project decisions.",
    }
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Skill features: {len(train_features.columns)}; PCA components: {pca.n_components_} "
          f"({audit['pca_variance_retained']:.1%} variance)")
    print(f"Development bench queries matched to starters: {len(summary_frame)}")
    print(f"High-salary, below-benchmark comp flags: {high_value_comps}")
    print(pd.read_csv(RESULTS / "development_bench_comp_metrics.csv").to_string(index=False))
    print("Results: " + str(RESULTS))


if __name__ == "__main__":
    main()
