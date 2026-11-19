"""Compare PCA feature definitions using actual TRAINING targets only.

This is a feature-design audit, not a new prediction experiment. It does not
change any prepared arrays, trained models, or frozen test results.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from player_matching import keep_combined_seasons

ROOT = Path(__file__).resolve().parent
BOOTSTRAPS = 20
SEED = 42

SHOOTING_COLUMNS = ["two_point_makes", "two_point_misses", "three_point_makes", "three_point_misses", "free_throw_makes", "free_throw_misses"]
BOX_COLUMNS = SHOOTING_COLUMNS + ["offensive_rebounds", "defensive_rebounds", "assists", "steals", "blocks", "turnovers"]
FEATURE_SETS = {
    "advanced_3": ["pie", "ws_48", "bpm"],
    "basic_6": ["points", "rebounds", "assists", "steals", "blocks", "turnovers"],
    "raw_12": BOX_COLUMNS,
    "raw_14_with_minutes": BOX_COLUMNS + ["personal_fouls", "minutes_per_game"],
    "raw_15_with_availability": BOX_COLUMNS + ["personal_fouls", "minutes_per_game", "games"],
    "raw_14_plus_advanced_17": BOX_COLUMNS + ["personal_fouls", "minutes_per_game", "pie", "ws_48", "bpm"],
}
RAW_COLUMNS = {
    "two_point_makes": "x2p_per_game",
    "three_point_makes": "x3p_per_game",
    "free_throw_makes": "ft_per_game",
    "offensive_rebounds": "orb_per_game",
    "defensive_rebounds": "drb_per_game",
    "assists": "ast_per_game",
    "steals": "stl_per_game",
    "blocks": "blk_per_game",
    "turnovers": "tov_per_game",
    "personal_fouls": "pf_per_game",
    "minutes_per_game": "mp_per_game",
    "games": "g",
    "points": "pts_per_game",
    "rebounds": "trb_per_game",
}


def main():
    processed = ROOT / "data" / "processed"
    rows = pd.read_csv(processed / "train_rows_rates.csv")
    source_path = ROOT / "data" / "raw" / "basketball_reference" / "Player Per Game.csv"
    source = pd.read_csv(source_path)
    source = source[(source["lg"] == "NBA") & source["season"].isin(rows["target_season"])].copy()
    source = keep_combined_seasons(source)
    source = source.rename(columns={"season": "target_season"})
    next_rows = rows[["player_id", "target_season"]].merge(source, on=["player_id", "target_season"], how="left", sort=False, validate="one_to_one", indicator=True)

    if(not next_rows["_merge"].eq("both").all()):
        raise ValueError("Every training target must match an actual next-season record.")

    frame = pd.DataFrame(index=rows.index)

    for name, column in RAW_COLUMNS.items():
        frame[name] = next_rows[column].astype(float)

    frame["two_point_misses"] = next_rows["x2pa_per_game"] - next_rows["x2p_per_game"]
    frame["three_point_misses"] = next_rows["x3pa_per_game"] - next_rows["x3p_per_game"]
    frame["free_throw_misses"] = next_rows["fta_per_game"] - next_rows["ft_per_game"]

    with np.load(processed / "model_data_rates.npz", allow_pickle=False) as data:
        mask = data["rate_mask_train"].astype(bool)

        for target in ["pie", "ws_48", "bpm"]:
            frame[target] = data["y_" + target + "_train"].astype(np.float64)

    # Same training cohort for every candidate; targets are never imputed.
    finite = np.isfinite(frame.to_numpy()).all(axis=1)
    valid = mask & finite
    retained = frame.loc[valid].reset_index(drop=True)

    for name in ["two_point_misses", "three_point_misses", "free_throw_misses"]:
        if((retained[name] < -1e-10).any()):
            raise ValueError("Shot makes cannot exceed attempts.")

    output = ROOT / "results" / "training" / "pca_feature_audit" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    retained.corr().to_csv(output / "feature_correlations.csv")
    summaries = []
    loadings = []

    for name, columns in FEATURE_SETS.items():
        values = retained[columns].to_numpy(dtype=np.float64)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(values)
        pca = PCA(svd_solver="full")
        pca.fit(standardized)
        weights = pca.components_[0].copy()

        if(weights.sum() < 0):
            weights = -weights

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        correlations = retained[columns].corr().to_numpy()
        strong_pairs = 0

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if(abs(correlations[i, j]) >= 0.9):
                    strong_pairs = strong_pairs + 1

        similarities = []
        rng = np.random.default_rng(SEED)

        for repeat in range(BOOTSTRAPS):
            sampled = values[rng.integers(0, len(values), len(values))]
            sampled_scaled = StandardScaler().fit_transform(sampled)
            fitted = PCA(n_components=1, svd_solver="full").fit(sampled_scaled)
            similarities.append(abs(float(np.dot(weights, fitted.components_[0]))))

        summaries.append({"feature_set": name, "features": len(columns), "training_examples": len(values), "pc1_variance_percent": float(100.0 * pca.explained_variance_ratio_[0]), "components_for_80_percent": int(np.searchsorted(cumulative, 0.8) + 1), "components_for_90_percent": int(np.searchsorted(cumulative, 0.9) + 1), "correlation_pairs_at_least_0_9": strong_pairs, "bootstrap_pc1_cosine_median": float(np.median(similarities)), "bootstrap_pc1_cosine_minimum": float(np.min(similarities))})

        for i, column in enumerate(columns):
            loadings.append({"feature_set": name, "feature": column, "pc1_weight": float(weights[i]), "mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])})

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "candidate_summary.csv", index=False)
    pd.DataFrame(loadings).to_csv(output / "pc1_loadings.csv", index=False)
    definitions = {"feature_sets": FEATURE_SETS, "raw_columns": RAW_COLUMNS, "derived_misses": "Attempts minus makes, per game", "cohort": "Original valid-rate training rows with finite raw targets; same rows for every candidate", "original_training_rows": len(rows), "retained_training_rows": int(valid.sum()), "excluded_rate_or_nonfinite_rows": int((~valid).sum()), "raw_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "bootstrap_repeats": BOOTSTRAPS, "seed": SEED, "bootstrap_interpretation": "Exploratory row-resampling stability, not player-clustered uncertainty or predictive validation", "salary_in_pca": False, "age_position_in_pca": False, "dev_evaluated": False, "test_evaluated": False}
    (output / "feature_definitions.json").write_text(json.dumps(definitions, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print (summary.to_string(index=False))
    print ("\n14-feature candidate loadings:")
    print (pd.DataFrame(loadings).query("feature_set == 'raw_14_with_minutes'")[["feature", "pc1_weight"]].to_string(index=False))
    print ("\nAudit saved to: " + str(output))
    return output


if(__name__ == "__main__"):
    main()
