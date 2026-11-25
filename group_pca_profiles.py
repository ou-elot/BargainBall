"""Cluster PCA-based performance profiles using training-only fit.

The scaler, PCA, and K-means clusters are fitted on actual training outcomes.
Development player profiles are assigned from the existing model forecasts;
observed development outcomes are used only for descriptive validation.
Salary is joined only for summaries and is not a PCA or clustering feature.
No test data are loaded.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "development" / "pca_profiles"
MODEL_DATA = PROCESSED / "model_data.npz"
PCA_PARAMETERS = ROOT / "models" / "pca" / "pca_parameters.json"
DEV_SCORES = ROOT / "results" / "development" / "pca_scores.csv"
SALARY_DATA = PROCESSED / "salary_cap_normalized_pairs.csv"
TARGETS = ["pie", "ws", "vorp"]
N_PROFILES = 4
SEED = 42


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    saved_parameters = json.loads(PCA_PARAMETERS.read_text(encoding="utf-8"))
    model_hash = hashlib.sha256(MODEL_DATA.read_bytes()).hexdigest()
    if saved_parameters["model_data_sha256"] != model_hash:
        raise ValueError("Saved PCA and current model arrays do not match.")

    with np.load(MODEL_DATA, allow_pickle=False) as arrays:
        training = np.column_stack(
            [arrays["y_" + target + "_train"].astype(float) for target in TARGETS]
        )
        if not np.isfinite(training).all():
            raise ValueError("Training outcome matrix contains missing/nonfinite values.")

    scaler = StandardScaler()
    train_standardized = scaler.fit_transform(training)
    pca = PCA(n_components=len(TARGETS), svd_solver="full")
    train_coordinates = pca.fit_transform(train_standardized)
    # PCA axis signs are arbitrary; orient PC1 toward higher average outcomes.
    if pca.components_[0].sum() < 0:
        pca.components_[0] *= -1
        train_coordinates[:, 0] *= -1

    clusterer = KMeans(n_clusters=N_PROFILES, random_state=SEED, n_init=50)
    train_cluster_raw = clusterer.fit_predict(train_coordinates)
    centers = clusterer.cluster_centers_
    order = np.argsort(centers[:, 0])
    profile_names = {int(cluster): f"Profile {rank + 1}" for rank, cluster in enumerate(order)}
    train_profiles = np.array([profile_names[int(value)] for value in train_cluster_raw])

    dev = pd.read_csv(DEV_SCORES)
    predicted_targets = dev[["predicted_" + target for target in TARGETS]].to_numpy(float)
    actual_targets = pd.read_csv(PROCESSED / "dev_rows.csv")[
        ["next_season_" + target for target in TARGETS]
    ].to_numpy(float)
    if len(dev) != len(actual_targets) or not np.isfinite(predicted_targets).all():
        raise ValueError("Development predictions and outcomes are missing or misaligned.")
    if not np.isfinite(actual_targets).all():
        raise ValueError("Development actual outcomes contain missing/nonfinite values.")

    predicted_coordinates = pca.transform(scaler.transform(predicted_targets))
    actual_coordinates = pca.transform(scaler.transform(actual_targets))
    predicted_cluster_raw = clusterer.predict(predicted_coordinates)
    actual_cluster_raw = clusterer.predict(actual_coordinates)
    predicted_profiles = np.array([profile_names[int(value)] for value in predicted_cluster_raw])
    actual_profiles = np.array([profile_names[int(value)] for value in actual_cluster_raw])

    players = dev[["player", "player_id", "season", "target_season"]].copy()
    for index, target in enumerate(TARGETS):
        players["predicted_" + target] = predicted_targets[:, index]
        players["actual_" + target] = actual_targets[:, index]
    for index in range(len(TARGETS)):
        players[f"predicted_pc{index + 1}"] = predicted_coordinates[:, index]
        players[f"actual_pc{index + 1}"] = actual_coordinates[:, index]
    players["predicted_profile"] = predicted_profiles
    players["actual_outcome_profile"] = actual_profiles
    players["profile_match"] = predicted_profiles == actual_profiles

    salary = pd.read_csv(SALARY_DATA)
    salary_fields = [
        "player_id", "target_season", "salary_cap_share_pct",
        "salary_available_check", "salary_needs_source_review",
    ]
    players = players.merge(
        salary[salary_fields],
        on=["player_id", "target_season"],
        how="left",
        validate="one_to_one",
    )
    players.to_csv(RESULTS / "development_player_profiles.csv", index=False)

    profile_rows = []
    for profile in sorted(profile_names.values(), key=lambda name: int(name.split()[-1])):
        selected = players[players["predicted_profile"] == profile]
        outcome_match = selected["profile_match"].mean()
        row = {
            "predicted_profile": profile,
            "players_assigned_by_forecast": int(len(selected)),
            "players_with_observed_outcome_in_profile": int(
                (players["actual_outcome_profile"] == profile).sum()
            ),
            "mean_predicted_pc1": float(selected["predicted_pc1"].mean()),
            "mean_actual_pc1": float(selected["actual_pc1"].mean()),
            "profile_assignment_match_rate": float(outcome_match),
            "mean_salary_cap_share_pct_available": float(
                selected["salary_cap_share_pct"].mean()
            ),
            "salary_rows_flagged_for_source_review": int(
                (selected["salary_needs_source_review"].fillna(True).astype(bool)
                 & selected["salary_available_check"].fillna(False).astype(bool)).sum()
            ),
        }
        for target in TARGETS:
            row["mean_predicted_" + target] = float(selected["predicted_" + target].mean())
            row["mean_actual_" + target] = float(selected["actual_" + target].mean())
        profile_rows.append(row)
    pd.DataFrame(profile_rows).to_csv(RESULTS / "profile_summary.csv", index=False)

    transition = pd.crosstab(
        pd.Series(actual_profiles, name="actual_outcome_profile"),
        pd.Series(predicted_profiles, name="predicted_profile"),
        normalize="index",
    )
    transition.to_csv(RESULTS / "profile_transition_rates.csv")

    model_metadata = {
        "targets": TARGETS,
        "training_rows": int(len(training)),
        "training_means": scaler.mean_.tolist(),
        "training_scales": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cluster_count": N_PROFILES,
        "random_seed": SEED,
        "training_cluster_centers": centers.tolist(),
        "profile_order": "Profiles are numbered by ascending training cluster-center PC1; clusters are multidimensional performance profiles, not equal-width tiers.",
        "salary_used_as_pca_or_clustering_feature": False,
        "test_data_loaded": False,
    }
    (RESULTS / "fitted_profile_model.json").write_text(
        json.dumps(model_metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit = {
        "split_policy": "Scaler, PCA, and K-means fitted on actual training outcomes only; development assignments use existing PCA model forecasts; test split not loaded.",
        "training_rows": int(len(training)),
        "development_rows": int(len(players)),
        "development_profile_match_rate": float(players["profile_match"].mean()),
        "development_adjusted_rand_index": float(
            adjusted_rand_score(actual_profiles, predicted_profiles)
        ),
        "training_silhouette_score": float(
            silhouette_score(train_coordinates, train_cluster_raw)
        ),
        "salary_is_context_only": True,
        "n_profiles": N_PROFILES,
        "test_evaluated": False,
        "model_data_sha256": model_hash,
    }
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Training-fitted PCA/K-means profiles: " + str(N_PROFILES))
    print("Training silhouette score: " + str(round(audit["training_silhouette_score"], 4)))
    print("Development forecast profile match rate vs observed outcomes: "
          + str(round(audit["development_profile_match_rate"], 4)))
    print("Development adjusted Rand index: "
          + str(round(audit["development_adjusted_rand_index"], 4)))
    print(pd.DataFrame(profile_rows).to_string(index=False))
    print("Salary excluded from clustering; test split not loaded.")
    print("Results: " + str(RESULTS))


if __name__ == "__main__":
    main()
