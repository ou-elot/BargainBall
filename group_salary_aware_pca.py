"""Find salary-aware player-season groups with a training-only PCA/K-means fit.

The four PCA inputs are next-season PIE, WS, VORP, and log salary-cap share.
The scaler, PCA, and K-means model use training rows only. Development rows
are projected and assigned for descriptive follow-up; the test split is not
loaded. Rows without a listed salary are omitted rather than imputed.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "development" / "salary_aware_pca"
MODEL_DATA = PROCESSED / "model_data.npz"
SALARY_DATA = PROCESSED / "salary_cap_normalized_pairs.csv"
N_PROFILES = 4
SEED = 42
PERFORMANCE = ["pie", "ws", "vorp"]
FEATURES = PERFORMANCE + ["log_salary_cap_share"]


def add_salary(rows, salary):
    fields = [
        "player_id", "target_season", "salary_cap_share",
        "salary_available_check", "salary_needs_source_review",
    ]
    joined = rows.merge(
        salary[fields],
        on=["player_id", "target_season"],
        how="left",
        validate="one_to_one",
    )
    joined["salary_cap_share"] = pd.to_numeric(joined["salary_cap_share"], errors="coerce")
    joined["salary_available_check"] = joined["salary_available_check"].fillna(False).astype(bool)
    joined["salary_needs_source_review"] = joined["salary_needs_source_review"].fillna(True).astype(bool)
    joined["log_salary_cap_share"] = np.nan
    positive = joined["salary_cap_share"].gt(0) & np.isfinite(joined["salary_cap_share"])
    joined.loc[positive, "log_salary_cap_share"] = np.log(joined.loc[positive, "salary_cap_share"])
    return joined


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    salary = pd.read_csv(SALARY_DATA)
    train_rows = pd.read_csv(PROCESSED / "train_rows.csv")
    dev_rows = pd.read_csv(PROCESSED / "dev_rows.csv")

    with np.load(MODEL_DATA, allow_pickle=False) as arrays:
        if len(train_rows) != len(arrays["y_pie_train"]):
            raise ValueError("Training rows do not align with the saved training targets.")
        train = train_rows[["player", "player_id", "season", "target_season"]].copy()
        for target in PERFORMANCE:
            train[target] = arrays["y_" + target + "_train"].astype(float)
    dev = dev_rows[["player", "player_id", "season", "target_season"]].copy()
    for target in PERFORMANCE:
        dev[target] = pd.to_numeric(dev_rows["next_season_" + target], errors="coerce")

    train = add_salary(train, salary)
    dev = add_salary(dev, salary)
    train_eligible = (
        train[FEATURES].notna().all(axis=1)
        & np.isfinite(train[FEATURES].to_numpy(float)).all(axis=1)
    )
    dev_eligible = (
        dev[FEATURES].notna().all(axis=1)
        & np.isfinite(dev[FEATURES].to_numpy(float)).all(axis=1)
    )
    train_fit = train.loc[train_eligible].copy()
    dev_fit = dev.loc[dev_eligible].copy()
    if len(train_fit) <= N_PROFILES:
        raise ValueError("Too few eligible training rows for salary-aware profiles.")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_fit[FEATURES].to_numpy(float))
    pca = PCA(n_components=len(FEATURES), svd_solver="full")
    train_pc = pca.fit_transform(train_scaled)
    if pca.components_[0].sum() < 0:
        pca.components_[0] *= -1
        train_pc[:, 0] *= -1

    clusterer = KMeans(n_clusters=N_PROFILES, random_state=SEED, n_init=50)
    train_labels = clusterer.fit_predict(train_pc)
    centers = clusterer.cluster_centers_
    order = np.argsort(centers[:, 0])
    profile_names = {int(label): f"Profile {rank + 1}" for rank, label in enumerate(order)}
    train_fit["profile"] = [profile_names[int(label)] for label in train_labels]
    train_fit[[f"pc{i + 1}" for i in range(len(FEATURES))]] = train_pc

    dev_scaled = scaler.transform(dev_fit[FEATURES].to_numpy(float))
    dev_pc = pca.transform(dev_scaled)
    dev_labels = clusterer.predict(dev_pc)
    dev_fit["profile"] = [profile_names[int(label)] for label in dev_labels]
    dev_fit[[f"pc{i + 1}" for i in range(len(FEATURES))]] = dev_pc
    train_fit["split"] = "training"
    dev_fit["split"] = "development"

    rows = pd.concat([train_fit, dev_fit], ignore_index=True)
    rows.to_csv(RESULTS / "salary_aware_player_groups.csv", index=False)

    summaries = []
    for profile in sorted(profile_names.values(), key=lambda name: int(name.split()[-1])):
        for split, group in rows.groupby("split"):
            selected = group[group["profile"].eq(profile)]
            if selected.empty:
                continue
            summary = {
                "split": split,
                "profile": profile,
                "player_seasons": int(len(selected)),
                "mean_salary_cap_share_pct": float(selected["salary_cap_share"].mean() * 100),
                "median_salary_cap_share_pct": float(selected["salary_cap_share"].median() * 100),
                "flagged_salary_rows": int(
                    (selected["salary_available_check"] & selected["salary_needs_source_review"]).sum()
                ),
                "pc1_center": float(selected["pc1"].mean()),
            }
            for target in PERFORMANCE:
                summary[f"mean_{target}"] = float(selected[target].mean())
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(RESULTS / "profile_summary.csv", index=False)

    metadata = {
        "features": FEATURES,
        "salary_feature_definition": "natural log of target-season salary divided by target-season salary cap",
        "salary_review_flags_included": "yes; current listed values retained, counts exported; missing salaries excluded, no imputation",
        "training_rows_total": int(len(train)),
        "training_rows_used": int(len(train_fit)),
        "training_missing_salary_or_outcome": int(len(train) - len(train_fit)),
        "training_salary_flagged_rows_used": int(
            (train_fit["salary_available_check"] & train_fit["salary_needs_source_review"]).sum()
        ),
        "development_rows_total": int(len(dev)),
        "development_rows_used": int(len(dev_fit)),
        "development_missing_salary_or_outcome": int(len(dev) - len(dev_fit)),
        "development_salary_flagged_rows_used": int(
            (dev_fit["salary_available_check"] & dev_fit["salary_needs_source_review"]).sum()
        ),
        "n_profiles": N_PROFILES,
        "seed": SEED,
        "scaler_means": scaler.mean_.tolist(),
        "scaler_scales": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cluster_centers": centers.tolist(),
        "training_silhouette_score": float(silhouette_score(train_pc, train_labels)),
        "test_data_loaded": False,
        "model_arrays_sha256": hashlib.sha256(MODEL_DATA.read_bytes()).hexdigest(),
        "salary_data_sha256": hashlib.sha256(SALARY_DATA.read_bytes()).hexdigest(),
        "interpretation": "Exploratory descriptive groups jointly reflect performance and salary; not causal or a dollar valuation.",
    }
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Training player-seasons used: {len(train_fit)} / {len(train)}")
    print(f"Development player-seasons assigned: {len(dev_fit)} / {len(dev)}")
    print(f"Training salary-flagged rows included: {metadata['training_salary_flagged_rows_used']}")
    print(f"Development salary-flagged rows included: {metadata['development_salary_flagged_rows_used']}")
    print(f"Training silhouette: {metadata['training_silhouette_score']:.4f}")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Test split not loaded. Results: " + str(RESULTS))


if __name__ == "__main__":
    main()
