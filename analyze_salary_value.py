"""Compare PCA-based performance forecasts with cap-share salary on train/dev.

The salary-performance benchmark is fit on the training split only. Shortlist
evaluation is development-only; the previously examined test split is never
loaded. Rows with missing or source-review salaries are excluded from the
primary comparison and counted in the audit.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "development" / "salary_value"
PCA_PARAMETERS_PATH = ROOT / "models" / "pca" / "pca_parameters.json"
MODEL_DATA_PATH = PROCESSED / "model_data.npz"
SALARY_PATH = PROCESSED / "salary_cap_normalized_pairs.csv"
SHORTLIST_SIZES = (25, 50, 100)


def combine(values, parameters):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(parameters["target_order"]):
        raise ValueError("PCA target columns do not match the saved training-fitted PCA.")
    standardized = (
        values - np.asarray(parameters["target_means"])
    ) / np.asarray(parameters["target_stds"])
    centered = standardized - np.asarray(parameters["pca_center"])
    return (
        centered @ np.asarray(parameters["component_weights"])
        - parameters["score_mean"]
    ) / parameters["score_std"]


def checked_join(rows, salary, label):
    keys = ["player_id", "target_season"]
    salary_fields = keys + [
        "target_salary",
        "salary_cap_share",
        "salary_cap_share_pct",
        "salary_needs_source_review",
        "salary_method",
        "salary_available_check",
    ]
    joined = rows.merge(
        salary[salary_fields],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(label + " rows did not all match normalized salary records.")
    return joined.drop(columns="_merge")


def fit_salary_benchmark(train):
    eligible = (
        train["salary_available_check"].fillna(False).astype(bool)
        & ~train["salary_needs_source_review"].fillna(True).astype(bool)
        & np.isfinite(train["actual_pca_score"])
        & np.isfinite(train["salary_cap_share"])
        & train["salary_cap_share"].gt(0)
    )
    fit_rows = train.loc[eligible].copy()
    if len(fit_rows) < 20:
        raise ValueError("Too few non-flagged training salaries to fit a benchmark.")

    # Log cap share captures a diminishing performance association while
    # keeping the benchmark transparent and trained entirely on training data.
    x = np.log(fit_rows["salary_cap_share"].to_numpy(dtype=float))
    y = fit_rows["actual_pca_score"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2 or not np.isfinite(coefficients).all():
        raise ValueError("Salary benchmark fit was singular or nonfinite.")
    return coefficients, fit_rows


def expected_performance(cap_share, coefficients):
    share = np.asarray(cap_share, dtype=float)
    result = np.full(share.shape, np.nan, dtype=float)
    valid = np.isfinite(share) & (share > 0)
    result[valid] = coefficients[0] + coefficients[1] * np.log(share[valid])
    return result


def rank_shortlists(dev, score_columns):
    rows = []
    for k in SHORTLIST_SIZES:
        available_k = min(k, len(dev))
        ranked = {}
        for label, column in score_columns.items():
            ranked[label] = set(
                dev.nlargest(available_k, column, keep="first").index.tolist()
            )
        actual_top = set(
            dev.nlargest(available_k, "actual_value_residual", keep="first").index.tolist()
        )
        for label, selected in ranked.items():
            chosen = dev.loc[list(selected)]
            rows.append(
                {
                    "shortlist_size": available_k,
                    "ranking": label,
                    "mean_actual_pca_score": float(chosen["actual_pca_score"].mean()),
                    "mean_actual_value_residual": float(chosen["actual_value_residual"].mean()),
                    "mean_salary_cap_share_pct": float(chosen["salary_cap_share_pct"].mean()),
                    "median_salary_cap_share_pct": float(chosen["salary_cap_share_pct"].median()),
                    "mean_salary_cap_share_pct_all_eligible": float(dev["salary_cap_share_pct"].mean()),
                    "overlap_with_actual_top_value": len(selected & actual_top),
                    "overlap_fraction_of_actual_top_value": len(selected & actual_top) / available_k,
                }
            )
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(PCA_PARAMETERS_PATH.read_text(encoding="utf-8"))
    salary = pd.read_csv(SALARY_PATH)
    data_hash = hashlib.sha256(MODEL_DATA_PATH.read_bytes()).hexdigest()
    if parameters["model_data_sha256"] != data_hash:
        raise ValueError("PCA parameters do not match current model arrays.")
    if parameters["fit_split"] != "train":
        raise ValueError("PCA parameters were not fitted on training data.")

    train_rows = pd.read_csv(PROCESSED / "train_rows.csv")
    dev_scores = pd.read_csv(ROOT / "results" / "development" / "pca_scores.csv")
    train = train_rows[["player", "player_id", "season", "target_season"]].copy()
    dev = dev_scores[
        [
            "player", "player_id", "season", "target_season",
            "actual_pca_score", "predicted_pca_score", "baseline_pca_score",
        ]
    ].copy()

    with np.load(MODEL_DATA_PATH, allow_pickle=False) as arrays:
        if len(train) != len(arrays["X_train"]):
            raise ValueError("Training metadata does not match training arrays.")
        train_targets = np.column_stack(
            [arrays["y_" + target + "_train"].astype(float) for target in parameters["target_order"]]
        )
    train["actual_pca_score"] = combine(train_targets, parameters)
    train = checked_join(train, salary, "Training")
    dev = checked_join(dev, salary, "Development")

    coefficients, fit_rows = fit_salary_benchmark(train)
    train["benchmark_expected_pca_score"] = expected_performance(
        train["salary_cap_share"], coefficients
    )
    dev["benchmark_expected_pca_score"] = expected_performance(
        dev["salary_cap_share"], coefficients
    )
    train["observed_value_residual"] = (
        train["actual_pca_score"] - train["benchmark_expected_pca_score"]
    )
    dev["actual_value_residual"] = (
        dev["actual_pca_score"] - dev["benchmark_expected_pca_score"]
    )
    dev["forecast_value_residual"] = (
        dev["predicted_pca_score"] - dev["benchmark_expected_pca_score"]
    )
    dev["prior_performance_value_residual"] = (
        dev["baseline_pca_score"] - dev["benchmark_expected_pca_score"]
    )

    train.to_csv(RESULTS / "training_salary_benchmark_rows.csv", index=False)
    dev.to_csv(RESULTS / "development_salary_value_rows_all.csv", index=False)

    eligible = (
        dev["salary_available_check"].fillna(False).astype(bool)
        & ~dev["salary_needs_source_review"].fillna(True).astype(bool)
        & np.isfinite(dev["actual_pca_score"])
        & np.isfinite(dev["predicted_pca_score"])
        & np.isfinite(dev["baseline_pca_score"])
        & np.isfinite(dev["salary_cap_share"])
        & dev["salary_cap_share"].gt(0)
    )
    primary = dev.loc[eligible].copy()
    if len(primary) < max(SHORTLIST_SIZES):
        raise ValueError("Not enough non-flagged development salary rows for all shortlist sizes.")
    primary.to_csv(RESULTS / "development_salary_value_rows.csv", index=False)
    comparisons = rank_shortlists(
        primary,
        {
            "PCA forecast value": "forecast_value_residual",
            "prior-season PCA value": "prior_performance_value_residual",
        },
    )
    comparisons.to_csv(RESULTS / "shortlist_comparison.csv", index=False)

    model_rmse = float(np.sqrt(np.mean(
        (primary["predicted_pca_score"] - primary["actual_pca_score"]) ** 2
    )))
    baseline_rmse = float(np.sqrt(np.mean(
        (primary["baseline_pca_score"] - primary["actual_pca_score"]) ** 2
    )))
    audit = {
        "split_policy": "Benchmark fitted on training only; shortlist comparisons evaluated on development only; test split not loaded.",
        "performance_score": "Existing training-fitted PCA of next-season PIE, WS, and VORP; higher is better.",
        "salary_benchmark": "OLS linear fit of actual training PCA score on log(target-season salary / target-season salary cap), using available salaries not flagged for source review.",
        "benchmark_coefficients": {
            "intercept": float(coefficients[0]),
            "log_cap_share": float(coefficients[1]),
        },
        "benchmark_training_rows": int(len(fit_rows)),
        "training_rows_total": int(len(train)),
        "training_salary_available_not_flagged": int(
            (train["salary_available_check"].fillna(False).astype(bool)
             & ~train["salary_needs_source_review"].fillna(True).astype(bool)).sum()
        ),
        "development_rows_total": int(len(dev)),
        "development_salary_available_not_flagged": int(eligible.sum()),
        "development_salary_missing": int((~dev["salary_available_check"].fillna(False).astype(bool)).sum()),
        "development_salary_available_flagged_for_source_review": int(
            (dev["salary_available_check"].fillna(False).astype(bool)
             & dev["salary_needs_source_review"].fillna(True).astype(bool)).sum()
        ),
        "development_pca_rmse_on_nonflagged_salary_rows": model_rmse,
        "development_prior_pca_rmse_on_nonflagged_salary_rows": baseline_rmse,
        "development_rows_excluded_for_missing_or_review_salary": int((~eligible).sum()),
        "shortlist_sizes": list(SHORTLIST_SIZES),
        "comparison_csv": str((RESULTS / "shortlist_comparison.csv").relative_to(ROOT)),
        "interpretation": "Value residual is performance above/below the training-only average associated with cap-share salary; it is not causal or a contract valuation. Shortlist findings are exploratory because development data informed prior model selection.",
        "test_evaluated": False,
        "salary_normalized_file_sha256": hashlib.sha256(SALARY_PATH.read_bytes()).hexdigest(),
        "pca_parameters_sha256": hashlib.sha256(PCA_PARAMETERS_PATH.read_bytes()).hexdigest(),
    }
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("Training-only salary benchmark rows: " + str(len(fit_rows)))
    print("Eligible (salary available, not flagged) development rows: " + str(len(primary)) + " / " + str(len(dev)))
    print("Development PCA RMSE / prior baseline RMSE: "
          + str(round(model_rmse, 4)) + " / " + str(round(baseline_rmse, 4)))
    print(comparisons.to_string(index=False))
    print("Test split not loaded. Results: " + str(RESULTS))


if __name__ == "__main__":
    main()
