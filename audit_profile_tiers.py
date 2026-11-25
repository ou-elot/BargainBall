"""Audit player-quality balance and development errors by prior-season tier.

Tier boundaries are quantiles of training-set prior-season PIE only. No test
split is loaded, and no model is retrained or reweighted by this script.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS_ROOT = ROOT / "results" / "development" / "multitask_profile"
QUANTILES = [0.20, 0.40, 0.60, 0.80]
TIER_NAMES = ["lowest", "lower-middle", "upper-middle", "high", "highest"]


def main():
    result_dir = max(
        (path for path in RESULTS_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    train = pd.read_csv(PROCESSED / "train_rows_rates.csv")
    dev = pd.read_csv(PROCESSED / "dev_rows_rates.csv")
    predictions = pd.read_csv(result_dir / "development_profile_predictions.csv")
    if len(dev) != len(predictions):
        raise ValueError("Development metadata and predictions do not align.")
    if not dev[["player_id", "season", "target_season"]].equals(
        predictions[["player_id", "season", "target_season"]]
    ):
        raise ValueError("Development player-season order differs from predictions.")

    train_pie = train["pie"].to_numpy(dtype=float)
    if not np.isfinite(train_pie).all():
        raise ValueError("Training prior-season PIE must be complete for tiering.")
    boundaries = np.quantile(train_pie, QUANTILES)
    labels = np.searchsorted(boundaries, train_pie, side="right")
    train_distribution = []
    for tier_id, name in enumerate(TIER_NAMES):
        values = train_pie[labels == tier_id]
        train_distribution.append(
            {
                "prior_pie_tier": name,
                "training_rows": int(len(values)),
                "training_share": float(len(values) / len(train_pie)),
                "prior_pie_min": float(values.min()) if len(values) else None,
                "prior_pie_median": float(np.median(values)) if len(values) else None,
                "prior_pie_max": float(values.max()) if len(values) else None,
            }
        )

    dev_pie = dev["pie"].to_numpy(dtype=float)
    dev_tiers = np.searchsorted(boundaries, dev_pie, side="right")
    dev["prior_pie_tier"] = [TIER_NAMES[index] for index in dev_tiers]
    dev["prior_pie_training_percentile"] = np.searchsorted(
        np.sort(train_pie), dev_pie, side="right"
    ) / len(train_pie)
    predictions["prior_pie_tier"] = dev["prior_pie_tier"].to_numpy()
    predictions["prior_pie_training_percentile"] = (
        dev["prior_pie_training_percentile"].to_numpy()
    )

    target_names = [
        column.removeprefix("actual_")
        for column in predictions.columns
        if column.startswith("actual_")
        and "predicted_" + column.removeprefix("actual_") in predictions.columns
        and "prior_" + column.removeprefix("actual_") in predictions.columns
    ]
    by_tier = []
    for tier_name in TIER_NAMES:
        group = predictions[predictions["prior_pie_tier"] == tier_name]
        for target in target_names:
            actual = group["actual_" + target].to_numpy(dtype=float)
            predicted = group["predicted_" + target].to_numpy(dtype=float)
            baseline = group["prior_" + target].to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(baseline)
            if not valid.any():
                continue
            model_error = predicted[valid] - actual[valid]
            baseline_error = baseline[valid] - actual[valid]
            by_tier.append(
                {
                    "prior_pie_tier": tier_name,
                    "target": target,
                    "rows": int(valid.sum()),
                    "model_rmse": float(np.sqrt(np.mean(model_error**2))),
                    "model_mae": float(np.mean(np.abs(model_error))),
                    "persistence_rmse": float(np.sqrt(np.mean(baseline_error**2))),
                    "persistence_mae": float(np.mean(np.abs(baseline_error))),
                    "model_rmse_gain_pct": float(
                        (1 - np.sqrt(np.mean(model_error**2))
                         / np.sqrt(np.mean(baseline_error**2))) * 100
                    ),
                }
            )

    tails = []
    for target in target_names:
        actual = predictions["actual_" + target].to_numpy(dtype=float)
        predicted = predictions["predicted_" + target].to_numpy(dtype=float)
        baseline = predictions["prior_" + target].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(baseline)
        model_squared = (predicted[valid] - actual[valid]) ** 2
        baseline_squared = (baseline[valid] - actual[valid]) ** 2
        n_tail = max(1, int(np.ceil(0.05 * len(model_squared))))
        model_order = np.argsort(model_squared)
        baseline_order = np.argsort(baseline_squared)
        tails.append(
            {
                "target": target,
                "rows": int(valid.sum()),
                "model_rmse": float(np.sqrt(np.mean(model_squared))),
                "model_rmse_without_largest_5pct_errors": float(
                    np.sqrt(np.mean(model_squared[model_order[:-n_tail]]))
                ),
                "model_top_5pct_squared_error_share": float(
                    model_squared[model_order[-n_tail:]].sum()
                    / model_squared.sum()
                ),
                "persistence_rmse": float(np.sqrt(np.mean(baseline_squared))),
                "persistence_top_5pct_squared_error_share": float(
                    baseline_squared[baseline_order[-n_tail:]].sum()
                    / baseline_squared.sum()
                ),
            }
        )

    distribution_frame = pd.DataFrame(train_distribution)
    tier_frame = pd.DataFrame(by_tier)
    tail_frame = pd.DataFrame(tails)
    distribution_frame.to_csv(result_dir / "training_prior_pie_tiers.csv", index=False)
    tier_frame.to_csv(result_dir / "development_metrics_by_prior_pie_tier.csv", index=False)
    tail_frame.to_csv(result_dir / "development_tail_error_audit.csv", index=False)

    report = {
        "tier_feature": "prior-season PIE",
        "tier_boundary_fit": "training split only",
        "training_quantile_boundaries": {
            str(q): float(value) for q, value in zip(QUANTILES, boundaries)
        },
        "training_tiers": train_distribution,
        "development_rows": int(len(dev)),
        "development_tier_counts": {
            name: int((dev["prior_pie_tier"] == name).sum()) for name in TIER_NAMES
        },
        "test_split_loaded": False,
        "model_retrained_or_reweighted": False,
        "files": [
            "training_prior_pie_tiers.csv",
            "development_metrics_by_prior_pie_tier.csv",
            "development_tail_error_audit.csv",
        ],
    }
    (result_dir / "prior_pie_tier_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Results directory:", result_dir)
    print("\nTraining prior-PIE distribution:")
    print(distribution_frame.to_string(index=False))
    print("\nDevelopment metrics by tier (key outcomes):")
    key_metrics = tier_frame[tier_frame["target"].isin(["pie", "bpm", "ws_48", "minutes"])]
    print(key_metrics.to_string(index=False))
    print("\nExtreme-error sensitivity:")
    print(tail_frame.to_string(index=False))
    print("\nNo oversampling or reweighting performed; test split not loaded.")


if __name__ == "__main__":
    main()
