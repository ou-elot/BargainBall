"""Estimate cap share from historical nearest-player salary comparisons.

This is a descriptive nearest-comps summary, not a trained salary model.
Missing and source-review-flagged salary observations are excluded.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "development" / "multitask_profile"
MODEL_ROOT = ROOT / "models" / "numpy_multitask_profile"
SALARY_PATH = ROOT / "data" / "processed" / "salary_cap_normalized_pairs.csv"
MIN_USABLE_COMPS = 5
QUERY_KEYS = [
    "query_player",
    "query_player_id",
    "query_prior_season",
    "query_target_season",
]


def find_latest_unrestricted_run():
    candidates = []
    for result_dir in RESULTS_ROOT.iterdir():
        metadata_path = MODEL_ROOT / result_dir.name / "metadata.json"
        comps_path = result_dir / "development_profile_neighbors.csv"
        if not metadata_path.exists() or not comps_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("training_minimum_prior_games", 0)) == 0:
            candidates.append(result_dir)
    if not candidates:
        raise FileNotFoundError("No unrestricted profile-model result run was found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_comps(comps, cap_by_target_season):
    rows = []
    for key, group in comps.groupby(QUERY_KEYS, sort=False, dropna=False):
        group = group.sort_values("comp_rank", kind="stable")
        share = pd.to_numeric(
            group["comp_target_salary_cap_share_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        review = group["comp_salary_review_flag"].fillna(True).astype(bool).to_numpy()
        usable = np.isfinite(share) & (share >= 0) & ~review
        values = share[usable]
        valid_comps = group.loc[usable]
        target_season = int(key[3])
        cap = cap_by_target_season.get(target_season, np.nan)
        enough = len(values) >= MIN_USABLE_COMPS
        estimate_pct = float(np.median(values)) if enough else np.nan
        q25 = float(np.quantile(values, 0.25)) if enough else np.nan
        q75 = float(np.quantile(values, 0.75)) if enough else np.nan
        row = dict(zip(QUERY_KEYS, key))
        row.update(
            {
                "salary_estimate_method": (
                    "median salary-cap share of nearest historical profiles"
                ),
                "nearest_profile_comp_count": int(len(group)),
                "usable_salary_comp_count": int(len(values)),
                "minimum_usable_comps": MIN_USABLE_COMPS,
                "salary_estimate_available": bool(enough),
                "estimated_salary_cap_share_pct": estimate_pct,
                "salary_cap_share_q25_pct": q25,
                "salary_cap_share_q75_pct": q75,
                "target_season_salary_cap_usd": float(cap) if pd.notna(cap) else np.nan,
                "estimated_salary_usd": (
                    estimate_pct / 100 * float(cap)
                    if enough and pd.notna(cap)
                    else np.nan
                ),
                "salary_q25_usd": (
                    q25 / 100 * float(cap)
                    if enough and pd.notna(cap)
                    else np.nan
                ),
                "salary_q75_usd": (
                    q75 / 100 * float(cap)
                    if enough and pd.notna(cap)
                    else np.nan
                ),
                "usable_comps": "; ".join(
                    f"{comp.historical_comp} ({comp.comp_target_salary_cap_share_pct:.2f}%)"
                    for comp in valid_comps.itertuples()
                ),
                "estimate_warning": (
                    ""
                    if enough
                    else f"Only {len(values)} eligible comp salaries; fewer than {MIN_USABLE_COMPS}."
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main(run_id=None):
    result_dir = RESULTS_ROOT / run_id if run_id else find_latest_unrestricted_run()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {result_dir}")
    metadata_path = MODEL_ROOT / result_dir.name / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("training_minimum_prior_games", 0)) != 0:
        raise ValueError("Salary estimates should use the unrestricted primary model.")

    salary_data = pd.read_csv(SALARY_PATH)
    cap_table = salary_data[
        ["target_season", "salary_cap_usd"]
    ].dropna().drop_duplicates()
    cap_counts = cap_table.groupby("target_season")["salary_cap_usd"].nunique()
    if cap_counts.gt(1).any():
        raise ValueError("Salary-cap values are inconsistent within target season.")
    cap_by_target_season = cap_table.drop_duplicates("target_season").set_index(
        "target_season"
    )["salary_cap_usd"].to_dict()

    result_paths = [
        (
            "development_profile_neighbors.csv",
            "development_comp_salary_estimates.csv",
        ),
        (
            "development_bench_to_starter_neighbors.csv",
            "development_bench_to_starter_salary_estimates.csv",
        ),
    ]
    report = {
        "run_directory": str(result_dir.relative_to(ROOT)),
        "method": "Median salary-cap share among nearest profile comps with an available, non-flagged source salary.",
        "similarity_model": "Salary is not used in forecast training, PCA, distance calculation, or comp ranking.",
        "minimum_usable_comps": MIN_USABLE_COMPS,
        "test_split_loaded": False,
        "outputs": [],
    }
    for comps_file, output_file in result_paths:
        comps_path = result_dir / comps_file
        comps = pd.read_csv(comps_path)
        estimates = summarize_comps(comps, cap_by_target_season)
        output_path = result_dir / output_file
        estimates.to_csv(output_path, index=False)
        report["outputs"].append(
            {
                "file": output_file,
                "query_count": int(len(estimates)),
                "estimates_available": int(estimates["salary_estimate_available"].sum()),
                "estimates_insufficient_comps": int(
                    (~estimates["salary_estimate_available"]).sum()
                ),
            }
        )

    (result_dir / "comp_salary_estimate_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    all_estimates = pd.read_csv(result_dir / result_paths[0][1])
    bench_estimates = pd.read_csv(result_dir / result_paths[1][1])
    print("Primary model run:", result_dir)
    print("\nExample nearest-comp salary estimates:")
    print(
        all_estimates[
            [
                "query_player",
                "query_target_season",
                "usable_salary_comp_count",
                "estimated_salary_cap_share_pct",
                "salary_cap_share_q25_pct",
                "salary_cap_share_q75_pct",
                "estimated_salary_usd",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(
        f"\nAll-player estimates available: "
        f"{all_estimates['salary_estimate_available'].sum()} / {len(all_estimates)}"
    )
    print(
        f"Bench-to-starter estimates available: "
        f"{bench_estimates['salary_estimate_available'].sum()} / {len(bench_estimates)}"
    )
    print("Salary is comp context only; it does not alter similarity rankings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        help="Profile-model run folder name; defaults to latest unrestricted-training run.",
    )
    main(run_id=parser.parse_args().run_id)
