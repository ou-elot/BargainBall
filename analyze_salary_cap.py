"""Join target-season salary caps and calculate cap-share fields.

Run after `prepare_data.py`:
    python analyze_salary_cap.py

This is a valuation-data preparation step. It does not use salary as a model
input, impute missing salary, or remove source-review rows.
"""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
CAP_PATH = ROOT / "data" / "raw" / "salary_cap_history.csv"
PAIRS_PATH = PROCESSED / "prediction_pairs.csv"
OUTPUT_PATH = PROCESSED / "salary_cap_normalized_pairs.csv"


def main():
    caps = pd.read_csv(CAP_PATH)
    pairs = pd.read_csv(PAIRS_PATH)
    caps["target_season"] = pd.to_numeric(caps["target_season"], errors="raise").astype(int)
    caps["salary_cap_usd"] = pd.to_numeric(caps["salary_cap_usd"], errors="raise")
    pairs["target_season"] = pd.to_numeric(pairs["target_season"], errors="raise").astype(int)
    pairs["target_salary"] = pd.to_numeric(pairs["target_salary"], errors="coerce")

    if(caps["target_season"].duplicated().any()):
        raise ValueError("Salary-cap source has duplicate target-season rows.")
    if(caps["salary_cap_usd"].isna().any() or (caps["salary_cap_usd"] <= 0).any()):
        raise ValueError("Every salary-cap value must be finite and positive.")
    if(pairs.duplicated(["player_id", "target_season"]).any()):
        raise ValueError("Player-season pairs must be unique before cap join.")

    expected_seasons = list(range(int(pairs["target_season"].min()), int(pairs["target_season"].max()) + 1))
    if(sorted(caps["target_season"].tolist()) != expected_seasons):
        raise ValueError("Cap table must contain each target season exactly once, without gaps.")

    cap_seasons = caps[["target_season", "season", "salary_cap_usd", "source", "source_url", "accessed_date"]].rename(columns={"season": "salary_cap_season"})
    normalized = pairs.merge(cap_seasons, on="target_season", how="left", validate="many_to_one", indicator=True)

    if(not normalized["_merge"].eq("both").all()):
        raise ValueError("Some target seasons did not match a salary-cap value.")

    normalized = normalized.drop(columns=["_merge"])
    normalized["salary_cap_share"] = normalized["target_salary"] / normalized["salary_cap_usd"]
    normalized["salary_cap_share_pct"] = 100.0 * normalized["salary_cap_share"]
    normalized["salary_available_check"] = normalized["target_salary"].notna() & normalized["target_salary"].gt(0)
    recorded_available = normalized["salary_available"].astype("boolean").fillna(False)

    if(not normalized["salary_available_check"].eq(recorded_available).all()):
        raise ValueError("Computed salary availability does not match the saved availability flag.")

    normalized["salary_needs_source_review"] = normalized["salary_needs_source_review"].astype("boolean").fillna(True)
    normalized["salary_reviewed_and_available"] = normalized["salary_available_check"] & ~normalized["salary_needs_source_review"]
    normalized["source_review_available_salary"] = normalized["salary_available_check"] & normalized["salary_needs_source_review"]
    normalized["source_review_missing_salary"] = ~normalized["salary_available_check"] & normalized["salary_needs_source_review"]
    normalized["salary_cap_share_over_100pct"] = normalized["salary_cap_share"].gt(1.0)
    normalized.to_csv(OUTPUT_PATH, index=False)

    coverage = normalized.groupby(["target_season", "salary_cap_season"], sort=True).agg(
        prediction_pairs=("player_id", "size"),
        salary_available=("salary_available_check", "sum"),
        salary_missing=("salary_available_check", lambda values: int((~values).sum())),
        available_salary_source_review=("source_review_available_salary", "sum"),
        missing_salary_flagged_for_review=("source_review_missing_salary", "sum"),
        reviewed_salary_available=("salary_reviewed_and_available", "sum"),
        cap_share_over_100pct=("salary_cap_share_over_100pct", "sum"),
        cap_share_median_pct=("salary_cap_share_pct", "median"),
        cap_share_mean_pct=("salary_cap_share_pct", "mean"),
    ).reset_index()
    coverage.to_csv(PROCESSED / "salary_cap_coverage_by_season.csv", index=False)

    source_review_audit = normalized.groupby(["salary_method", "salary_available_check", "salary_needs_source_review"], dropna=False).agg(
        records=("player_id", "size"),
        cap_share_median_pct=("salary_cap_share_pct", "median"),
        cap_share_over_100pct=("salary_cap_share_over_100pct", "sum"),
    ).reset_index()
    source_review_audit.to_csv(PROCESSED / "salary_cap_source_review_audit.csv", index=False)

    high_shares = normalized[normalized["salary_cap_share_over_100pct"]].copy()
    high_shares.to_csv(PROCESSED / "salary_cap_shares_over_100pct.csv", index=False)

    available = normalized[normalized["salary_available_check"]]
    review_rows = available[available["salary_needs_source_review"]]
    reviewed_rows = available[~available["salary_needs_source_review"]]
    summary = {
        "cap_source": "Basketball-Reference NBA Salary Cap History",
        "cap_source_url": "https://www.basketball-reference.com/contracts/salary-cap-history.html",
        "cap_source_accessed_date": "2026-11-21",
        "season_key": "target_season is the season ending year; e.g. target_season 2026 maps to 2025-26",
        "normalization": "salary_cap_share = target_salary / salary_cap_usd; percentage is 100 * salary_cap_share",
        "prediction_pairs": int(len(normalized)),
        "first_target_season": int(normalized["target_season"].min()),
        "last_target_season": int(normalized["target_season"].max()),
        "cap_seasons": int(caps["target_season"].nunique()),
        "rows_missing_cap": int(normalized["salary_cap_usd"].isna().sum()),
        "salary_available": int(normalized["salary_available_check"].sum()),
        "salary_missing_preserved": int((~normalized["salary_available_check"]).sum()),
        "available_salary_rows_flagged_for_source_review": int(len(review_rows)),
        "available_salary_rows_not_flagged_for_source_review": int(len(reviewed_rows)),
        "available_salary_cap_shares_over_100pct": int(len(high_shares)),
        "over_100pct_interpretation": "This is a descriptive ratio, not an invalid-record rule: NBA is a soft-cap league and an individual salary may exceed the team cap.",
        "salary_missing_rows_removed_or_imputed": 0,
        "source_review_rows_removed": 0,
        "output_csv": str(OUTPUT_PATH.relative_to(ROOT)),
        "coverage_csv": str((PROCESSED / "salary_cap_coverage_by_season.csv").relative_to(ROOT)),
        "source_review_audit_csv": str((PROCESSED / "salary_cap_source_review_audit.csv").relative_to(ROOT)),
        "over_100pct_examples_csv": str((PROCESSED / "salary_cap_shares_over_100pct.csv").relative_to(ROOT)),
    }
    (PROCESSED / "salary_cap_normalization_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Salary-cap normalization complete.")
    print("Target seasons: " + str(summary["first_target_season"]) + "-" + str(summary["last_target_season"]) + " (" + str(summary["cap_seasons"]) + " cap values)")
    print("Salary rows: " + str(summary["salary_available"]) + " available; " + str(summary["salary_missing_preserved"]) + " missing preserved")
    print("Available salaries flagged for source review: " + str(summary["available_salary_rows_flagged_for_source_review"]))
    print("Salary/cap shares over 100% (valid in a soft-cap league): " + str(summary["available_salary_cap_shares_over_100pct"]))
    print("Wrote: " + str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
