"""Cross-check flagged salary rows against independent downloadable datasets.

Run after prepare_data.py and analyze_salary_cap.py. Only exact matches from
another dataset are promoted to verified; conflicts and unmatched rows remain
flagged for manual source review.
"""

from pathlib import Path
import pandas as pd

from player_matching import name_key

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "salaries"
PROCESSED = ROOT / "data" / "processed"
BBREF_URL = "https://github.com/ramvibhakar/basketball-analytics/blob/master/data/player_salary.csv"
ALT_URL = "https://github.com/erikgregorywebb/datasets/blob/master/nba-salaries.csv"
HIST_URL = "https://github.com/logan-lauton/NBA-Player-HUD/blob/main/NBA%20Salaries(1990-2023).csv"


def dollars(series):
    return pd.to_numeric(
        series.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce",
    )


def main():
    pairs = pd.read_csv(PROCESSED / "salary_cap_normalized_pairs.csv")
    flagged = pairs[pairs["salary_available_check"] & pairs["salary_needs_source_review"]].copy()
    flagged["name_key"] = flagged["player"].map(name_key)

    bbref = pd.read_csv(RAW / "bbref_historical_salaries.csv")
    bbref["player_id"] = bbref["url"].str.extract(r"/players/[a-z]/([^/.]+)")
    bbref["target_season"] = bbref["season"].str[:4].astype(int) + 1
    bbref["bbref_amount"] = dollars(bbref["salary"])
    bbref = bbref.groupby(["player_id", "target_season"], as_index=False)["bbref_amount"].sum()

    alt = pd.read_csv(RAW / "alternative_salaries.csv")
    alt["name_key"] = alt["name"].map(name_key)
    alt["target_season"] = pd.to_numeric(alt["season"], errors="raise").astype(int)
    alt["alt_amount"] = pd.to_numeric(alt["salary"], errors="coerce")
    alt = alt.groupby(["name_key", "target_season"], as_index=False)["alt_amount"].sum()

    # Resolve names through the project crosswalk, and reject source name-season
    # collisions instead of summing salaries that may belong to different players.
    names = pairs[["player_id", "target_season", "name_key"]].drop_duplicates()
    names = names[~names.duplicated(["name_key", "target_season"], keep=False)]
    historical = pd.read_csv(RAW / "historical_salaries.csv").drop_duplicates()
    historical["name_key"] = historical["playerName"].map(name_key)
    historical["target_season"] = historical["seasonStartYear"].astype(int) + 1
    historical["hist_amount"] = dollars(historical["salary"])
    historical = historical.merge(names, on=["name_key", "target_season"], how="inner")
    historical = historical.groupby(["player_id", "target_season"], as_index=False).agg(
        hist_amount=("hist_amount", "first"),
        hist_source_rows=("hist_amount", "size"),
    )
    historical.loc[historical["hist_source_rows"].ne(1), "hist_amount"] = pd.NA

    audit = flagged.merge(bbref, on=["player_id", "target_season"], how="left")
    audit = audit.merge(alt, on=["name_key", "target_season"], how="left")
    audit = audit.merge(historical, on=["player_id", "target_season"], how="left")

    # Don't count the originating dataset as corroboration.
    audit["bbref_exact"] = audit["bbref_amount"].notna() & (audit["target_salary"] - audit["bbref_amount"]).abs().le(1)
    audit["alt_exact"] = audit["alt_amount"].notna() & (audit["target_salary"] - audit["alt_amount"]).abs().le(1)
    audit["hist_exact"] = audit["hist_amount"].notna() & (audit["target_salary"] - audit["hist_amount"]).abs().le(1)
    audit["independent_exact_sources"] = ""
    audit["verification_source_urls"] = ""

    for idx, row in audit.iterrows():
        exact_sources = []
        urls = []
        method = row["salary_method"]
        if method not in ("historical_bbref_team_sum",) and row["bbref_exact"]:
            exact_sources.append("Basketball-Reference-ID archive")
            urls.append(BBREF_URL)
        if method != "alternative_listed_salary" and row["alt_exact"]:
            exact_sources.append("alternative salary archive")
            urls.append(ALT_URL)
        if method != "historical_listed_salary" and row["hist_exact"]:
            exact_sources.append("historical listed-salary archive")
            urls.append(HIST_URL)
        audit.at[idx, "independent_exact_sources"] = "; ".join(exact_sources)
        audit.at[idx, "verification_source_urls"] = "; ".join(urls)

    audit["review_outcome"] = "unmatched"
    audit.loc[audit["bbref_amount"].notna() | audit["alt_amount"].notna() | audit["hist_amount"].notna(), "review_outcome"] = "source_difference_or_non-independent_match"
    audit.loc[audit["independent_exact_sources"].ne(""), "review_outcome"] = "verified_exact_crossmatch"
    columns = [
        "player", "player_id", "target_season", "target_salary", "salary_method",
        "salary_source_url", "bbref_amount", "bbref_exact", "alt_amount", "alt_exact",
        "hist_amount", "hist_exact", "independent_exact_sources",
        "verification_source_urls", "review_outcome",
    ]
    audit[columns].to_csv(PROCESSED / "salary_source_review_audit.csv", index=False)

    verified = audit[audit["review_outcome"].eq("verified_exact_crossmatch")].copy()
    overrides = verified[["player_id", "target_season", "target_salary", "verification_source_urls"]].copy()
    overrides = overrides.rename(columns={
        "verification_source_urls": "salary_source_url",
    })
    overrides["salary_needs_source_review"] = False
    overrides["salary_method"] = "verified_external_crossmatch"
    overrides_path = RAW / "verified_salary_overrides.csv"
    if overrides_path.exists():
        previous = pd.read_csv(overrides_path)
        overrides = pd.concat([previous, overrides], ignore_index=True)
        overrides = overrides.drop_duplicates(["player_id", "target_season"], keep="last")
    overrides.to_csv(overrides_path, index=False)

    print("Flagged rows audited: " + str(len(audit)))
    print("Exact independent crossmatches: " + str(len(verified)))
    print("Still unresolved/conflicting: " + str(len(audit) - len(verified)))
    print(audit["review_outcome"].value_counts().to_string())
    print("Audit: " + str(PROCESSED / "salary_source_review_audit.csv"))


if __name__ == "__main__":
    main()
