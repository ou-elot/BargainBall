"""Build consecutive regular-season prediction pairs, keeping missing salaries visible."""

from pathlib import Path
import json
import pandas as pd
from player_matching import keep_combined_seasons, name_key, explicit_ids

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIRST_SEASON = 1997
LAST_SEASON = 2026


def load_statistics():
    advanced = pd.read_csv(RAW / "basketball_reference" / "Advanced.csv")
    traditional = pd.read_csv(RAW / "basketball_reference" / "Player Per Game.csv")
    advanced = advanced[(advanced["lg"] == "NBA") & advanced["season"].between(FIRST_SEASON, LAST_SEASON)].copy()
    traditional = traditional[(traditional["lg"] == "NBA") & traditional["season"].between(FIRST_SEASON, LAST_SEASON)].copy()
    advanced = keep_combined_seasons(advanced)
    traditional = keep_combined_seasons(traditional)
    columns = ["player_id", "season"]

    for column in traditional.columns:
        if column not in advanced.columns:
            columns.append(column)

    stats = advanced.merge(traditional[columns], on=["player_id", "season"], how="outer", validate="one_to_one", indicator=True)

    if not stats["_merge"].eq("both").all():
        raise ValueError("Traditional and advanced records do not match.")

    stats = stats.drop(columns="_merge")
    frames = []
    nba_columns = ["player_id", "player_name", "season", "age", "gp", "pie", "off_rating", "def_rating", "net_rating", "pace"]

    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        path = RAW / "sportsdataverse" / ("player_season_stats_" + str(year) + ".parquet")
        season = pd.read_parquet(path)
        season = season[(season["season_type"] == "regular-season") & (season["measure_type"] == "advanced") & (season["per_mode"] == "pergame")].copy()

        if len(season) == 0 or not season["season"].eq(year).all():
            raise ValueError("Missing or mislabeled NBA season: " + str(year))

        frames.append(season[nba_columns])

    nba = pd.concat(frames, ignore_index=True)
    nba = nba.rename(columns={"player_id": "nba_player_id", "player_name": "nba_player_name", "age": "nba_age", "gp": "nba_games"})
    nba["nba_player_id"] = nba["nba_player_id"].astype("string")

    if nba.duplicated(["nba_player_id", "season"]).any():
        raise ValueError("Duplicate NBA player-seasons after filtering measures and modes.")

    stats["name_key"] = stats["player"].map(name_key)
    nba["name_key"] = nba["nba_player_name"].map(name_key)
    return stats, nba


def match_players(stats, nba):
    keys = ["name_key", "season"]
    stats_unique = stats[~stats.duplicated(keys, keep=False)].copy()
    nba_unique = nba[~nba.duplicated(keys, keep=False)].copy()
    named = stats_unique.merge(nba_unique[keys + ["nba_player_id"]], on=keys, how="inner", validate="one_to_one")
    crosswalk = named[["player_id", "nba_player_id"]].drop_duplicates()
    manual = []

    for bbref_id, nba_id in explicit_ids.items():
        manual.append({"player_id": bbref_id, "nba_player_id": nba_id})

    crosswalk = pd.concat([crosswalk, pd.DataFrame(manual)], ignore_index=True).drop_duplicates()

    if crosswalk["player_id"].duplicated().any() or crosswalk["nba_player_id"].duplicated().any():
        raise ValueError("Inconsistent player ID mappings need review.")

    matched = stats.merge(crosswalk, on="player_id", how="left", validate="many_to_one")
    matched = matched.merge(nba.drop(columns="name_key"), on=["nba_player_id", "season"], how="left", validate="one_to_one", indicator=True)
    unmatched = matched[matched["_merge"] != "both"].copy()
    unmatched.to_csv(PROCESSED / "unmatched_players.csv", index=False)

    if len(unmatched) > 0:
        print(unmatched[["player", "player_id", "season"]].to_string(index=False))
        raise ValueError("Some players require an alias or explicit ID mapping.")

    matched = matched.drop(columns="_merge")

    if len(matched) != len(nba) or matched.duplicated(["nba_player_id", "season"]).any():
        raise ValueError("The two sources do not cover the same unique player-seasons.")

    matched["age_difference"] = (matched["age"] - matched["nba_age"]).abs()
    review = matched[(matched["age_difference"] > 1) | matched["age_difference"].isna() | matched["g"].ne(matched["nba_games"])].copy()
    review.to_csv(PROCESSED / "player_source_review.csv", index=False)
    crosswalk.to_csv(PROCESSED / "player_id_crosswalk.csv", index=False)
    return matched, review


def build_pairs(matched):
    targets = matched[["nba_player_id", "season", "pie", "ws", "vorp"]].rename(columns={"season": "target_season", "pie": "next_season_pie", "ws": "next_season_ws", "vorp": "next_season_vorp"})
    targets["season"] = targets["target_season"] - 1
    pairs = matched.merge(targets, on=["nba_player_id", "season"], how="inner", validate="one_to_one")
    missing = pairs["pie"].isna() | pairs["next_season_pie"].isna()
    pairs[missing].to_csv(PROCESSED / "pairs_missing_pie.csv", index=False)
    pairs = pairs[~missing].copy()
    pairs["target_season"] = pairs["target_season"].astype(int)
    return pairs


def attach_salaries(pairs, matched):
    # Preserve all previously reviewed salary amounts as the first choice.
    salaries = pd.read_csv(RAW / "salaries" / "BargainBall_listed_salaries.csv", dtype={"player_id": "string"})
    columns = ["player_id", "target_season", "target_salary", "source_url", "salary_needs_source_review"]
    salaries = salaries[columns].rename(columns={"source_url": "salary_source_url"})
    salaries["salary_method"] = "previous_listed_lookup"
    frames = []

    for year in range(2010, LAST_SEASON + 1):
        path = RAW / "bbref_salaries" / ("bbref_salaries_" + str(year) + ".parquet")
        frame = pd.read_parquet(path)

        if not frame["season"].eq(year).all() or frame.duplicated(["season", "team_id", "bbref_id"]).any():
            raise ValueError("Unexpected salary archive records: " + str(year))

        frames.append(frame)

    team_salaries = pd.concat(frames, ignore_index=True)
    team_salaries.to_csv(PROCESSED / "salary_team_records.csv", index=False)
    archive_rows = []

    for key, group in team_salaries.groupby(["bbref_id", "season"]):
        if group["salary"].isna().any():
            continue

        amount = group["salary"].sum()

        if amount > 0:
            archive_rows.append({"player_id": key[0], "target_season": int(key[1]), "target_salary": amount, "salary_source_url": "https://basketball-data.billmill.org/nba/bbref/bbref_salaries/season=" + str(key[1]) + "/data.parquet", "salary_needs_source_review": False, "salary_method": "bbref_team_salary_sum"})

    archive = pd.DataFrame(archive_rows)
    salaries = pd.concat([salaries, archive], ignore_index=True).drop_duplicates(["player_id", "target_season"], keep="first")

    # Historical salaries label seasons by their STARTING year.
    historical = pd.read_csv(RAW / "salaries" / "historical_salaries.csv").drop_duplicates()
    historical["target_season"] = historical["seasonStartYear"].astype(int) + 1
    historical["name_key"] = historical["playerName"].map(name_key)
    historical["target_salary"] = pd.to_numeric(historical["salary"].str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="raise")
    names = matched[["player_id", "season", "name_key"]].rename(columns={"season": "target_season"})
    nba_names = matched[["player_id", "season", "nba_player_name"]].rename(columns={"season": "target_season"})
    nba_names["name_key"] = nba_names["nba_player_name"].map(name_key)
    names = pd.concat([names, nba_names.drop(columns="nba_player_name")], ignore_index=True).drop_duplicates()
    names = names[~names.duplicated(["name_key", "target_season"], keep=False)]
    historical = historical.merge(names, on=["name_key", "target_season"], how="inner", validate="many_to_one")
    historical = historical[["player_id", "target_season", "target_salary"]].drop_duplicates()
    ambiguous = historical.duplicated(["player_id", "target_season"], keep=False)
    historical[ambiguous].to_csv(PROCESSED / "historical_salary_conflicts.csv", index=False)
    historical = historical[~ambiguous & historical["target_salary"].gt(0)].copy()
    historical["salary_source_url"] = "https://github.com/logan-lauton/NBA-Player-HUD/blob/main/NBA%20Salaries(1990-2023).csv"
    historical["salary_needs_source_review"] = True
    historical["salary_method"] = "historical_listed_salary"
    salaries = pd.concat([salaries, historical], ignore_index=True).drop_duplicates(["player_id", "target_season"], keep="first")

    # Another archive contains Basketball-Reference IDs, avoiding name guesses.
    older = pd.read_csv(RAW / "salaries" / "bbref_historical_salaries.csv")
    older["player_id"] = older["url"].str.extract(r"/players/[a-z]/([^/.]+)")
    older["target_season"] = older["season"].str[:4].astype(int) + 1
    older["amount"] = pd.to_numeric(older["salary"].str.replace("$", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    older = older.drop_duplicates(["team_url", "season", "player_id", "amount"])
    older_rows = []

    for key, group in older.groupby(["player_id", "target_season"]):
        if group["amount"].isna().any() or group["team_url"].duplicated().any():
            continue

        amount = group["amount"].sum()

        if amount > 0:
            older_rows.append({"player_id": key[0], "target_season": int(key[1]), "target_salary": amount, "salary_source_url": "https://github.com/ramvibhakar/basketball-analytics/blob/master/data/player_salary.csv", "salary_needs_source_review": True, "salary_method": "historical_bbref_team_sum"})

    salaries = pd.concat([salaries, pd.DataFrame(older_rows)], ignore_index=True).drop_duplicates(["player_id", "target_season"], keep="first")

    # This archive's season is the ENDING year (e.g. Shaq's 1999-2000 salary).
    alternative = pd.read_csv(RAW / "salaries" / "alternative_salaries.csv")
    alternative["name_key"] = alternative["name"].map(name_key)
    alternative = alternative.rename(columns={"season": "target_season", "salary": "target_salary"})
    alternative = alternative.merge(names, on=["name_key", "target_season"], how="inner", validate="many_to_one")
    alternative = alternative[["player_id", "target_season", "target_salary"]].drop_duplicates()
    alternative = alternative[~alternative.duplicated(["player_id", "target_season"], keep=False) & alternative["target_salary"].gt(0)].copy()
    alternative["salary_source_url"] = "https://github.com/erikgregorywebb/datasets/blob/master/nba-salaries.csv"
    alternative["salary_needs_source_review"] = True
    alternative["salary_method"] = "alternative_listed_salary"
    salaries = pd.concat([salaries, alternative], ignore_index=True).drop_duplicates(["player_id", "target_season"], keep="first")

    # HoopsHype snapshot mirrored by MrBridge; retain its published amounts.
    snapshot = pd.read_csv(RAW / "salaries" / "hoopshype_snapshot.csv").drop_duplicates()
    salary_aliases = {"Patrick Baldwin": "Patrick Baldwin Jr.", "Wendell Moore": "Wendell Moore Jr.", "TyTy Washington": "TyTy Washington Jr.", "Kevin McCullar": "Kevin McCullar Jr."}
    snapshot["name_key"] = snapshot["player"].replace(salary_aliases).map(name_key)
    snapshot["target_season"] = snapshot["season_year"].astype(int) + 1
    snapshot = snapshot[snapshot["target_season"].between(FIRST_SEASON + 1, LAST_SEASON)].copy()
    snapshot = snapshot.merge(names, on=["name_key", "target_season"], how="inner", validate="many_to_one", suffixes=("_hoopshype", ""))
    snapshot.to_csv(PROCESSED / "hoopshype_salary_records.csv", index=False)
    snapshot_rows = []

    for key, group in snapshot.groupby(["player_id", "target_season"]):
        group = group.drop_duplicates(["team_id", "salary"])

        if group["team_id"].duplicated().any() or group["salary"].isna().any() or not group["salary"].gt(0).all():
            continue

        snapshot_rows.append({"player_id": key[0], "target_season": int(key[1]), "target_salary": group["salary"].sum(), "salary_source_url": "https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026/blob/main/player_salaries.csv", "salary_needs_source_review": True, "salary_method": "hoopshype_snapshot_team_sum"})

    salaries = pd.concat([salaries, pd.DataFrame(snapshot_rows)], ignore_index=True).drop_duplicates(["player_id", "target_season"], keep="first")

    # Resolve the same-name collision in the NBA-Player-HUD historical file:
    # smithch04 is the Clippers player, while the $4,375,143 amount belongs
    # to the separate Spurs player with the same name.
    verified_corrections = pd.DataFrame([
        {
            "player_id": "smithch04",
            "target_season": 1999,
            "target_salary": 677080,
            "salary_source_url": "https://www.hoopshype.com/salaries/players/charles-smith/3176",
            "salary_needs_source_review": False,
            "salary_method": "verified_name_collision_correction",
        },
        {
            "player_id": "bryanma01",
            "target_season": 2001,
            "target_salary": 3029268,
            "salary_source_url": "https://www.hoopshype.com/salaries/players/mark-bryant/42",
            "salary_needs_source_review": False,
            "salary_method": "verified_source_confirmation",
        },
        {
            "player_id": "richaja01",
            "target_season": 2009,
            "target_salary": 12664335,
            "salary_source_url": "https://www.hoopshype.com/salaries/players/jason-richardson/3496",
            "salary_needs_source_review": False,
            "salary_method": "verified_source_confirmation",
        },
    ])
    external_verifications_path = RAW / "salaries" / "verified_salary_overrides.csv"
    if external_verifications_path.exists():
        external_verifications = pd.read_csv(external_verifications_path)
        verified_corrections = pd.concat([verified_corrections, external_verifications], ignore_index=True)
    salaries = pd.concat([salaries, verified_corrections], ignore_index=True)
    salaries = salaries.drop_duplicates(["player_id", "target_season"], keep="last")
    pd.DataFrame([{
        "player_id": "smithch04",
        "player": "Charles Smith",
        "target_season": 1999,
        "previous_salary": 5052223,
        "corrected_salary": 677080,
        "reason": "The historical name match combined the Clippers player's salary with a different Charles Smith's Spurs salary.",
        "verification_sources": "https://www.hoopshype.com/salaries/players/charles-smith/3176; https://www.basketball-reference.com/players/s/smithch04.html",
    }, {
        "player_id": "bryanma01",
        "player": "Mark Bryant",
        "target_season": 2001,
        "previous_salary": 3029268,
        "corrected_salary": 3029268,
        "reason": "HoopsHype shows $2,200,000 with Cleveland and $829,268 with Dallas; their sum confirms the listed season salary.",
        "verification_sources": "https://www.hoopshype.com/salaries/players/mark-bryant/42; https://www.basketball-reference.com/players/b/bryanma01.html",
    }, {
        "player_id": "richaja01",
        "player": "Jason Richardson",
        "target_season": 2009,
        "previous_salary": 12664335,
        "corrected_salary": 12664335,
        "reason": "HoopsHype shows $12,222,221 with Phoenix and $442,114 with Miami; their sum confirms the listed season salary.",
        "verification_sources": "https://www.hoopshype.com/salaries/players/jason-richardson/3496; https://www.basketball-reference.com/players/r/richaja01.html",
    }]).to_csv(PROCESSED / "verified_salary_corrections.csv", index=False)
    salaries.to_csv(PROCESSED / "salary_lookup.csv", index=False)
    pairs = pairs.merge(salaries, on=["player_id", "target_season"], how="left", validate="one_to_one")
    pairs["salary_available"] = pairs["target_salary"].notna() & pairs["target_salary"].gt(0)
    pairs["salary_needs_source_review"] = pairs["salary_needs_source_review"].astype("boolean").fillna(True)
    return pairs


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    stats, nba = load_statistics()
    matched, review = match_players(stats, nba)
    pairs = attach_salaries(build_pairs(matched), matched)
    pairs = pairs.sort_values(["player_id", "season"]).reset_index(drop=True)

    if pairs.duplicated(["player_id", "season"]).any() or not pairs["target_season"].eq(pairs["season"] + 1).all():
        raise ValueError("Invalid or duplicate consecutive-season pairs.")

    # Replace the main dataset only after the entire CSV has been written.
    temporary = PROCESSED / "prediction_pairs.csv.part"
    pairs.to_csv(temporary, index=False)
    temporary.replace(PROCESSED / "prediction_pairs.csv")
    pairs[~pairs["salary_available"]].to_csv(PROCESSED / "missing_salaries.csv", index=False)
    pairs[pairs["salary_available"]].to_csv(PROCESSED / "salary_analysis_pairs.csv", index=False)
    coverage = pairs.groupby("target_season").agg(prediction_pairs=("player_id", "size"), pairs_with_salary=("salary_available", "sum"))
    coverage["missing_salary"] = coverage["prediction_pairs"] - coverage["pairs_with_salary"]
    coverage.to_csv(PROCESSED / "coverage_by_season.csv")
    summary = {"first_input_season": int(pairs["season"].min()), "last_input_season": int(pairs["season"].max()), "first_target_season": int(pairs["target_season"].min()), "last_target_season": int(pairs["target_season"].max()), "matched_player_seasons": len(matched), "prediction_pairs": len(pairs), "pairs_with_salary": int(pairs["salary_available"].sum()), "missing_salary": int((~pairs["salary_available"]).sum()), "salaries_needing_review": int(pairs["salary_needs_source_review"].sum()), "source_discrepancies_retained": len(review)}
    summary["salaries_needing_review"] = int((pairs["salary_needs_source_review"] & pairs["salary_available"]).sum())
    (PROCESSED / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Matched player-seasons: " + str(len(matched)))
    print("Prediction pairs: " + str(len(pairs)))
    print("Pairs with salary: " + str(summary["pairs_with_salary"]))
    print("Pairs missing salary (retained for performance training): " + str(summary["missing_salary"]))
    print("Saved dataset to: " + str(PROCESSED / "prediction_pairs.csv"))


if __name__ == "__main__":
    main()
