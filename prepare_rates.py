"""Add rate-statistic targets and observed target-season minutes to existing pairs."""

from pathlib import Path
import numpy as np
import pandas as pd
from player_matching import keep_combined_seasons

ROOT = Path(__file__).resolve().parent


def main():
    processed = ROOT / "data" / "processed"
    pairs = pd.read_csv(processed / "prediction_pairs.csv")
    source = pd.read_csv(ROOT / "data" / "raw" / "basketball_reference" / "Advanced.csv")
    source = source[(source["lg"] == "NBA") & source["season"].between(pairs["target_season"].min(), pairs["target_season"].max())].copy()
    source = keep_combined_seasons(source)
    columns = ["player_id", "season", "ws_48", "bpm", "mp", "g", "ws", "vorp"]
    targets = source[columns].rename(columns={"season": "target_season", "ws_48": "next_season_ws_48", "bpm": "next_season_bpm", "mp": "next_season_minutes", "g": "next_season_games", "ws": "source_next_ws", "vorp": "source_next_vorp"})
    extended = pairs.merge(targets, on=["player_id", "target_season"], how="left", sort=False, validate="one_to_one", indicator=True)

    if len(extended) != len(pairs) or not extended["_merge"].eq("both").all():
        raise ValueError("Some target-season records did not match the raw statistics.")

    pd.testing.assert_frame_equal(extended[["player_id", "season"]], pairs[["player_id", "season"]])

    for target in ["ws", "vorp"]:
        np.testing.assert_allclose(extended["source_next_" + target], extended["next_season_" + target], rtol=0, atol=0)

    extended = extended.drop(columns=["_merge", "source_next_ws", "source_next_vorp"])
    extended["rate_valid"] = True

    # Never impute an undefined target or a missing baseline for comparison.
    for column in ["next_season_pie", "next_season_ws_48", "next_season_bpm", "pie", "ws_48", "bpm"]:
        extended["rate_valid"] = extended["rate_valid"] & np.isfinite(extended[column].to_numpy(dtype=float))

    path = processed / "prediction_pairs_rates.csv"
    temporary = path.with_suffix(".csv.tmp")
    extended.to_csv(temporary, index=False)
    temporary.replace(path)
    extended[~extended["rate_valid"]].to_csv(processed / "rate_excluded_rows.csv", index=False)
    print("Original examples retained: " + str(len(extended)))
    print("Examples with defined rate targets and baselines: " + str(int(extended["rate_valid"].sum())))
    print("Undefined rate rows (retained in original experiment): " + str(int((~extended["rate_valid"]).sum())))
    print("Saved to: " + str(path))


if __name__ == "__main__":
    main()
