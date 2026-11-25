"""Attach next-season skill/efficiency labels and minutes to existing pairs."""

from pathlib import Path
import numpy as np
import pandas as pd
from player_matching import keep_combined_seasons
from profile_targets import NEXT_SEASON_COLUMNS, PCA_TARGETS

ROOT = Path(__file__).resolve().parent


def main():
    processed = ROOT / "data" / "processed"
    pairs = pd.read_csv(processed / "prediction_pairs.csv")
    season_range = (pairs["target_season"].min(), pairs["target_season"].max())
    advanced = pd.read_csv(ROOT / "data" / "raw" / "basketball_reference" / "Advanced.csv")
    advanced = advanced[
        (advanced["lg"] == "NBA")
        & advanced["season"].between(*season_range)
    ].copy()
    advanced = keep_combined_seasons(advanced)
    advanced_fields = [
        "player_id", "season", "ws_48", "bpm", "mp", "g", "ws", "vorp",
        "ts_percent", "usg_percent", "trb_percent", "ast_percent",
        "stl_percent", "blk_percent", "tov_percent", "x3p_ar", "f_tr",
    ]
    advanced_targets = advanced[advanced_fields].rename(
        columns={
            "season": "target_season",
            "ws_48": "next_season_ws_48",
            "bpm": "next_season_bpm",
            "mp": "next_season_minutes",
            "g": "next_season_games",
            "ws": "source_next_ws",
            "vorp": "source_next_vorp",
            **{name: "next_season_" + name for name in PCA_TARGETS
               if name in advanced_fields},
        }
    )
    extended = pairs.merge(
        advanced_targets,
        on=["player_id", "target_season"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if len(extended) != len(pairs) or not extended["_merge"].eq("both").all():
        raise ValueError("Some target-season records did not match Advanced.csv.")

    per_game = pd.read_csv(
        ROOT / "data" / "raw" / "basketball_reference" / "Player Per Game.csv"
    )
    per_game = per_game[
        (per_game["lg"] == "NBA")
        & per_game["season"].between(*season_range)
    ].copy()
    per_game = keep_combined_seasons(per_game)
    per_game_fields = [
        "player_id", "season", "e_fg_percent", "x3p_percent", "ft_percent",
        "mp_per_game",
    ]
    per_game_targets = per_game[per_game_fields].rename(
        columns={
            "season": "target_season",
            "e_fg_percent": "next_season_e_fg_percent",
            "x3p_percent": "next_season_x3p_percent",
            "ft_percent": "next_season_ft_percent",
            "mp_per_game": "next_season_mp_per_game",
        }
    )
    extended = extended.merge(
        per_game_targets,
        on=["player_id", "target_season"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator="per_game_merge",
    )
    if not extended["per_game_merge"].eq("both").all():
        raise ValueError("Some target-season records did not match Player Per Game.csv.")
    extended = extended.drop(columns=["per_game_merge"])

    pd.testing.assert_frame_equal(extended[["player_id", "season"]], pairs[["player_id", "season"]])

    for target in ["ws", "vorp"]:
        np.testing.assert_allclose(extended["source_next_" + target], extended["next_season_" + target], rtol=0, atol=0)

    extended = extended.drop(columns=["_merge", "source_next_ws", "source_next_vorp"])
    extended["rate_valid"] = True

    # Never impute an undefined target or a missing baseline for comparison.
    for column in ["next_season_pie", "next_season_ws_48", "next_season_bpm", "pie", "ws_48", "bpm"]:
        extended["rate_valid"] = extended["rate_valid"] & np.isfinite(extended[column].to_numpy(dtype=float))

    profile_cols = [NEXT_SEASON_COLUMNS[target] for target in PCA_TARGETS]
    extended["skill_profile_valid"] = np.isfinite(
        extended[profile_cols].to_numpy(dtype=float)
    ).all(axis=1)
    extended["skill_profile_valid"] = (
        extended["skill_profile_valid"]
        & extended["next_season_minutes"].ge(0)
    )

    path = processed / "prediction_pairs_rates.csv"
    temporary = path.with_suffix(".csv.tmp")
    extended.to_csv(temporary, index=False)
    temporary.replace(path)
    extended[~extended["rate_valid"]].to_csv(processed / "rate_excluded_rows.csv", index=False)
    print("Original examples retained: " + str(len(extended)))
    print("Examples with defined rate targets and baselines: " + str(int(extended["rate_valid"].sum())))
    print("Undefined rate rows (retained in original experiment): " + str(int((~extended["rate_valid"]).sum())))
    print("Complete next-season skill profiles for PCA: " + str(int(extended["skill_profile_valid"].sum())))
    print("Per-target missing labels:")
    print(extended[[NEXT_SEASON_COLUMNS[target] for target in NEXT_SEASON_COLUMNS]].isna().sum().to_string())
    print("Saved to: " + str(path))


if __name__ == "__main__":
    main()
