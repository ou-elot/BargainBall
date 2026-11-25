"""Compare unrestricted training with minimum-prior-games training."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS_ROOT = ROOT / "results" / "development" / "multitask_profile"
THRESHOLD = 10


def discover_runs():
    runs = []
    for folder in RESULTS_ROOT.iterdir():
        report_path = folder / "development_report.json"
        metadata_path = ROOT / "models" / "numpy_multitask_profile" / folder.name / "metadata.json"
        if not report_path.exists() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        runs.append((int(metadata.get("training_minimum_prior_games", 0)), folder))
    unrestricted = [folder for games, folder in runs if games == 0]
    filtered = [folder for games, folder in runs if games == THRESHOLD]
    if not unrestricted or not filtered:
        raise FileNotFoundError(
            "Need both an unrestricted run and a filtered run with "
            f"training_minimum_prior_games={THRESHOLD}."
        )
    return unrestricted[-1], filtered[-1]


def main():
    unrestricted_dir, filtered_dir = discover_runs()
    dev = pd.read_csv(PROCESSED / "dev_rows_rates.csv")
    groups = {
        "all_development": np.ones(len(dev), dtype=bool),
        f"prior_games_gte_{THRESHOLD}": dev["g"].ge(THRESHOLD).to_numpy(),
        f"prior_games_lt_{THRESHOLD}": dev["g"].lt(THRESHOLD).to_numpy(),
    }
    results = []
    target_names = None
    for model_name, folder in [
        ("unrestricted_training", unrestricted_dir),
        (f"training_prior_games_gte_{THRESHOLD}", filtered_dir),
    ]:
        predictions = pd.read_csv(folder / "development_profile_predictions.csv")
        keys = ["player_id", "season", "target_season"]
        if len(predictions) != len(dev) or not predictions[keys].equals(dev[keys]):
            raise ValueError(f"Development rows do not align for {model_name}.")
        if target_names is None:
            target_names = [
                col.removeprefix("actual_")
                for col in predictions.columns
                if col.startswith("actual_")
                and "predicted_" + col.removeprefix("actual_") in predictions
                and "prior_" + col.removeprefix("actual_") in predictions
            ]
        for cohort, cohort_mask in groups.items():
            for target in target_names:
                actual = predictions["actual_" + target].to_numpy(dtype=float)
                forecast = predictions["predicted_" + target].to_numpy(dtype=float)
                persistence = predictions["prior_" + target].to_numpy(dtype=float)
                valid = (
                    cohort_mask
                    & np.isfinite(actual)
                    & np.isfinite(forecast)
                    & np.isfinite(persistence)
                )
                if not valid.any():
                    continue
                model_error = forecast[valid] - actual[valid]
                persistence_error = persistence[valid] - actual[valid]
                results.append(
                    {
                        "model": model_name,
                        "cohort": cohort,
                        "target": target,
                        "rows": int(valid.sum()),
                        "model_rmse": float(np.sqrt(np.mean(model_error**2))),
                        "model_mae": float(np.mean(np.abs(model_error))),
                        "persistence_rmse": float(
                            np.sqrt(np.mean(persistence_error**2))
                        ),
                        "persistence_mae": float(np.mean(np.abs(persistence_error))),
                    }
                )

    comparison = pd.DataFrame(results)
    report_dir = filtered_dir
    comparison.to_csv(report_dir / "minimum_games_sensitivity_comparison.csv", index=False)
    summary = {
        "training_comparison": {
            "unrestricted_run": str(unrestricted_dir.relative_to(ROOT)),
            "filtered_run": str(filtered_dir.relative_to(ROOT)),
            "filtered_training_minimum_prior_games": THRESHOLD,
        },
        "development_cohorts": {
            name: int(mask.sum()) for name, mask in groups.items()
        },
        "test_split_loaded": False,
        "development_split_used_for_model_selection": True,
        "output": "minimum_games_sensitivity_comparison.csv",
    }
    (report_dir / "minimum_games_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    key = comparison[
        comparison["target"].isin(["pie", "ws_48", "bpm", "minutes"])
    ]
    print("Unrestricted run:", unrestricted_dir)
    print("Filtered run:", filtered_dir)
    print("\nKey outcomes by development eligibility cohort:")
    print(key.to_string(index=False))
    print("\nFull per-target comparison:", report_dir / "minimum_games_sensitivity_comparison.csv")
    print("Test split not loaded.")


if __name__ == "__main__":
    main()
