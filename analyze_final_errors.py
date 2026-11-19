"""Describe errors in saved final test predictions without changing the models.

Run: python analyze_final_errors.py --results results/test/numpy_final/BUNDLE_ID
Minute groups are descriptive, not training filters or model-selection rules.
"""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from evaluate_final_models import calculate_metrics

ROOT = Path(__file__).resolve().parent
TARGETS = ["pie", "ws_48", "bpm", "pca_score"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    predictions_path = args.results / "test_predictions.csv"
    table = pd.read_csv(predictions_path)
    targets = TARGETS.copy()

    if("predicted_minutes" in table.columns):
        targets.insert(-1, "minutes")

    metadata = pd.read_csv(ROOT / "data" / "processed" / "test_rows_rates.csv")
    keys = ["player_id", "season", "target_season"]
    table = table.merge(metadata[keys + ["g", "mp", "next_season_games", "next_season_minutes"]], on=keys, how="left", validate="one_to_one")

    if(table["next_season_minutes"].isna().any()):
        raise ValueError("Saved predictions must match the test minute metadata.")

    group_rows = []
    concentration = []

    for target in targets:
        actual = table["actual_" + target].to_numpy()
        predicted = table["predicted_" + target].to_numpy()
        baseline = table["baseline_" + target].to_numpy()
        valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(baseline)
        model_squared = (predicted - actual) ** 2
        baseline_squared = (baseline - actual) ** 2
        error_rows = table.copy()
        error_rows["squared_error"] = model_squared
        error_rows = error_rows.sort_values("squared_error", ascending=False)
        error_rows.head(20).to_csv(args.results / (target + "_largest_errors.csv"), index=False)

        for label, squared in [("model", model_squared), ("baseline", baseline_squared)]:
            index = int(np.argmax(np.where(valid, squared, -np.inf)))
            total = float(np.sum(squared[valid]))
            share = 0.0

            if(total > 0):
                share = float(squared[index] / total)

            row = table.iloc[index]
            concentration.append({"target": target, "predictor": label, "largest_error_player": row["player"], "input_season": int(row["season"]), "target_season": int(row["target_season"]), "prior_minutes": float(row["mp"]), "target_minutes": float(row["next_season_minutes"]), "actual": float(actual[index]), "prediction": float(predicted[index]) if label == "model" else float(baseline[index]), "largest_row_share_of_squared_error": share})

        for label, low, high in [("under_250", 0, 250), ("250_to_999", 250, 1000), ("1000_or_more", 1000, float("inf"))]:
            mask = valid & table["next_season_minutes"].ge(low).to_numpy() & table["next_season_minutes"].lt(high).to_numpy()

            if(mask.any()):
                row = calculate_metrics(actual[mask], predicted[mask], baseline[mask], "test", target)
                row["observed_target_minutes_group"] = label
                group_rows.append(row)

    pd.DataFrame(group_rows).to_csv(args.results / "errors_by_minutes.csv", index=False)
    pd.DataFrame(concentration).to_csv(args.results / "squared_error_concentration.csv", index=False)
    lines = ["# Descriptive test-error diagnostics", "", "These diagnostics use the saved frozen-model test predictions. All rows remain", "in the headline evaluation. No model, PCA weight, or inclusion rule was changed.", "The minute groups describe observed next-season appearances; they are not", "predictors or selection criteria.", "", "## Largest squared-error contributors", "", "| Target | Predictor | Player | Target season ending | Prior minutes | Target minutes | Share of total squared error |", "|---|---|---|---:|---:|---:|---:|"]

    for row in concentration:
        lines.append("| " + row["target"].upper() + " | " + row["predictor"] + " | " + row["largest_error_player"] + " | " + str(row["target_season"]) + " | " + str(row["prior_minutes"]) + " | " + str(row["target_minutes"]) + " | " + str(round(100.0 * row["largest_row_share_of_squared_error"], 2)) + "% |")

    lines.extend(["", "## Interpretation", "", "- Mindaugas Kuzminskas's target season ending in 2018 contains only two minutes and a source-listed PIE of -4.0. This extreme target dominates PIE squared error and also affects the PCA target.", "- Naz Mitrou-Long's prior season ending in 2018 contains one minute and source-listed BPM of 242.2. Carrying this value forward creates an extreme baseline error.", "- These records illustrate the instability of rate statistics for very short appearances. Their presence does not by itself prove a source error.", "- RMSE gives extreme errors high weight. Read MAE alongside RMSE; large baseline RMSE reductions are not equivalent to equally large improvements for a typical player.", "- Minute-group metrics are in `errors_by_minutes.csv`; full-data metrics remain in `metrics.csv`.", ""])
    (args.results / "ERROR_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    audit = {"saved_test_predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(), "test_rows": len(table), "diagnostics_only": True, "models_changed": False, "rows_removed_from_headline_metrics": 0}
    (args.results / "diagnostic_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print ("Saved descriptive error diagnostics to: " + str(args.results))

    for row in concentration:
        print (row["target"].upper() + " | " + row["predictor"] + " | largest squared-error share " + str(round(100.0 * row["largest_row_share_of_squared_error"], 2)) + "% | " + row["largest_error_player"])


if(__name__ == "__main__"):
    main()
