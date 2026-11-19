"""Compare original and rate composites on the same development rows."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from train import create_model
from pca_rates import combine

ROOT = Path(__file__).resolve().parent


def correlation(x, y, method):
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))

    if x.nunique() < 2 or y.nunique() < 2:
        return None

    return float(x.corr(y, method=method))


def main():
    torch.set_num_threads(1)
    processed = ROOT / "data" / "processed"
    output = ROOT / "results" / "development" / "rates"
    output.mkdir(parents=True, exist_ok=True)
    rate_path = processed / "model_data_rates.npz"
    rate_hash = hashlib.sha256(rate_path.read_bytes()).hexdigest()
    rate_parameters = json.loads((ROOT / "models" / "rates" / "pca" / "pca_parameters.json").read_text())
    original_parameters = json.loads((ROOT / "models" / "pca" / "pca_parameters.json").read_text())

    if rate_parameters["model_data_sha256"] != rate_hash:
        raise ValueError("Run pca_rates.py again after changing the rate arrays.")

    with np.load(rate_path, allow_pickle=False) as data:
        mask = data["rate_mask_dev"].astype(bool)
        inputs = torch.from_numpy(data["X_dev"][mask].astype(np.float32))
        base_hash = str(data["base_model_data_sha256"].item())
        feature_names = data["feature_names"].tolist()

        if original_parameters["model_data_sha256"] != base_hash:
            raise ValueError("Run the original pca_score.py before comparing composites.")

        definitions = {"original": ["pie", "ws", "vorp"], "rates": ["pie", "ws_48", "bpm"]}
        scores = pd.read_csv(processed / "dev_rows_rates.csv").loc[mask, ["player", "player_id", "season", "target_season", "mp", "next_season_minutes"]].reset_index(drop=True)
        metrics = []
        minute_groups = []

        for name, targets in definitions.items():
            parameters = original_parameters

            if name == "rates":
                parameters = rate_parameters

            if parameters["target_order"] != targets:
                raise ValueError("PCA target order does not match.")

            actual_columns = []
            predicted_columns = []
            baseline_columns = []

            for target in targets:
                path = ROOT / "models" / target / "best_model.pt"
                expected_hash = base_hash

                if target in ["ws_48", "bpm"]:
                    path = ROOT / "models" / "rates" / target / "best_model.pt"
                    expected_hash = rate_hash

                checkpoint = torch.load(path, map_location="cpu", weights_only=True)

                if checkpoint["target"] != target or checkpoint["model_data_sha256"] != expected_hash or checkpoint["feature_names"] != feature_names:
                    raise ValueError("Checkpoint mismatch for " + target)

                model = create_model(checkpoint["input_size"], checkpoint["dropout"])
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                with torch.no_grad():
                    predicted = model(inputs).squeeze(1) * checkpoint["target_scale"] + checkpoint["target_mean"]

                predicted_columns.append(predicted.numpy().astype(float))
                actual_columns.append(data["y_" + target + "_dev"][mask].astype(float))
                baseline_columns.append(data["baseline_" + target + "_dev"][mask].astype(float))

            actual_score = combine(np.column_stack(actual_columns), parameters)
            predicted_score = combine(np.column_stack(predicted_columns), parameters)
            baseline_score = combine(np.column_stack(baseline_columns), parameters)
            errors = predicted_score - actual_score
            baseline_errors = baseline_score - actual_score
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            baseline_rmse = float(np.sqrt(np.mean(baseline_errors ** 2)))
            improvement = None

            if baseline_rmse > 0:
                improvement = 100.0 * (1.0 - rmse / baseline_rmse)

            row = {"composite": name, "examples": len(scores), "model_rmse": rmse, "baseline_rmse": baseline_rmse, "model_mae": float(np.mean(np.abs(errors))), "baseline_mae": float(np.mean(np.abs(baseline_errors))), "rmse_improvement_percent": improvement, "variance_captured": parameters["explained_variance_ratio"]}

            for method in ["pearson", "spearman"]:
                row["predicted_score_vs_next_minutes_" + method] = correlation(predicted_score, scores["next_season_minutes"], method)
                row["actual_score_vs_next_minutes_" + method] = correlation(actual_score, scores["next_season_minutes"], method)
                row["predicted_score_vs_prior_minutes_" + method] = correlation(predicted_score, scores["mp"], method)

            metrics.append(row)
            scores[name + "_actual_score"] = actual_score
            scores[name + "_predicted_score"] = predicted_score
            scores[name + "_error"] = errors

            # Descriptive groups only. Next-season minutes are not model inputs.
            for label, low, high in [("under_250", 0, 250), ("250_to_999", 250, 1000), ("1000_or_more", 1000, float("inf"))]:
                selected = scores["next_season_minutes"].ge(low) & scores["next_season_minutes"].lt(high)

                if selected.any():
                    minute_groups.append({"composite": name, "target_minutes_group": label, "examples": int(selected.sum()), "model_rmse": float(np.sqrt(np.mean(errors[selected] ** 2))), "baseline_rmse": float(np.sqrt(np.mean(baseline_errors[selected] ** 2))), "model_mae": float(np.mean(np.abs(errors[selected])))})

            print(name + " | RMSE " + str(round(rmse, 6)) + " | own-baseline improvement (%) " + str(round(improvement, 2)))
            print("  Predicted score vs next-season minutes, Spearman: " + str(round(row["predicted_score_vs_next_minutes_spearman"], 4)))

    for filename, table in [("composite_comparison.csv", pd.DataFrame(metrics)), ("errors_by_minutes.csv", pd.DataFrame(minute_groups)), ("matched_composite_scores.csv", scores)]:
        path = output / filename
        temporary = path.with_suffix(".csv.tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(path)

    report = {"split": "development", "examples": len(scores), "test_evaluated": False, "interpretation": "These composites define different targets. Compare each with its own baseline; raw RMSE differences do not identify the better basketball metric. Correlation with minutes is descriptive, not causal evidence of independence or ability at a different role."}
    (output / "comparison_notes.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved comparisons to: " + str(output))


if __name__ == "__main__":
    main()
