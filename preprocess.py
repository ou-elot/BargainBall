from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


processed_folder = Path(__file__).resolve().parent / "data" / "processed"

numeric_features = [
    "age",
    "g",
    "mp_per_game",
    "pts_per_game",
    "trb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
    "ts_percent",
    "usg_percent",
    "per",
    "ws",
    "ws_48",
    "bpm",
    "vorp",
    "pie"
]

target_columns = {"pie": "next_season_pie", "ws": "next_season_ws", "vorp": "next_season_vorp"}


def main():
    dataset_path = processed_folder / "prediction_pairs.csv"
    data = pd.read_csv(dataset_path)
    summary = json.loads((processed_folder / "dataset_summary.json").read_text(encoding="utf-8"))

    if len(data) != summary["prediction_pairs"]:
        raise ValueError("Dataset row count differs from the preparation summary. Run prepare_data.py again.")

    for target, column in target_columns.items():
        if column not in data.columns:
            raise ValueError("Missing " + column + ". Run the updated prepare_data.py first.")

        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError("Some " + target + " targets are missing or infinite. Do not impute targets.")

    if data.duplicated(["player_id", "season"]).any():
        raise ValueError("Duplicate prediction examples.")

    if not data["target_season"].eq(data["season"] + 1).all():
        raise ValueError("Targets must come from the immediately following season.")

    # Randomly split the rows: 70% training, 15% development, 15% test.
    train_data, remaining_data = train_test_split(data, test_size=0.30, random_state=42)
    dev_data, test_data = train_test_split(remaining_data, test_size=0.50, random_state=42)
    datasets = {"train": train_data.copy(), "dev": dev_data.copy(), "test": test_data.copy()}

    # Older seasons also have a few missing PER, usage, WS/48, and BPM values.
    # Learn every replacement value using training data only.
    medians = {}

    for column in numeric_features:
        median = datasets["train"][column].median()

        if not np.isfinite(median):
            raise ValueError("No finite training median for " + column)

        medians[column] = float(median)

    inputs = {}
    targets = {}

    for split_name, rows in datasets.items():
        for column in numeric_features:
            rows[column] = rows[column].fillna(medians[column])

        inputs[split_name] = rows[numeric_features + ["pos"]].copy()
        targets[split_name] = rows["next_season_pie"].copy()

    # Standardize within each input season using training statistics only.
    training_means = {}
    training_stds = {}

    for column in numeric_features:
        season_means = datasets["train"].groupby("season")[column].mean()
        season_stds = datasets["train"].groupby("season")[column].std(ddof=0)
        season_stds = season_stds.replace(0, 1)
        training_means[column] = season_means.to_dict()
        training_stds[column] = season_stds.to_dict()

        for split_name, rows in datasets.items():
            means = rows["season"].map(season_means)
            stds = rows["season"].map(season_stds)
            inputs[split_name][column] = (rows[column] - means) / stds

    # Learn positions from training data and use the same columns in every split.
    positions = sorted(datasets["train"]["pos"].unique())

    for split_name, rows in datasets.items():
        if not rows["pos"].isin(positions).all():
            raise ValueError("An unknown position appeared in the " + split_name + " split.")

        for position in positions:
            column_name = "pos_" + position
            inputs[split_name][column_name] = (rows["pos"] == position).astype(int)

        inputs[split_name] = inputs[split_name].drop(columns=["pos"])

    # Use original PIE values to evaluate the previous-season PIE baseline.
    baseline_errors = datasets["dev"]["pie"].to_numpy() - targets["dev"].to_numpy()
    baseline_rmse = np.sqrt(np.mean(baseline_errors ** 2))
    baseline_mae = np.mean(np.abs(baseline_errors))

    # Save arrays and corresponding player rows in the same order.
    model_data = {}

    for split_name, rows in datasets.items():
        model_inputs = inputs[split_name].to_numpy(dtype=np.float32)
        model_targets = targets[split_name].to_numpy(dtype=np.float32)

        if not np.isfinite(model_inputs).all() or not np.isfinite(model_targets).all():
            raise ValueError("Missing or infinite model values in the " + split_name + " split.")

        model_data["X_" + split_name] = model_inputs
        model_data["y_" + split_name] = model_targets

        for target, column in target_columns.items():
            # All target arrays share exactly the same input rows and row order.
            model_data["y_" + target + "_" + split_name] = rows[column].to_numpy(dtype=np.float32)
            model_data["baseline_" + target + "_" + split_name] = rows[target].to_numpy(dtype=np.float32)

        rows.to_csv(processed_folder / (split_name + "_rows.csv"), index=False)

    model_data["baseline_dev"] = datasets["dev"]["pie"].to_numpy(dtype=np.float32)
    model_data["baseline_test"] = datasets["test"]["pie"].to_numpy(dtype=np.float32)
    model_data["feature_names"] = inputs["train"].columns.to_numpy(dtype=str)
    model_data["target_names"] = np.array(list(target_columns), dtype=str)

    output_path = processed_folder / "model_data.npz"
    np.savez_compressed(output_path, **model_data)

    # Preserve the learned transformations for reuse.
    preprocessing = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "examples": len(data),
        "random_state": 42,
        "split_method": "random prediction rows; 70% train, 15% dev, 15% test",
        "numeric_features": numeric_features,
        "target_columns": target_columns,
        "positions": positions,
        "feature_names": list(inputs["train"].columns),
        "medians": medians,
        "season_means": training_means,
        "season_stds": training_stds
    }

    settings_path = processed_folder / "preprocessing.json"
    settings_path.write_text(json.dumps(preprocessing, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Print one concise summary.
    print("Examples: " + str(len(data)))

    for split_name in datasets:
        print(split_name + " input shape: " + str(inputs[split_name].shape))

    for target, column in target_columns.items():
        errors = datasets["dev"][target].to_numpy() - datasets["dev"][column].to_numpy()
        print(target.upper() + " development baseline RMSE: " + str(float(np.sqrt(np.mean(errors ** 2)))))
        print(target.upper() + " development baseline MAE: " + str(float(np.mean(np.abs(errors)))))
    print("Saved model data to: " + str(output_path))


if __name__ == "__main__":
    main()
