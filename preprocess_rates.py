"""Keep the existing input arrays and split rows; attach new targets and masks."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    folder = ROOT / "data" / "processed"
    base_path = folder / "model_data.npz"
    pairs = pd.read_csv(folder / "prediction_pairs_rates.csv")
    pairs = pairs.set_index(["player_id", "season"], verify_integrity=True)
    arrays = {}
    sizes = {}

    with np.load(base_path, allow_pickle=False) as original:
        for key in original.files:
            arrays[key] = original[key].copy()

    for split in ["train", "dev", "test"]:
        rows = pd.read_csv(folder / (split + "_rows.csv"))
        keys = pd.MultiIndex.from_frame(rows[["player_id", "season"]])
        aligned = pairs.loc[keys].reset_index()

        if len(rows) != arrays["X_" + split].shape[0]:
            raise ValueError("Split metadata and model inputs have different row counts.")

        pd.testing.assert_frame_equal(rows[["player_id", "season"]], aligned[["player_id", "season"]])
        np.testing.assert_array_equal(arrays["y_pie_" + split], aligned["next_season_pie"].to_numpy(dtype=np.float32))

        for target in ["ws_48", "bpm"]:
            arrays["y_" + target + "_" + split] = aligned["next_season_" + target].to_numpy(dtype=np.float32)
            arrays["baseline_" + target + "_" + split] = aligned[target].to_numpy(dtype=np.float32)

        mask = aligned["rate_valid"].to_numpy(dtype=bool)
        arrays["rate_mask_" + split] = mask
        arrays["next_season_minutes_" + split] = aligned["next_season_minutes"].to_numpy(dtype=np.float64)
        arrays["prior_minutes_" + split] = aligned["mp"].to_numpy(dtype=np.float64)
        path = folder / (split + "_rows_rates.csv")
        temporary = path.with_suffix(".csv.tmp")
        aligned.to_csv(temporary, index=False)
        temporary.replace(path)
        sizes[split] = {"original": len(rows), "valid_rate_examples": int(mask.sum()), "excluded": int((~mask).sum())}
        print(split + ": " + str(int(mask.sum())) + " valid rate examples out of " + str(len(rows)))

    arrays["base_model_data_sha256"] = np.array(hashlib.sha256(base_path.read_bytes()).hexdigest())
    path = folder / "model_data_rates.npz"
    np.savez_compressed(path, **arrays)
    report = {"base_model_data_sha256": str(arrays["base_model_data_sha256"]), "splits": sizes, "input_arrays": "copied unchanged from model_data.npz", "excluded_rows": "Undefined next-season rate target or prior-season baseline; no target imputation", "next_season_minutes": "Observed evaluation metadata only; not a model input", "test_metrics_computed": False}
    (folder / "rate_preprocessing.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved new arrays to: " + str(path))


if __name__ == "__main__":
    main()
