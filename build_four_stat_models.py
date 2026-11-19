"""Train total minutes and package PIE / WS48 / BPM / minutes with fixed PCA.

Run: python build_four_stat_models.py
Training and comparison use train/dev only. Settings are collected below.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from train_numpy import train_target, load_target_arrays
from numpy_network import load_model, predict
from numpy_pipeline import FOUR_TARGETS, fit_pca, combine_scores, load_bundle
from evaluate_four_stat_development import evaluate_bundle

ROOT = Path(__file__).resolve().parent
BASE_BUNDLE = ROOT / "models" / "numpy_final" / "20261109_150650_343315"

# HYPERPARAMETERS for the new total-minute network.
MINUTES_CONFIGURATIONS = [
    {"name": "small_16_8", "hidden": [16, 8]},
    {"name": "reference_32_16", "hidden": [32, 16]},
    {"name": "wide_64_32", "hidden": [64, 32]},
]
DROPOUT_CANDIDATES = [0.0, 0.1]
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001
SEED = 42


def main():
    base_manifest, base_models, base_pca = load_bundle(BASE_BUNDLE)
    processed = ROOT / "data" / "processed"
    data_path = processed / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    if(data_hash != base_manifest["model_data_sha256"]):
        raise ValueError("The existing effectiveness networks use different arrays.")

    # Fit the scaler and four-stat PCA only on actual training targets.
    with np.load(data_path, allow_pickle=False) as data:
        train_columns = []
        mask = data["rate_mask_train"].astype(bool)

        for target in FOUR_TARGETS:
            actual, baseline = load_target_arrays(data, target, "train")
            mask = mask & np.isfinite(actual) & np.isfinite(baseline)
            train_columns.append(actual.astype(np.float64))

        training = np.column_stack(train_columns)[mask]

    pca = fit_pca(training, data_hash, FOUR_TARGETS)
    training_scores = combine_scores(training, pca)
    np.testing.assert_allclose(training_scores.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(training_scores.std(ddof=0), 1.0, atol=1e-12)

    if((np.array(pca["component_weights"]) <= 0).any()):
        raise ValueError("Mixed PCA signs require review before interpreting higher as more contribution.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = ROOT / "models" / "numpy_four_stat" / run_id
    results = ROOT / "results" / "development" / "numpy_four_stat" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    settings = {"minutes_configurations": MINUTES_CONFIGURATIONS, "dropout_candidates": DROPOUT_CANDIDATES, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "maximum_epochs": MAX_EPOCHS, "patience": PATIENCE, "weight_decay": WEIGHT_DECAY, "seed": SEED, "base_bundle": str(BASE_BUNDLE), "target_order": FOUR_TARGETS, "minutes_target": "Next-season regular-season total minutes", "minutes_prediction_minimum": 0.0, "minutes_upper_clipping": False, "feature_policy": "Reuse the same 22 prior-season inputs, including age, games and minutes per game; no next-season inputs", "pca_fit": "Actual training targets only on the shared valid rate cohort", "test_evaluated": False, "prior_test_status": "Original split already examined for the previous three-stat design; this is development-only follow-up."}
    (results / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    candidates = []

    for config in MINUTES_CONFIGURATIONS:
        args = SimpleNamespace(dataset="rates", data_dir=processed, output=folder / "minutes_experiments" / config["name"], hidden=config["hidden"], dropout=DROPOUT_CANDIDATES, epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, seed=SEED)
        train_target("minutes", args)
        model_folder = args.output / "rates" / "minutes"
        report = json.loads((model_folder / "development_results.json").read_text(encoding="utf-8"))

        for run in report["runs"]:
            row = run.copy()
            row["configuration"] = config["name"]
            row["hidden_layers"] = "-".join(map(str, config["hidden"]))
            row["learning_rate"] = LEARNING_RATE
            row["baseline_dev_rmse"] = report["baseline_dev_rmse"]
            row["baseline_dev_mae"] = report["baseline_dev_mae"]
            row["checkpoint_path"] = ""

            if(run["dropout"] == report["selected"]["dropout"]):
                row["checkpoint_path"] = str(model_folder / "best_model.npz")

            candidates.append(row)

        pd.DataFrame(candidates).to_csv(results / "minutes_experiment_summary.csv", index=False)

    winner = min(candidates, key=lambda row: row["dev_rmse"])
    minutes_source = Path(winner["checkpoint_path"])
    models = {}
    artifacts = {}

    for target in FOUR_TARGETS:
        source = BASE_BUNDLE / base_manifest["models"][target]["checkpoint"] if target != "minutes" else minutes_source
        parameters, metadata = load_model(source)

        if(metadata["model_data_sha256"] != data_hash or metadata["target"] != target or metadata["feature_names"] != base_manifest["feature_names"]):
            raise ValueError("Checkpoint target, arrays, or feature order differs: " + target)

        target_folder = folder / target
        target_folder.mkdir()
        shutil.copyfile(source, target_folder / "best_model.npz")
        shutil.copyfile(source.parent / "training_history.csv", target_folder / "training_history.csv")
        models[target] = {"checkpoint": target + "/best_model.npz", "source_checkpoint": str(source), "layer_dims": metadata["layer_dims"], "learning_rate": metadata["learning_rate"], "dropout": metadata["selected"]["dropout"], "best_epoch": metadata["selected"]["best_epoch"], "seed": metadata["seed"], "training_examples": metadata["training_examples"], "prediction_minimum": metadata.get("prediction_minimum"), "development_selection": metadata["selected"]}

        for name in [target + "/best_model.npz", target + "/training_history.csv"]:
            artifacts[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()

    (folder / "pca_parameters.json").write_text(json.dumps(pca, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (folder / "selected_models.json").write_text(json.dumps(models, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copyfile(BASE_BUNDLE / "pca_parameters.json", folder / "reference_pca_parameters.json")
    shutil.copyfile(BASE_BUNDLE / "manifest.json", folder / "base_manifest.json")

    for name in ["preprocessing.json", "rate_preprocessing.json"]:
        shutil.copyfile(processed / name, folder / name)

    for name in ["pca_parameters.json", "selected_models.json", "reference_pca_parameters.json", "base_manifest.json", "preprocessing.json", "rate_preprocessing.json"]:
        artifacts[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()

    snapshot = folder / "source_snapshots"
    snapshot.mkdir()

    for name in ["numpy_network.py", "numpy_pipeline.py", "train_numpy.py", "build_four_stat_models.py", "evaluate_four_stat_development.py", "evaluate_final_models.py"]:
        shutil.copyfile(ROOT / name, snapshot / name)
        artifacts["source_snapshots/" + name] = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()

    manifest = {"bundle_id": run_id, "selection_frozen": True, "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection_basis": "Reuse the prior three effectiveness checkpoints; select total-minutes architecture, dropout, and epoch by development RMSE only", "target_order": FOUR_TARGETS, "feature_names": base_manifest["feature_names"], "model_data_sha256": data_hash, "models": models, "artifact_sha256": artifacts, "base_bundle": str(BASE_BUNDLE), "minutes_used_as_model_input": False, "minutes_used_as_training_target_and_pca_input": True, "pca_training_examples": len(training), "test_evaluated_at_freeze": False, "evaluation_status": "Development-only follow-up; original test split was already examined for the previous design", "cohort": "Players with an observed immediately following season; does not model leaving the NBA"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_bundle(folder)
    print ("\nFour-stat model bundle: " + str(folder), flush=True)
    print ("Selected total-minute model: " + winner["configuration"] + " | dropout " + str(winner["dropout"]), flush=True)
    evaluate_bundle(folder, results)
    return folder, results


if(__name__ == "__main__"):
    main()
