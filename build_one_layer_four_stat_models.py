"""Train and package the four development-selected one-hidden-layer models.

Uses existing train/development splits only; does not load test arrays.
Run: python build_one_layer_four_stat_models.py
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
from train_numpy import train_target
from numpy_network import load_model
from numpy_pipeline import FOUR_TARGETS, load_bundle
from evaluate_four_stat_development import evaluate_bundle

ROOT = Path(__file__).resolve().parent
REFERENCE_BUNDLE = ROOT / "models" / "numpy_four_stat" / "20261109_204731_573737"
HIDDEN_WIDTHS = {"pie": [32], "ws_48": [32], "bpm": [64], "minutes": [32]}
LEARNING_RATES = {"pie": 0.001, "ws_48": 0.001, "bpm": 0.003, "minutes": 0.001}
SEED = 42
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 25
WEIGHT_DECAY = 0.0001


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    processed = ROOT / "data" / "processed"
    data_path = processed / "model_data_rates.npz"
    data_hash = sha256(data_path)
    reference_manifest, unused_models, pca = load_bundle(REFERENCE_BUNDLE)

    if(reference_manifest["model_data_sha256"] != data_hash):
        raise ValueError("Reference bundle and current model arrays differ.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    bundle = ROOT / "models" / "numpy_one_layer_four_stat" / run_id
    result_root = ROOT / "results" / "development" / "numpy_one_layer_four_stat" / run_id
    bundle.mkdir(parents=True, exist_ok=False)
    result_root.mkdir(parents=True, exist_ok=False)
    settings = {
        "bundle_id": run_id,
        "candidate_basis": "Lowest mean development RMSE in the five-seed architecture comparison.",
        "hidden_widths": HIDDEN_WIDTHS,
        "learning_rates": LEARNING_RATES,
        "dropout_probability": DROPOUT,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "split": "Existing train/development split; test arrays not accessed.",
        "pca": "Reuse the frozen four-stat PCA fitted only on actual training targets.",
        "model_data_sha256": data_hash,
        "reference_bundle": str(REFERENCE_BUNDLE),
    }
    (result_root / "experiment_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    models = {}
    artifacts = {}

    for target in FOUR_TARGETS:
        args = SimpleNamespace(
            dataset="rates",
            data_dir=processed,
            output=bundle / "training" / target,
            hidden=HIDDEN_WIDTHS[target],
            dropout=[DROPOUT],
            epochs=MAX_EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATES[target],
            weight_decay=WEIGHT_DECAY,
            seed=SEED,
        )
        print("\nBUILD CANDIDATE: " + target.upper() + " | hidden width " + str(HIDDEN_WIDTHS[target]), flush=True)
        train_target(target, args)
        source_dir = args.output / "rates" / target
        target_dir = bundle / target
        target_dir.mkdir()
        shutil.copyfile(source_dir / "best_model.npz", target_dir / "best_model.npz")
        shutil.copyfile(source_dir / "training_history.csv", target_dir / "training_history.csv")
        parameters, metadata = load_model(target_dir / "best_model.npz")

        if(metadata["target"] != target or metadata["layer_dims"] != [22, HIDDEN_WIDTHS[target][0], 1] or metadata["model_data_sha256"] != data_hash):
            raise ValueError("Unexpected checkpoint metadata for " + target)

        models[target] = {
            "checkpoint": target + "/best_model.npz",
            "layer_dims": metadata["layer_dims"],
            "learning_rate": metadata["learning_rate"],
            "dropout": metadata["selected"]["dropout"],
            "best_epoch": metadata["selected"]["best_epoch"],
            "seed": metadata["seed"],
            "training_examples": metadata["training_examples"],
            "prediction_minimum": metadata.get("prediction_minimum"),
            "development_selection": metadata["selected"],
        }

        for filename in [target + "/best_model.npz", target + "/training_history.csv"]:
            artifacts[filename] = sha256(bundle / filename)

    shutil.copyfile(REFERENCE_BUNDLE / "pca_parameters.json", bundle / "pca_parameters.json")
    shutil.copyfile(REFERENCE_BUNDLE / "reference_pca_parameters.json", bundle / "reference_pca_parameters.json")
    shutil.copyfile(REFERENCE_BUNDLE / "manifest.json", bundle / "reference_manifest.json")
    shutil.copyfile(processed / "preprocessing.json", bundle / "preprocessing.json")
    shutil.copyfile(processed / "rate_preprocessing.json", bundle / "rate_preprocessing.json")
    (bundle / "selected_models.json").write_text(json.dumps(models, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    for filename in ["pca_parameters.json", "reference_pca_parameters.json", "reference_manifest.json", "preprocessing.json", "rate_preprocessing.json", "selected_models.json"]:
        artifacts[filename] = sha256(bundle / filename)

    snapshot = bundle / "source_snapshots"
    snapshot.mkdir()

    for filename in ["numpy_network.py", "numpy_pipeline.py", "train_numpy.py", "build_one_layer_four_stat_models.py", "evaluate_four_stat_development.py"]:
        shutil.copyfile(ROOT / filename, snapshot / filename)
        artifacts["source_snapshots/" + filename] = sha256(snapshot / filename)

    manifest = {
        "bundle_id": run_id,
        "selection_frozen": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "Four one-hidden-layer architecture widths chosen by five-seed mean development RMSE; seed 42 checkpoints used for this bundle.",
        "target_order": FOUR_TARGETS,
        "feature_names": reference_manifest["feature_names"],
        "model_data_sha256": data_hash,
        "models": models,
        "artifact_sha256": artifacts,
        "reference_bundle": str(REFERENCE_BUNDLE),
        "minutes_used_as_model_input": False,
        "minutes_used_as_training_target_and_pca_input": True,
        "pca_training_examples": pca["training_examples"],
        "test_evaluated_at_freeze": False,
        "evaluation_status": "New one-layer candidate; development-only evaluation. Test split not accessed.",
        "cohort": "Players with an observed immediately following season; does not model leaving the NBA",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_bundle(bundle)
    (result_root / "build_settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print("\nOne-layer bundle: " + str(bundle), flush=True)
    evaluate_bundle(bundle, result_root)
    return bundle, result_root


if __name__ == "__main__":
    main()
