"""Freeze the selected trained NumPy models and training-fitted PCA.

Run this BEFORE evaluating the test set. No test arrays are accessed here.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import numpy as np
from numpy_network import load_model
from numpy_pipeline import TARGETS, fit_pca, combine_scores, load_bundle

ROOT = Path(__file__).resolve().parent
SELECTION_FILE = ROOT / "results" / "development" / "numpy_experiments" / "20261109_144912_284443" / "best_by_target.csv"


def main():
    processed = ROOT / "data" / "processed"
    data_path = processed / "model_data_rates.npz"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    with SELECTION_FILE.open(newline="", encoding="utf-8") as file:
        selection_rows = list(csv.DictReader(file))

    selections = {}

    for row in selection_rows:
        if(row["target"] in selections):
            raise ValueError("Duplicate model selection: " + row["target"])
        selections[row["target"]] = row

    if(set(selections) != set(TARGETS)):
        raise ValueError("Select exactly one checkpoint for each of PIE, WS48, and BPM.")

    # Read only training arrays to learn the PCA transformation.
    with np.load(data_path, allow_pickle=False) as data:
        feature_names = data["feature_names"].tolist()
        mask = data["rate_mask_train"].astype(bool)
        columns = []

        for target in TARGETS:
            columns.append(data["y_" + target + "_train"][mask].astype(np.float64))

        training_targets = np.column_stack(columns)

    pca_parameters = fit_pca(training_targets, data_hash)
    training_scores = combine_scores(training_targets, pca_parameters)
    np.testing.assert_allclose(training_scores.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(training_scores.std(ddof=0), 1.0, atol=1e-12)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = ROOT / "models" / "numpy_final" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    models = {}
    artifacts = {}

    for target in TARGETS:
        row = selections[target]
        source = Path(row["checkpoint_path"])
        parameters, metadata = load_model(source)

        if(metadata["target"] != target or metadata["dataset"] != "rates" or metadata["model_data_sha256"] != data_hash or metadata["feature_names"] != feature_names):
            raise ValueError("Selected model differs from the current arrays: " + target)
        if(metadata["selected"]["dropout"] != float(row["dropout"]) or metadata["learning_rate"] != float(row["learning_rate"])):
            raise ValueError("Selected model settings differ from the selection table.")

        np.testing.assert_allclose(metadata["selected"]["dev_rmse"], float(row["dev_rmse"]), rtol=1e-10)
        target_folder = folder / target
        target_folder.mkdir()
        shutil.copyfile(source, target_folder / "best_model.npz")
        shutil.copyfile(source.parent / "training_history.csv", target_folder / "training_history.csv")
        checkpoint = target + "/best_model.npz"
        artifacts[checkpoint] = hashlib.sha256((folder / checkpoint).read_bytes()).hexdigest()
        artifacts[target + "/training_history.csv"] = hashlib.sha256((target_folder / "training_history.csv").read_bytes()).hexdigest()
        models[target] = {"checkpoint": checkpoint, "source_checkpoint": str(source), "configuration": row["configuration"], "layer_dims": metadata["layer_dims"], "learning_rate": metadata["learning_rate"], "dropout": metadata["selected"]["dropout"], "best_epoch": metadata["selected"]["best_epoch"], "seed": metadata["seed"], "batch_size": metadata["batch_size"], "weight_decay": metadata["weight_decay"], "training_examples": metadata["training_examples"], "development_selection": metadata["selected"]}

    (folder / "pca_parameters.json").write_text(json.dumps(pca_parameters, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copyfile(SELECTION_FILE, folder / "model_selection.csv")

    for name in ["preprocessing.json", "rate_preprocessing.json"]:
        shutil.copyfile(processed / name, folder / name)

    for name in ["pca_parameters.json", "model_selection.csv", "preprocessing.json", "rate_preprocessing.json"]:
        artifacts[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()

    source_folder = folder / "source_snapshots"
    source_folder.mkdir()

    for name in ["numpy_network.py", "numpy_pipeline.py", "build_final_models.py", "evaluate_final_models.py"]:
        shutil.copyfile(ROOT / name, source_folder / name)
        artifacts["source_snapshots/" + name] = hashlib.sha256((source_folder / name).read_bytes()).hexdigest()

    manifest = {"bundle_id": run_id, "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selection_frozen": True, "selection_basis": "Per-target development RMSE from the completed 69-candidate comparison; no test-based selection", "selection_file": str(SELECTION_FILE), "model_data_sha256": data_hash, "target_order": TARGETS, "feature_names": feature_names, "models": models, "artifact_sha256": artifacts, "test_evaluated_at_freeze": False, "pca_fitted_on": "Actual next-season training targets on the valid rate cohort only", "split_interpretation": "Random held-out historical season pairs; players can appear in multiple splits in different seasons."}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    load_bundle(folder)
    print ("Frozen model bundle: " + str(folder))
    print ("PCA fitted on " + str(len(training_targets)) + " actual training rows.")
    print ("PCA variance captured (%): " + str(100.0 * pca_parameters["explained_variance_ratio"]))

    for target in TARGETS:
        print (target.upper() + " | dimensions " + str(models[target]["layer_dims"]) + " | learning rate " + str(models[target]["learning_rate"]) + " | dropout " + str(models[target]["dropout"]))

    return folder


if(__name__ == "__main__"):
    main()
