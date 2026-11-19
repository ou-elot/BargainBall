# BargainBall: forecasting NBA player performance and value

BargainBall uses prior-season player statistics to forecast next-season NBA
performance and playing time. The project combines four independently trained
NumPy neural networks—PIE, WS/48, BPM, and total regular-season minutes—using a
PCA transformation fitted on training targets. Salary-cap normalization and
salary-based value comparisons are planned next.

## Current development results

The primary models use two ReLU hidden layers. They are compared with a
persistence baseline that repeats each player's previous-season statistics and
minutes.

| Target | Model RMSE | Persistence RMSE |
|---|---:|---:|
| PIE | 0.03548 | 0.04584 |
| WS/48 | 0.07068 | 0.08279 |
| BPM | 3.13535 | 3.70394 |
| Total minutes | 589.94 | 677.54 |
| Four-stat PCA composite | 0.75676 | 0.90660 |

The historical test split has been examined during earlier model development;
its follow-up results are not independent confirmation. Architecture and width
comparisons use the training/development split only. See
[`milestone/main.pdf`](milestone/main.pdf) for the report and limitations.

## Project contents

- `milestone/`: report source, references, figure, and generated PDF.
- `numpy_network.py`, `train_numpy.py`, `numpy_pipeline.py`: NumPy model,
  training, and PCA prediction pipeline.
- `prepare_data.py`, `preprocess.py`, `prepare_rates.py`,
  `preprocess_rates.py`: dataset preparation and preprocessing.
- `build_four_stat_models.py`, `evaluate_four_stat_development.py`,
  `evaluate_final_models.py`: model packaging and evaluation.
- `compare_*`: architecture and width experiments on train/development data.
- `data/raw/download_manifest.json`: source URLs and checksums. Raw and
  processed datasets, checkpoints, and detailed prediction outputs are kept
  out of this portfolio repository.

## Reproduce locally

Use Python 3.12 or a compatible Python version. Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the source datasets and prepare model arrays:

```powershell
python download_data.py
python prepare_data.py
python preprocess.py
python prepare_rates.py
python preprocess_rates.py
```

The raw data are not included here. Review source terms and attributions before
redistributing downloaded or derived data. The current full four-stat builder
also reuses the frozen three-stat checkpoints described in
`README_FOUR_STAT.md`; those generated artifacts are not included in this
portfolio snapshot.

## Data and source attribution

The project combines NBA/Basketball-Reference statistics and listed-salary
snapshots from multiple sources. Public accessibility does not make the
underlying data public domain, and no blanket license is asserted for the
combined dataset. Source descriptions and URLs are documented in the report
references and in `data/raw/download_manifest.json`. The manifest records
source URLs and local snapshot hashes when data are downloaded.
