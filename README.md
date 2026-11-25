# BargainBall: forecasting NBA player performance and value

BargainBall uses prior-season player statistics to forecast next-season NBA
performance. The current candidate is one shared, multi-output NumPy neural
network that predicts a broad skill/efficiency profile plus minutes. A
training-fitted PCA of the predicted skill profile supports similar-player
search; minutes and salary are reported as context, not used for similarity.
Salary-cap normalization and salary-cap shares have been added as a separate
valuation field. A training-only
salary-performance benchmark and exploratory development-set value-shortlist
comparison are implemented in `analyze_salary_value.py`.

## Current development results

The shared candidate uses two ReLU hidden layers and is compared with a
persistence baseline.

The latest model predicts 16 targets with one shared network. PCA similarity
uses 13 next-season skill/efficiency targets, excluding minutes and salary.
The 3P% and FT% predictions are auxiliary outputs but omitted from PCA because
they are undefined for players with no attempts. The fit retained 9 PCA
components (96.0% variance); development profile-coordinate RMSE was 0.8928
versus 0.9714 for persistence. Results and player comps are in
`results/development/multitask_profile/20261125_111617_124760/`.
`audit_profile_tiers.py` adds a training-defined prior-PIE tier and tail-error
audit to that run directory; it does not retrain or reweight the model.

For example, Isaiah Jackson's top bench-to-starter skill-profile comps include
Andris Biedriņš, Mark Blount, and Emeka Okafor. That list is a nearest-profile
search, not a claim of identical career outcomes. Similarity uses predicted
next-season skill profiles compared with historical observed profiles; salary
is only displayed as context. `estimate_salary_from_comps.py` summarizes
eligible comps' salary-cap shares as a median and interquartile range, with an
approximate dollar equivalent for the query season. This is a comp-based market
reference, not a separate salary-trained forecast.

The historical test split has been examined during earlier model development;
its follow-up results are not independent confirmation. Architecture and width
comparisons use the training/development split only. See
[`milestone/main.pdf`](milestone/main.pdf) for the report and limitations.

## Project contents

- `milestone/`: report source, references, figure, and generated PDF.
- `numpy_network.py`, `train_numpy.py`, `numpy_pipeline.py`: original
  single-target NumPy models and PCA pipeline.
- `multitask_network.py`, `multitask_profile_pipeline.py`,
  `train_multitask_profile.py`, `profile_targets.py`: current shared
  multi-output model, PCA-profile inference, and target definitions.
- `build_four_stat_models.py`, `evaluate_four_stat_development.py`,
  `evaluate_final_models.py`, `train_numpy.py`: historical separate-target
  experiments and evaluation.
- `prepare_data.py`, `audit_salary_sources.py`,
  `record_online_salary_checks.py`, `preprocess.py`, `prepare_rates.py`,
  `preprocess_rates.py`: dataset preparation and preprocessing.
- `find_*`, `group_*`, `analyze_salary_*.py`: optional exploratory analyses;
  they are not part of the model-training path.
- `compare_*`: architecture and width experiments on train/development data.
- `data/raw/download_manifest.json`: source URLs and checksums. Raw and
  processed datasets, checkpoints, and detailed prediction outputs are kept
  out of this portfolio repository.
- `data/raw/salary_cap_history.csv`: target-season salary-cap values used for
  nominal salary-share normalization, sourced from Basketball-Reference.

## Salary-cap normalization

The salary cap is joined by target-season ending year (for example, target year
2026 maps to the 2025–26 cap). `analyze_salary_cap.py` creates
`salary_cap_normalized_pairs.csv` with salary as a fraction and percentage of
the target-season cap. All 11,272 examples have a cap match; 11,176 have a
listed salary, while 96 remain missing without imputation. Among available
salaries, 194 remain flagged for source review. An exact cross-check against
independent salary archives verified 4,265 records; 24 additional records
received direct source checks, including one corrected name collision. The 194
unmatched or conflicting records remain flagged rather than being assumed
correct. One salary exceeds 100% of the cap; this is possible in the NBA's
soft-cap system and is retained as a descriptive ratio, not treated as an error.

## Salary-based value comparison

`analyze_salary_value.py` uses the existing training-fitted PCA score for
next-season PIE, WS, and VORP. It fits a simple linear benchmark of training
PCA performance against log salary-cap share, using salaries available and not
flagged for source review. On development rows, it ranks players by forecast
performance above the benchmark and compares those lists with lists ranked by
prior-season performance above the benchmark. The benchmark fit uses 7,680
training rows; the shortlist comparison has 1,650 eligible development rows
(salary available and not flagged). Missing and flagged salaries are excluded
from this primary comparison, not imputed. Outputs are in
`results/development/salary_value/`. These are exploratory results, not causal
contract-value estimates or independent test results; the test split was not
loaded.

PCA profiles, salary-aware clusters, and player-comparison scripts remain as
optional exploratory work. They are not part of the active neural-network
training path; the player-comparison prototype did not outperform persistence
consistently on the development split.

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
python train_multitask_profile.py
```

The raw data are not included here. Review source terms and attributions before
redistributing downloaded or derived data. The historical separate-target
models and generated datasets are local artifacts and are not included in this
portfolio snapshot.

## Data and source attribution

The project combines NBA/Basketball-Reference statistics and listed-salary
snapshots from multiple sources. Public accessibility does not make the
underlying data public domain, and no blanket license is asserted for the
combined dataset. Source descriptions and URLs are documented in the report
references and in `data/raw/download_manifest.json`. The manifest records
source URLs and local snapshot hashes when data are downloaded.
