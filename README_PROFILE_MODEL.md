# Shared next-season profile model

The active pipeline trains one shared NumPy MLP to forecast 16 next-season
outputs: PIE, WS/48, BPM, TS%, usage, rebound/assist/steal/block/turnover rates,
three-point attempt rate, free-throw attempt rate, eFG%, 3P%, FT%, and total
minutes. Target labels are standardized using observed training labels and a
masked, target-balanced loss allows partially labeled rows to contribute.
Missing labels are not imputed.

## Similar-player profiles

PCA is fit only on jointly observed training outcomes and transforms predicted
skill/efficiency outcomes into the same profile space. The 13 PCA features are
PIE, WS/48, BPM, TS%, usage, TRB%, AST%, STL%, BLK%, TOV%, 3P attempt rate,
free-throw attempt rate, and eFG%. 3P% and FT% remain auxiliary predictions,
but are omitted from PCA because they are undefined for players with no
attempts. Minutes are predicted by the shared model but excluded from skill
similarity; salary is displayed only as optional context.

To calculate a comp-based market reference from those neighbors, use the
comp-salary section in notebook 03. For each query it reports the median
salary-cap share and 25th–75th percentile range among the nearest historical
comps that have an available, non-review-flagged salary. It also converts those
shares to approximate dollars using the query target season's cap. At least
five usable salary comps are required. Salary is not used in the similarity
ranking. This is a descriptive market comparison, not a salary-trained model
or a guarantee of a fair contract.

Historical comparisons require the comp's observed outcome season to be no
later than the query player's prior season. The optional bench-to-starter
search filters comps to at least 28 minutes per game and 20 games, while the
query uses prior-season minutes under 24 minutes per game and at least 20 games.
These role thresholds are descriptive filters, not model inputs.

## Current development run

The November 25, 2026 run retained 9 PCA components (96.03% of training
variance). Development PCA-coordinate RMSE was 0.8928 for the model versus
0.9714 for the prior-season persistence profile. The run produced 7,230
bench-to-starter comp rows; salary and minutes do not affect their distances.
Detailed per-target metrics, predictions, and comp lists are in
`results/development/multitask_profile/20261125_111617_124760/`.

To inspect performance by prior-season production tier and the contribution
from extreme errors, use the tier-audit section in notebook 03.

It defines five prior-PIE tiers from training-split quintiles and applies those
fixed boundaries to development. It writes tier-level metrics,
training-tier counts, and a top-5%-squared-error audit alongside the run. It
does not tune or retrain the model.

The follow-up sensitivity experiment retrains after excluding training
player-seasons with fewer than 10 prior-season games, then compares predictions
on all development rows and on development players above/below that same
prior-season cutoff. It removes 320 of 7,890 training rows. On all development
rows, average per-target RMSE rose slightly (about 0.8%; only 3 of 16 targets
improved). On the 1,623-row development cohort with at least 10 prior games,
10 of 16 targets improved, but the average change was very small. Errors
increased more on the 68 development rows below the cutoff. This suggests a
global eligibility filter would sacrifice the limited-history population for
little overall benefit; keep those players in the training set for now.
Detailed results are saved in
`results/development/multitask_profile/20261125_114303_894490/`.
The saved sensitivity comparison and the legacy architecture experiment
summaries are reviewed together in `analysis/06_model_experiments.ipynb`.

## Experimental salary model

`analysis/04_salary_model.ipynb` fits a small Ridge regression to log
salary-cap share using forecast production, age, position, prior games/minutes,
and target season. It excludes prior salary and hometown. Training production
features are generated out-of-fold; source-review-flagged salaries are
excluded. On 1,597 eligible development rows, the model's mean absolute error
was 3.83 salary-cap percentage points versus 5.19 for the nearest-comp median.
This is an exploratory result on the development split, not a chronological
future-season test. The OOF folds are random, so a time-ordered backtest is
needed before treating the improvement as evidence of future performance.
The test split was not loaded.

## 2026 salary holdout

`analysis/05_temporal_salary_holdout.ipynb` now fits using target seasons
through 2025 and evaluates only target season 2026. It retrains the profile
forecaster using earlier outcomes, uses expanding time-causal forecasts for
salary-model fitting, and rebuilds the PCA basis and historical comp pool from
pre-2026 profiles. On 55 eligible 2026 development rows, Ridge MAE was 3.28
cap percentage points (RMSE 4.33), versus 4.26 (RMSE 5.75) for the
historical-comp median. The 2026 rows were excluded from fitting. This is a
chronological backtest, but not a fresh blind test because 2026 development
results were examined previously. The historical test split was not loaded.
Predictions, metrics, and the retrained model are saved under the current
development run directory and `models/numpy_salary_ridge/temporal_<run-id>/`.

This is development-set evidence, not independent test confirmation. The
historical test split was not loaded for this run, but was examined earlier in
the project, so it should not be presented as a pristine holdout.

## Reproduce

Run notebooks `01_data_preparation.ipynb` through
`08_optional_exploratory_analyses.ipynb` in order. Data rebuilding is opt-in in notebook 01,
experiment reruns are opt-in in notebook 06, and downloads/audit writes are
opt-in in notebook 07. Notebook 02 trains the primary model and displays saved
diagnostic figures inline. Older Python scripts are retained as reference
copies; optional legacy comps and clustering workflows are in notebook 08.

Optional low-game sensitivity run and comparison:

For the optional low-game sensitivity run, set the `main()` argument to
`min_prior_games=10` in notebook 02 and compare the resulting outputs in
notebook 03. The unrestricted model remains the primary model.

The training notebook includes checkpoint saving, development inference, and
PCA-coordinate generation. Its output columns follow the target order defined
near the top of notebook 02; PCA coordinates use only the designated skill
targets, excluding minutes and salary.
