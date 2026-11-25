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

To calculate a comp-based market reference from those neighbors, run
`python estimate_salary_from_comps.py`. For each query it reports the median
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
from extreme errors, run:

```powershell
python audit_profile_tiers.py
```

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

This is development-set evidence, not independent test confirmation. The
historical test split was not loaded for this run, but was examined earlier in
the project, so it should not be presented as a pristine holdout.

## Reproduce

```powershell
python prepare_rates.py
python preprocess_rates.py
python train_multitask_profile.py
python estimate_salary_from_comps.py
```

Optional low-game sensitivity run and comparison:

```powershell
python train_multitask_profile.py --min-prior-games 10
python compare_min_games_runs.py
```

Inference using the saved checkpoint:

```python
from multitask_profile_pipeline import load_bundle, predict_profile

metadata, parameters, pca = load_bundle("models/numpy_multitask_profile/<run-id>")
predictions, profile_coordinates = predict_profile(
    X_rows, metadata["feature_names"], metadata, parameters, pca
)
```

Prediction columns follow `metadata["target_order"]`; PCA coordinates use the
ordered skill targets in `metadata["skill_pca_target_order"]`.
