# Rate-statistic comparison experiment

Copy the five Python scripts in this package into your existing CS230_Final
folder beside the existing `train.py`, `pca_score.py`, and `player_matching.py`.
Keep your original scripts, arrays, models, and raw files. None are replaced.
This update needs no new packages or downloads.

Run in your project terminal:

```powershell
.\.venv\Scripts\python.exe prepare_rates.py
.\.venv\Scripts\python.exe preprocess_rates.py
.\.venv\Scripts\python.exe train_rates.py
.\.venv\Scripts\python.exe pca_rates.py
.\.venv\Scripts\python.exe compare_rates.py
```

The final comparison requires the original PCA parameters from your previous
`pca_score.py` run. If you have not run that script yet, run it once before
`compare_rates.py`.

## What changes

- Existing composite: predicted PIE + total Win Shares + total VORP.
- New composite: the same predicted PIE + predicted WS/48 + predicted BPM.
- WS/48 measures Win Shares per 48 minutes. BPM estimates impact per 100
  possessions. These targets focus on rates rather than accumulated season
  contribution. They do not establish ability independent of role or context.
- The existing PIE network is reused. Only WS/48 and BPM networks are trained.
  They use the same 22 prior-season inputs, architecture, Adam optimizer,
  mini-batches, dropout comparison, and development-based early stopping.
- Existing input arrays, preprocessing transformations, and split assignments
  are copied exactly. No new split is drawn. Prior-season minutes per game,
  games, and total statistics remain among the original inputs.

## Undefined rates and minutes

All 11,272 original examples remain in the original experiment. Two training
examples have an undefined next-season WS/48 or BPM, or an undefined previous
rate needed for a fair baseline. They are excluded from the rate experiment,
leaving 7,888 rate-training, 1,691 development, and 1,691 test examples. Actual
rate targets are never filled with a median or zero. The new arrays retain
these missing values and provide `rate_mask_train`, `rate_mask_dev`, and
`rate_mask_test`; use the masks with rate targets.

`rate_excluded_rows.csv` lists both rows. They involve a zero-minute Damion
James season, either as the input season or the target season.

No general minimum-minutes cutoff is applied. Rate statistics can be unstable
for short appearances. `compare_rates.py` therefore reports development
errors in descriptive target-season minute groups: under 250, 250–999, and
1,000 or more. Those observed next-season minutes are used only for error
analysis and correlations, never as model inputs, model-training weights,
or model-selection criteria. The scripts do not train a playing-time model.

## PCA and comparison

The new PCA scaler and weights are fit only to actual training-set PIE, WS/48,
and BPM on the valid rate cohort. The same fixed transformation combines
predictions, observed next-season targets, and prior-season baseline values.
Scores are centered on the training composite and expressed in training
standard deviations. The original PCA parameters are preserved.

The original composite is recomputed from its original models on exactly the
same development rows as the new composite. Both composites are compared
with their own prior-season baselines. Pearson and Spearman correlations
with observed next-season minutes, and with prior-season total minutes, are
reported. Correlations are descriptive: low correlation is not proof of a
better ability metric; high correlation is not proof that a rate is a total.
Strong players often receive more playing time. All target predictions remain
evaluated against observed performance, not hypothetical performance at a new
role or minute allocation.

Because the composites summarize different statistics, their raw RMSEs do
not decide which one is the better basketball metric. Read each model's
improvement over its own baseline alongside the minute correlations and
short-appearance errors. Salary comparison is still a separate later step.
No test-set performance is calculated by these scripts.

## Outputs

- `data/processed/prediction_pairs_rates.csv`: augmented pairs, including
  actual next-season rate statistics and minutes.
- `data/processed/model_data_rates.npz`: unchanged input arrays plus new
  targets, baselines, masks, and minute metadata. `model_data.npz` is preserved.
- `data/processed/rate_preprocessing.json`: retained and excluded split counts.
- `models/rates/ws_48/best_model.pt` and `models/rates/bpm/best_model.pt`:
  new checkpoints. Target scaling parameters are stored in each checkpoint.
- `models/rates/pca/pca_parameters.json`: new PCA parameters.
- `results/development/rates/pca_scores.csv`: new composite predictions.
- `results/development/rates/composite_comparison.csv`: baseline comparisons
  and minute correlations for both composites.
- `results/development/rates/errors_by_minutes.csv`: errors by target minutes.
- `results/development/rates/matched_composite_scores.csv`: both composites
  for the same players, plus their observed minutes.

`train_rates.py --target ws_48` or `--target bpm` can rerun one new target.
Use the existing same virtual-environment Python command before the script.

The scripts use explicit loops, string concatenation, and ordinary indentation,
without comprehensions or f-strings.

Definitions: https://www.basketball-reference.com/about/glossary.html and
https://www.basketball-reference.com/about/bpm2.html
