# Current model: PIE, WS/48, BPM, and total playing time

## Pipeline

```text
22 prior-season inputs (including age, games and minutes per game)
                            |
           Four separately trained NumPy networks
                            |
        Predicted PIE / WS48 / BPM / total season minutes
                            |
             Fixed training-derived standardization
                            |
                Fixed four-variable PCA component
                            |
        Predicted effectiveness + playing-time composite
```

The fourth output is **next-season regular-season total minutes**, not minutes
per game. Higher predicted minutes now increases the composite when the other
predictions are fixed. Salary remains a separate valuation input.

## Completed run

```text
models/numpy_four_stat/20261109_204731_573737/
results/development/numpy_four_stat/20261109_204731_573737/
```

| Target | Hidden layers | Learning rate | Drop probability | Selected epoch |
|---|---|---:|---:|---:|
| PIE | 32 -> 16 | 0.001 | 0.1 | 127 |
| WS/48 | 128 -> 64 | 0.001 | 0.1 | 10 |
| BPM | 32 -> 16 | 0.003 | 0.1 | 17 |
| Total minutes | 64 -> 32 | 0.001 | 0.1 | 25 |

The first three checkpoints are reused from the frozen three-stat bundle.
The new minutes network was selected from 16/8, 32/16, and 64/32 hidden layers,
each trying dropout 0 and 0.1. All minute fits used Adam, batch size 64,
learning rate 0.001, weight decay 0.0001, seed 42, maximum 300 epochs, and
development early stopping with patience 25. Settings are near the top of
`build_four_stat_models.py`.

## Development results

| Target | Model RMSE | Prior-season baseline RMSE | RMSE reduction |
|---|---:|---:|---:|
| Total minutes | 589.9364 minutes | 677.5360 minutes | 12.93% |
| Four-stat composite | 0.756761 | 0.906598 | 16.53% |

The minute model's MAE is 473.0763 minutes, versus baseline 512.7167 minutes.
This remains a noisy forecast of playing time; injuries, future roster changes,
and future coaching decisions are not explicitly modeled.

The three-stat composite, evaluated on the same development rows with the
same effectiveness networks, has RMSE 0.801016 versus its own baseline
0.970270 (17.44% reduction). These composites have different definitions.
Their raw RMSEs do not determine which is a better basketball/value metric.

## Four-stat PCA definition

Fit actual next-season training targets on 7,888 valid joint training rows.
Standardize every target using this training cohort, then fit PCA. The first
component has these coefficients on standardized values:

| Target | Coefficient |
|---|---:|
| PIE | 0.516020 |
| WS/48 | 0.527763 |
| BPM | 0.561244 |
| Total minutes | 0.374426 |

Finally, center and scale the component to mean zero and standard deviation
one on the actual training composite. Coefficients are not percentages and
their sum is not constrained to one. PCA's first component captures 72.36%
of standardized training-target variance, which is not prediction accuracy.

This is a hybrid of effectiveness and opportunity/availability. It is not a
pure rate metric, a causal estimate of contribution under a different role,
or a dollar-value score. It does not turn the component into estimated wins.

## Data handling

- The existing `model_data_rates.npz`, row order, and split assignments are
  reused. No replacement array file or new split is necessary.
- `load_target_arrays` in `train_numpy.py` exposes `next_season_minutes_*` as
  the new target and `prior_minutes_*` as its baseline.
- Observed next-season minutes are never a predictor input. They are used
  only as training labels, actual PCA targets, and evaluation outcomes.
- Minutes training uses all 7,890 training rows, including defined zero-minute
  targets. The four-stat PCA uses 7,888 rows where all rate targets and prior
  baselines are defined. Targets are not imputed.
- The network learns standardized minute targets with a linear output and
  MSE. Decoded minute predictions are floored at zero during prediction and
  development selection. There is no upper clipping. This policy is stored
  in the checkpoint metadata; three development predictions needed the floor.
- All contract types remain in the cohort, including rookie deals. Salary is
  not among the performance-network inputs or PCA inputs.
- Examples require an observed immediately following NBA season. The minutes
  forecast is conditional on this cohort; league exit is not modeled.

## Commands

Build a new timestamped bundle and run the development comparison:

```powershell
.\.venv\Scripts\python.exe build_four_stat_models.py
```

Train only the new target with the general trainer:

```powershell
.\.venv\Scripts\python.exe train_numpy.py --target minutes --hidden 64 32 --dropout 0.1 --output models/numpy_minutes
```

The general trainer's `--dataset rates --target all` now includes the fourth
target. Existing experiment scripts that explicitly list only three targets
still run those three targets.

Inspect the selected minutes network:

```powershell
.\.venv\Scripts\python.exe inspect_layers.py --model models/numpy_four_stat/20261109_204731_573737/minutes/best_model.npz
```

Run a separate development-only evaluation of a saved four-stat bundle:

```powershell
.\.venv\Scripts\python.exe evaluate_four_stat_development.py --bundle models/numpy_four_stat/20261109_204731_573737
```

Predict from already-preprocessed feature rows:

```python
from numpy_pipeline import load_bundle, predict_bundle

manifest, models, pca = load_bundle("models/numpy_four_stat/20261109_204731_573737")
statistics, scores = predict_bundle(X_rows, models, pca, manifest["feature_names"])
```

`statistics` has columns in `pca["target_order"]`: PIE, WS/48, BPM, minutes.
`scores` contains the four-stat PCA composite. `X_rows` must use the original
preprocessing and feature order; this function does not accept raw statistics.

## Evaluation status

The frozen version now also has a follow-up output-loss evaluation on the
original test split. No model parameters or PCA settings were updated during
that evaluation. Results are saved under:

```text
results/test/numpy_four_stat/20261109_204731_573737/
```

| Split | Composite MSE | Baseline composite MSE | Composite RMSE | Composite MAE |
|---|---:|---:|---:|---:|
| Training | 0.584137 | 0.956047 | 0.764289 | 0.464435 |
| Development | 0.572687 | 0.821920 | 0.756761 | 0.487815 |
| Test | 1.183963 | 2.051982 | 1.088101 | 0.493758 |

Test composite MSE is 42.30% below baseline; equivalently, RMSE is 24.04%
below baseline. MSE and RMSE percentage reductions are different quantities.
The previously discussed two-minute Kuzminskas season contributes 57.84% of
the new composite's test squared error, so MAE and the descriptive error
diagnostics remain important alongside RMSE.

The original test split was already examined for the earlier three-stat
design before this revision was chosen. This is a follow-up evaluation on a
previously examined split, not fresh independent confirmation of the revised
method. A genuinely new confirmation requires a fresh evaluation design.

`evaluate_final_models.py` now supports both three- and four-stat bundles and
reports losses for training, development, and test rows. It does no training
or PCA fitting. The original test report and model bundle remain separate.

```powershell
.\.venv\Scripts\python.exe evaluate_final_models.py --bundle models/numpy_four_stat/20261109_204731_573737
```

The completed evaluation is not overwritten on rerun. Read `output_loss.csv`
for final-composite errors and `metrics.csv` for all individual outputs.

The original three-stat bundle still loads through the same inference API.
Its saved artifacts and test report retain their original definitions.

The earlier 14-raw-stat feature audit remains exploratory background. The
current chosen PCA inputs are the four targets listed above.

## Train / development / test loss plots

Run `plot_loss_diagnostics.py` to visualize the saved results without training
or generating new model predictions:

```powershell
.\.venv\Scripts\python.exe plot_loss_diagnostics.py
```

Plots are in the completed test result directory's `loss_diagnostics/` folder:

- `composite_loss_comparison.png`: final-score MSE, RMSE, and MAE by split.
- `network_split_losses.png`: individual-output MSE and MAE by split.
- `network_learning_curves.png`: recorded train/dev MSE curves with one test
  marker per selected checkpoint; no per-epoch test curve is implied.
- `calibration_diagnostics.png`: predicted versus observed bin averages.
- `DIAGNOSTICS.md`: interpretation and bias/variance limitations.

The plots use fixed epoch-end/inference errors with dropout off, not noisy
minibatch training losses. No joint-composite epoch history was recorded
because its constituent networks were trained separately.
