# BargainBall: notebook-style NumPy network

**Current workflow:** the four-stat model adds predicted total season minutes
to PIE, WS/48, and BPM. See `README_FOUR_STAT.md` for the completed run,
training settings, PCA weights, development results, and follow-up test output
loss. The three-stat
experiments and final-test results below document the earlier design.

The network follows the organization and notation in the local copies of
`Building_your_Deep_Neural_Network_Step_by_Step.ipynb` and
`Regularization.ipynb` listed in your shared CS230 Drive folder. The Adam
function follows the notation in your local `Optimization_methods.ipynb`.

## Start by seeing the hidden layers

Run these commands from `CS230_Final` in PowerShell:

```powershell
.\.venv\Scripts\python.exe inspect_layers.py
```

This uses a real **training** example and a small, untrained network:
`22 -> 4 -> 2 -> 1`. It prints the input features, each layer's weights/bias
shapes, the weighted sum for its first neuron, all pre-activation values `Z`,
and the activations `A` passed to the next layer. It then calculates gradients
and demonstrates one Adam update on a copy of the parameters.

To see dropout's masks and scaling:

```powershell
.\.venv\Scripts\python.exe inspect_layers.py --keep-prob 0.75
```

The random initial predictions are not trained estimates. The single-example
update is a teaching demonstration and does not save a model. Use `--row 1`
to choose the second training example, or `--hidden 32 16` for larger layers.

## Read the functions in notebook order

1. `initialize_parameters_deep`: create `W1`, `b1`, ..., `WL`, `bL`.
2. `linear_forward`: calculate `Z = np.dot(W, A_prev) + b`.
3. `linear_activation_forward`: apply ReLU, or the linear output activation.
4. `L_model_forward`: connect all the layers and save their caches.
5. `compute_cost`: measure mean squared prediction error.
6. `linear_backward`: calculate `dW`, `db`, and `dA_prev`.
7. `linear_activation_backward`: include the activation's derivative.
8. `L_model_backward`: work backward from the output to the first layer.
9. `update_parameters_with_adam`: update the parameters using the gradients.

All these functions are in `numpy_network.py`. Arrays use the course convention:
`X` is `(features, examples)` and `Y` is `(1, examples)`. Parameters use keys
such as `W1` and `b1`; gradients use `dW1` and `db1`. Adam uses `v`, `s`,
`v_corrected`, and `s_corrected`. A basic `update_parameters` function is also
included to illustrate ordinary gradient descent.

### Changes needed for this project

- The notebook classifier ends in sigmoid and uses cross-entropy. This network
  predicts continuous PIE, WS/48, or BPM, so its output is **linear** and its
  cost is **MSE**.
- For MSE, backpropagation starts with `dAL = 2 * (AL - Y)`. As in the notebook,
  `linear_backward` averages `dW` and `db` over the examples using `1 / m`.
  Do not divide `dAL` by `m` as well: that would average the gradients twice.
- Initialization uses He scaling, `sqrt(2 / incoming_neurons)`, appropriate for
  the ReLU hidden layers.
- Optional `weight_decay` retains the experiment's coupled L2 penalty on
  weights and biases. Set it to zero for an unregularized learning exercise.

### Dropout, using the notebook's four steps

Inside each hidden layer during training:

```python
D = rng.random(A.shape)
D = (D < keep_prob).astype(A.dtype)
A = A * D
A = A / keep_prob
```

During backpropagation, reuse that layer's saved mask:

```python
dA = dA * D
dA = dA / keep_prob
```

`keep_prob=0.9` means keep 90%, or drop 10%. The trainer's existing CLI uses
the **drop** probability: `--dropout 0.1` becomes `keep_prob=0.9` internally.
Prediction and development evaluation disable dropout. The output layer never
uses dropout. Training creates the random generator once per candidate so masks
change across batches, rather than resetting the random seed on every call.

`forward_propagation_with_dropout` and `backward_propagation_with_dropout` are
also available as notebook-style entry points. Unlike the fixed three-layer
notebook example, they support any number of hidden layers.

## Train a small network

```powershell
.\.venv\Scripts\python.exe check_numpy.py
.\.venv\Scripts\python.exe train_numpy.py --target pie --hidden 4 2 --dropout 0 --weight-decay 0 --output models/numpy_learning
```

Inspect the resulting trained layers:

```powershell
.\.venv\Scripts\python.exe inspect_layers.py --model models/numpy_learning/rates/pie/best_model.npz
```

The checkpoint determines the target, dimensions, and target scaling when
`--model` is supplied. The walkthrough verifies its dataset hash and feature
order. The inspected row is a training example, so this is not an evaluation
of generalization.

## Full experiments

The trainer defaults to `22 -> 32 -> 16 -> 1`, separate PIE/WS48/BPM/minutes networks,
and dropout candidates 0 and 0.1:

```powershell
.\.venv\Scripts\python.exe train_numpy.py
```

Use `--dataset original` for the earlier PIE/WS/VORP targets. Use a distinct
`--output` directory for each architecture experiment.

The trainer reuses your prepared arrays and split assignments, selects models
on development RMSE, and saves checkpoints, predictions, training history,
and development reports under `<output>/<dataset>/<target>/`. Rate targets
use the saved validity masks; PIE uses its full training split. Non-PIE target
scaling is learned from training data only. Training never accesses test arrays.

The old PyTorch PCA scripts load `.pt` models. These NumPy models use `.npz`
and can be read with `load_model` and `predict` from `numpy_network.py`.

## Compare hyperparameters

```powershell
.\.venv\Scripts\python.exe run_experiments.py
```

The experiment settings are together near the top of `run_experiments.py`.
It compares four hidden-layer sizes at learning rate 0.001, plus learning rates
0.0003 and 0.003 for the reference 32/16 architecture. Each setting tries
dropout 0 and 0.1 for PIE, WS/48, and BPM: 36 fitted candidates in total.

Each invocation creates a timestamped folder under
`results/development/numpy_experiments/` with all candidate scores,
`best_by_target.csv`, comparison plots, selected-model learning curves,
experiment settings, and source snapshots. Models and full histories are
under the matching timestamp in `models/numpy_experiments/`. Each
configuration/target folder retains the winning dropout checkpoint and the
history and scores for both dropout candidates.

The trainer reports two different kinds of training loss:

- `train_mse_scaled`: the original average minibatch error accumulated while
  weights change, using scaled targets and training-time dropout.
- `train_mse` and `dev_mse`: unpenalized errors in original target units,
  evaluated at the same fixed epoch-end weights with dropout disabled.

Use the second pair when comparing learning curves. Summary scores are from
the checkpoint with the lowest development RMSE, not necessarily the last
epoch. Loss values from different targets have different units and should not
be compared directly. All comparisons in a run use the same split and seed.

The first completed comparison is documented in
`results/development/numpy_experiments/20261109_143702_820846/RESULTS.md`.

### More dropout probabilities and a larger network

```powershell
.\.venv\Scripts\python.exe run_dropout_experiments.py
```

This follow-up runs dropout 0.2, 0.3, and 0.5 for hidden dimensions 32/16 and
64/32, plus dropout 0, 0.1, 0.2, 0.3, and 0.5 for 128/64. All use learning rate
0.001 and the same other settings as the first round: 33 new fits across the
three targets. It reuses the first round's scores instead of repeating those
fits. `experiment_summary.csv` contains only the new fits;
`combined_experiment_summary.csv` combines all 69 candidates.
`best_by_target.csv` selects across the combined comparison.

The follow-up adds `dropout_vs_width.png`, which compares the complete
five-probability sweep for 32/16, 64/32, and 128/64 at learning rate 0.001.
Its results are documented in
`results/development/numpy_experiments/20261109_144912_284443/RESULTS.md`.

## Final three-model bundle and held-out test evaluation

The selected trained checkpoints and a training-fitted PCA transformation are
frozen together under:

```text
models/numpy_final/20261109_150650_343315/
```

The final choices are PIE 32/16 at learning rate 0.001, WS/48 128/64 at 0.001,
and BPM 32/16 at 0.003, all with 10% training dropout. Each checkpoint retains
its development-selected epoch. PCA fits actual training targets on 7,888
valid rate examples; its weights are fixed before evaluating test data.

`build_final_models.py` packages the already-trained models and fits PCA
without accessing test arrays. `numpy_pipeline.py` provides the reusable
`load_bundle`, `predict_bundle`, and `combine_scores` functions. The bundle
includes preprocessing metadata, source snapshots, hashes, model settings,
and a manifest recording when selection was frozen.

The completed final evaluation is saved here:

```text
results/test/numpy_final/20261109_150650_343315/
```

Start with `RESULTS.md`, `metrics.csv`, and `ERROR_DIAGNOSTICS.md`. The folder
also contains row-aligned development/test predictions, test plots, and an
evaluation audit linking the results to the frozen manifest.

Commands used for final evaluation and descriptive diagnostics:

```powershell
.\.venv\Scripts\python.exe evaluate_final_models.py --bundle models/numpy_final/20261109_150650_343315
.\.venv\Scripts\python.exe analyze_final_errors.py --results results/test/numpy_final/20261109_150650_343315
```

The evaluation command creates a new result directory and does not overwrite
the completed run. It performs no training or model/PCA selection. The test
set has now been evaluated; these test results were not used to change the
frozen configuration. The random historical-pair split does not establish
performance on chronologically future seasons.

Example inference from already-preprocessed feature rows:

```python
from numpy_pipeline import load_bundle, predict_bundle

manifest, models, pca = load_bundle("models/numpy_final/20261109_150650_343315")
predicted_statistics, composite_scores = predict_bundle(X_rows, models, pca, manifest["feature_names"])
```

`X_rows` must have shape `(examples, 22)` and use the stored preprocessing and
feature order. Output statistic columns are PIE, WS/48, BPM; the composite
uses training-composite standard-deviation units, not salary-value units.
