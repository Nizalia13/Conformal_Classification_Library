# Conformal Envelopes

A Python library for classification with a separate conformal envelopes per class. For each test sample it returns a prediction set, that is the classes whose envelopes accept the sample's score vector.

Envelope methods:
- **Collapsed:** averages the observed nonconformity scores.
- **Radial:** learns a direction dependent boundary.
- **Strip:** learns constraints between pairs of observed score dimensions.

## Input data

Training data can be a pandas DataFrame or a CSV file:

| ID | predictor_1 | predictor_2 | label |
|---|---|---|---|
| sample_1 | 0.85 | 0.72 | A |
| sample_2 | 0.31 | NaN | B |

The library takes predictor scores as input; it does not train the predictors. The predictors must be trained on data that are separate from the data passed to `fit` and from the test data.

Missing scores can be represented by NaN. Handling of the missing scores depends on the method and on which scores are available. A sample whose scores are all missing needs particular care; the collapsed method rejects it, while the radial and strip methods accept it. No coverage statement is made for that case.

By default, higher scores mean better agreement with a class, and scores must lie between 0 and 1. The library converts these to nonconformity scores using `1 - score`.

Use `score_direction="lower_is_better"` for nonnegative scores where smaller values already mean better agreement.


## Installation

From the project directory:

```bash
python -m pip install -e .
```

To include testing and notebook example dependencies:

```bash
python -m pip install -e ".[test,examples]"
```

Python 3.10 or later is required.


## Fit and predict

```python
from conformal_envelopes import ConformalSetModel

model = ConformalSetModel(method="strip", alpha=0.10, random_state=13, force_nonempty=False, n_bins=8, min_samples=3,)

model.fit(training, id_col="ID", label_col="label", score_cols=["predictor_1", "predictor_2"],)

predictions = model.predict(test_data)
envelopes = model.get_envelopes()
```

Here, `training` and `test_data` are DataFrames or CSV paths. Test labels are not needed for prediction.

Each prediction set contains the classes whose envelopes accept the sample. Sets can contain several classes or be empty. The returned DataFrame has one row per sample, with the columns `prediction_set`, `set_size`, `forced` and `tau_per_label`, next to the ID column.

For a dictionary mapping sample IDs to prediction sets:

```python
prediction_sets = model.predict(test_data, output="dict")
```

Metrics for set-valued predictions (coverage, average set size, singleton rate, empty rate and per-class coverage):

```python
from conformal_envelopes import evaluate_prediction_sets

metrics = evaluate_prediction_sets(prediction_sets, true_labels)
```

A fitted model can be stored with `model.save(path)` and loaded with `ConformalSetModel.load(path)`.

## Class-specific predictor columns

By default, all classes use the same predictor columns.

When predictors provide separate scores for each candidate class, supply `label_to_columns` to `fit`:

```python
model.fit(training, label_to_columns={"A": ["A_predictor_1", "A_predictor_2"],
                                      "B": ["B_predictor_1", "B_predictor_2"],},)
```

## Method parameters

- Collapsed: no additional geometry parameters.
- Radial: `n_directions` (default 250), `smoothing` (default 8), `angle_deg` (default 30) and `neighbor_fraction` (default 0.2).
- Strip: `n_bins` (default 8) and `min_samples` (default 3).

Common parameters are `alpha`, `shape_fraction` (default 0.5), `random_state`, `score_direction`, `force_nonempty` and `min_class_size`.

## Visualisation

There are three plotting options:

- `model.plot(...)`: fits a separate two dimensional envelope using the two selected predictors. Test point categories describe decisions from this separate 2D envelope.

- `model.plot_slice(...)`: displays a cross section of the existing fitted envelope without refitting. Use `slice_values` to specify their fixed nonconformity values.

- `plot_inclusion_heatmap(...)`: shows how frequently each candidate class appears in prediction sets, grouped by the true class. Diagonal entries give per class coverage. Rows need not sum to one because prediction sets can contain multiple labels.

Envelope plots use nonconformity coordinates. With `score_direction="higher_is_better"`, these equal `1 - score`.

In slice plots, test point categories use each sample's actual full-dimensional scores. Their acceptance may therefore differ from what their displayed position relative to the slice suggests.

Use `test_data` to show test samples, `show_training` to display training samples, and `show_missing` to display dotted lines for samples missing one plotted coordinate.

Pass `ax` to arrange plots in separate figures or a shared figure.

## Examples

All notebooks are in the `examples/` folder. Run them with a Python kernel that has the `[examples]` dependencies installed. Each notebook uses the local library.

- [`synthetic_example.ipynb`](examples/synthetic_example.ipynb): simulated two-class data with four predictors per class and 15% of the scores hidden at random. The experiment is repeated over 500 seeds and reports the mean and standard error of coverage, set size, singleton rate and empty rate.

- [`iris_guided_example.ipynb`](examples/iris_guided_example.ipynb): a short usage tutorial on the Iris data. The sample is too small to judge coverage, so it only shows how to use the library.

- [`football_player_positions_example.ipynb`](examples/football%20example/football_player_positions_example.ipynb): the main football example. It classifies 18,147 distinct FIFA 19 players as Defender, Midfielder, Attacker, or Goalkeeper from 34 individual skill ratings. Four classifiers (logistic regression, random forest, gradient boosting, and an RBF support-vector classifier) produce the scores. Position-specific suitability ratings and identity fields are excluded from the predictors. This is classification of recorded roles in a historical game-data snapshot, not a forecast of match outcomes or future player performance. The data are split into four disjoint partitions of approximately 40% classifier training, 20% validation, 30% conformal fitting, and 10% testing. After a single illustrative run, the conformal fit and the test draw are repeated 100 times with the same trained classifiers.

- [`football_all_positions_example.ipynb`](examples/football%20example/football_all_positions_example.ipynb): a separate experiment with the recorded position codes. Positions with fewer than 100 players (LWB, RWB, LAM, RAM, CF, LF, RF) are excluded, which leaves 20 positions and 17,835 players. It includes the five goalkeeper skill attributes, reports per-position coverage, and shows inclusion heatmaps. The conformal fit and the test draw are repeated 100 times.

The football notebooks download one CSV on first run and then use the hash-verified cache in `examples/football example/data/`, so later runs work offline. Results and run information are written to `examples/football example/outputs/`.

## Tests

```bash
python -m pytest test_library.py -v
```

The tests check software behaviour, including prediction formats, input validation, saving/loading, and whether plotting preserves predictions.

## Coverage and limitations

`alpha=0.10` specifies a nominal coverage target of 90%.

Conformal coverage relies on assumptions such as exchangeability between calibration and future samples, with the score construction fixed independently of the calibration data. It does not guarantee exactly 90% coverage on every observed test set.

The guarantee is per class (it holds for samples of each true class) and assumes that the missingness pattern has the same distribution in calibration and test data. Distribution changes, candidate filtering, and missing-score handling need to be considered when interpreting results.

Small class-specific calibration sets can produce infinite thresholds and consequently broad prediction sets.

For strip envelopes, zero or very small shape limits can produce large calibration scaling factors, weakening other constraints.

No method is claimed to be best in general. On the included examples the three constructions give similar coverage and set sizes.