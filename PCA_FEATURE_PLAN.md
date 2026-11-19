# Broader PCA feature design: proposed starting point

**Later design decision:** the current chosen composite uses four inputs:
PIE, WS/48, BPM, and predicted next-season total minutes. See
`README_FOUR_STAT.md`. The raw-stat audit below is retained as exploratory
background, rather than the active model specification.

## Recommendation

Start the broader design with **14 inputs** representing observed per-game
production and opportunity. This is a practical, available feature definition,
not a claim that 14 is a universally optimal number. Interpret the components
before defining a final quality score or training a new target.

| Number | Feature | Source / definition |
|---:|---|---|
| 1 | Two-point field goals made per game | `x2p_per_game` |
| 2 | Two-point field goals missed per game | `x2pa_per_game - x2p_per_game` |
| 3 | Three-point field goals made per game | `x3p_per_game` |
| 4 | Three-point field goals missed per game | `x3pa_per_game - x3p_per_game` |
| 5 | Free throws made per game | `ft_per_game` |
| 6 | Free throws missed per game | `fta_per_game - ft_per_game` |
| 7 | Offensive rebounds per game | `orb_per_game` |
| 8 | Defensive rebounds per game | `drb_per_game` |
| 9 | Assists per game | `ast_per_game` |
| 10 | Steals per game | `stl_per_game` |
| 11 | Blocks per game | `blk_per_game` |
| 12 | Turnovers per game | `tov_per_game` |
| 13 | Personal fouls per game | `pf_per_game` |
| 14 | Minutes per game | `mp_per_game` |

Made/missed shooting categories distinguish productive outcomes from missed
opportunities without adding both makes and their attempt totals. Do not also
add points and total field goals: points are already determined by 2PM, 3PM,
and FTM, and total field goals by 2PM and 3PM. Similarly, total rebounds repeat
information in offensive and defensive rebounds. Published per-game rounding
can make the numerical identities approximate, but the conceptual overlap
remains.

Salary and contract terms stay in the valuation stage. Age and position remain
useful prediction/context inputs, rather than defining how good a performance
was. Games played and total minutes should be forecast or analyzed separately
for availability and season-long contribution. A per-game composite alone
does not represent total season value.

## Training-only audit

`audit_pca_features.py` matched the original training examples to their actual
next-season Basketball-Reference records by player ID and season. It compared
all candidates on the same 7,888 training examples with defined rate targets
and finite raw statistics. No development or test scores were used.

| Candidate | Inputs | First component's variance | Components to reach 80% | Components to reach 90% |
|---|---:|---:|---:|---:|
| Existing PIE / WS48 / BPM | 3 | 86.20% | 1 | 2 |
| Basic box score | 6 | 59.60% | 2 | 4 |
| Raw shooting and box score | 12 | 51.71% | 3 | 5 |
| Raw plus fouls and minutes/game | 14 | 53.60% | 3 | 6 |
| Raw plus fouls, minutes/game, games | 15 | 52.00% | 4 | 7 |
| 14 raw variables plus three advanced metrics | 17 | 50.10% | 4 | 7 |

Explained variance is a description of retained information, not a quality or
prediction-accuracy score. More varied inputs can naturally require more
components. The 80% threshold is an exploratory retention reference, not a
proof of an optimal number of components.

The audit used 20 row-level bootstrap resamples to check first-component
loading stability. The first axis was stable across these resamples, but that
does not validate it as a basketball-quality score. Repeated player-seasons
also mean this is not a player-clustered confidence analysis.

Two training rows were excluded to make the raw and advanced candidates
comparable. Their undefined advanced rates need not automatically exclude
them from a future raw-stat-only definition.

## Important interpretation problem

The 14-feature first component had positive loadings for every included
variable, including misses, turnovers, and fouls. Examples:

- Minutes/game: +0.3371.
- Two-point makes/game: +0.3354.
- Turnovers/game: +0.3258.
- Two-point misses/game: +0.3224.

This is consistent with a dominant **volume/opportunity** dimension: players
who play and handle the ball more accumulate both positive and negative
events. It is not a scoring rule under which a turnover improves basketball
quality. Simply reversing a harmful variable's sign before PCA does not solve
this: PCA can reverse the corresponding loading while preserving the same
axis.

Therefore, do not replace the existing performance score with raw PC1 and
label higher universally better. A useful next design step is to inspect and
interpret the first few components, with three as an initial candidate for
this 14-variable set. Any final one-dimensional quality index needs an
explicit basketball interpretation and validation; retaining multiple
components does not itself define a good-to-bad ordering.

## Next implementation decisions

1. Inspect the first three components and representative player profiles.
2. Decide whether the desired output is a production profile, a single quality
   score, or separate effectiveness/opportunity forecasts.
3. Define the target using actual training statistics and freeze all fitting
   transformations before development evaluation.
4. Train the corresponding next-season model(s) using prior-season inputs.
   The number of PCA inputs does not require the same number of networks.
5. Evaluate on development data with a relevant baseline. Different target
   definitions cannot be ranked merely by their raw RMSE.

The previously evaluated test split remains historical evaluated data. A new
final confirmation of a redesigned method requires a fresh evaluation design.

## Audit artifacts

```text
results/training/pca_feature_audit/20261109_202345_984114/
```

- `candidate_summary.csv`: all six feature-set comparisons.
- `pc1_loadings.csv`: first-component coefficients and scaling parameters.
- `feature_correlations.csv`: training correlations.
- `feature_definitions.json`: definitions, cohort counts, and source hash.
