# Interchangeable engines and comparable predictive effect sizes

> **Exploratory mode** supports only OLS, Elastic-Net and LightGBM. Random
> forest and gradient boosting were dropped on 2026-10-19; see
> [exploratory-mode-engines.md](exploratory-mode-engines.md). This page
> describes the earlier subsample pipeline.

OPEN retains elastic net as its default. A fixed engine can instead be
`random_forest`, `gbm` (scikit-learn gradient boosting), or `lightgbm` (optional
dependency). One engine is used throughout a recursive construction. Comparing
engines is a separate fixed-target analysis, not automatic selection of a winner
on the same evaluation observations or averaging different networks.

## Shared quantities

- Cross-validated R² uses pooled held-out squared error and outcome variation
  on those same observation keys. Negative values are preserved; a constant
  outcome has undefined R².
- Unique ΔCV R² compares the full model with a refitted model excluding one
  predictor, on identical observations. Explicit joint removal sets can assess
  shared information. Ablations reuse full-model tuning parameters.
- Standardized first-order accumulated local effects (ALE) describe the fitted
  prediction function. The signed contrast is ALE at the predictor's 75th
  percentile minus ALE at its 25th percentile, divided by a common outcome SD.
  It is neither a beta coefficient nor Cohen's d, and is not causal.
- The entire ALE curve accompanies the contrast: opposite slopes or interactions
  can be hidden by a small endpoint contrast. Shared quantile boundaries include
  both contrast endpoints. Local differences use held-out reference observations
  from each outer fold; other covariates remain at their observed values.
- Percentile points at P0, P10 … P100 show centered ALE levels in outcome SD
  units. Signed adjacent contrasts show P0→P10 … P90→P100 changes. They describe
  different parts of the same fitted curve, not separate models, individual-person
  scatterplots, or absolute expected outcome scores. Percentiles refer to the
  predictor distribution; equal percentile steps need not span equal score distances.
- Points interpolate supported piecewise-linear estimation bins. Adjacent changes
  integrate their local increments and never bridge an unsupported bin. A supported
  interval outside a gap can remain available while the centered curve is unavailable.
  Tied endpoints are explicitly unavailable, not a flat relationship. P0/P100 are
  observed extremes and can be sensitive to outliers. No fold-derived confidence
  interval or precision guarantee is attached to these descriptive points.
- **Overall curve magnitude** (`standardized_curve_sd`) is the empirical SD of
  ALE levels evaluated at observed predictor values, divided by the common outcome
  SD. Levels are interpolated along the supported binned curve; SD uses ddof=0.
  This unsigned typical spread is positive for a U shape even when Q25→Q75 is
  zero. A flat supported curve has magnitude zero. `curve_sd` reports raw outcome
  units. No universal small/medium/large labels are assigned.
- Aggregate magnitude is calculated from the combined curve on the common
  reference predictor distribution, not by averaging fold magnitudes. Opposing
  fold shapes can therefore cancel in the combined curve; fold-level magnitudes
  and curves remain available for inspection. Magnitude has an independent
  `magnitude_status`: tied Q25/Q75 can leave it available, but empty bins, constant
  predictors/outcomes or missing references leave it unavailable.
- Reference endpoints and outcome SD are fixed across engines using the supplied
  target's observed rows. They describe this reference population, and are not
  used for tuning or fitting. Missing predictor values are excluded from that
  predictor's ALE reference calculation, but model imputation remains fold-local.
- Empty ALE bins are marked unsupported, never silently interpolated. A contrast
  is unavailable if any intervening bin lacks support in any outer fold. Curves
  outside the contrast can be partially unavailable. Constant predictors/outcomes
  are explicitly flagged. No fabricated zero or confidence interval is produced.
- Elastic net and the opt-in OLS reference can additionally report raw b and standardized beta from their actual
  fitted coefficients. Tree engines do not receive surrogate linear coefficients.

OLS (`learner.library: [ols]`) fits unpenalized least squares with an intercept
and the same fold-local imputation/scaling as Elastic Net. It has no tuning grid
and performs zero inner tuning fits. It does not select coefficients using
p-values. Its network is still pruned by the existing held-out gain, unique/shared
relevance and layer-consensus rules; OLS itself does not shrink coefficients.
This separates learner regularization from network selection. The default remains
Elastic Net. OLS predictions use sklearn's least-squares solution, including a
minimum-norm solution for rank-deficient designs; individual coefficients are not
uniquely identified in that situation. Adding this reference is implementation
support, not evidence that its recursive networks are already reliable.

## Comparison protocol

All engines share data, candidate sets, seeds, outer/inner folds, reference
endpoints, scale, and loss. Fold-local imputation/scaling occurs inside tuning.
Each engine has a documented finite grid: elastic net retains its historical
25 candidates. Tree engines use four candidates, except RF searches with one
numeric predictor, which use two: both `min_samples_leaf` settings with
`max_features=0.7`. At one predictor, the former feature fractions 0.7 and 1.0
both select that same feature. Keeping the first fraction preserves the original
tie order and selected parameter representation. Numeric imputation preserves
columns even when an inner training fold contains only missing values.

The effective grid is chosen independently for each tuned model. Thus a broad
single-predictor model can search two settings while its multivariable full model
searches four. Three-fold inner CV uses six singleton tuning fits instead of
twelve; outer prediction fits remain separate. Risk counts use the tuned clone's
actual search size, and the effect supplement exports the full model's effective
grid and size. Fit counts and grids are saved;
these are transparent development defaults, not an equal-budget benchmark.
Stochastic engines use explicit seeds and one worker. LightGBM imports lazily
and fails with an installation instruction when requested but unavailable.

The comparison exports a separate versioned JSON/CSV supplement with fold
parameters, raw curves, statuses, counts, dataset/config/split identity, package
versions and engine-specific metrics. Existing network bundle schemas and
historical results remain readable. Effects describe fixed local comparisons;
different engine networks can have different conditioning sets and must not be
treated as matched effects. Effect sizes are not summed or multiplied into
causal paths. No post-selection interval or universal small/medium/large cutoff
is claimed.

Magnitude summarizes the first-order curve, not unique predictive usefulness,
variance explained, or all interactions. An interaction can average to a flat
first-order curve while still helping prediction. Retain ΔCV R² and the curve
alongside this summary. Magnitude depends on the reference distribution and bin
resolution; it is not a new form of Cohen's d or a scientifically calibrated cutoff.

An optional [bounded pair-screening supplement](pair-interaction-screening.md)
examines joint patterns in the existing fitted models, without product features
or additional fits. It exports descriptive local mixed contrasts and a separate
exploratory junction view. This does not certify population interactions or
standalone main effects, and does not change recursive graph selection.

## Scientific boundaries

For corresponding rescaled predictions, unit changes do not change standardized
ALE or R². Refitting after a unit change can affect tuning, especially with
elastic-net's fixed penalty grid; training invariance is not claimed. Different populations,
measurement reliability, predictor distributions, or adjustment sets can still
change their values. ALE reduces extrapolation under correlation but does not
identify unique ownership of redundant information; joint ablation remains
necessary. Software verification is separate from scientific calibration.

References: [ALE, Apley and Zhu](https://arxiv.org/abs/1612.08468),
[random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html),
[LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html).

## Usage

Choose an engine for the existing local/recursive APIs through YAML:

```yaml
learner:
  mode: fixed
  library: [random_forest]  # elastic_net, gbm, or lightgbm also supported
  retune_ablation: false
```

Install the optional engine with `pip install ".[lightgbm]"` from the repository.
To compare the same local problem across engines:

```python
from open.config import load_config
from open.splits import make_split_plan
from open.effects import compare_target_engines, save_engine_comparison

config = load_config("configs/default.yaml")  # root/seed must suit your data
plan = make_split_plan(data, config)
comparison = compare_target_engines(
    data, config.analysis.root, candidates, config, plan,
    engines=("elastic_net", "random_forest", "gbm", "lightgbm"),
    joint_groups=(), bins=10, adaptive_bins=True,
)
save_engine_comparison(comparison, "results/engine-comparison")
```

`joint_groups` accepts explicit tuples of at least two candidate names. Joint
removal measures the total contribution of that set, including its members'
individual contributions; it does not by itself establish redundancy. Comparison
does not promote these sets into the network. Common reference endpoints are
shown as `low` and `high`, and outcome SD as `outcome_sd`. `effects.csv` contains
one row per engine/predictor, including unavailable statuses; `comparison.json`
also contains fold curves, bin support, parameters, joint scores and provenance.
Aggregate ALE contrasts and curves are weighted by nonmissing held-out reference
counts; any unsupported contributing fold withholds the aggregate contrast.
Beta coefficients use raw fitted slopes and reference predictor/outcome SDs.
No engine is selected as a winner by this API.

`percentile_points.csv` contains eleven rows per engine/predictor, with actual
predictor scores, raw and standardized centered ALE levels and support status.
`interval_contrasts.csv` contains ten rows, with percentile/score endpoints, raw
and standardized signed changes and interval-specific support status. The JSON
also retains these summaries for every outer fold. Aggregates require support in
all contributing folds and use nonmissing held-out reference counts as weights;
an unsupported fold is never silently dropped. A valid flat interval yields zero.

`bins` sets the nominal estimation resolution; Q25 and Q75 boundaries are added,
so the actual number of bins can differ. `adaptive_bins=False` is the API default
and preserves fixed-resolution behavior. With `adaptive_bins=True`, the largest
resolution up to `bins` with nonempty bins in every observed-target held-out fold
and nonconstant predictor is selected before engine fitting, using covariates
only. At least two nominal bins are attempted; impossible support remains explicit.
`requested_bins`, `estimation_bins` and the exact reference edges record the choice.
Nonempty bins are a minimum support check, not a guarantee of stable estimates.
Coarser bins yield interpolated decile points, not additional independently
estimated detail. All engines use the same selected resolution.

The demonstration workflow enables adaptation and defaults to `--bins 10`:

```text
python examples/run_engine_comparison.py --output NEW_DIRECTORY --bins 10
```

It reports both percentile curves and adjacent changes, while retaining Q25→Q75,
curve magnitude, unique/joint ΔCV R² and the original score-axis curves. These
supplements do not alter recursive selection or stability thresholds.
