# Phase B protocol: local detector per engine

Date: 2026-10-18. Status: **FROZEN 2026-10-18.**

- Manifest: `examples/study_manifests/phase-b-local-detector.json`, canonical
  SHA-256 `73aef5e6…febd`. It was generated on the windows-2022 runner during
  the passing check.
- Source: `examples/phase_b_local_detector.py`, `7f2da249…5af6`.
- Fixture check passed on Actions run 39113988219
  (`examples/results/2026-10-18-phase-b-local-detector-check/`).
- Execution and budget: amendment B1 (§11).

Plan: `2026-10-17-detection-method-test-plan.md` §6 Phase B, decisions D1–D7.
Inputs: Phase A results (`examples/results/2026-10-17-phase-a-linear-reference/`)
and Phase A′ results (`examples/results/2026-10-17-phase-a-prime-small-sample/`,
prerelease `phase-a-prime-small-sample-2026-10-17-39105797558`), both in the
phase-2 checkout.

## 1. Why this phase exists

The linear reference is settled:

- OLS with HC4 standard errors and a t(n − k) reference is the linear default.
  It is validated from N ≥ 275 and valid at N = 1,000 (Phase A′).

What is not settled is inference with flexible engines, which use the
cross-fitted GCM and PCM statistics.

- **GCM is anti-conservative at small N even with perfect linear nuisances.**
  Phase A's M4 (cross-fitted GCM, OLS nuisances, K = 5) had pooled null rates
  of .081 at N = 100 (p = 10), .064 at 200, .058 at 385, .055 at 500 and .053 at
  1,000. It passed only from N ≥ 1,000 (criterion A6).
- **Flexible nuisances add error.** The informal LightGBM check (plan §4) had
  null rates of .10–.13 at N = 1,000 on the correlated nulls.
- **GCM is blind to symmetric effects** (U-shapes, pure interactions), so PCM is
  required for flexible engines.

The question:

> For each flexible engine, at which N are its p-values trustworthy, how much
> power does it give up against the HC4 reference on linear effects, and what
> does it gain on nonlinear ones?

### What this phase decides

| Outcome | Consequence |
|---|---|
| An engine passes C1, C3 and C4 from some ladder N₀ ≥ 1,000, and C2 | It enters the supported-N table as "inference supported for N ≥ N₀". It can be offered in Phase C at those N. |
| An engine fails at every N on the ladder | "Not supported for inference" at N ≤ 5,000. Phase D may test larger N. It can still be used for prediction and descriptives. |
| A small-sample correction qualifies (§2.4) and an engine passes the small-N cells with it | The supported N for that engine extends below 1,000, down to its lowest passing small-N step. |
| No correction qualifies | Flexible-engine inference stays limited to N ≥ 1,000. Below that, OPEN uses OLS + HC4. |
| The HC4 reference itself fails C1 in Phase B's linear cells | Contradicts Phase A′. Stop, treat as a defect, and investigate (§7.6). |

Production defaults change only after Phase E confirmation.

### Out of scope

- Network-level error control and recursion (Phase C).
- N above 5,000 and effects below r = .05 (Phase D).
- Missing data (Phase M), binary outcomes, clustered data, p ≥ n.
- XGBoost, CatBoost, PLS, SVM, GAM. They go through this same conformance
  suite later, each in its own protocol, after the engine registry exists
  (plan §5).

## 2. Methods

Every dataset is analyzed by every method (a paired design). Each method gives
a two-sided p-value for every candidate j, for H0(j): E[T | Xj, C−j] = E[T | C−j].

### 2.1 Linear reference

| ID | Method | Role |
|---|---|---|
| R1 | OLS + HC4, t(n − k) | **Reference.** Phase A′ default. Paired comparator for power and calibration |
| R2 | Cross-fitted GCM with OLS nuisances, K = 5, normal reference | Bridge: separates problems of the GCM statistic from problems of the engine (Phase A's M4) |

### 2.2 Flexible engines (nuisance configurations, frozen here)

No grid search. Each engine uses one fixed, regularized configuration. Boosters
stop early on an inner 20% validation split of each training set. *(Owner
review Q1: fixed configurations. An inner-CV grid search costs about 12× more
fits.)*

| ID | Engine | Configuration |
|---|---|---|
| F1 | Random forest (scikit-learn) | 100 trees; `min_samples_leaf = max(5, ⌈.005·n_train⌉)`; `max_features = .5`; `n_jobs = 1` |
| F2 | Gradient boosting (scikit-learn `GradientBoostingRegressor`) | `max_depth = 2`, `learning_rate = .05`, `subsample = .8`, up to 1,000 trees, `n_iter_no_change = 20`, `validation_fraction = .2` |
| F3 | LightGBM | `num_leaves = 15`, `learning_rate = .05`, `min_child_samples = max(20, ⌈.01·n_train⌉)`, `bagging_fraction = .8` (freq 1), `lambda_l2 = 1`, up to 1,000 rounds, early stopping patience 50; `num_threads = 1`, `deterministic = true` |
| F4 | LightGBM, linear trees | F3 with `linear_tree = true`, `linear_lambda = 1`, `num_leaves = 7` |

- These replace the repository grids for Phase B only. In particular, the
  LightGBM default `min_child_samples = 5` is not used (plan §6, Phase B).
- Every engine seed is derived deterministically from the dataset seed, the
  method ID, the fold and the fit role (§4).

### 2.3 Test statistics for flexible engines

**GCM** (Shah & Peters, 2020), cross-fitted, K = 5, as plan §3.2:

- For each fold k and candidate j, fit f̂ ≈ E[T | C−j] and ĝ ≈ E[Xj | C−j] on
  the other folds.
- Compute residuals on fold k and set Rᵢ = eTᵢ·eXᵢ over all N rows.
- Z = √N·mean(R)/sd(R), referred to N(0, 1).

**PCM** (Lundborg, Kim, Shah & Samworth, 2022), unweighted (the variance
function is fixed at 1). For each of the two halves A and B, and each j:

1. On A: fit ĝ ≈ E[T | C] (once per half, shared across j). Then fit
   m̂ ≈ E[ĝ(X) | C−j] and set ĥⱼ = ĝ − m̂.
2. On B: fit E[T | C−j] and E[ĥⱼ | C−j]. Form the GCM statistic of T against
   ĥⱼ on B's rows and take its one-sided p-value: positive covariance is the
   alternative, as in the paper.
   *(Clarified at implementation: these two regressions are cross-fitted
   within B, using B's rows of the same K-fold labels. In-sample tree fits
   would shrink the residuals and bias the statistic.)*
3. Swap A and B. The PCM p-value is p_PCM = min(1, 2·min(p_A, p_B)).

**Reported p-value for flexible engines:**
p = min(1, 2·min(p_GCM, p_PCM)), a Bonferroni combination over the two
statistics. All criteria apply to this combined p. GCM and PCM are also
reported separately.

**Descriptive alternative (owner review Q3):** the Cauchy combination
(Liu & Xie, 2020) of p_GCM and p_PCM with equal weights,
p_C = ½ − arctan(mean(tan((½ − pᵢ)π)))/π. It is computed from the same
p-values at no extra cost. Its null rates, BH FDR and power are reported beside
Bonferroni's, but don't affect any criterion. They are evidence for a later,
separately frozen switch.

Fit count per dataset at p = 5 and K = 5: GCM 2·K·p = 50 fits on 80% of N.
PCM needs 2·(1 + p·(1 + 2K)) = 112 fits, on 40–50% of N. Small-N cells add the
K = 10 fits for G1/G2: 100 GCM and 212 PCM.

### 2.4 Small-sample correction screen (GCM below N = 1,000)

Flexible-engine inference is limited to N ≥ 1,000 **unless a correction
qualifies here**. Candidates, all computed for R2 (OLS nuisances, where the
truth is exact) and for each flexible engine in the small-N cells:

| ID | Variant | Rationale |
|---|---|---|
| G0 | K = 5, normal reference (the Phase A M4 form) | Baseline |
| G1 | K = 10, normal reference | Larger training folds, so less nuisance error per fold |
| G2 | K = 10, fold-cluster-robust variance with a t(K − 1) = t(9) reference | Folds share nuisance-fit error, which the i.i.d. sd(R) ignores. Clustering by fold captures it, at a cost in power |

*(Owner review Q4: G0–G2, no others.)*

**Qualification (mechanical):**

1. A candidate qualifies from N₀ if R2 with that candidate passes C1 (§7.2) in
   every screen cell at N ≥ N₀ (§3.3).
2. Its paired power loss against R1 must also be ≤ .05 + 2.58·SE at ρp = .10
   and N = N₀.
3. If several candidates qualify, pick the one with the smallest N₀. On ties,
   take the higher mean power, then the order G0, G1, G2.
4. The chosen candidate (if any) is the one applied to flexible engines in the
   small-N cells. Those cells then count for criteria; otherwise they are
   descriptive.
5. PCM uses the same K and variance rule as the chosen GCM candidate. It isn't
   screened separately: its small-N calibration is judged through the combined
   p.

## 3. Data-generating designs

### 3.1 Candidate structure (all cells, p = 5)

Draw (X1, X0, Z1, Z2, M) jointly Gaussian, all with unit variance:

- corr(X1, Z1) = .6: a strongly correlated null;
- corr(X1, X0) = .3: a weakly correlated null;
- Z2 and M independent of everything.

T = β·s(X1, M) + ε, with ε ~ N(0, σ²) unless stated. Roles:

| Variable | Role |
|---|---|
| X1 | Signal (null when r = 0) |
| Z1, X0, Z2 | Null in every shape except E (see below) |
| M | Null, except in shape I, where it is a moderator (signal) |

Each null is exact: T ⟂ Z1, X0, Z2 | X1 by construction. Z1 and X0 are
marginally associated with T, which is the case where nuisance errors leak
into false rejections (plan §4).

### 3.2 Shapes (s standardized to mean 0 and variance 1 under X1 ~ N(0, 1))

| ID | Shape | s(·) | Why |
|---|---|---|---|
| L | Linear | x | Efficiency against HC4 |
| Mo | Monotone nonlinear | std(tanh(1.5x)) | Where a flexible engine should match or beat OLS |
| U | U-shape | (x² − 1)/√2 | GCM and OLS are blind to it; PCM should find it |
| I | Interaction | x·m (M is the moderator) | Pure interaction with a symmetric moderator |
| E | Linear, measurement error | T depends on the latent X1*; observed X1 = √.8·X1* + √.2·e | Reliability .8 |

In shape E, Z1 and X0 carry information about X1* beyond the noisy X1. So they
are **not** null under the estimand. Their rejection rates are reported as
"measurement-error leakage" (role `measurement`, as in Phase A) and excluded
from the null criteria.

**Effect size r:** the population partial correlation ratio of X1:

> r² = β²·V / (β²·V + σ²), where V = Var(s − E[s | C−1]).

V is computed from a 10⁶-row check draw. For L this equals ρp². Realized values
are reported, not assumed. In shape I the target is X1, and M's partial ratio
is reported.

**Heteroskedastic null:** shape L with r = 0 and
Var(ε | X) ∝ 1 + Z2² + X0²: conditional variance that depends on candidates
under an exact null.

### 3.3 Grid

| Block | N | Cells | Reps | Datasets |
|---|---|---|---:|---:|
| B2 criterion: null (L, r = 0) | 1,000; 2,000; 5,000 | 3 | 1,000 | 3,000 |
| B2 criterion: heteroskedastic null | 1,000; 2,000; 5,000 | 3 | 500 | 1,500 |
| B2 criterion: signal, 5 shapes × r ∈ {.05, .10} | 1,000; 2,000; 5,000 | 30 | 400 | 12,000 |
| Small-N: null (L, r = 0) | 300; 500 | 2 | 1,000 | 2,000 |
| Small-N: signal, 5 shapes × r ∈ {.10, .20} | 300; 500 | 20 | 400 | 8,000 |
| Correction screen (R1, R2 and G0–G2 only), p ∈ {5, 10} × ρp ∈ {0, .10}, shape L | 100; 200; 300; 500; 1,000 | 20 | 2,000 (null) / 1,000 (signal) | 30,000 |
| **Total** | | **78** | | **56,500** |

- Screen cells with p = 10 add five independent nulls to §3.1. The screen runs
  OLS methods only (cheap), so it can use more replications.
- Flexible engines run only in the B2 and small-N blocks (36,500 datasets
  × 4 configurations).
- In small-N cells, flexible engines compute G0, G1 and G2. In B2 cells they
  use K = 5 (G0) only.

## 4. Replications, seeds and compute

**Seeds (fresh block):**

- Dataset seed = 3,740,000,000 + 10,000·cell_index + rep.
- Fixtures: 3,749,000,000–3,749,999,999. Defect reruns:
  3,750,000,000–3,759,999,999.
- One `numpy.random.default_rng(seed)` per dataset, drawn in this order:
  1. candidates;
  2. latent or measurement error;
  3. ε;
  4. fold assignments (K = 5, then K = 10, then the PCM half split).
- Engine seed = the first 31 bits of SHA-256(dataset seed, method ID, K, fold,
  fit role).
- **Seed audit** as in Phases A and A′: the range 3,740,000,000–3,759,999,999
  against every manifest, result and script, including both earlier phases'
  outputs. It must report zero overlaps before freezing.

**Stage B0, timing probe (D7, ≤ 300 runner-minutes, before freezing):**

- One Actions job per configuration (R1, R2, F1–F4), on the same runner image
  and pins as the study.
- Each job times 10 datasets each at N = 300, 1,000 and 5,000 (p = 5).
- The probe also times one N = 50,000, p = 15 target with GCM + PCM, which is
  the probe D7 asked for for Phase D.
- Outputs: per-cell cost for the chunk plan, and the projected total.

**Rough projection (to be replaced by B0):**

- Local single-thread fits: RF ≈ 0.20 s (N = 1,000) and 0.67 s (N = 5,000) at
  200 trees; GBM with early stopping ≈ 0.02–0.13 s. LightGBM is not installed
  locally and is assumed to be similar to GBM.
- With about 82 fits per dataset, a 1.5× runner slowdown and 4 worker processes
  per runner, this gives about **4,000–6,000 runner-minutes**, about 4–6 hours
  of wall time at 20 parallel jobs.

**Budget:**

- Ceiling: **15,000 runner-minutes** (D7), approved before dispatch.
  *Superseded by amendment B1 (§11): about 45,000 runner-minutes.*
- If B0 projects more, cuts are applied before freezing, in this order:
  1. signal reps 400 → 300;
  2. drop F4;
  3. drop N = 2,000;
  4. RF uses 50 trees.
- No changes are made mid-run.

## 5. Software and checks

**Code:**

- Script `examples/phase_b_local_detector.py`.
- Actions harness `examples/phase_b_actions.py`.
- Independent `examples/results/<date>-phase-b-local-detector/review.py`.
- Pins: Python 3.11.9, numpy 1.26.4, scipy 1.11.2, pandas 2.2.3,
  statsmodels 0.14.4, scikit-learn 1.3.0, lightgbm 4.6.0 (the earlier pilots'
  pins). Every process uses one BLAS/OpenMP thread.
- Imports Phase A′'s HC4 (verified by its frozen source SHA-256 `8329c91e…`)
  and Phase A's helpers (`d621a6e5…`).

**`--check` mode (fixtures only):**

1. R1 equals Phase A′'s HC4 on Phase A′ fixtures, to 1e-12.
2. R2 and G0 reproduce Phase A's M4 on its fixtures to 1e-10. G1/G2 match a
   separately written loop implementation.
3. GCM and PCM for every flexible engine match a separately written loop
   implementation (independent fold and half bookkeeping) to 1e-10 on
   20 fixtures each.
4. Determinism: running the same fixture twice in separate processes gives
   identical p-values for every engine (conformance (e)).
5. Population check draw (10⁶ rows per shape): realized r, the partial
   correlation of each null role (within 4 Monte Carlo SE of 0), and the
   measurement-error leakage of Z1/X0 in shape E.
6. Seed audit passes; manifest deterministic; run guard refuses existing
   output.
7. Smoke run: 6 cells × 20 reps on fixture seeds, every output path, with
   timing.

**On Actions** (lessons from Phase A′):

- The `prepare` job reruns checks 1–5 on the runner.
- Manifest regeneration is compared with float tolerance (A1.1).
- The frozen module is imported under its real name.
- The release target is the full commit SHA.
- BH on grid-valued p-values uses exact integer arithmetic.

## 6. Endpoints

| ID | Endpoint |
|---|---|
| E1 | Null rejection rate per cell × method × null role, and the pooled per-N rate per method × role, with dataset-clustered SE |
| E2 | Power for X1 (and M in shape I) per cell × method, with the paired difference from R1 |
| E3 | BH realized FDR and power at q = .05 and .20 within each dataset's family, per method |
| E4 | GCM-only and PCM-only rates beside the combined p; Cauchy-combination null rates, BH FDR and power (descriptive, Q3) |
| E5 | Measurement-error leakage (shape E, Z1/X0) per method |
| E6 | Runtime per method × N; early-stopping round counts; share of fits that hit the round cap |
| E7 | Correction screen: G0/G1/G2 null rates and power for R2 and each flexible engine |

## 7. Acceptance criteria

### 7.1 Changes from Phase A′

Prespecified before any Phase B data exist.

- **Multiplicity allowance on per-cell checks** (Phase A′ caveat). Each
  criterion family has three parts:
  - a pooled per-N test, which catches small systematic deviations;
  - a count allowance on per-cell failures, which accepts the chance failures a
    perfect method makes;
  - a gross bound, which stops one real local failure hiding inside the
    allowance.
- **Calibration is one-sided for support.** Too-high null rates fail.
  Too-low rates (conservative) are reported and reflected in power, but don't
  withdraw support.
- **A practical margin of .005** on calibration and FDR, so trivial excesses
  that are invisible to users don't fail an engine.

### 7.2 Criteria (per method; R1 is checked under the same rules, §7.6)

Notation: z₉₉ = 2.58 (one-sided level .005) and z_g = 3.89. SE is the
dataset-clustered Monte Carlo SE. a(K, π) is the smallest a with
P(Binomial(K, π) > a) ≤ .01.

**C1. Calibration** (null roles in all criterion cells; the heteroskedastic
null cells count):

- (a) **Pooled:** for each N and each null role, the rate pooled over that N's
  cells is ≤ .055 + z₉₉·SE_pooled.
- (b) **Count:** a cell fails if any null role's rate is > .055 + z₉₉·SE. The
  number of failing cells at that N is ≤ a(K_N, .02). Here .02 bounds the
  per-cell chance failure over at most 4 roles.
- (c) **Gross:** no role in any cell has a rate > .06 + z_g·SE.

**C2. Linear efficiency** (engine-wide; shape L, N = 5,000, r = .05 and .10):
power ≥ .90·power(R1) − z₉₉·SE_paired.

**C3. Robustness:** C1 holds in the heteroskedastic null cells and on the
shape-U and shape-I null roles at every N on the ladder. This is already
inside C1(a–c), listed so it is reported separately.

**C4. FDR** (BH on the combined p, every criterion cell, q ∈ {.05, .20}):

- (a) **Pooled:** realized FDR pooled per N is ≤ q + .005 + z₉₉·SE_pooled.
- (b) **Count:** the number of failing cells (FDR > q + .005 + z₉₉·SE) is
  ≤ a(K_N, .01).
- (c) **Gross:** no cell has FDR > q + .01 + z_g·SE.

**Reported, not gating:**

- Nonlinear advantage: power of the combined p against R1 on Mo, U and I.
- The Cauchy combination, against Bonferroni.
- GCM against PCM shares.
- Measurement-error leakage.
- Runtime.

These feed OPEN's "when to use a flexible engine" guidance.

### 7.3 Ladder and supported N

- Ladder N ∈ {1,000, 2,000, 5,000}. It extends down to {500, 300} only if a
  correction qualified (§2.4).
- A flexible configuration's **supported minimum N** is the smallest ladder N
  at and above which C1, C3 and C4 pass. It is "None" if it fails at 5,000.
  C2 must also pass, or the configuration is "None" overall.
- **Per engine:** if both LightGBM configurations are supported, the table
  lists the one with the smaller supported N. On ties it takes the higher mean
  linear power, then F3. The other is reported.

### 7.4 Output: supported-N table

One row per engine, with these columns:

- supported minimum N (or "None");
- configuration;
- GCM correction used (if below 1,000);
- linear efficiency;
- nonlinear advantage (Mo/U/I);
- runtime per target at each N.

This is the evidence D5 needs to enable flexible engines by N.

### 7.5 What does not change support

- Conservativeness.
- Low power on U/I for GCM alone (expected by design).
- Measurement-error leakage: it is a property of the estimand, not of the
  engine, and is disclosed in OPEN's output text.

### 7.6 Defects versus failures

- **R1 must pass C1 and C4** in the shape-L, -Mo and heteroskedastic cells.
  Linear OLS is misspecified on U and I, but those nulls are still exact. If R1
  fails, it's a simulation or harness defect: fix it and rerun on the
  3,750,000,000 block, with the failed output preserved.
- **R2 must reproduce Phase A's M4 pattern** at N = 1,000 (within C1).
- Statistical failures of flexible configurations are reported, and resolved
  only by §7.3. No configuration is retuned after seeing results; a retuned
  engine is a new frozen study.

## 8. Analytic reference values (R1, shape L, two-sided .05, p = 5)

| N | r = .05 | r = .10 | r = .20 |
|---:|---:|---:|---:|
| 300 | — | .404 | .937 |
| 500 | — | .606 | .995 |
| 1,000 | .351 | .886 | — |
| 2,000 | .608 | .994 | — |
| 5,000 | .943 | 1.000 | — |

Noncentral t with df = N − 6. The manifest stores the exact values.

## 9. Outputs and preservation

As in Phase A′:

- `protocol.json`, `seed-audit.json`, `check.json`, `b0-timing.json`;
- `pvalues/*.npz` (a prerelease asset; inventory and hashes committed);
- `summary.csv`, `criteria.json`, `supported-n.json`;
- `review.py`, `verification.json`, `interpretation.md`.

**The independent review:**

- recomputes every summary and criterion from saved p-values;
- re-derives the correction choice and the supported-N table;
- regenerates 1% of datasets *(amended in B1: see §11)* (stratified by N and shape) with the loop
  implementations of GCM and PCM, matching to 1e-10 on the same runner image.

**Actions layout** (as Phase A′, amendment A1):

- `register` → `prepare` → `shard`, a matrix of up to 60 shards with
  max-parallel 20, 4 workers per shard and a 5-hour deadline →
  `aggregate` + `review` → `publish`.
- A `continue_from` dispatch input recomputes only the missing chunks of an
  earlier run, under the same frozen manifest. Chunks are deterministic, so a
  continued run is identical to a single run.

## 10. Owner review (resolved 2026-10-18)

| # | Question | Resolution |
|---|---|---|
| Q1 | Tuning | Fixed regularized configurations with early stopping; no grid search |
| Q2 | Configurations | 4: F1 RF, F2 GBM, F3 LightGBM, F4 LightGBM-linear |
| Q3 | Combining GCM and PCM | Bonferroni (2·min) is primary and gates every criterion. The Cauchy combination is computed and reported descriptively (§2.3) |
| Q4 | Small-N corrections | G0–G2 as drafted |
| Q5 | Allowance settings | Margin .005, count level .01, gross bound 3.89 SE |
| Q6 | Replications | 1,000 per null cell, 400 per signal cell |

Steps:

1. ~~Implement the script and run `--check`.~~ Done 2026-10-18. Every part
   passed on Actions run 39113988219.
2. ~~B0 and the budget.~~ Done by amendment B1.
3. ~~Freeze the manifest.~~ Done: `73aef5e6…febd`.
4. Dispatch the workflow `.github/workflows/phase-b-local-detector.yml`.

## 11. Amendment B1: budget and execution (2026-10-18, owner decision)

Made before any study data exist. Nothing in the design, seeds or criteria
changes.

**B0 and the budget:**

- The check's `runtime` part ran on the study's runner image (windows-2022,
  4 concurrent workers). It measured seconds per dataset of:
  - 70 at N = 1,000, 124 at N = 2,000 and 238 at N = 5,000;
  - 105 at N = 300 and 97 at N = 500 (small-N cells, including the K = 10
    fits).
  - The screen cells are negligible.
- Total: about 3.39 × 10⁶ thread-seconds, or **about 14,100 runner-minutes**
  for the full grid. This replaces the draft's rough projection of 4,000–6,000.
- **Owner decision:** keep the full grid, with no §4 cuts, and raise the
  ceiling to **about 45,000 runner-minutes** (3× the measured figure, owner
  rule). The repository is public, so standard runners cost nothing.
- The check's runtime part serves as B0 for Phase B. The N = 50,000, p = 15
  probe that D7 asked for belongs to Phase D and moves to that protocol.

**Execution** (harness `examples/phase_b_actions.py`, an execution detail and
not design):

- **Chunks:** replications are split into chunks of 5 (N = 5,000), 10
  (N = 2,000 and small-N cells), 15 (N = 1,000) or 250 (screen cells). Each
  chunk takes about 20 measured minutes or less, and is run by the frozen
  `run_chunk()`.
- **Shards:** chunks are spread over **125 shards** by longest-processing-time
  assignment, each about 116 measured minutes at 4 workers. Each job's limit
  is 360 minutes, about 3× its expected time. No new chunks start after
  280 minutes, which leaves 80 minutes, more than 3× the longest chunk.
  Shards run 20 at a time, so expected wall time is about 12 hours; allowing
  3×, up to about 36 hours.
- **Continuation:** an unfinished chunk is never estimated or dropped.
  `aggregate` refuses to compute criteria until every chunk is present. A new
  dispatch with `continue_from = <run id>` verifies the earlier run's chunk
  files (seeds, shapes, finite p in [0, 1]), reuses them and computes only
  the rest.
- **Cross-platform checks:** `prepare` repeats the HC4 identity, GCM identity,
  determinism and run-guard checks on the runner. The full check already
  passed there for this exact source.

**Review regeneration** (replaces the last bullet of §9's review):

- 1% of datasets, stratified by cell (565), are regenerated with the frozen
  `generate` and `analyze` and must match the stored p-values to 1e-10.
- A further 20 datasets (N = 1,000 signal cells) are recomputed with the
  separately written loop implementations of GCM and PCM for all four engines,
  also to 1e-10.
- Loop equivalence on fixtures was already shown by the check (max difference
  4.4e-16).
- Regeneration runs as 8 parallel Actions jobs.
- `review.py` doesn't import the study script. It recomputes every per-cell
  statistic with its own BH, Bonferroni and Cauchy code, and re-derives the
  correction choice, ladder, C2 and the supported-N table with its own
  implementation of §7.
