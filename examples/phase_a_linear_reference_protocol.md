# Phase A protocol: linear-reference calibration

Date: 2026-10-17. Status: **FROZEN and RUN 2026-10-17. Verdict under the frozen
rules: FAIL.** A2 failed at N = 100, p = 10 (HC3 conservative). A1 is scoped to
N ≥ 385 and A6 to N ≥ 1,000; A3, A4 and A5 pass. Manifest
`examples/study_manifests/phase-a-linear-reference.json` (canonical SHA-256
`cd2f459b…5c107`). Results and interpretation are in
`examples/results/2026-10-17-phase-a-linear-reference/` (phase-2 checkout).
Phase B does not start until a Phase A′ protocol is approved. Draft: `2026-10-17-phase-a-prime-small-sample-protocol.md`
(option b: HC4, HC5 and wild-bootstrap HC3 at N = 100–385).

Parent plan: `2026-10-17-detection-method-test-plan.md` (§6 Phase A), decisions
D1–D7 approved 2026-10-17.

## 1. Purpose and what this phase decides

Phase A checks that the full-sample linear test behaves as theory predicts.
Every later phase uses it as its yardstick: Phase B efficiency targets, Phase C
power targets, and the "detectable effect" values in OPEN's reports.

| Outcome | Consequence |
|---|---|
| All primary criteria pass | OLS + HC3 adopted as the default inferential reference (D5). The analytic tables in §8 become official Phase B/C targets. |
| A criterion fails only below some N | The supported scope is set by the prespecified ladder in §7.3 (e.g. "HC3 inference supported for N ≥ 500"). It is not re-tuned. |
| A criterion fails at all N | Phase B does not start. Report the failure; a new frozen protocol is required. |
| Descriptive cells (Likert, missing data) | No pass/fail. Their results feed two new owner decisions: D8 (missing data for inference) and D9 (measurement-scale guidance). |

This phase does **not** test recursion, flexible engines, network FDR or
production code. It uses a research script only, like the archived noise
pilots.

## 2. Methods compared

All methods run on the same dataset within a cell (paired).

| ID | Method | Role |
|---|---|---|
| M1 | OLS, classical t-test per coefficient | Comparison; expected to fail under heteroskedasticity |
| **M2** | **OLS, HC3-robust t-test per coefficient** | **Primary reference** |
| M3 | GCM with OLS nuisance fits, in-sample | Checks GCM = partial-correlation identity |
| M4 | GCM with OLS nuisance fits, cross-fitted K = 5 | Bridge to Phase B; measures the cost of cross-fitting |
| M5 | HC3 Wald test of a predictor group (group cells only) | Group-first hypothesis (D3) |

Multiplicity, applied afterwards to saved p-values (no refits):

- BH within each dataset's family of p candidates, at q = .05 and q = .20.
- Group cells: hierarchical. Test the group with M5 at .05; if it rejects, test
  the members with M2 at .05 inside the group.

All tests are two-sided with unadjusted α = .05 unless stated.

## 3. Data-generating designs

### 3.1 Primary cells (Gaussian, homoskedastic)

Candidates X = (X1, …, Xp) are multivariate normal with unit variances and
correlation matrix Σ:

| Structure | Σ |
|---|---|
| S1 independent | Identity |
| S2 moderate | All pairs .3 |
| S3 overlap | X1 (true) and X2 (null) correlated .6; all other pairs .3 |

S3 is the hard case: a null predictor that substantially overlaps with the true
one. The tree signal study's false links came from this kind of predictor.

Target: T = β·X1 + σ·ε, with ε ~ N(0, 1) and Var(T) = 1. Only X1 has an effect;
X2…Xp are exact conditional nulls.

The effect size is the **population partial correlation** ρp of X1 with T given
the other candidates, so that analytic power is exact:

- v = 1 / (Σ⁻¹)₁₁, the variance of X1 given the others;
- β² = ρp² / (v(1 − ρp²) + ρp²);
- σ² = 1 − β²·(Σ)₁₁ = 1 − β².

Grid: N ∈ {100, 200, 385, 500, 1,000, 5,000} × structure ∈ {S1, S2, S3} ×
p ∈ {4, 10} × ρp ∈ {0, .05, .10, .20}. **144 cells.** *(N = 100 added at
owner review, Q2.)*

### 3.2 Robustness cells (error distributions)

Same as §3.1 with p = 4, structure ∈ {S2, S3}, ρp ∈ {0, .10},
N ∈ {100, 200, 500, 1,000, 5,000}, and ε replaced by:

| Error | Definition (standardized to unit variance) |
|---|---|
| Heteroskedastic | ε = z·√h / √E[h], with z ~ N(0, 1) and h = 1 + X1² + X2² |
| Skewed | (χ²₃ − 3) / √6 |
| Heavy-tailed | t₅ · √(3/5) |

3 errors × 5 N × 2 structures × 2 effects = **60 cells.**

### 3.2b Coarsened-target cells (single-item outcome)

*Added at owner review, Q5.*

- S2 design, p = 4. The latent T is generated as in §3.1, then the observed
  target is cut into 5 categories coded 1–5, under the same two threshold sets
  as §3.4. Predictors stay continuous.
- Grid: 2 threshold sets × N ∈ {200, 1,000, 5,000} × ρp ∈ {0, .10}.
  **12 cells.**
- The nulls stay exact for the OLS linear projection: with Gaussian X, Stein's
  lemma gives a zero projection coefficient on X2…X4 for any function of X1.
  A3 therefore applies to their type I error.
- X1's power is descriptive, since the analytic latent-scale value doesn't
  apply.

### 3.3 Group cells (redundant pair)

- p = 4 candidates: G1 and G2 correlated .8, plus two independent nulls.
- T = b·(G1 + G2) + σ·ε, Var(T) = 1.
- Effect: the group's population partial R² η² ∈ {0, .005, .01, .02}, with
  η² = 3.6·b², because the nulls are independent.
- N ∈ {200, 385, 500, 1,000, 5,000}. **20 cells.**

Endpoints for these cells:

- the group test's power against the noncentral F;
- how often each member is individually supported (expected to be low: the
  "shared, allocation unresolved" case);
- the hierarchical procedure's error rate.

### 3.4 Descriptive cells (no pass/fail)

**Likert-coarsened predictors.** The S1 and S3 latent designs are generated as
in §3.1. Each observed candidate is the latent value cut into 5 categories, coded 1–5, under
two threshold sets *(skewed set added at owner review, Q3)*:

- symmetric (−1.5, −.5, .5, 1.5): about 7 / 24 / 38 / 24 / 7%;
- skewed agreement-scale (−2.0, −1.3, −.5, .4): about 2 / 7 / 21 / 35 / 34%.

T stays continuous (a scale-score target). Grid: 2 threshold sets ×
N ∈ {200, 1,000, 5,000} × {S1, S3} × ρp ∈ {0, .10}. **24 cells.**

- In S3, X2 is null only given the *latent* X1. Given coarsened X1, X2 carries
  real information about T on the observed scale. Its rejection rate is
  therefore reported as **measurement-induced association**, not type I error.
- Type I error is evaluated only for predictors independent of X1: all
  predictors when ρp = 0, and X2…X4 in S1 when ρp = .10. In S3 every predictor
  correlates with X1 (.3 or .6), so at ρp = .10 all of them are
  measurement-induced. *(Corrected at implementation; the draft listed X3/X4
  in S3 as independent.)*

**Missing data, MCAR 10%.** Each candidate value is independently missing with
probability .10, under the S3 design. Two analyses: complete-case, and median
imputation (the current production default). Grid: N ∈ {500, 1,000, 5,000} ×
ρp ∈ {0, .10} × 2 analyses. **12 cells.**

- Median imputation is expected to inflate X2's rejection rate: residual
  confounding from the imputed X1. This is a planned diagnostic that informs D8.
  It is not a criterion.
- MAR mechanisms and multiple imputation (spec §30) are deferred to a separate
  missing-data phase in the parent plan *(owner review, Q4)*.

**Total: 272 cells.**

## 4. Replications and seeds

**Replications:**

- 5,000 per cell with ρp = 0 or η² = 0 (95 cells). Monte Carlo SE at 5%
  ≈ .0031; 99% band ±.0080. *(Raised from 2,000 at owner review, Q1.)*
- 1,000 per signal cell (177 cells). SE ≤ .016.
- About 652,000 datasets in total.

**Seed scheme:**

- Cells are enumerated in the frozen manifest's order, as `cell_index`
  0–271.
- Dataset seed = 3,700,000,000 + 10,000·cell_index + rep, with rep < 10,000.
- Maximum 3,702,710,999 (the last cell has 1,000 replications), below 2³².
- Reserved, never used for the run: fixture/smoke seeds 3,709,000,000–3,709,999,999;
  defect-rerun base 3,710,000,000 (§7.4).
- Generator: `numpy.random.default_rng(seed)`.
- Cross-fitting folds for M4 come from the same generator, after data
  generation.
- Cells are independent; no prefix pairing across N.

**Seed audit before freezing:**

- Confirm no overlap with any seed value or block in
  `examples/study_manifests/*.json` or the 2026-10-16/17 result protocols, in
  the same format as `lightgbm-effect-seed-audit.json`.
- Implemented in `--check`: scans every 10-digit integer starting with 3 in
  `examples/study_manifests/*.json`, `examples/results/**/*.json` and
  `examples/**/*.py` (including underscore literals), plus JSON
  `start`/`end_inclusive` blocks, over 3,700,000,000–3,719,999,999. Digit runs
  inside hex digests are excluded. **Result 2026-10-17: 2,318 files scanned,
  0 overlaps, 0 skipped.**

## 5. Software and environment

- Research script `examples/phase_a_linear_reference.py`, which is not a
  package API.
- Independent reviewer `examples/results/<date>-phase-a-linear-reference/review.py`.
- Pinned runtime: Python 3.11.9, numpy, scipy, and statsmodels 0.14.4, with
  versions and source SHA-256 recorded in the manifest.
- Primary computations (M1–M5) are implemented directly with numpy linear
  algebra for speed.
- Local only: 16 worker processes, one numerical thread each. Wall-clock cap
  60 minutes. **No Actions dispatch.**

**`--check` mode (run before the real run; fits only on small fixed fixtures):**

1. On 200 fixed fixtures, M1/M2/M5 p-values match statsmodels
   `OLS(...).fit()`, `fit(cov_type="HC3", use_t=True)` and
   `wald_test(use_f=True)`, to 1e-10.
2. M3 satisfies the Frisch–Waugh–Lovell identity Σ(eT·eX)/Σ(eX²) = OLS
   coefficient to 1e-9 relative. M4 folds are balanced and seed-reproducible.
   *(Corrected at implementation: GCM's Z isn't algebraically equal to the
   classical t, so the draft's "equals the t-statistic" check was wrong.)*
3. Generated populations: on one 10⁶-row draw per design, sample partial
   correlations and Var(T) match the specified values within
   max(.003, 4 Monte Carlo SE). *(Amended: a flat ±.003 is under 2 SE for the
   heavy-tailed variance, so it would fail by chance.)* Group R² and Likert
   category proportions within ±.003.
4. The seed audit passes, and the manifest is deterministic (two builds give
   the same SHA-256).
5. The run refuses an output directory already holding `pvalues/`,
   `summary.csv` or `criteria.json`. Check output goes to its own
   `…-phase-a-linear-reference-check/` directory.
6. A smoke run of 4 cells × 60 replications on fixture seeds exercises
   generation, all methods, the summary and criteria code. It is not
   evidence.

## 6. Endpoints

| ID | Endpoint | Unit |
|---|---|---|
| E1 | Rejection rate for each null hypothesis (unadjusted), with Clopper–Pearson 99% interval | Per cell × method × null predictor |
| E2 | Power for X1 versus analytic noncentral-t power (§8) | Per signal cell × method |
| E3 | BH realized FDR (mean false-discovery proportion; 0 when nothing is rejected) and power, at q = .05 and .20 | Per cell, M2 |
| E4 | Group test power versus noncentral F; member-level support; hierarchical error rate | Group cells |
| E5 | M4 versus M2: p-value correlation, decision agreement, power difference | Per cell |
| E6 | Runtime per method | Total and per cell |
| D1 | Likert: type I error on independent nulls; measurement-induced rate on X2 | Descriptive |
| D2 | Missing data: complete-case versus median imputation, X2 rejection rate and X1 power | Descriptive |

## 7. Acceptance criteria (primary method M2 unless stated)

### 7.1 Criteria

- **A1. Calibration, primary cells.** For each null hypothesis × cell, the
  rejection rate must lie within the 99% Monte Carlo band around .05:
  - ±2.58·√(.05·.95/R) for R replications, i.e. [.042, .058] at R = 5,000 and
    [.032, .068] at R = 1,000.
  - With K such hypothesis-cells, at most Q99 = the 99th percentile of
    Binomial(K, .01) may fall outside the band.
  - None may fall outside [.037, .063] at R = 5,000, or [.025, .075] at
    R = 1,000 (about 4 SE).
- **A2. Power, primary cells.** For each signal cell,
  |observed − analytic| ≤ 2.58·√(π(1 − π)/1,000) + .01, where π is the analytic
  power. The +.01 allows for the noncentral-t approximation with random
  predictors.
- **A3. Robustness.** Heteroskedastic, skewed, heavy-tailed and coarsened-target
  (§3.2b) cells meet A1 at
  N ≥ 500. At N = 100 and 200, rejection rates must be ≤ .075. M1 is reported but
  carries no criterion.
- **A4. BH.** In each cell, realized FDR ≤ q + 2.58·SE (SE estimated from the
  per-dataset FDP), for q = .05 and q = .20.
- **A5. Group test (M5).** The A1 analog when η² = 0. The A2 analog against
  noncentral-F power, with λ = N·η²/(1 − η²), df₁ = 2, df₂ = N − p − 1. The
  hierarchical procedure's family-wise error ≤ .05 + 2.58·SE at η² = 0.
- **A6. Cross-fitting (M4).** Meets A1 in primary cells, and its power is at
  least M2's power − .03 in every signal cell with N ≥ 385.

### 7.2 Checks with no criterion

- M3 should match M2's classical counterpart (M1) up to small-sample
  differences. This is reported, not gated.

### 7.3 Prespecified scope ladder

- If A1, A3 or A6 fails only at small N, the supported scope is the smallest N
  on the ladder {100, 200, 385, 500, 1,000, 5,000} at and above which every
  criterion passes.
- No other remedy (a different variance estimator, a df correction, a changed
  band) may be adopted from this phase's results. Any such change needs a new
  frozen protocol and fresh seeds.

### 7.4 Defects versus failures

- A **software defect** found after the run (wrong formula, seed bug) is fixed
  and regression-tested. The failed output is preserved, and the whole phase is
  rerun on the reserved block based at 3,710,000,000. *(Corrected: the draft's
  3,701,000,000 fell inside the run's own seed range.)*
- A **statistical failure** is reported as a failure and handled only through
  §7.3.

## 8. Analytic reference values

Two-sided α = .05, noncentral t, df = N − p − 1. These are the E2/A2 targets
for X1 in the primary cells.

| N | p = 4: ρp = .05 | .10 | .20 | p = 10: ρp = .05 | .10 | .20 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | .077 | .163 | .504 | .075 | .155 | .478 |
| 200 | .107 | .287 | .810 | .105 | .280 | .797 |
| 385 | .164 | .498 | .978 | .162 | .492 | .976 |
| 500 | .199 | .607 | .995 | .197 | .602 | .995 |
| 1,000 | .351 | .886 | 1.000 | .350 | .884 | 1.000 |
| 5,000 | .943 | 1.000 | 1.000 | .942 | 1.000 | 1.000 |

Partial correlation detectable with 80% power (p = 4, unadjusted). This feeds
the "detectable effect" line in OPEN's reports and is confirmed by A2:

| N | 100 | 200 | 300 | 500 | 1,000 | 2,000 | 5,000 | 10,000 | 20,000 | 50,000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ρp | .279 | .198 | .161 | .125 | .089 | .063 | .040 | .028 | .020 | .013 |

The manifest will store these values, computed by the frozen script, so A2
compares against the identical numbers.

## 9. Outputs and preservation

Output directory: `examples/results/<run-date>-phase-a-linear-reference/`.

| File | Contents |
|---|---|
| `protocol.json` | Frozen manifest: grid, seeds, replications, criteria, analytic table, software hashes |
| `seed-audit.json` | Overlap check against prior manifests |
| `check.json` | `--check` results |
| `pvalues/*.npz` | Per-dataset p-values for every method and hypothesis; compressed, about 60–150 MB. Published as a dedicated GitHub prerelease asset, verified by download and SHA-256, with only the inventory and hashes committed (Q6). The upload is confirmed with the owner before it happens. |
| `summary.csv` | One row per cell × method × hypothesis: rate, interval, analytic value, pass flag |
| `criteria.json` | A1–A6 verdicts, the K/Q99 counts, and the scope-ladder result |
| `review.py`, `verification.json` | Independent recomputation of every summary from the saved p-values; a 2% random sample of datasets regenerated from seeds and recomputed with statsmodels |
| `interpretation.md` | Plain-language results, written after the review passes |

Unknown, failed or incomplete cells stay in denominators and are reported as
such.

## 10. Owner review (resolved 2026-10-17)

| Question | Resolution |
|---|---|
| Q1 Replications | 5,000 per null cell, 1,000 per signal cell |
| Q2 N = 100 | Added to the primary and robustness grids; the scope ladder starts at 100 |
| Q3 Likert thresholds | Symmetric and skewed agreement-scale sets |
| Q4 Missing data | MCAR descriptive cells kept here; MAR and multiple imputation moved to Phase M in the parent plan |
| Q5 Coarsened target | 12 single-item-outcome cells added, under the A3 criterion |
| Q6 Storage | Prerelease asset plus committed inventory and hashes |

Remaining steps before freezing:

1. ~~Implement the script and its `--check` mode.~~ Done 2026-10-17:
   `examples/phase_a_linear_reference.py` in the phase-2 checkout.
2. ~~Run the formal seed audit.~~ Done: 0 overlaps
   (`examples/results/2026-10-17-phase-a-linear-reference-check/seed-audit.json`).
   All `--check` items passed in 33 s; the projected full run is about
   4 minutes of wall time on 16 workers.
3. Write the manifest with `--write-manifest examples/study_manifests/phase-a-linear-reference.json`
   and record its SHA-256. The current draft manifest SHA is `cd2f459b…5c107`,
   from source `d621a6e5…8bba`; it changes with any script edit.
4. Owner sign-off on the frozen manifest.
5. Write `review.py` (independent recomputation), required before results are
   interpreted, per §9.

The full run happens only after step 4.
