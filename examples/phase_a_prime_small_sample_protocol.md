# Phase A′ protocol: small-sample variance estimators

Date: 2026-10-17. Status: **FROZEN 2026-10-17** (manifest
`examples/study_manifests/phase-a-prime-small-sample.json`, canonical SHA-256
`044bf386…cb01`, source `8329c91e…984b`). **Execution moved to GitHub Actions by
owner decision; see §11.**

Phase A ran 2026-10-17 and failed under its frozen rules. Results are in
`examples/results/2026-10-17-phase-a-linear-reference/` (phase-2 checkout).

## 1. Why this phase exists

Phase A found HC3 calibrated and matching analytic power from N ≥ 385, but
**conservative at N = 100–200, especially with 10 predictors**:

- null rejection averaged .041 (min .024) at N = 100, p = 10;
- power for ρp = .20 was about 6 points below the exact test (criterion A2
  failed).

HC3 was robust to heteroskedastic, skewed, heavy-tailed and coarsened-target
errors at every N from 100 up. Classical OLS was exact under homoskedasticity
but rejected about 10% under heteroskedasticity.

Small samples are common in the intended applications, so the question is:

> Is there a heteroskedasticity-robust linear test that keeps HC3's robustness
> but is neither too conservative nor too liberal at N = 100–385?

### What this phase decides

| Outcome | Consequence |
|---|---|
| A candidate passes every criterion from N = 100 | It becomes the linear default for all N (if it also passes the N = 1,000 bridge cells), or for N < 385 with HC3 above. Phase A is closed; Phase B can be protocolled. |
| The best candidate passes only from N₀ > 100 | Supported scope is N ≥ N₀ via the ladder (§7.3). Below N₀, OPEN warns that p-values are approximate. Phase B can be protocolled once you confirm. |
| No candidate passes even at N = 385 | Contradicts Phase A. Stop and investigate; no adoption. |

### Out of scope

- The GCM small-N inflation (Phase A criterion A6). It concerns flexible
  engines and is handled in the Phase B protocol: GCM inference limited to
  N ≥ 1,000, or a separately validated small-sample correction.
- Missing data (Phase M).
- Phase A's passing results (A3 robustness, A4 BH, A5 groups). They stand.

## 2. Methods

All methods run on the same dataset (paired). Each is a two-sided test of
H0: βj = 0 for every candidate j, at α = .05.

| ID | Method | Reference distribution | Definition |
|---|---|---|---|
| M1 | Classical OLS t | t(n − k) | Exact control under Gaussian homoskedastic errors. Used for paired comparisons; **not a candidate** (fails under heteroskedasticity, Phase A) |
| M2 | HC3 | t(n − k) | ωᵢ = eᵢ² / (1 − hᵢ)². Phase A baseline |
| M6 | HC4 (Cribari-Neto, 2004) | t(n − k) | ωᵢ = eᵢ² / (1 − hᵢ)^δᵢ, with δᵢ = min(4, hᵢ / h̄) and h̄ = k/n |
| M7 | HC5 (Cribari-Neto, Souza & Vasconcellos, 2007) | t(n − k) | ωᵢ = eᵢ² / (1 − hᵢ)^(αᵢ/2), with αᵢ = min(hᵢ / h̄, max(4, 0.7·h_max / h̄)) |
| M8 | Restricted wild bootstrap of the HC3 t (Davidson & Flachaire, 2008) | Bootstrap, B = 999 | See below |

Here k = p + 1 (with intercept), hᵢ is the leverage, and each candidate's
variance is V = (AᵀA)⁻¹ Aᵀ diag(ω) A (AᵀA)⁻¹.

**M8 bootstrap, for each hypothesis j:**

1. Fit the restricted model without Xj, giving fitted values f̃ and residuals
   ũ.
2. For b = 1…999, set y*ᵇ = f̃ + ũ ⊙ wᵇ, with Rademacher weights wᵢᵇ ∈ {−1, +1}.
3. Refit the full model and compute the HC3 t*ᵇ for βj.
4. p = (1 + #{|t*ᵇ| ≥ |t|}) / 1,000.

- The same weight matrix W (n × 999) serves all hypotheses in a dataset.
- The minimum attainable p is .001. That is below the smallest BH threshold
  used (.05/10 = .005), so discreteness doesn't block discoveries.

Excluded, with reasons:

- **Pretest-then-choose rules** (classical t unless a heteroskedasticity test
  rejects): pretesting distorts error rates.
- **HC4m and HC2:** not requested. HC2 is reported as a diagnostic only (§6
  E5), not a candidate.

Multiplicity, applied afterwards to saved p-values: BH within each dataset's
family, at q = .05 and q = .20, for every method.

## 3. Data-generating designs

Generation follows Phase A §3 (same Σ structures S1/S2/S3, same β/σ² formula
for population partial correlation ρp, same error definitions). One new
predictor design is added: the high-leverage cells.

### 3.1 Primary cells (Gaussian, homoskedastic)

N ∈ {100, 150, 200, 275, 385} × S ∈ {S1, S2, S3} × p ∈ {4, 10} ×
ρp ∈ {0, .10, .20, .30}. **120 cells.**

### 3.2 Robustness cells

Errors ∈ {heteroskedastic, skewed, heavy-tailed} × N ∈ {100, 200, 385} ×
S ∈ {S2, S3} × p ∈ {4, 10} × ρp ∈ {0, .20}. **72 cells.**

### 3.3 High-leverage cells (new)

- **Why:** skewed predictors such as income, tenure, or counts create
  high-leverage observations. That's where HC3 can become liberal, and what
  HC4/HC5 were designed for.
- **Generation:** draw Z ~ N(0, Σ_S3). Set X1 = std(exp(s·Z1)) and
  X3 = std(exp(s·Z3)), with lognormal shape s ∈ {0.5, 0.8} (skewness about
  1.75 and 3.7; s = 0.5 added at owner review, Q2. *The draft's 2.9 for
  s = 0.8 was wrong; lognormal theory and the check draw both give 3.7.*), where std() standardizes by the analytic lognormal
  mean and SD. All other Xj = Zj.
- **Coefficients:** β and σ² use the Gaussian-Σ formula (Phase A §3.1).
  Realized partial correlations are reported from a 10⁶-row check draw, not
  assumed.
- **Nulls stay exact:** E[T | X] = β·X1, so X2…Xp have zero coefficients for
  any predictor distribution.
- **Grid:** s ∈ {0.5, 0.8} × errors ∈ {gaussian, heteroskedastic
  (h = 1 + X1² + X2²)} × N ∈ {100, 200, 385} × p ∈ {4, 10} × ρp ∈ {0, .20}.
  **48 cells.**
- The power criterion doesn't apply here; calibration does.

### 3.4 Bridge cells

N = 1,000 × S3 × p ∈ {4, 10} × ρp ∈ {0, .10}. **4 cells.** These confirm a
candidate adopted for all N doesn't misbehave above the small-N range.

### 3.5 Many-predictor stress cells (descriptive; owner review, Q4)

- p = 15 × N ∈ {100, 150, 200} × S ∈ {S2, S3} × ρp ∈ {0, .20} × errors ∈
  {gaussian, heteroskedastic}. **24 cells.**
- Observations per predictor: about 6.7 to 13.3.
- **Excluded from C1–C4 and from selection.** Their calibration and power,
  per method, inform a runtime N/p warning rule for OPEN.

**Total: 268 cells** (244 criterion-bearing, 24 descriptive).

## 4. Replications, seeds and compute

**Replications:**

- 5,000 per null cell (104 cells).
- 1,000 per signal cell (164 cells).
- 684,000 datasets in total.

**Seeds (fresh block; no reuse of Phase A data):**

- Dataset seed = 3,720,000,000 + 10,000·cell_index + rep, for cell_index
  0–267 (descriptive p = 15 cells last).
- Maximum ≤ 3,722,674,999.
- One `numpy.random.default_rng(seed)` per dataset, drawing in order: data
  (Phase A order), then bootstrap weights W = 2·rng.integers(0, 2, (n, 999)) − 1.
- Reserved: fixtures 3,729,000,000–3,729,999,999; defect-rerun base
  3,730,000,000.
- **Seed audit** (same implementation as Phase A): range
  3,720,000,000–3,739,999,999 against all manifests, results and scripts,
  **including Phase A's manifest and outputs.** It must report zero overlaps
  before freezing.

**Compute:**

- Timing probe of the vectorized bootstrap: about 2.5 ms (N = 100, p = 4) to
  48 ms (N = 385, p = 10) per dataset, and about 150 ms at N = 1,000, p = 10.
- Projected about 20–30 min wall time on 16 local workers (B = 999 confirmed
  at owner review, Q1). Wall cap
  90 minutes. **No Actions.**
- If `--check` projects more than 75 minutes, the run is split into two
  sequential local batches. B, replications and grid don't change.

## 5. Software and checks

- Script `examples/phase_a_prime_small_sample.py` and independent
  `examples/results/<date>-phase-a-prime-small-sample/review.py`.
- Same pinned runtime as Phase A (Python 3.11.9, numpy 1.26.4, scipy 1.11.2,
  statsmodels 0.14.4).
- Imports Phase A's helpers (correlation structures, error draws, OLS design,
  BH, partial correlations), verified by its frozen source SHA-256
  (`d621a6e5…8bba`). Generation is reimplemented so the bootstrap weights can
  follow from the same generator. `--check` verifies that X and T match Phase A's
  generator exactly for shared designs. *(Clarified at implementation.)*

**`--check` mode (fixtures only):**

1. M1/M2 match statsmodels to 1e-10, as in Phase A.
2. HC4/HC5 match a separately written loop-based implementation of the
   published formulas to 1e-12. Two property checks, on a balanced ±1
   factorial design with constant leverage hᵢ = h̄:
   - HC4 equals statsmodels HC2;
   - HC5 equals the HC0 weights divided by √(1 − h̄).
3. M8's vectorized bootstrap equals a brute-force loop (refit statsmodels
   HC3 for each replicate, with identical weights) to 1e-10, on 20 fixtures.
4. High-leverage population draw (10⁶ rows): realized partial correlations,
   Var(T), and max leverage are reported. Exact nulls have partial correlation
   within 4 Monte Carlo SE of 0.
5. Seed audit passes; manifest deterministic; run guard refuses existing
   output.
6. Smoke run (4 cells × 60 replications, fixture seeds) exercises every
   output path. Timing projection included.

## 6. Endpoints

| ID | Endpoint |
|---|---|
| E1 | Null rejection rate per cell × method × null predictor, and the **pooled** rate per cell × method (average over the cell's null predictors), with a dataset-clustered SE (SD over datasets of the per-dataset rejection proportion, ÷ √R) |
| E2 | Power for X1: versus analytic noncentral t (primary cells), and **paired difference from M1** on the same datasets (primary cells), with paired SE |
| E3 | BH realized FDR and power at q = .05 and .20, per method |
| E4 | Runtime per method; bootstrap p-value granularity (share of p = .001) |
| E5 | Diagnostic only: HC2 null rate and power, stored for reference |

## 7. Acceptance criteria and selection

### 7.1 Changes from Phase A's rules

Both changes are prespecified here, before any A′ data exist.

- **Calibration is judged per cell on the pooled null rate, with
  dataset-clustered SE.** Phase A's per-hypothesis Binomial(K, .01) allowance
  ignored correlation among hypotheses that share a dataset. It nearly failed
  the exact test (15 of 17 allowed).
- **Every criterion gets the N ladder,** including power (requested).

### 7.2 Criteria (for each candidate method M2, M6, M7, M8)

- **C1. Calibration** (primary, robustness, high-leverage and bridge cells):
  - the pooled null rate lies within [.040, .060] in null cells (R = 5,000),
    or [.035, .065] in signal cells (R = 1,000, on their null predictors);
  - no single null predictor's rate lies outside [.030, .070] (R = 5,000) or
    [.025, .075] (R = 1,000).
  - The margins are at least about 4 clustered SE, so chance failures are
    rare.
- **C2. Power** (primary signal cells):
  - the paired power loss versus M1 is at most .03 + 2.58·SE_paired;
  - and |power − analytic| ≤ 2.58·√(π(1 − π)/R) + .01.
  - M1 must itself satisfy the analytic check. Otherwise the cell is flagged
    as a simulation defect, not as a candidate failure.
- **C3. Robustness** (robustness and high-leverage cells): C1 applies at every
  N on the ladder; there is no separate small-N relaxation.
- **C4. BH:** in every criterion cell, realized FDR ≤ q + 2.58·SE for
  q ∈ {.05, .20}.

### 7.3 Ladder and scope

- Ladder N ∈ {100, 150, 200, 275, 385} (bridge cells count at 385+).
- For each candidate, its **supported minimum N** is the smallest ladder value
  at and above which C1–C4 all pass. "None" if it fails at 385.

### 7.4 Selection rule (applied mechanically)

1. Discard candidates with supported minimum N = None.
2. Choose the candidate with the **smallest supported minimum N**.
3. Ties: choose the highest mean power over primary signal cells with
   N ≤ 385.
4. If still within .01: prefer analytic over bootstrap, then the order HC3,
   HC4, HC5. This favors continuity and simplicity.
5. **Outcome:**
   - if the winner passes the bridge cells, it becomes the linear default at
     all N;
   - otherwise, the winner applies for N < 385 and HC3 above (Phase A
     validated HC3 at N ≥ 385).

### 7.5 Defects versus failures

As Phase A §7.4. Defects are fixed and rerun on the 3,730,000,000 block, with
the failed output preserved. Statistical failures are reported and resolved
only by §7.3–7.4.

## 8. Analytic reference values (two-sided .05, primary cells)

| N | p = 4: ρp = .10 | .20 | .30 | p = 10: ρp = .10 | .20 | .30 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | .163 | .504 | .859 | .155 | .478 | .835 |
| 150 | .225 | .685 | .964 | .218 | .666 | .957 |
| 200 | .287 | .810 | .992 | .280 | .797 | .990 |
| 275 | .377 | .917 | .999 | .370 | .911 | .999 |
| 385 | .498 | .978 | 1.000 | .492 | .976 | 1.000 |

## 9. Outputs and preservation

Same layout as Phase A:

- `protocol.json` (manifest), `seed-audit.json`, `check.json`;
- `pvalues/*.npz` (about 90 MB; prerelease asset after owner confirmation,
  with inventory and hashes committed);
- `summary.csv`, `criteria.json` (including the selection result);
- `review.py`, `verification.json`, `interpretation.md`.

The independent review recomputes all summaries and criteria from saved
p-values, re-derives the selection, and regenerates 2% of datasets: M1/M2
with statsmodels, HC4/HC5 with the loop implementation, M8 with the
brute-force bootstrap.

## 10. Owner review (resolved 2026-10-17)

| Question | Resolution |
|---|---|
| Q1 Bootstrap size | B = 999 |
| Q2 High-leverage strength | Lognormal shapes s = 0.5 and 0.8 (48 cells) |
| Q3 Selection tie-break | Continuity, as drafted: analytic before bootstrap, then HC3 → HC4 → HC5 |
| Q4 p = 15 at small N | Added as 24 descriptive stress cells; they inform an N/p warning and don't affect C1–C4 or selection |

Remaining steps before freezing:

1. ~~Implement the script and `--check`.~~ Done 2026-10-17:
   `examples/phase_a_prime_small_sample.py` (phase-2 checkout). All checks
   passed in 31 s:
   - M1/M2/HC2 match statsmodels to about 1e-14;
   - HC4/HC5 match the naive loop to about 1e-14, and on a constant-leverage
     ±1 factorial they equal HC2 and HC0/√(1 − h̄);
   - the vectorized bootstrap equals the brute-force statsmodels refit on 20
     fixtures (max p difference 0);
   - generation is identical to Phase A on 32 shared-design cells;
   - leverage draws: realized partial ρ for X1 is .205–.218 (target .20 under
     Gaussian Σ; power is descriptive there), exact nulls within 4 SE of 0, and
     the median max-leverage ratio at N = 100 is 2.8–7.2× average;
   - run guard and smoke run pass;
   - projected run time about 14 min on 16 workers.
2. ~~Run the seed audit (including Phase A outputs).~~ Done: 2,328 files, 0
   overlaps, Phase A outputs included
   (`examples/results/2026-10-17-phase-a-prime-small-sample-check/`).
   Draft manifest SHA `044bf386…cb01`, source `8329c91e…984b`; these change
   with any script edit.
3. Write the manifest to `examples/study_manifests/phase-a-prime-small-sample.json`
   and record its SHA-256.
4. Owner sign-off on the frozen manifest.
5. Write `review.py`, then run.

## 11. Amendment A1: execution on GitHub Actions (2026-10-17, owner decision)

**What happened locally:**

- The frozen run started locally at 23:05 on 16 workers.
- `--check` had projected about 14 minutes, but it timed datasets one at a time
  on an idle machine. Under 16 concurrent processes the memory-bound bootstrap
  ran about 7–8× slower: about 102,000 CPU-seconds in total, about 107 minutes
  of wall time, beyond the 90-minute cap in §4.
- The owner stopped the run at about 23:43, with 15 of 268 cells complete. Those
  15 cells are preserved as a record in
  `examples/results/2026-10-17-phase-a-prime-small-sample-local-aborted/`
  (inventory and hashes). They aren't used for any criterion.
- **Lesson:** runtime projections must be measured under the planned parallel
  load.

**What changes (execution only):**

- The frozen script, manifest, seeds, cells, methods, replications and criteria
  are unchanged. No scientific setting differs.
- New harness `examples/phase_a_prime_actions.py`. It verifies the frozen source
  and manifest hashes, and checks that the runner regenerates the identical
  manifest under the same pinned versions.
- It splits each cell's replications into chunks of 250 (2,736 chunks).
  Chunks are assigned to 20 shards by longest-processing-time first; shard
  loads are balanced within 0.1%.
- Each windows-2022 runner (4 vCPU) computes its chunks with 4 processes. The
  aggregate job reassembles each cell in replication order. Per-dataset seeds
  are unchanged, so a reassembled cell equals `run_cell()`.
- Verified locally before dispatch:
  - one cell rebuilt from 20 chunks is bit-identical to `run_cell()`;
  - a chunk reproduces the aborted local run's rows bit-identically.
- The runner repeats the fixture cross-checks before computing: estimators
  versus statsmodels and the naive loop, brute-force bootstrap, generation
  identity with Phase A, run guard and smoke run.
- The seed audit and the 10⁶-row population draws aren't repeated; they ran
  locally before freezing.

**Workflow** `.github/workflows/phase-a-prime-small-sample.yml` on
`codex/multiengine-preflight`:

| Job | Contents |
|---|---|
| prepare | Verify, run checks, freeze the plan |
| 20 × shard | 60-minute job cap; no new chunks started after 45 minutes |
| aggregate | Reassemble, summarize, apply the frozen criteria, then run `review.py` on the runner |
| publish | Durable prerelease, verified by download and SHA-256 |

- Incomplete chunks leave their cells unknown, and selection is reported as
  INCOMPLETE. There are no automatic retries.
- **Budget ceiling:** 20 × 60 + 20 + 60 + 30 + 1 = **1,311 runner-minutes**,
  free on this public repository. The expected use is about 15–35 minutes per
  shard.
- **Numerics:** runners use the same OS family (Windows) and pinned wheels. Any
  last-bit floating-point differences from the local machine have no scientific
  effect. The review regenerates its 2% sample on the runner itself.


**Amendment A1.1 (first dispatch, run 39105672167):** the prepare job stopped
because the runner did not regenerate a byte-identical manifest. The frozen
file itself passed its SHA-256 check. The harness now compares the regenerated
manifest field by field: non-float values must match exactly, and floats must
match to 1e-12 relative (CPU-specific BLAS kernels can change last bits of the
computed β, σ² and analytic power). Every run reads those values from the
frozen file. No computation had started; nothing was retried automatically.
