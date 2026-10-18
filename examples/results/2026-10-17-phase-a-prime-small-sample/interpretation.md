# Phase A′ small-sample variance estimators: results

- [Actions run 39105797558](https://github.com/imh-ds/open/actions/runs/39105797558),
  at commit `50e82152` on `codex/multiengine-preflight`.
- Frozen manifest `examples/study_manifests/phase-a-prime-small-sample.json`
  (canonical SHA-256 `044bf386…cb01`); source `8329c91e…984b`.
- Protocol: `examples/phase_a_prime_small_sample_protocol.md`, with amendments A1
  (Actions execution) and A1.1 (float-tolerant manifest regeneration).
- Raw evidence:
  [prerelease](https://github.com/imh-ds/open/releases/tag/phase-a-prime-small-sample-2026-10-17-39105797558),
  verified by download (`publication-verification.json`).

## What ran

- All 268 cells and 684,000 datasets completed: 2,736 chunks over 20 Windows
  runners, 4–9 minutes per shard, no missing chunks and no retries.
- Before computing, the runner repeated the fixture cross-checks (estimators,
  brute-force bootstrap, generation identity with Phase A, run guard, smoke
  run). All passed.

## Verification

**The in-workflow review failed one cross-check; the cause was the check, not the
study.** It recomputed the false discovery rate (FDR) for each cell with
statsmodels' BH and found differences of up to 1.1 × 10⁻⁴. All of them were in
the wild bootstrap at q = .20, in 3 cells.

- Bootstrap p-values lie on a 1/1,000 grid. In one dataset in each of those cells,
  a p-value sat exactly on a BH threshold (q·k/m), and statsmodels' floating-point
  comparison rounded the wrong way.
- Checked with exact rational arithmetic: the frozen study code was right in all
  3 cells, and statsmodels was wrong in one dataset each.
- The review now uses exact integer BH for bootstrap p-values. Rerun on the
  run's own downloaded artifacts, it passes every check:
  - integrity;
  - recount of all 12,408 summary rows;
  - cell statistics;
  - an independently re-derived selection;
  - regeneration of 13,680 datasets (2%), matching to 3.6 × 10⁻¹⁴;
  - a brute-force statsmodels bootstrap on 40 datasets, identical.
- The original failed review is kept as `verification-actions-original.json`;
  the passing one is `verification.json`.
- Nothing was recomputed. The published results are exactly the run's outputs.

## Result under the frozen selection rule

| Candidate | Supported from | Valid at N = 1,000 | Mean power (primary signal cells) | Failing checks |
|---|---:|---|---:|---|
| **HC4** | **N ≥ 275** | Yes | .673 | 6 calibration (high-leverage, N = 100–200), 4 FDR (N = 150–200) |
| HC3 | none | Yes | .666 | 6 power, 13 calibration, 4 FDR |
| HC5 | none | No | .683 | 6 calibration (primary), 12 calibration (robustness/leverage), 93 FDR |
| Wild bootstrap of HC3 | none (one failure at N = 385) | Yes | .669 | 3 power (two at N = 100, one at N = 385) |

**Selected: HC4, supported from N ≥ 275, valid at all N** (it passes the
N = 1,000 bridge cells). Below N = 275, no candidate passed every check, so the
protocol's outcome is SCOPED: below 275, OPEN warns that p-values are
approximate.

## What the numbers show (plain language)

**HC3 is too cautious at small N, as Phase A found.**

- Its null rejection rate was .041 at N = 100 with 10 predictors, and .034–.035
  with 15 predictors.
- Power for ρp = .20 at N = 100 was .44, against .48 for the exact test.

**HC4 fixes HC3's small-sample caution.**

- Null rates averaged .049–.051 at every N (100–385) and p in the Gaussian cells.
- Under heteroskedastic errors they averaged .048–.051.
- Power was within about .01–.015 of the exact test.

HC4's two weaknesses are small:

- **Conservative with strongly skewed predictors plus heteroskedasticity**
  (lognormal shape 0.8): about .032–.043 at N ≤ 200. That costs some power but
  doesn't inflate false findings.
- **Slightly liberal in the far tail at N = 150–200.** In all-null cells, the
  chance of *any* BH discovery was .059–.061, versus .053–.056 for the exact
  test. Its rate at α = .05 is still nominal (.049–.050). The excess is at the
  very small thresholds BH uses for its first rejection.

**HC5 is too liberal.** Null rates averaged .051–.064, and FDR exceeded its target in
93 cells.

**The wild bootstrap was the most uniformly calibrated.**

- Null rates averaged .045–.051 in every design group, including the high-leverage and
  15-predictor cells.
- It had no calibration or FDR failures.
- Its 3 failures are small power shortfalls. The disqualifying one, at N = 385
  (cell 102), was power .952 versus analytic .976. In the same cell the exact
  test reached .959, so this miss is very likely chance (see the caveat below).

**Classical OLS** remains exact under homoskedasticity, but rejected .057–.139 of
true nulls under heteroskedasticity. It isn't a candidate.

**Stress cells** (15 predictors at N = 100–200; descriptive only): HC4 and the
bootstrap stayed at .048–.050, HC3 at .034–.043, and HC5 at .056–.064. With an
observations-per-predictor ratio as low as 6.7, HC4 and the bootstrap stay
calibrated. That evidence informs the runtime N/p warning rule.

## Multiplicity caveat

The power (C2) and FDR (C4) checks compared each cell with a 99% bound
(2.58 SE), with no allowance for how many cells were checked: about 120 C2 and
about 490 C4 checks per candidate. Even a perfect method should fail roughly
1% of them by chance, i.e. several per candidate. So:

- **The selection of HC4 over the wild bootstrap is not a demonstrated
  superiority.** The bootstrap was excluded by a single borderline power cell,
  and HC4's 4 FDR failures are partly a real but small tail effect and partly
  within chance range.
- **The broad pattern is robust; the fine ranking isn't.** HC3 is too
  conservative at small N, HC5 too liberal, and HC4 and the bootstrap both
  close to nominal.
- The frozen rule's verdict (HC4, from N ≥ 275) stands as recorded. It was not
  re-scored after seeing these results.
- **For future protocols:** apply a multiplicity allowance to C2/C4 (for
  example, a Binomial(K, .01) count allowance calibrated on the exact test, or
  pooled per-N tests). That replaces cell-by-cell pass/fail. Phase A already
  showed the per-hypothesis version of this problem.

## Implications

1. **Linear inference default:** HC4 with a t(n − k) reference, validated at
   N ≥ 275, and passing the N = 1,000 bridge cells. This replaces HC3, which
   Phase A had supported only from N ≥ 385.
2. **Below N = 275:** OPEN should report HC4 p-values with a warning that they
   are approximate. Its known deviations there are conservative with very
   skewed predictors, and slightly liberal for the strongest BH discovery.
3. **The wild bootstrap is a credible small-N option.** It was the best
   calibrated, but costs about 2–50 ms per target, and Phase A′ can't formally
   select it. Use it as an optional small-sample setting, or validate it in a
   separate protocol with the multiplicity allowance above.
4. **Phase B** can now be protocolled. Its linear reference is HC4. It must
   also respect Phase A's A6 finding: GCM inference is limited to N ≥ 1,000
   unless a small-sample correction is validated.
