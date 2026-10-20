# Phase B: engine-based p-values (interpretation)

Actions run [39144406136](https://github.com/imh-ds/open/actions/runs/39144406136).
Frozen manifest `73aef5e6…febd`; source `7f2da249…5af6`; protocol
`examples/phase_b_local_detector_protocol.md`.

## Bottom line

The question was: when a machine-learning engine adjusts for the other
predictors, does the cross-fitted GCM/PCM test give honest p-values, and from
what N?

| Engine (configuration) | Honest p-values from | Correction |
|---|---|---|
| LightGBM (F3; F4 with linear trees is an equivalent alternative) | **N ≥ 300** | G2: K = 10 folds, fold-cluster variance, t(9) reference |
| Random forest (F1) | N ≥ 5,000 (the next ladder step after 2,000) | G0 |
| scikit-learn gradient boosting (F2, depth-2 trees) | No supported N | — |

The OLS + HC4 reference passed every check at every N (300 to 5,000), and no
reference defects were found.

**Owner decision (2026-10-19):** OPEN's engines are OLS, Elastic-Net and
LightGBM. Random forest and gradient boosting are dropped
(`docs/exploratory-mode-engines.md`, on `codex/exploratory-mode`). For
confirming links on independent data, OLS + HC4 (from N ≥ 275, Phase A′) and
LightGBM GCM with G2 (from N ≥ 300) are the calibrated options. Elastic-Net was
not tested here; its shrunken linear adjustment adds nothing over OLS + HC4 at
typical numbers of predictors.

## Details

- **Execution:** all 78 cells and 3,141 chunks completed, none missing or
  invalid. Every cross-platform check repeated in `prepare` passed: HC4 identity
  with Phase A′, GCM identity with Phase A, determinism and the run guard.
- **Random forest:** at N = 1,000 and 2,000, false-positive rates were slightly
  above the limit: pooled about 6.3–6.6% against a limit of about 6.4%. A few
  cells exceeded the allowance, all with curved (Mo, U), interaction (I) or
  measurement-error (E) shapes. At N = 5,000 every rate was about 5%. The cause
  is that the averaged forest predictions leave some curved signal in the GCM
  residuals. That bias does not shrink with N while the test sharpens, so it
  disappears only at large N.
- **Gradient boosting:** it passed at N = 300 to 2,000, then failed at N = 5,000
  in the interaction cell (cell 33). Null predictors were rejected 25–31% of the
  time there. A supported range must hold at every larger N, so none is
  supported. The likely cause is that depth-2 trees cannot fit a strong
  interaction. That leftover bias becomes detectable at large N.
- **LightGBM power** (combined test, nonlinear shapes, r = .1): monotone 0.79 at
  N = 1,000; U-shaped 0.16 at 1,000 and 0.89 at 5,000; interaction 0.11 at 1,000
  and 0.63 at 5,000. Power for linear effects is close to the OLS reference
  (efficiency 0.98 at r = .05).

## Deviation: regeneration check (accepted, owner decision 2026-10-19)

The independent review (`review.py`, `verification.json`) passed integrity,
recomputation of every per-cell statistic, the summary-row check and
independent re-derivation of every criterion, including the supported-N table.
It **failed only the bit-exact regeneration check**: 42 of 585 sampled datasets
differed from their stored p-values by more than 1e-10.

| | |
|---|---|
| Items sampled | 565 frozen-path datasets, plus 20 checked with the loop implementation |
| Items that differed | 40 frozen, 2 loop |
| Where | Only the random-forest (F1) and gradient-boosting (F2) GCM tests. None in the screen cells, none in LightGBM, none in OLS. |
| Size | Largest 0.23 (one item); most below 0.03 |

**Diagnosis:**
- Regenerating a differing dataset is deterministic. It gives identical results
  when repeated and when run after other datasets in the same process.
- The review runner's regenerated value and a local machine's value agree
  exactly (for example cell 52, rep 14: both differ from storage by 0.030737).
- Only the original shard values differ. The shards ran on 125 different
  GitHub-hosted machines.
- The likely cause, not proven, is floating-point differences between runner
  CPUs. They change near-tied tree splits in scikit-learn's random forest and
  gradient boosting, and the GCM test amplifies the change. LightGBM was run
  with `deterministic=True` and was unaffected.

**Why the results are accepted:** the affected engines are the two that are
dropped, and their conclusions (supported only at large N, or not at all) do
not depend on these small differences. The LightGBM and OLS conclusions
involve no differing values. Re-running those cells on a single machine type
was considered and not needed.

## Files

- Committed here: summary statistics and criteria (`summary.csv`,
  `cell-stats.json`, `criteria.json`, `supported-n.json`), the review
  (`verification.json`, `review.py`), the execution records (`aggregate.json`,
  `plan.json`, `prepare.json`, `regen-plan.json`) and the manifest.
- Raw per-dataset p-values (`pvalues/`, 3,140 files) and regeneration outputs
  (`regen/`) are in the GitHub prerelease
  `phase-b-local-detector-2026-10-18-39144406136`. They are not committed.
