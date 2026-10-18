# Phase A linear-reference calibration: results

Run 2026-10-17, local, 16 workers, 545 s.

- Frozen manifest: `examples/study_manifests/phase-a-linear-reference.json`
  (canonical SHA-256 `cd2f459b…5c107`, file SHA-256 `f21725f2…74cc7`).
- Source: `d621a6e5…8bba`.
- Protocol: `outline/2026-10-17-phase-a-linear-reference-protocol.md` (main
  checkout).

All 272 cells and 652,000 datasets completed. The independent `review.py`
reproduced every rejection count, the BH and group statistics, and each
criterion verdict. It also regenerated 13,040 datasets (2%) from their seeds,
and statsmodels reproduced the stored p-values to 1.5e-13 (`verification.json`).

## Verdict under the frozen rules: FAIL

| Criterion | Result | What failed |
|---|---|---|
| A1 HC3 calibration (primary) | Fail at all N; **passes from N ≥ 385** (ladder) | 51/900 hypotheses outside the 99% band (17 allowed), 4 outside the hard band. Almost all at N = 100–200 and **too low** (conservative): mean .041 at N = 100, p = 10 |
| A2 HC3 power vs analytic | **Fail** (no ladder for A2) | 3 cells, all N = 100, p = 10, ρp = .20: .42 vs analytic .48 |
| A3 Robustness | Pass at all N | — |
| A4 BH FDR (q = .05, .20) | Pass | — |
| A5 Group test and hierarchy | Pass | — |
| A6 Cross-fitted GCM (M4) | Fail; **passes from N ≥ 1,000** (ladder) | 225/900 outside the band, all at small N and **too high** (anti-conservative): mean .081, max .105 at N = 100, p = 10 |

The protocol scopes A1/A3/A6 failures by N, but has no ladder for A2. A2's
failure at N = 100 therefore makes the overall verdict FAIL. Per protocol §1,
Phase B does not start until a new frozen protocol addresses it. This review
does not reinterpret or relax the rule.

## What the results mean

1. **The simulation and analytic targets are correct.** Classical OLS (M1),
   which is exact for Gaussian homoskedastic data, matched analytic power in
   every cell (deviations within ±.03, i.e. Monte Carlo noise). Its type I
   error passed the A1 rule (15 outside the band, 17 allowed). The A2 failure
   is therefore a property of HC3, not of the reference values.
2. **HC3 is conservative at small N with many predictors.** At N = 100 with
   p = 10, its null rejection rate averages .041 (min .024), and power for
   ρp = .20 drops by about 6 points versus the exact test. By N = 385 it is
   calibrated, matching analytic power within noise. This is a known
   small-sample property of HC3, which overcorrects leverage.
3. **HC3 does its job under heteroskedasticity.** With variance depending on
   the predictors, classical OLS rejected true nulls about 10% of the time,
   even at N = 1,000. HC3 stayed at about .05 at every N from 100 up (A3 pass,
   N = 100 included). Skewed, heavy-tailed and single-item (coarsened) targets
   also passed.
4. **The GCM's normal reference is anti-conservative at small N, even with
   perfect linear nuisance models.** Cross-fitted GCM (M4) reached .08–.10 at
   N = 100 with p = 10 and was calibrated only from about N = 1,000. In-sample
   GCM (M3) showed the same pattern, less strongly. **Implication for
   Phase B:** flexible-engine GCM (with imperfect nuisances) can't be trusted
   below N ≈ 1,000 without a small-sample correction. This is consistent with
   the earlier informal LightGBM result (.11–.13 at N = 1,000).
5. **Error control and groups work.** BH realized FDR stayed within q plus Monte
   Carlo error (q + 2.58 SE) in every criterion cell, at both .05 and .20. The redundant-pair group test
   matched noncentral-F power, and its hierarchical error rate was nominal.
6. **Descriptive cells (no criteria) behaved as predicted:**
   - **Median imputation (MCAR 10%):** an overlapping null predictor was
     rejected 9.0% of the time at N = 5,000 (complete-case: 5.1%). The
     inflation grows with N. Median imputation shouldn't be used for inference
     (input to D8 / Phase M).
   - **Likert-coarsened predictors:** a predictor overlapping with a coarsened
     true predictor picked up measurement-induced association, 7.9–8.8% at
     N = 5,000. This is a real association on the observed scale, but readers
     shouldn't take it as a separate effect (input to D9).
   - **Coarsened targets:** power fell modestly (r = .10 at N = 1,000: .84–.86
     versus .89 analytic on the latent scale).

## A note on the A1 rule itself

The exact test (M1) came close to the rule's limit: 15 of 17 allowed overall,
and 11 of 12 at N ≥ 385. Hypotheses within a cell share datasets, so their
band exceedances are positively correlated, while the Binomial(K, .01)
allowance assumes independence. This makes the rule slightly too strict, which
is why HC3 at N ≥ 385 sits exactly at its limit (12/12). Future protocols
should count exceedances per cell, or calibrate the allowance by simulating
the exact test. This is recorded as a lesson; the frozen rule is unchanged.

## Options for the owner (none taken)

- **(a) Restrict the linear reference to N ≥ 385.** Freeze a Phase A′ that
  adds an A2 ladder and re-tests at fresh seeds (a new block, not the defect
  rerun block). Below 385, OPEN would warn that p-values are conservative.
- **(b) Test a small-sample alternative in Phase A′.** Candidates: the
  classical t when no heteroskedasticity is detected (needs care: pretesting
  distorts error rates), HC4/HC5, or wild-bootstrap HC3. Compare against HC3
  at N = 100–385 under the same criteria.
- **(c) Accept HC3's conservativeness as the safe direction** (fewer false
  edges, lower power at N < 385). Freeze Phase A′ with a one-sided calibration
  criterion at small N, then re-run on fresh seeds.

Any option needs a new frozen protocol and fresh seeds. Changing the rule in
place for this run isn't allowed (protocol §7.3).
