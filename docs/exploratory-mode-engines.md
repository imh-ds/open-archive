# Engines in exploratory mode

Decision date: 2026-10-19 (owner decision).

## Supported engines

| Engine | Exploratory mode | Confirmatory p-values (Phase B) |
|---|---|---|
| OLS | Validated, N 300–1,000 | Calibrated from N ≥ 275 (OLS + HC4, Phase A′) |
| Elastic-Net | Validated, N 300–1,000 | Not tested |
| LightGBM | Validated; finds real effects reliably from about N = 1,000 and is the only validated engine that detects U-shaped effects | Calibrated from N ≥ 300 with the K = 10 cluster correction (G2) |

`open.explore` accepts only these engines (`SUPPORTED_ENGINES`).

## Dropped: random forest and scikit-learn gradient boosting

Both were removed from exploratory mode. The code stays in the development
archive, so nothing is lost.

- **Redundant with LightGBM.** All three are tree ensembles and capture the
  same effects: curves, thresholds and interactions. LightGBM is itself a
  gradient-boosting implementation, and it is faster. One validated tree engine
  is enough.
- **Not validated for exploratory mode.** Neither was part of the validation
  study (Actions run 39184696821), because both were too slow for it.
- **Weak for confirmatory p-values (Phase B, Actions run 39144406136).**
  - **Random forest:** calibrated only at N = 5,000, the next ladder step after
    2,000. At N = 1,000 and 2,000 its false-positive rates were slightly over
    the limit (about 6.3–6.6% against about 6.4%), in cells with curved or
    interaction effects.
  - **scikit-learn gradient boosting (depth-2 trees):** passed from N = 300 to
    2,000. It then failed badly at N = 5,000 in the interaction cell, where
    null predictors were rejected 25–31% of the time, so it has no supported
    range. The likely cause is the shallow trees under-fitting a strong
    interaction.
- **Bad defaults for users.** An engine that needs 5,000 or more rows, or that
  fails at large N, would mislead the typical user with N = 300–1,000.

## Adding engines later

PLS, SVR, XGBoost, or random forest again are added only after passing the same
validation study (`examples/exploratory_validation.py`, run for that engine
alone). They also need a Phase B–style calibration check if they are to be
offered for confirmatory p-values.
