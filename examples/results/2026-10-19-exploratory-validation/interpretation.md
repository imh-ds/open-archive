# Exploratory-mode validation study (interpretation)

Actions run [39184696821](https://github.com/imh-ds/open/actions/runs/39184696821),
branch `codex/exploratory-mode`, script `examples/exploratory_validation.py`,
workflow `.github/workflows/exploratory-validation.yml`. Timing probe: run
39182459285.

## Design

- **Cells:** 3 engines (OLS, Elastic-Net, LightGBM) × N (300, 500, 1,000) ×
  4 shapes (linear, saturating, threshold, U-shaped). 200 simulated datasets
  per cell, 7,200 in total. Each N × shape × replication dataset is shared
  across engines.
- **Truth:** 10 candidates per dataset.
  - Y ← A (r² .09), B (.04), C (.01);
  - a substitute pair S1/S2 (shared latent, r² .04);
  - an interaction C × D (r² .04; D has no main effect);
  - A ← E (.09), B ← F (.01);
  - N1 and N2 are pure noise.

  Every link uses the cell's shape. The interaction is bilinear in linear cells
  and curved (V-shaped in C, with the sign set by D) otherwise.
- **Method:** defaults (10 folds, kept ≥ 80%, flagged ≥ 40%, 95% noise bar,
  hide floor .001, interactions on), max depth 2.
- **Tables:**
  - `true-link-rates.csv`: share of datasets where each true link was kept,
    flagged, hidden, dropped, absent or not tested (its target was not reached).
  - `false-link-rates.csv`: counts and rates of false links per cell.

## Findings

**1. Solid lines are reliable.** False links kept per dataset:

| | OLS | Elastic-Net | LightGBM |
|---|---|---|---|
| Kept false links per dataset | 0.015 | 0.068 | 0.013 |

The Elastic-Net figure excludes S1 → S2 and S2 → S1. The scoring counts those
as false, but they reflect real shared structure; including them gives 0.16.
As a share of all false links tested, kept false links are ≤ 0.4% in every cell.

**2. Dotted lines contain noise, mostly interactions.** Flagged false
interactions average 1.1–1.4 per dataset for every engine. Other flagged false
links average 0.9 (OLS), 0.3 (Elastic-Net) and 0.1 (LightGBM). Dotted
interaction lines should therefore be read as weak leads.

**3. Linear engines.** OLS and Elastic-Net detect linear, saturating and
threshold effects well:
- A (r² .09) is kept in 70–100% of datasets at N 500–1,000.
- B (.04) is kept in 33–89%.

Elastic-Net keeps more than OLS at N ≤ 500 (A, linear, N = 300: 71% vs 36%).
Neither detects U-shaped effects: A is kept in 0–1% of datasets.

**4. LightGBM detects U-shaped effects** (A kept in 81% of datasets at
N = 1,000) and other shapes (78–88%). It needs more data: at N = 300–500, real
effects are mostly flagged rather than kept (A kept in 5–33%, flagged or kept
in 66–96%).

**5. Small effects (r² .01: C, F) are rarely kept at N ≤ 1,000** by any engine
(0–35%). At best they are usually flagged. This is a power limit, not a flaw in
the method.

**6. Interactions are kept only at about N = 1,000:**
- OLS and Elastic-Net keep the interaction in about 50% of linear cells.
- LightGBM keeps it in 42%.
- Curved interactions are kept in 14–24%.

Interactions in U-shaped cells are rarely tested. The search only starts from
kept or flagged variables, and C's r² .01 main effect seldom gets there.

**7. Substitute pair.** S1/S2 is found as a shared group or through a member at
N = 1,000 in most cases: OLS keeps it 28–69% of the time and flags or keeps it
97–100% in non-U-shaped cells.

## Consequences (owner decisions 2026-10-19)

- Engines: OLS, Elastic-Net and LightGBM. Random forest and gradient boosting
  are dropped (`docs/exploratory-mode-engines.md`).
- Guidance: OLS or Elastic-Net for typical psychology samples (N 300–1,000).
  Add LightGBM at about N ≥ 1,000, or when effects may be U-shaped or otherwise
  non-monotone.
- Open: how to show flagged interactions (hide by default, or require more
  folds).

## Raw outputs

Per-dataset link and dataset tables for all 36 cells are in the GitHub
prerelease `exploratory-validation-2026-10-19-39184696821`. They are not
committed.
