# open (development archive)

> **This is the development archive of OPEN.** It was named `imh-ds/open` until
> 2026-10-19. The published method, its validation evidence and reproduction
> instructions are in **[imh-ds/open](https://github.com/imh-ds/open)**.
>
> Links in this repository's results files of the form `github.com/imh-ds/open/...`
> (commits, Actions runs, releases) refer to **this** repository. The final work
> is on branches `codex/exploratory-mode` and `codex/multiengine-preflight`.

Outcome-centered Predictive Expansion Network

## Development usage

OPEN is a development package. The current configuration contract covers one
cross-sectional analysis with independent rows, a continuous outcome using MSE,
numeric continuous or ordinal scale-level predictors, and one fixed deterministic
elastic-net learner. Statistical analysis is not implemented in this phase.

After installing the package in a Python 3.11+ environment, load and inspect the
example configuration:

```python
from open.config import load_config

config = load_config("configs/default.yaml")
print(config.analysis.root, config.performance_cv.folds)
```

Set `analysis.root` to the outcome column and set `analysis.random_seed` explicitly.
Thresholds are configurable engineering defaults, not calibrated scientific
constants. A future OPEN arrow represents predictive contribution under a stated
learner, candidate set, loss, splits, and thresholds; it does not establish
causal direction. Fitted preprocessing, dependence estimation, and tuning must
use training folds only. This package has no claim of methodological validation,
simulation calibration, or applied validity yet.
