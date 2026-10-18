"""Simple recursive OLS network: fold-stable predictors, expanded depth by depth.

Depth 1: regress the outcome on every predictor in each of K training sets
(each leaves one fold out). A predictor is kept when its HC4 p-value is below
``alpha`` in at least ``threshold`` of the K fits.

Depth d + 1: each variable kept at depth d is regressed, the same way, on the
variables not yet placed in the network. Whatever is kept becomes depth d + 1
and leaves the pool. This repeats until the pool is empty or nothing is kept.
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Phase A' found HC4 p-values calibrated from N >= 275.
HC4_MIN_N = 275


def hc4_pvalues(X, y):
    """Two-sided HC4 p-values for the slopes of OLS of y on X (intercept added)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    A = np.column_stack([np.ones(n), X])
    k = A.shape[1]
    if n <= k:
        raise ValueError(f'{n} rows cannot fit {k - 1} predictors plus an intercept')
    if np.linalg.matrix_rank(A) < k:
        raise ValueError('predictors are perfectly collinear (or one is constant)')
    C = np.linalg.solve(A.T @ A, A.T)
    beta = C @ y
    e = y - A @ beta
    h = np.einsum('ij,ji->i', A, C)
    w = e ** 2 / (1 - h) ** np.minimum(4.0, h / (k / n))
    t = beta / np.sqrt((C ** 2) @ w)
    return 2 * stats.t.sf(np.abs(t[1:]), n - k), beta[1:]


@dataclass
class RecursiveOLSResult:
    nodes: pd.DataFrame      # variable, depth
    edges: pd.DataFrame      # kept predictor -> target links
    tests: pd.DataFrame      # every predictor tested for every target
    settings: dict

    def summary(self):
        lines = [f"Outcome: {self.settings['outcome']}  (N = {self.settings['n']}, "
                 f"{self.settings['folds']} folds, p < {self.settings['alpha']}, "
                 f"kept if selected in >= {self.settings['threshold']:.0%} of folds)"]
        for depth, group in self.edges.groupby('depth'):
            lines.append(f'Depth {depth}:')
            for _, e in group.iterrows():
                lines.append(f"  {e.predictor} -> {e.target}   selected {e.selection_rate:.0%}, "
                             f"std. beta {e.beta_std:+.3f}, full-sample p {e.p_full:.3g}")
        if self.edges.empty:
            lines.append('No predictor passed the threshold.')
        return '\n'.join(lines)


def _fold_ids(n, folds, seed):
    rng = np.random.default_rng(seed)
    ids = np.arange(n) % folds
    rng.shuffle(ids)
    return ids


def _test_target(Z, target, pool, fold, folds, alpha):
    """Selection rate, full-sample std. beta and p of each pool variable for target."""
    X, y = Z[pool].to_numpy(), Z[target].to_numpy()
    hits = np.zeros(len(pool))
    for k in range(folds):
        train = fold != k
        p, _ = hc4_pvalues(X[train], y[train])
        hits += p < alpha
    p_full, beta = hc4_pvalues(X, y)
    return pd.DataFrame({'target': target, 'predictor': pool, 'selection_rate': hits / folds,
                         'beta_std': beta, 'p_full': p_full})


def recursive_ols(data, outcome, predictors=None, folds=10, alpha=.05, threshold=.80,
                  max_depth=None, seed=0):
    """Build the recursive network for ``outcome``. See the module docstring.

    Rows with a missing value in any used column are dropped. Variables are
    z-scored, so ``beta_std`` is a standardized coefficient.
    """
    if predictors is None:
        predictors = [c for c in data.columns if c != outcome]
    predictors = list(predictors)
    if outcome in predictors:
        raise ValueError('the outcome cannot also be a predictor')
    if not 0 < threshold <= 1:
        raise ValueError('threshold must be in (0, 1]')
    if folds < 2:
        raise ValueError('folds must be at least 2')
    Z = data[[outcome] + predictors].apply(pd.to_numeric, errors='raise').dropna()
    n = len(Z)
    if n < HC4_MIN_N:
        warnings.warn(f'N = {n}: HC4 p-values were found calibrated from N >= {HC4_MIN_N}; '
                      'below that, selections are less trustworthy.', stacklevel=2)
    Z = (Z - Z.mean()) / Z.std(ddof=1)
    fold = _fold_ids(n, folds, seed)

    nodes = [{'variable': outcome, 'depth': 0}]
    tests, edges = [], []
    frontier, pool, depth = [outcome], predictors, 0
    while frontier and pool and (max_depth is None or depth < max_depth):
        depth += 1
        kept = []
        for target in frontier:
            t = _test_target(Z, target, pool, fold, folds, alpha)
            t['depth'] = depth
            # Small tolerance so 8/10 counts as 80%.
            t['selected'] = t.selection_rate >= threshold - 1e-9
            tests.append(t)
            edges.append(t[t.selected])
            kept += [v for v in t.predictor[t.selected] if v not in kept]
        nodes += [{'variable': v, 'depth': depth} for v in kept]
        frontier, pool = kept, [v for v in pool if v not in kept]

    cols = ['depth', 'target', 'predictor', 'selection_rate', 'beta_std', 'p_full']
    tests = pd.concat(tests, ignore_index=True)[cols + ['selected']] if tests else \
        pd.DataFrame(columns=cols + ['selected'])
    edges = pd.concat(edges, ignore_index=True)[cols] if edges else pd.DataFrame(columns=cols)
    settings = dict(outcome=outcome, n=n, folds=folds, alpha=alpha, threshold=threshold,
                    max_depth=max_depth, seed=seed)
    return RecursiveOLSResult(pd.DataFrame(nodes), edges.reset_index(drop=True), tests, settings)
