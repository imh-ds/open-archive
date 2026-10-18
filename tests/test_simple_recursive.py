"""Simple recursive OLS: HC4 arithmetic, threshold rule and depth-by-depth pool."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from open.simple_recursive import hc4_pvalues, recursive_ols


def _chain_data(n=1500, seed=1):
    # X4 -> X1 -> Y <- X3 ; X2 -> X3 ; X5..X8 noise
    r = np.random.default_rng(seed)
    d = {f'X{i}': r.normal(size=n) for i in (2, 4, 5, 6, 7, 8)}
    d['X1'] = .5 * d['X4'] + r.normal(size=n)
    d['X3'] = .5 * d['X2'] + r.normal(size=n)
    d['Y'] = .4 * d['X1'] + .4 * d['X3'] + r.normal(size=n)
    return pd.DataFrame(d)


def test_hc4_matches_loop_formula():
    r = np.random.default_rng(3)
    X = r.normal(size=(60, 3))
    y = X @ [1, 0, -.5] + r.normal(size=60) * (1 + X[:, 0] ** 2)
    A = np.column_stack([np.ones(60), X])
    inv = np.linalg.inv(A.T @ A)
    b = inv @ A.T @ y
    e = y - A @ b
    V = np.zeros((4, 4))
    for i in range(60):
        h = A[i] @ inv @ A[i]
        V += np.outer(A[i], A[i]) * e[i] ** 2 / (1 - h) ** min(4, h / (4 / 60))
    se = np.sqrt(np.diag(inv @ V @ inv))
    p_ref = 2 * stats.t.sf(np.abs(b / se), 56)[1:]
    p, beta = hc4_pvalues(X, y)
    np.testing.assert_allclose(p, p_ref, rtol=1e-10)
    np.testing.assert_allclose(beta, b[1:], rtol=1e-10)


def test_recovers_chain_depth_by_depth():
    res = recursive_ols(_chain_data(), 'Y')
    depth = dict(zip(res.nodes.variable, res.nodes.depth))
    assert depth == {'Y': 0, 'X1': 1, 'X3': 1, 'X4': 2, 'X2': 2}
    links = set(zip(res.edges.predictor, res.edges.target))
    assert links == {('X1', 'Y'), ('X3', 'Y'), ('X4', 'X1'), ('X2', 'X3')}
    # Depth-2 regressions use only the variables left after depth 1.
    d2 = res.tests[res.tests.depth == 2]
    assert set(d2.predictor) == {'X2', 'X4', 'X5', 'X6', 'X7', 'X8'}
    assert 'Y' not in set(res.tests.predictor)
    assert 'Depth 2:' in res.summary()


def test_threshold_is_inclusive_and_rates_are_fold_counts():
    res = recursive_ols(_chain_data(n=400, seed=5), 'Y', max_depth=1)
    rates = res.tests.selection_rate * 10
    np.testing.assert_allclose(rates, np.round(rates))
    assert (res.tests.selected == (res.tests.selection_rate >= .8 - 1e-9)).all()
    strict = recursive_ols(_chain_data(n=400, seed=5), 'Y', max_depth=1, threshold=1.0)
    assert set(strict.edges.predictor) <= set(res.edges.predictor)


def test_seed_reproducible_and_missing_rows_dropped():
    df = _chain_data(n=600)
    df.loc[:4, 'X5'] = np.nan
    a = recursive_ols(df, 'Y', seed=11)
    b = recursive_ols(df, 'Y', seed=11)
    pd.testing.assert_frame_equal(a.tests, b.tests)
    assert a.settings['n'] == 595


def test_small_n_warns_and_bad_inputs_raise():
    df = _chain_data(n=200)
    with pytest.warns(UserWarning, match='275'):
        recursive_ols(df, 'Y', max_depth=1)
    with pytest.raises(ValueError):
        recursive_ols(df, 'Y', predictors=['Y', 'X1'])
    big = _chain_data(n=400)
    big['X9'] = big.X1 * 2
    with pytest.raises(ValueError, match='collinear'):
        recursive_ols(big, 'Y')
