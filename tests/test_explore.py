"""Exploratory mode: fold-level noise bar, tiers and depth-by-depth recursion."""
import numpy as np
import pandas as pd
import pytest

from open.explore import ExploreConfig, assess_target, explore, tier
from open.learners.ols import OLSPredictor


def _chain(n=800, seed=2026):
    # X4 -> X1 -> Y <- X3 <- X2 ; X5/X11 substitute pair -> Y ; X6..X10 noise
    r = np.random.default_rng(seed)
    d = {f'X{i}': r.normal(size=n) for i in (2, 4, 5, 6, 7, 8, 9, 10)}
    d['X1'] = .5 * d['X4'] + r.normal(size=n)
    d['X3'] = .5 * d['X2'] + r.normal(size=n)
    d['X11'] = d['X5'] + .3 * r.normal(size=n)
    d['Y'] = .3 * d['X1'] + .3 * d['X3'] + .25 * (d['X5'] + d['X11']) / 2 + r.normal(size=n)
    return pd.DataFrame(d)


@pytest.fixture(scope='module')
def chain_result():
    return explore(_chain(), 'Y', engine='ols', config=ExploreConfig(max_depth=2, interactions=False))


def test_tier_boundaries_are_inclusive():
    assert tier(.8, .8, .4) == 'kept'
    assert tier(.79, .8, .4) == 'flagged'
    assert tier(.4, .8, .4) == 'flagged'
    assert tier(.39, .8, .4) == 'dropped'
    assert tier(8 / 10, .8, .4) == 'kept'


def test_recovers_planted_chain_and_shared_pair(chain_result):
    res = chain_result
    depth = dict(zip(res.nodes.variable, res.nodes.depth))
    assert depth == {'Y': 0, 'X1': 1, 'X3': 1, 'X5': 1, 'X11': 1, 'X4': 2, 'X2': 2}
    kept = res.links[res.links.tier == 'kept']
    assert set(zip(kept.source, kept.target, kept.kind)) == {
        ('X1', 'Y', 'unique'), ('X3', 'Y', 'unique'), ('X5+X11', 'Y', 'shared'),
        ('X4', 'X1', 'unique'), ('X2', 'X3', 'unique')}
    parents = dict(zip(res.nodes.variable, res.nodes.parents))
    assert parents['X5'] == parents['X11'] == ('Y',)
    assert res.stops['X4'] == res.stops['X2'] == 'MAX_DEPTH'
    assert '[solid ] X4 -> X1' in res.summary()


def test_pool_excludes_kept_but_returns_flagged_and_dropped(chain_result):
    res = chain_result
    d1 = res.links[res.links.depth == 1]
    not_kept = set(d1.source[(d1.kind == 'unique') & (d1.tier != 'kept')]) - {'X5', 'X11'}
    d2 = res.folds[(res.folds.depth == 2) & ~res.folds.noise & (res.folds.contrast == 'unique')]
    for target in ('X1', 'X3', 'X5', 'X11'):
        pool = set(d2.source[d2.target == target])
        assert pool == not_kept
        assert not pool & {'Y', 'X1', 'X3', 'X5', 'X11'}


def test_noise_bar_is_pooled_quantile_and_pass_is_strictly_above(chain_result):
    f = chain_result.folds
    f = f[(f.depth == 1) & (f.contrast == 'unique')]
    bar = f[f.noise].value.quantile(.95)
    assert np.allclose(f.bar, bar)
    assert (f[f.noise].shape[0]) == 10 * 11
    assert (f.passed == (f.value > bar)).all()
    links = chain_result.links
    x1 = links[(links.depth == 1) & (links.source == 'X1')].iloc[0]
    real = f[(f.source == 'X1') & ~f.noise]
    assert x1.pass_rate == pytest.approx(real.passed.mean())
    assert x1.r2_median == pytest.approx(real.value.median())


def test_pure_noise_outcome_keeps_nothing():
    r = np.random.default_rng(7)
    df = pd.DataFrame(r.normal(size=(600, 9)), columns=[f'X{i}' for i in range(1, 10)])
    df['Y'] = r.normal(size=600)
    res = explore(df, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False))
    assert (res.links.tier != 'kept').all()
    assert res.stops == {'Y': 'NOTHING_KEPT'}
    assert list(res.nodes.variable) == ['Y']


def test_missing_target_rows_are_left_out_of_that_targets_folds():
    df = _chain(n=400)
    df.loc[:19, 'X1'] = np.nan
    fold_ids = np.arange(400) % 10
    folds, links, _ = assess_target(df, 'X1', ('X2', 'X4', 'X6'), fold_ids, OLSPredictor(),
                                    ExploreConfig(interactions=False), depth=2)
    assert set(folds.fold) == set(range(10))
    assert 'X4' in set(links.source[links.tier == 'kept'])


def test_seed_reproducible_and_engine_instance_accepted():
    df = _chain(n=400, seed=3)
    a = explore(df, 'Y', engine='ols', config=ExploreConfig(max_depth=1, seed=5))
    b = explore(df, 'Y', engine=OLSPredictor(), config=ExploreConfig(max_depth=1, seed=5))
    pd.testing.assert_frame_equal(a.links, b.links)
    pd.testing.assert_frame_equal(a.folds, b.folds)
    c = explore(df, 'Y', engine='ols', config=ExploreConfig(max_depth=1, seed=6, interactions=False))
    unique = [r.folds[r.folds.contrast == 'unique'].value.to_numpy() for r in (a, c)]
    assert not np.allclose(*unique)


def test_elastic_net_engine_runs():
    res = explore(_chain(n=400), 'Y', engine='elastic_net',
                  config=ExploreConfig(max_depth=1, folds=5, interactions=False))
    assert {'X1', 'X3'} <= set(res.links.source[res.links.tier != 'dropped'])


def test_invalid_inputs_raise():
    df = _chain(n=400)
    with pytest.raises(ValueError):
        ExploreConfig(keep=.4, flag=.8)
    with pytest.raises(ValueError):
        ExploreConfig(noise_quantile=1)
    with pytest.raises(ValueError):
        explore(df, 'Y', predictors=['Y', 'X1'])
    bad = df.copy()
    bad.loc[0, 'Y'] = np.nan
    with pytest.raises(ValueError, match='missing'):
        explore(bad, 'Y')
    with pytest.raises(ValueError, match='too small'):
        explore(df.iloc[:40], 'Y')
    with pytest.raises(ValueError, match='reserved|without'):
        explore(df.rename(columns={'X1': 'X1+X2'}), 'Y')


def test_pair_classes_shared_and_partner_screening():
    r = np.random.default_rng(11)
    n = 800
    a = r.normal(size=n)
    d = pd.DataFrame({'S1': a + .1 * r.normal(size=n), 'S2': a + .1 * r.normal(size=n),  # substitutes
                      'P': r.normal(size=n)})
    d['Q'] = d.P + .6 * r.normal(size=n)                                   # proxy of P only
    d['N1'], d['N2'] = r.normal(size=n), r.normal(size=n)
    d['Y'] = .35 * a + .4 * d.P + r.normal(size=n)
    res = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1, interactions=False))
    pairs = res.pairs.set_index(['a', 'b'])
    assert pairs.loc[('S1', 'S2'), 'relation'] == 'shared_redundant'
    assert pairs.loc[('P', 'Q'), 'relation'] == 'partner_screening'
    assert pairs.loc[('P', 'Q'), 'screened_member'] == 'Q'
    assert pairs.loc[('P', 'Q'), 'screened_by'] == 'P'
    kept = res.links[res.links.tier == 'kept']
    assert set(kept.source) >= {'P', 'S1+S2'}
    assert 'Q' not in set(res.nodes.variable)
    assert 'Correlated pairs:' in res.summary()


def test_planted_interaction_is_detected_without_main_effect_of_partner():
    r = np.random.default_rng(4)
    n = 800
    d = pd.DataFrame(r.normal(size=(n, 6)), columns=['A', 'B', 'C', 'D', 'E', 'F'])
    d['Y'] = .4 * d.A + .3 * d.C + .35 * d.A * d.B + r.normal(size=n)
    res = explore(d, 'Y', engine='ols', config=ExploreConfig(max_depth=1))
    inter = res.links[res.links.kind == 'interaction'].set_index('source')
    assert inter.loc['A*B', 'tier'] == 'kept'
    assert inter.loc['A*B', 'members'] == ('A', 'B')
    f = res.folds[res.folds.contrast == 'interaction']
    assert f.noise.sum() == f[~f.noise].shape[0]            # one noise comparator per real pair
    assert 'B' not in set(res.nodes.variable)              # interactions promote nothing


def test_write_explore_outputs(tmp_path):
    from open.explore_report import write_explore
    res = explore(_chain(n=400), 'Y', engine='ols', config=ExploreConfig(max_depth=2, interactions=False))
    out = write_explore(res, tmp_path / 'out')
    for name in ('nodes.csv', 'links.csv', 'pairs.csv', 'folds.csv', 'noise-bars.csv', 'settings.json',
                 'summary.txt', 'network.png'):
        assert (out / name).stat().st_size > 0
    links = pd.read_csv(out / 'links.csv')
    assert {'pass_rate', 'tier', 'r2_median', 'x_noise', 'noise_bar'} <= set(links.columns)
