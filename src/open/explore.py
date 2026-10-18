"""Exploratory mode: depth-by-depth recursion with fold-level noise tests.

The rows are split into K folds once. For each target at a depth, every fold
fits the engine on the other K - 1 folds with the candidates plus one noise
column per candidate (a row-shuffled copy, unrelated to anything). On the
held-out fold it measures, in R² units of the mean-only model's error:

- ``unique``: R² lost when a column is left out (LOCO);
- ``broad``: R² gained by a column on its own;
- ``group``: R² lost when a dependence component (correlated candidates)
  is left out together.

Noise bar: the ``noise_quantile`` of the noise columns' values for the same
contrast, pooled over all folds of that target. A real column passes a fold
when its value is above the bar. The pass rate across folds gives the tier:
kept (>= ``keep``), flagged (>= ``flag``) or dropped. Kept variables become
the next depth's targets. Flagged and dropped ones stay in the pool.

Every contrast reuses the parameters tuned for that fold's full model, so
real and noise columns are treated identically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from open.config import AnalysisConfig, LearnerConfig, OpenConfig
from open.dependence.spearman import score_dependence
from open.learners import make_predictor
from open.learners.base import Predictor

NOISE_PREFIX = "__noise__"
TIERS = ("kept", "flagged", "dropped")


@dataclass(frozen=True)
class ExploreConfig:
    folds: int = 10
    keep: float = 0.80
    flag: float = 0.40
    noise_quantile: float = 0.95
    tuning_folds: int = 3
    dependence_pair: float = 0.70
    noise_groups: int = 10
    max_depth: int | None = None
    seed: int = 0

    def __post_init__(self):
        if type(self.folds) is not int or self.folds < 2:
            raise ValueError("folds must be an integer of at least 2")
        if not 0 < self.flag <= self.keep <= 1:
            raise ValueError("need 0 < flag <= keep <= 1")
        if not 0 < self.noise_quantile < 1:
            raise ValueError("noise_quantile must be in (0, 1)")
        if type(self.tuning_folds) is not int or self.tuning_folds < 2:
            raise ValueError("tuning_folds must be an integer of at least 2")
        if not 0 <= self.dependence_pair <= 1:
            raise ValueError("dependence_pair must be in [0, 1]")
        if type(self.noise_groups) is not int or self.noise_groups < 1:
            raise ValueError("noise_groups must be a positive integer")
        if self.max_depth is not None and (type(self.max_depth) is not int or self.max_depth < 1):
            raise ValueError("max_depth must be None or a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


@dataclass(frozen=True)
class ExploreResult:
    nodes: pd.DataFrame     # variable, depth, parents
    links: pd.DataFrame     # one row per (target, source) unique or group relation
    folds: pd.DataFrame     # every fold-level contrast, real and noise
    bars: pd.DataFrame      # noise bar per target and contrast
    stops: dict
    settings: dict

    def summary(self) -> str:
        s = self.settings
        lines = [f"Outcome: {s['outcome']}  (engine {s['engine']}, N = {s['n']}, {s['folds']} folds; "
                 f"kept >= {s['keep']:.0%}, flagged >= {s['flag']:.0%} of folds above the "
                 f"{s['noise_quantile']:.0%} noise bar)"]
        shown = self.links[self.links.tier != "dropped"]
        for depth, group in shown.groupby("depth"):
            lines.append(f"Depth {depth}:")
            for _, r in group.iterrows():
                style = "solid " if r.tier == "kept" else "dotted"
                ratio = "" if pd.isna(r.x_noise) else f", {r.x_noise:.1f}x noise"
                lines.append(f"  [{style}] {r.source} -> {r.target}  ({r.kind}) passed {r.pass_rate:.0%}, "
                             f"R2 lost {r.r2_median:.4f}{ratio}")
        if shown.empty:
            lines.append("No link reached the flagged tier.")
        return "\n".join(lines)


def tier(rate: float, keep: float, flag: float) -> str:
    eps = 1e-9  # so 8/10 counts as 80%
    if rate >= keep - eps:
        return "kept"
    if rate >= flag - eps:
        return "flagged"
    return "dropped"


def _predictor(engine, outcome: str, seed: int) -> Predictor:
    if isinstance(engine, Predictor):
        return engine
    config = OpenConfig(analysis=AnalysisConfig(root=outcome, random_seed=seed),
                        learner=LearnerConfig(library=(engine,)))
    return make_predictor(config)


def _sub_seed(seed: int, *parts) -> int:
    text = "|".join(str(p) for p in (seed, *parts))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def _components(data: pd.DataFrame, candidates: tuple[str, ...], tau: float) -> tuple[tuple[str, ...], ...]:
    """Connected components (size >= 2) of candidate pairs with |Spearman| >= tau.

    Uses predictor columns only, never the target.
    """
    if len(candidates) < 2:
        return ()
    edges = [(p.a, p.b) for p in score_dependence(data[list(candidates)], candidates, tau_dep=tau)
             if p.candidate]
    parent = {c: c for c in candidates}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[root(a)] = root(b)
    groups = {}
    for c in candidates:
        groups.setdefault(root(c), []).append(c)
    order = {c: i for i, c in enumerate(candidates)}
    return tuple(sorted((tuple(sorted(g, key=order.get)) for g in groups.values() if len(g) > 1),
                        key=lambda g: order[g[0]]))


def _risk(model: Predictor, params, Xtr, ytr, Xte, yte, columns) -> float:
    if not columns:
        return float(np.mean((yte - ytr.mean()) ** 2))
    m = model.clone()
    m.fit(Xtr[list(columns)], ytr, params)
    pred = np.asarray(m.predict(Xte[list(columns)]), dtype=float)
    if pred.shape != yte.shape or not np.isfinite(pred).all():
        raise ValueError("engine returned invalid held-out predictions")
    return float(np.mean((yte - pred) ** 2))


def assess_target(data: pd.DataFrame, target: str, candidates: tuple[str, ...], fold_ids: np.ndarray,
                  engine, config: ExploreConfig, depth: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold-level contrasts and per-source tiers for one target. Returns (folds, links)."""
    candidates = tuple(candidates)
    if not candidates or target in candidates:
        raise ValueError("candidates must be nonempty and exclude the target")
    model = _predictor(engine, target, config.seed)
    rng = np.random.default_rng(_sub_seed(config.seed, "noise", depth, target))
    noise = {f"{NOISE_PREFIX}{c}": rng.permutation(data[c].to_numpy(dtype=float, na_value=np.nan))
             for c in candidates}
    X = pd.concat([data[list(candidates)].astype(float).reset_index(drop=True),
                   pd.DataFrame(noise)], axis=1)
    noise_cols = tuple(noise)
    columns = candidates + noise_cols
    groups = _components(data, candidates, config.dependence_pair)
    y = data[target].to_numpy(dtype=float, na_value=np.nan)
    observed = ~np.isnan(y)
    group_rng = np.random.default_rng(_sub_seed(config.seed, "groups", depth, target))
    noise_sets = {m: [tuple(sorted(group_rng.choice(len(noise_cols), m, replace=False)))
                      for _ in range(config.noise_groups)]
                  for m in sorted({len(g) for g in groups})}

    rows = []
    for k in range(config.folds):
        train = observed & (fold_ids != k)
        test = observed & (fold_ids == k)
        if train.sum() < config.tuning_folds * 2 or test.sum() < 1:
            raise ValueError(f"target {target!r}: fold {k} has too few observed rows")
        Xtr, Xte = X[train].reset_index(drop=True), X[test].reset_index(drop=True)
        ytr, yte = y[train], y[test]
        tuner = model.clone()
        inner = KFold(config.tuning_folds, shuffle=True, random_state=_sub_seed(config.seed, "tune", k))
        params = dict(tuner.tune(Xtr[list(columns)], ytr, inner, "mse"))
        base = _risk(model, params, Xtr, ytr, Xte, yte, ())
        full = _risk(model, params, Xtr, ytr, Xte, yte, columns)
        if base <= 0:
            raise ValueError(f"target {target!r} is constant in held-out fold {k}")

        def add(contrast, source, value, is_noise):
            rows.append(dict(depth=depth, target=target, fold=k, contrast=contrast, source=source,
                             noise=is_noise, value=value))

        for c in columns:
            without = _risk(model, params, Xtr, ytr, Xte, yte, tuple(x for x in columns if x != c))
            alone = _risk(model, params, Xtr, ytr, Xte, yte, (c,))
            name = c.removeprefix(NOISE_PREFIX)
            add("unique", name, (without - full) / base, c in noise)
            add("broad", name, (base - alone) / base, c in noise)
        for g in groups:
            without = _risk(model, params, Xtr, ytr, Xte, yte, tuple(x for x in columns if x not in g))
            add(f"group{len(g)}", "+".join(g), (without - full) / base, False)
        for m, sets in noise_sets.items():
            for i, idx in enumerate(sets):
                drop = {noise_cols[j] for j in idx}
                without = _risk(model, params, Xtr, ytr, Xte, yte, tuple(x for x in columns if x not in drop))
                add(f"group{m}", f"noise-set-{i}", (without - full) / base, True)

    folds = pd.DataFrame(rows)
    bars = (folds[folds.noise].groupby("contrast").value
            .quantile(config.noise_quantile, interpolation="linear"))
    folds["bar"] = folds.contrast.map(bars)
    folds["passed"] = folds.value > folds.bar
    real = folds[~folds.noise]
    links = []
    for (contrast, source), g in real.groupby(["contrast", "source"], sort=False):
        if contrast == "broad":
            continue
        rate = float(g.passed.mean())
        bar = float(g.bar.iloc[0])
        med = float(g.value.median())
        links.append(dict(depth=depth, target=target, source=source,
                          kind="unique" if contrast == "unique" else "shared",
                          members=tuple(source.split("+")), pass_rate=rate,
                          tier=tier(rate, config.keep, config.flag),
                          r2_median=med, r2_min=float(g.value.min()), r2_max=float(g.value.max()),
                          noise_bar=bar, x_noise=med / bar if bar > 0 else np.nan,
                          broad_pass_rate=np.nan))
    broad = real[real.contrast == "broad"].groupby("source").passed.mean()
    for link in links:
        if link["kind"] == "unique":
            link["broad_pass_rate"] = float(broad[link["source"]])
    return folds, pd.DataFrame(links)


def explore(data: pd.DataFrame, outcome: str, predictors=None, engine="ols",
            config: ExploreConfig | None = None) -> ExploreResult:
    """Build the exploratory network for ``outcome``, one depth at a time.

    A dependence group (correlated candidates) counts only when at least one
    member isn't kept by itself. A kept group promotes all its members.
    """
    config = config or ExploreConfig()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    predictors = tuple(c for c in data.columns if c != outcome) if predictors is None else tuple(predictors)
    if outcome in predictors or len(set(predictors)) != len(predictors) or not predictors:
        raise ValueError("predictors must be distinct, nonempty and exclude the outcome")
    variables = (outcome, *predictors)
    missing = [v for v in variables if v not in data]
    if missing:
        raise ValueError(f"unknown columns: {missing}")
    for v in variables:
        s = data[v]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            raise ValueError(f"variable {v!r} must be numeric")
        if np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"variable {v!r} contains infinity")
    data = data[list(variables)].reset_index(drop=True)
    if data[outcome].isna().any():
        raise ValueError("the outcome must have no missing values")
    n = len(data)
    if n < config.folds * config.tuning_folds * 2:
        raise ValueError(f"N = {n} is too small for {config.folds} folds")
    fold_ids = np.arange(n) % config.folds
    np.random.default_rng(_sub_seed(config.seed, "folds")).shuffle(fold_ids)

    depths, parents, stops = {outcome: 0}, {}, {}
    all_folds, all_links = [], []
    frontier, depth = (outcome,), 0
    while frontier:
        if config.max_depth is not None and depth >= config.max_depth:
            stops.update((t, "MAX_DEPTH") for t in frontier)
            break
        pool = tuple(v for v in predictors if v not in depths)
        if not pool:
            stops.update((t, "NO_CANDIDATES") for t in frontier)
            break
        depth += 1
        next_parents = {}
        for target in frontier:
            folds, links = assess_target(data, target, pool, fold_ids, engine, config, depth)
            kept_alone = set(links.source[(links.kind == "unique") & (links.tier == "kept")])
            shared = links.kind == "shared"
            links = links[~shared | links.members.map(lambda m: not set(m) <= kept_alone)]
            all_folds.append(folds)
            all_links.append(links)
            promoted = set()
            for _, r in links[links.tier == "kept"].iterrows():
                promoted.update(r.members)
            if not promoted:
                stops[target] = "NOTHING_KEPT"
            for v in promoted:
                next_parents.setdefault(v, set()).add(target)
        frontier = tuple(v for v in predictors if v in next_parents)
        for v in frontier:
            depths[v] = depth
            parents[v] = tuple(sorted(next_parents[v]))

    nodes = pd.DataFrame([dict(variable=v, depth=d, parents=parents.get(v, ())) for v, d in depths.items()])
    links = pd.concat(all_links, ignore_index=True) if all_links else pd.DataFrame()
    folds = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    bars = (folds[folds.noise].groupby(["depth", "target", "contrast"]).bar.first().reset_index()
            if len(folds) else pd.DataFrame())
    name = engine.engine_name if isinstance(engine, Predictor) else engine
    settings = dict(outcome=outcome, engine=name, n=n, **asdict(config))
    return ExploreResult(nodes, links, folds, bars, stops, settings)
