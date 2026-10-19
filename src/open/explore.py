"""Exploratory mode: depth-by-depth recursion with fold-level noise tests.

See explore_report for tables and the graph.

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
TIERS = ("kept", "flagged", "dropped", "hidden")
# "x noise" is shown only when the noise bar is at least this; below it the
# ratio divides by about zero (e.g. Elastic-Net shrinks noise columns to 0).
RATIO_MIN_BAR = 0.001


@dataclass(frozen=True)
class ExploreConfig:
    folds: int = 10
    keep: float = 0.80
    flag: float = 0.40
    noise_quantile: float = 0.95
    tuning_folds: int = 3
    dependence_pair: float = 0.70
    noise_groups: int = 10
    interactions: bool = True
    max_interaction_pairs: int = 100
    hide_below: float | None = 0.001
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
        if type(self.max_interaction_pairs) is not int or self.max_interaction_pairs < 0:
            raise ValueError("max_interaction_pairs must be a nonnegative integer")
        if self.hide_below is not None and (isinstance(self.hide_below, bool)
                                            or not 0 <= self.hide_below < 1):
            raise ValueError("hide_below must be None or in [0, 1)")
        if self.max_depth is not None and (type(self.max_depth) is not int or self.max_depth < 1):
            raise ValueError("max_depth must be None or a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


@dataclass(frozen=True)
class ExploreResult:
    nodes: pd.DataFrame     # variable, depth, parents
    links: pd.DataFrame     # one row per unique, shared-group or interaction relation
    pairs: pd.DataFrame     # dependence-pair classes (C4)
    folds: pd.DataFrame     # every fold-level contrast, real and noise
    bars: pd.DataFrame      # noise bar per target and contrast
    stops: dict
    settings: dict

    def summary(self) -> str:
        s = self.settings
        lines = [f"Outcome: {s['outcome']}  (engine {s['engine']}, N = {s['n']}, {s['folds']} folds; "
                 f"kept >= {s['keep']:.0%}, flagged >= {s['flag']:.0%} of folds above the "
                 f"{s['noise_quantile']:.0%} noise bar)"]
        shown = self.links[self.links.tier.isin(["kept", "flagged"])]
        for depth, group in shown.groupby("depth"):
            lines.append(f"Depth {depth}:")
            for _, r in group.iterrows():
                style = "solid " if r.tier == "kept" else "dotted"
                ratio = f", {r.r2_above_noise:+.4f} above noise"
                what = "R2 gained" if r.kind == "interaction" else "R2 lost"
                lines.append(f"  [{style}] {r.source} -> {r.target}  ({r.kind}) passed {r.pass_rate:.0%}, "
                             f"{what} {r.r2_median:.4f}{ratio}")
        flagged_pairs = self.pairs[self.pairs.tier != "dropped"]
        if len(flagged_pairs):
            lines.append("Correlated pairs:")
            for _, r in flagged_pairs.iterrows():
                extra = f" ({r.screened_member} screened by {r.screened_by})" if isinstance(r.screened_member, str) else ""
                lines.append(f"  {r.a} ~ {r.b} -> {r.target}: {r.relation} in {r.rate:.0%} of folds "
                             f"[{r.tier}]{extra}")
        if shown.empty:
            lines.append("No link reached the flagged tier.")
        hidden = int((self.links.tier == "hidden").sum()) if len(self.links) else 0
        if hidden:
            lines.append(f"Hidden: {hidden} relation(s) passed enough folds but have median R2 below "
                         f"{s['hide_below']} (hide_below).")
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


def _dependence(data: pd.DataFrame, candidates: tuple[str, ...], tau: float):
    """Candidate pairs with |Spearman| >= tau, and their connected components (size >= 2).

    Uses predictor columns only, never the target.
    """
    if len(candidates) < 2:
        return (), ()
    order = {c: i for i, c in enumerate(candidates)}
    edges = tuple(sorted(((*sorted((p.a, p.b), key=order.get), p.score)
                          for p in score_dependence(data[list(candidates)], candidates, tau_dep=tau)
                          if p.candidate), key=lambda e: (order[e[0]], order[e[1]])))
    parent = {c: c for c in candidates}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in edges:
        parent[root(a)] = root(b)
    groups = {}
    for c in candidates:
        groups.setdefault(root(c), []).append(c)
    components = tuple(sorted((tuple(sorted(g, key=order.get)) for g in groups.values() if len(g) > 1),
                              key=lambda g: order[g[0]]))
    return edges, components


def _risk(model: Predictor, params, Xtr, ytr, Xte, yte, columns) -> float:
    if not columns:
        return float(np.mean((yte - ytr.mean()) ** 2))
    m = model.clone()
    m.fit(Xtr[list(columns)], ytr, params)
    pred = np.asarray(m.predict(Xte[list(columns)]), dtype=float)
    if pred.shape != yte.shape or not np.isfinite(pred).all():
        raise ValueError("engine returned invalid held-out predictions")
    return float(np.mean((yte - pred) ** 2))


def _apply_bars(folds: pd.DataFrame, q: float) -> pd.DataFrame:
    folds = folds.copy()
    bars = folds[folds.noise].groupby("contrast").value.quantile(q, interpolation="linear")
    folds["bar"] = folds.contrast.map(bars)
    folds["passed"] = folds.value > folds.bar
    return folds


def _link(depth, target, kind, source, members, g, config, **extra):
    """One relation's tally.

    ``fold_tier`` comes from the fold pass rate alone. ``tier`` is final: a kept
    or flagged relation whose median R² is below ``hide_below`` becomes
    "hidden", is not drawn and promotes nothing.
    """
    rate = float(g.passed.mean())
    bar = float(g.bar.iloc[0])
    med = float(g.value.median())
    fold_tier = tier(rate, config.keep, config.flag)
    final = fold_tier
    if fold_tier != "dropped" and config.hide_below is not None and med < config.hide_below:
        final = "hidden"
    return dict(depth=depth, target=target, source=source, kind=kind, members=tuple(members),
                pass_rate=rate, fold_tier=fold_tier, tier=final, r2_median=med,
                r2_min=float(g.value.min()), r2_max=float(g.value.max()), noise_bar=bar,
                r2_above_noise=med - bar, x_noise=med / bar if bar >= RATIO_MIN_BAR else np.nan, **extra)


PAIR_COLUMNS = ["depth", "target", "a", "b", "spearman", "relation", "rate", "tier", "screened_member",
                "screened_by", "rate_both_proximal", "rate_partner_screening", "rate_shared_redundant"]


def _classify_pairs(folds, edges, depth, target, config):
    """Per-fold relation of each dependence pair, tallied across folds (C4).

    both_proximal: both pass unique. partner_screening: one passes unique and
    the other passes only broad (it is screened by the first).
    shared_redundant: neither passes unique but dropping both passes.
    """
    real = folds[~folds.noise]
    passed = {(r.contrast, r.source, r.fold): bool(r.passed) for r in real.itertuples()}
    rows = []
    for a, b, score in edges:
        counts = {"both_proximal": 0, "partner_screening": 0, "shared_redundant": 0}
        screened = {}
        for k in range(config.folds):
            ua, ub = passed[("unique", a, k)], passed[("unique", b, k)]
            ba, bb = passed[("broad", a, k)], passed[("broad", b, k)]
            if ua and ub:
                counts["both_proximal"] += 1
            elif (ua and bb) or (ub and ba):
                counts["partner_screening"] += 1
                member, by = (b, a) if ua else (a, b)
                screened[(member, by)] = screened.get((member, by), 0) + 1
            elif not ua and not ub and passed[("pair", f"{a}+{b}", k)]:
                counts["shared_redundant"] += 1
        order = list(counts)
        relation = max(order, key=lambda r: (counts[r], -order.index(r)))
        rate = counts[relation] / config.folds
        member, by = max(screened, key=screened.get) if relation == "partner_screening" and screened \
            else (None, None)
        rows.append(dict(depth=depth, target=target, a=a, b=b, spearman=score,
                         relation=relation if rate > 0 else "none", rate=rate,
                         tier=tier(rate, config.keep, config.flag) if rate > 0 else "dropped",
                         screened_member=member, screened_by=by,
                         **{f"rate_{r}": c / config.folds for r, c in counts.items()}))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _pair_features(xa, xb, train, knots=3):
    """Centered product plus the spline tensor block with the pair's own curves projected out.

    Everything is fitted on the training rows only.
    """
    from sklearn.preprocessing import SplineTransformer

    def spline(x):
        x = np.where(np.isnan(x), np.nanmedian(x[train]), x)
        st = SplineTransformer(n_knots=knots, degree=3, knots="quantile", extrapolation="linear",
                               include_bias=False).fit(x[train][:, None])
        z = (x - x[train].mean()) / (x[train].std() or 1.0)
        return st.transform(x[:, None]), z

    A, za = spline(xa)
    B, zb = spline(xb)
    tensor = (A[:, :, None] * B[:, None, :]).reshape(len(xa), -1)
    marginal = np.column_stack([np.ones(len(xa)), A, B])
    projection = np.linalg.lstsq(marginal[train], tensor[train], rcond=None)[0]
    return np.column_stack([za * zb, tensor - marginal @ projection])


def _interactions(Xo, yo, fold_o, active, candidates, noise_of, depth, target, config):
    """Held-out R² gained by a pair's joint term over an additive spline model (C5).

    Engine-independent. In each fold an additive spline ridge model of all
    candidates and noise columns is tuned on the training folds. For each pair,
    a ridge model of its joint term (centered product plus the spline tensor
    with the pair's own curves projected out, with its own cross-validated
    penalty) is fitted to that model's training residuals. The gain is the
    drop in held-out error relative to the mean-only model.

    Real pairs are each kept or flagged source with every other candidate, so
    an interaction partner needs no main effect of its own. Each real pair
    (a, b) has one noise comparator, (a, noise b) or (noise a, b) at random,
    and these are pooled across folds for the bar. Returns fold rows and the
    member tuple of each real pair.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from open.pair_validation import PairValidationConfig, _tune
    real = []
    for a in active:
        for c in candidates:
            pair = tuple(sorted((a, c), key=candidates.index))
            if c != a and pair not in real:
                real.append(pair)
    real = tuple(real[:config.max_interaction_pairs])
    if not real:
        return pd.DataFrame(), {}
    flip = np.random.default_rng(_sub_seed(config.seed, "pair-noise", depth, target)).random(len(real)) < .5
    fake = tuple((a, noise_of[b]) if f else (noise_of[a], b) for (a, b), f in zip(real, flip))
    alphas = np.logspace(-2, 4, 13)
    rows = []
    for k in range(config.folds):
        train, test = fold_o != k, fold_o == k
        pv = PairValidationConfig(seed=_sub_seed(config.seed, "pairs", depth, target, k))
        base, _, _ = _tune(Xo[train], yo[train], None, pv)
        fitted = base.predict(Xo)
        res_tr, res_te = yo[train] - fitted[train], yo[test] - fitted[test]
        ref_mse = float(np.mean((yo[test] - yo[train].mean()) ** 2))
        inner = KFold(3, shuffle=True, random_state=_sub_seed(config.seed, "pair-ridge", depth, target, k))
        for i, (ma, mb) in enumerate(real + fake):
            F = _pair_features(Xo[ma].to_numpy(dtype=float), Xo[mb].to_numpy(dtype=float), train)
            scale = StandardScaler().fit(F[train])
            ridge = RidgeCV(alphas=alphas, cv=inner).fit(scale.transform(F[train]), res_tr)
            pred = ridge.predict(scale.transform(F[test]))
            gain = (float(np.mean(res_te ** 2)) - float(np.mean((res_te - pred) ** 2))) / ref_mse
            rows.append(dict(depth=depth, target=target, fold=k, contrast="interaction",
                             source="*".join(m.removeprefix(NOISE_PREFIX) for m in (ma, mb)),
                             noise=i >= len(real), value=gain))
    return pd.DataFrame(rows), {"*".join(p): p for p in real}


def assess_target(data: pd.DataFrame, target: str, candidates: tuple[str, ...], fold_ids: np.ndarray,
                  engine, config: ExploreConfig, depth: int = 1):
    """Fold-level contrasts, links and pair classes for one target.

    Returns (folds, links, pairs).
    """
    candidates = tuple(candidates)
    if not candidates or target in candidates:
        raise ValueError("candidates must be nonempty and exclude the target")
    model = _predictor(engine, target, config.seed)
    rng = np.random.default_rng(_sub_seed(config.seed, "noise", depth, target))
    noise_of = {c: f"{NOISE_PREFIX}{c}" for c in candidates}
    noise = {noise_of[c]: rng.permutation(data[c].to_numpy(dtype=float, na_value=np.nan))
             for c in candidates}
    X = pd.concat([data[list(candidates)].astype(float).reset_index(drop=True),
                   pd.DataFrame(noise)], axis=1)
    noise_cols = tuple(noise)
    columns = candidates + noise_cols
    edges, groups = _dependence(data, candidates, config.dependence_pair)
    y = data[target].to_numpy(dtype=float, na_value=np.nan)
    observed = ~np.isnan(y)
    group_rng = np.random.default_rng(_sub_seed(config.seed, "groups", depth, target))
    noise_sets = {m: [tuple(sorted(group_rng.choice(len(noise_cols), m, replace=False)))
                      for _ in range(config.noise_groups)]
                  for m in sorted({len(g) for g in groups})}
    pair_sets = [tuple(sorted(group_rng.choice(len(noise_cols), 2, replace=False)))
                 for _ in range(config.noise_groups)] if edges else []

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

        def lost(drop):
            kept = tuple(x for x in columns if x not in drop)
            return (_risk(model, params, Xtr, ytr, Xte, yte, kept) - full) / base

        for c in columns:
            alone = _risk(model, params, Xtr, ytr, Xte, yte, (c,))
            name = c.removeprefix(NOISE_PREFIX)
            add("unique", name, lost({c}), c in noise)
            add("broad", name, (base - alone) / base, c in noise)
        for a, b, _ in edges:
            add("pair", f"{a}+{b}", lost({a, b}), False)
        for i, idx in enumerate(pair_sets):
            add("pair", f"noise-set-{i}", lost({noise_cols[j] for j in idx}), True)
        for g in groups:
            add(f"group{len(g)}", "+".join(g), lost(set(g)), False)
        for m, sets in noise_sets.items():
            for i, idx in enumerate(sets):
                add(f"group{m}", f"noise-set-{i}", lost({noise_cols[j] for j in idx}), True)

    folds = _apply_bars(pd.DataFrame(rows), config.noise_quantile)
    real = folds[~folds.noise]
    links = []
    for c in candidates:
        g = real[(real.contrast == "unique") & (real.source == c)]
        b = real[(real.contrast == "broad") & (real.source == c)]
        links.append(_link(depth, target, "unique", c, (c,), g, config,
                           broad_pass_rate=float(b.passed.mean()), broad_r2_median=float(b.value.median())))
    for grp in groups:
        source = "+".join(grp)
        g = real[(real.contrast == f"group{len(grp)}") & (real.source == source)]
        links.append(_link(depth, target, "shared", source, grp, g, config))
    pairs = _classify_pairs(folds, edges, depth, target, config)

    if config.interactions:
        active = {l["source"] for l in links if l["kind"] == "unique" and l["fold_tier"] != "dropped"}
        for l in links:
            if l["kind"] == "shared" and l["fold_tier"] != "dropped":
                active.update(l["members"])
        inter, members = _interactions(X[observed].reset_index(drop=True), y[observed], fold_ids[observed],
                                       [c for c in candidates if c in active], candidates, noise_of,
                                       depth, target, config)
        if len(inter):
            inter = _apply_bars(inter, config.noise_quantile)
            folds = pd.concat([folds, inter], ignore_index=True)
            for source, pair in members.items():
                g = inter[~inter.noise & (inter.source == source)]
                links.append(_link(depth, target, "interaction", source, pair, g, config))
    return folds, pd.DataFrame(links), pairs


def explore(data: pd.DataFrame, outcome: str, predictors=None, engine="ols",
            config: ExploreConfig | None = None) -> ExploreResult:
    """Build the exploratory network for ``outcome``, one depth at a time.

    A dependence group (correlated candidates) counts only when none of its
    members is kept by itself; otherwise the kept member already carries the
    group's signal, and promoting the rest would promote a screened proxy.
    A kept group promotes all its members.
    Interactions are reported and drawn but promote nothing.
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
        if type(v) is not str or not v or v.startswith(NOISE_PREFIX) or "+" in v or "*" in v:
            raise ValueError(f"variable name {v!r} must be a string without '+', '*' or {NOISE_PREFIX!r}")
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
    all_folds, all_links, all_pairs = [], [], []
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
            folds, links, pairs = assess_target(data, target, pool, fold_ids, engine, config, depth)
            kept_alone = set(links.source[(links.kind == "unique") & (links.tier == "kept")])
            redundant = (links.kind == "shared") & links.members.map(lambda m: bool(set(m) & kept_alone))
            links = links[~redundant]
            all_folds.append(folds)
            all_links.append(links)
            all_pairs.append(pairs)
            promoted = set()
            for _, r in links[(links.tier == "kept") & (links.kind != "interaction")].iterrows():
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
    all_pairs = [p for p in all_pairs if len(p)]
    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=PAIR_COLUMNS)
    folds = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    bars = (folds[folds.noise].groupby(["depth", "target", "contrast"]).bar.first().reset_index()
            if len(folds) else pd.DataFrame())
    name = engine.engine_name if isinstance(engine, Predictor) else engine
    settings = dict(outcome=outcome, engine=name, n=n, **asdict(config))
    return ExploreResult(nodes, links, pairs, folds, bars, stops, settings)
