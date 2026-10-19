"""Exploratory-mode validation study (revision plan §5 and §8).

Each simulated dataset has 10 candidates and a known two-depth structure:

    Y <- A (r² .09), B (r² .04), C (r² .01), shared pair S1/S2 (latent, r² .04),
         interaction C x D (r² .04; D has no main effect)
    A <- E (r² .09)      B <- F (r² .01)
    N1, N2: pure noise

Every link in a dataset uses the cell's shape: linear, U-shaped, threshold
(step) or saturating. The interaction is bilinear in linear cells and curved
(a V-shape in C, switched on by D > 0) in the others. Each shape term is
standardized within the dataset, so r² is the variance share of the term.

Cells: engine (ols, elastic_net, lightgbm) x N (300, 500, 1000) x shape.
The method runs with its defaults (10 folds, keep 80%, flag 40%, 95% noise
bar, hide floor .001, interactions on), max_depth 2.

Commands:
  probe --out DIR [--workers W]   time one dataset per engine x N x {linear, U}
  run --engine E --n N --shape S --reps LO:HI --out DIR [--workers W]
  aggregate --inputs DIR --out DIR
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open.explore import ExploreConfig, explore  # noqa: E402

ENGINES = ("ols", "elastic_net", "lightgbm")
SIZES = (300, 500, 1000)
SHAPES = ("linear", "u_shape", "threshold", "saturating")
SEED_BASE = 4_100_000_000
CONFIG = ExploreConfig(max_depth=2)

# (source, target, kind, r²)
TRUTH = (
    ("A", "Y", "unique", .09),
    ("B", "Y", "unique", .04),
    ("C", "Y", "unique", .01),
    ("S1+S2", "Y", "shared", .04),
    ("C*D", "Y", "interaction", .04),
    ("E", "A", "unique", .09),
    ("F", "B", "unique", .01),
)
NOISE_VARS = ("N1", "N2")
# Not false links: S1/S2 alone carry the latent's signal, and D matters
# through the interaction, which LOCO credits to D for nonlinear engines.
NOT_FALSE = {("Y", "S1", "unique"), ("Y", "S2", "unique"), ("Y", "D", "unique")}
ORDER = {"kept": 0, "flagged": 1, "hidden": 2, "dropped": 3}


def _std(v):
    return (v - v.mean()) / v.std()


def shape_fn(shape, x):
    if shape == "linear":
        f = x
    elif shape == "u_shape":
        f = x ** 2
    elif shape == "threshold":
        f = (x > 0.5).astype(float)
    elif shape == "saturating":
        f = np.tanh(1.5 * x)
    else:
        raise ValueError(shape)
    return _std(f)


def _child(r2, term, rng):
    return np.sqrt(r2) * term + np.sqrt(1 - r2) * rng.normal(size=len(term))


def dataset_seed(engine, n, shape, rep):
    """Engines share datasets: the seed depends on N, shape and rep only."""
    text = f"{n}|{shape}|{rep}"
    return SEED_BASE + int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % 90_000_000


def generate(n, shape, seed):
    rng = np.random.default_rng(seed)
    z = lambda: rng.normal(size=n)  # noqa: E731
    E, F, C, D, L, N1, N2 = (z() for _ in range(7))
    A = _std(_child(.09, shape_fn(shape, E), rng))
    B = _std(_child(.01, shape_fn(shape, F), rng))
    S1, S2 = L + .3 * z(), L + .3 * z()   # substitutes: r(S1, S2) about .92
    if shape == "linear":
        inter = _std(C * D)
    else:   # V-shape in C whose sign flips with D: no main effect of C or D
        inter = _std((np.abs(C) - np.sqrt(2 / np.pi)) * np.where(D > 0, 1.0, -1.0))
    terms = {"A": (.09, shape_fn(shape, A)), "B": (.04, shape_fn(shape, B)), "C": (.01, shape_fn(shape, C)),
             "L": (.04, shape_fn(shape, L)), "CD": (.04, inter)}
    y = sum(np.sqrt(r2) * t for r2, t in terms.values())
    y = y + np.sqrt(1 - sum(r2 for r2, _ in terms.values())) * z()
    return pd.DataFrame({"Y": y, "A": A, "B": B, "C": C, "D": D, "S1": S1, "S2": S2, "E": E, "F": F,
                         "N1": N1, "N2": N2})


def score(result):
    """One row per true link and per false link that reached the flagged tier or above."""
    links = result.links
    rows = []
    tested = set(zip(links.target, links.source, links.kind)) if len(links) else set()
    for source, target, kind, r2 in TRUTH:
        hit = links[(links.target == target) & (links.source == source) & (links.kind == kind)] \
            if len(links) else links
        if not len(hit) and kind == "shared":   # group dropped because a member was kept alone
            hit = links[(links.target == target) & links.source.isin(source.split("+"))
                        & (links.kind == "unique")]
            hit = hit.iloc[hit.tier.map(ORDER).argsort()[:1]] if len(hit) else hit
        if len(hit):
            h = hit.iloc[0]
            rows.append(dict(link=f"{source}->{target}", kind=kind, true=True, r2_true=r2, tier=h.tier,
                             fold_tier=h.fold_tier, pass_rate=h.pass_rate, r2_median=h.r2_median))
        else:
            reached = target in set(result.nodes.variable)
            rows.append(dict(link=f"{source}->{target}", kind=kind, true=True, r2_true=r2,
                             tier="not_tested" if not reached else "absent", fold_tier=None,
                             pass_rate=np.nan, r2_median=np.nan))
    truth = {(t, s, k) for s, t, k, _ in TRUTH} | NOT_FALSE
    false_tested = 0
    for (target, source, kind) in tested - truth:
        h = links[(links.target == target) & (links.source == source) & (links.kind == kind)].iloc[0]
        false_tested += 1
        if h.fold_tier != "dropped":
            rows.append(dict(link=f"{source}->{target}", kind=kind, true=False, r2_true=0.0, tier=h.tier,
                             fold_tier=h.fold_tier, pass_rate=h.pass_rate, r2_median=h.r2_median))
    return rows, false_tested


def run_one(task):
    engine, n, shape, rep = task
    seed = dataset_seed(engine, n, shape, rep)
    data = generate(n, shape, seed)
    t = time.perf_counter()
    result = explore(data, "Y", engine=engine, config=CONFIG)
    seconds = time.perf_counter() - t
    rows, false_tested = score(result)
    base = dict(engine=engine, n=n, shape=shape, rep=rep, seed=seed)
    return dict(base, seconds=seconds, false_tested=false_tested, depth_reached=int(result.nodes.depth.max()),
                links=[dict(base, **r) for r in rows])


def _map(tasks, workers):
    if workers <= 1:
        return [run_one(t) for t in tasks]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_one, tasks))


def _write(out, records, name):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for rec in records for r in rec["links"]]).to_csv(out / f"{name}-links.csv", index=False)
    pd.DataFrame([{k: v for k, v in rec.items() if k != "links"} for rec in records]) \
        .to_csv(out / f"{name}-datasets.csv", index=False)


def probe(out, workers, engines=ENGINES):
    tasks = [(e, n, s, 0) for e in engines for n in SIZES for s in ("linear", "u_shape")]
    started = time.perf_counter()
    records = _map(tasks, workers)
    _write(out, records, "probe")
    per = pd.DataFrame([{k: r[k] for k in ("engine", "n", "shape", "seconds")} for r in records])
    summary = dict(workers=workers, wall_seconds=time.perf_counter() - started,
                   seconds=per.to_dict("records"),
                   mean_seconds_by_engine_n=per.groupby(["engine", "n"]).seconds.mean().round(1)
                   .reset_index().to_dict("records"))
    (out / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(per.to_string(index=False))
    return summary


def run(out, engine, n, shape, lo, hi, workers):
    records = _map([(engine, n, shape, rep) for rep in range(lo, hi)], workers)
    _write(out, records, f"{engine}-n{n}-{shape}-r{lo:04d}-{hi:04d}")


def aggregate(inputs, out):
    links = pd.concat([pd.read_csv(p) for p in sorted(Path(inputs).rglob("*-links.csv"))], ignore_index=True)
    datasets = pd.concat([pd.read_csv(p) for p in sorted(Path(inputs).rglob("*-datasets.csv"))],
                         ignore_index=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    true = links[links.true]
    rates = (true.assign(**{t: true.tier == t for t in ("kept", "flagged", "hidden", "absent", "not_tested")})
             .groupby(["engine", "n", "shape", "link"])[["kept", "flagged", "hidden", "absent", "not_tested"]]
             .mean().reset_index())
    rates["dropped"] = 1 - rates[["kept", "flagged", "hidden", "absent", "not_tested"]].sum(axis=1)
    false = links[~links.true]
    counts = datasets.groupby(["engine", "n", "shape"]).agg(datasets=("rep", "size"),
                                                           false_tested=("false_tested", "sum"),
                                                           seconds_mean=("seconds", "mean")).reset_index()
    for t in ("kept", "flagged", "hidden"):
        c = false[false.tier == t].groupby(["engine", "n", "shape"]).size().rename(f"false_{t}")
        counts = counts.merge(c, on=["engine", "n", "shape"], how="left").fillna({f"false_{t}": 0})
        counts[f"false_{t}_rate"] = counts[f"false_{t}"] / counts.false_tested
    rates.to_csv(out / "true-link-rates.csv", index=False)
    counts.to_csv(out / "false-link-rates.csv", index=False)
    return rates, counts


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("probe")
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--engines", default=",".join(ENGINES))
    b = sub.add_parser("run")
    b.add_argument("--engine", choices=ENGINES, required=True)
    b.add_argument("--n", type=int, choices=SIZES, required=True)
    b.add_argument("--shape", choices=SHAPES, required=True)
    b.add_argument("--reps", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("aggregate")
    c.add_argument("--inputs", required=True)
    c.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "probe":
        probe(Path(args.out), args.workers, tuple(args.engines.split(",")))
    elif args.cmd == "run":
        lo, hi = (int(x) for x in args.reps.split(":"))
        run(Path(args.out), args.engine, args.n, args.shape, lo, hi, args.workers)
    else:
        aggregate(args.inputs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
