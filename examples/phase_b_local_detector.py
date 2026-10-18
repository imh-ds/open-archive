"""Phase B local detector per engine; research script, not a package API.

Protocol: outline/2026-10-18-phase-b-local-detector-protocol.md (main checkout).
OLS + HC4 reference (R1), cross-fitted GCM with OLS nuisances (R2), and four
flexible nuisance engines (F1 random forest, F2 gradient boosting, F3 LightGBM,
F4 LightGBM with linear trees) tested with cross-fitted GCM and PCM, combined by
Bonferroni (Cauchy reported descriptively). Small-sample GCM corrections G0-G2
are screened with OLS nuisances and applied to flexible engines at N = 300/500.

Modes:
  --check --out DIR [--part P] HC4 identity with Phase A', GCM identity with Phase A,
                               loop cross-checks of GCM/PCM per engine, determinism,
                               population draws, seed audit, manifest determinism,
                               run guard, smoke run and runtime projection. With --part,
                               only that part runs (see CHECK_PARTS), so Actions jobs
                               can run the parts in parallel
  --combine-check DIR --out DIR  merge part-*.json files into check.json
  --write-manifest PATH        write the canonical manifest (freezing step)
  --run --manifest PATH --out DIR [--workers N]
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASE_A_SOURCE = HERE / "phase_a_linear_reference.py"
PHASE_A_SOURCE_SHA = "d621a6e55bb4697296bb7c29f8f8dd61faba3a1709dccd7c19aa6c0657128bba"
PHASE_A_PRIME_SOURCE = HERE / "phase_a_prime_small_sample.py"
PHASE_A_PRIME_SOURCE_SHA = "8329c91ee91460460e0787551dfdd3bdf9953e6e59a697e016f4d335642c984b"

SCHEMA = "open.phase_b_local_detector.v1"
SEED_BASE = 3_740_000_000
SEED_STRIDE = 10_000
FIXTURE_SEED_BASE = 3_749_000_000
RERUN_SEED_BASE = 3_750_000_000
AUDIT_RANGE = (3_740_000_000, 3_759_999_999)
ALPHA = 0.05
Z99 = float(stats.norm.ppf(0.995))
ZG = 3.89
MARGIN = 0.005
COUNT_LEVEL = 0.01
C1_CELL_CHANCE = 0.02
C4_CELL_CHANCE = 0.01
EFFICIENCY = 0.90
SCREEN_POWER_LOSS = 0.05

MAIN_N = (1000, 2000, 5000)
SMALL_N = (300, 500)
SCREEN_N = (100, 200, 300, 500, 1000)
SHAPES = ("L", "Mo", "U", "I", "E")
ENGINES = ("F1", "F2", "F3", "F4")
ENGINE_NAMES = {"F1": "random_forest", "F2": "gradient_boosting", "F3": "lightgbm",
                "F4": "lightgbm_linear_tree"}
ENGINE_CONFIG = {
    "F1": "RandomForestRegressor(n_estimators=100, min_samples_leaf=max(5, ceil(.005 n_train)), "
          "max_features=.5, n_jobs=1)",
    "F2": "GradientBoostingRegressor(max_depth=2, learning_rate=.05, subsample=.8, n_estimators=1000, "
          "n_iter_no_change=20, validation_fraction=.2)",
    "F3": "LGBMRegressor(num_leaves=15, learning_rate=.05, min_child_samples=max(20, ceil(.01 n_train)), "
          "subsample=.8, subsample_freq=1, reg_lambda=1, n_estimators=1000, num_threads=1, "
          "deterministic=True, force_col_wise=True); early stopping patience 50 on an inner 20% split",
    "F4": "F3 with linear_tree=True, linear_lambda=1, num_leaves=7",
}
CORRECTIONS = {"g0": "K = 5, normal reference", "g1": "K = 10, normal reference",
               "g2": "K = 10, fold-cluster-robust variance, t(K - 1) reference"}
CORRECTION_ORDER = ("g0", "g1", "g2")
BASE_NAMES = ("X1", "X0", "Z1", "Z2", "M")
SIGMA5 = np.array([[1.0, 0.3, 0.6, 0.0, 0.0],
                   [0.3, 1.0, 0.18, 0.0, 0.0],
                   [0.6, 0.18, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0]])
RELIABILITY = 0.8
GENERATION_ORDER = (
    "rng = numpy.random.default_rng(seed)",
    "V = rng.standard_normal((n, p)) @ cholesky(Sigma).T  (latent candidates X1, X0, Z1, Z2, M[, W1..W5])",
    "u = rng.standard_normal(n)  (measurement error; always drawn, used only in shape E)",
    "eps = rng.standard_normal(n)",
    "X = V, except shape E: X1 = sqrt(.8) V1 + sqrt(.2) u",
    "s = shape(V1, M); heteroskedastic null: eps *= sqrt((1 + Z2^2 + X0^2) / 3)",
    "T = beta * s + sqrt(sigma2) * eps",
    "fold5[perm] = arange(n) % 5 with perm = rng.permutation(n); then fold10 likewise (% 10); "
    "then half[perm] = arange(n) % 2",
)


def _load(path, sha, name):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"{path.name} changed; refusing to reuse it")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PA = _load(PHASE_A_SOURCE, PHASE_A_SOURCE_SHA, "phase_a_ref")


def _load_a_prime():
    return _load(PHASE_A_PRIME_SOURCE, PHASE_A_PRIME_SOURCE_SHA, "phase_a_prime_ref")


# ---------------------------------------------------------------- design

_HX, _HW = np.polynomial.hermite_e.hermegauss(120)
_HW = _HW / _HW.sum()
_MO_SD = float(np.sqrt(np.sum(_HW * np.tanh(1.5 * _HX) ** 2)))


def sigma(p):
    S = np.eye(p)
    S[:5, :5] = SIGMA5
    return S


def names(p):
    return list(BASE_NAMES) + [f"W{i}" for i in range(1, p - 4)]


def shape_fn(shape, x, m):
    if shape in ("L", "E"):
        return x
    if shape == "Mo":
        return np.tanh(1.5 * x) / _MO_SD
    if shape == "U":
        return (x * x - 1.0) / np.sqrt(2.0)
    if shape == "I":
        return x * m
    raise ValueError(shape)


def _tau2(p):
    return float(1.0 / np.linalg.inv(sigma(p))[0, 0])


def unique_variance(shape, p):
    """V = Var(s - E[s | C-1]) under the latent Gaussian design (Var(s) = 1)."""
    t2 = _tau2(p)
    if shape in ("L", "E", "I"):
        return t2
    if shape == "U":
        return 1.0 - (1.0 - t2) ** 2
    if shape == "Mo":
        mu = np.sqrt(1.0 - t2) * _HX
        g = (np.tanh(1.5 * (mu[:, None] + np.sqrt(t2) * _HX[None, :])) / _MO_SD) @ _HW
        return float(1.0 - np.sum(_HW * g * g))
    raise ValueError(shape)


def coefficients(shape, p, r):
    V = unique_variance(shape, p)
    b2 = r * r / (V * (1.0 - r * r) + r * r)
    return float(np.sqrt(b2)), float(1.0 - b2), V


def roles(shape, p, r):
    out = []
    for nm in names(p):
        if nm == "X1":
            out.append("signal" if r > 0 else "null")
        elif nm in ("X0", "Z1") and shape == "E" and r > 0:
            out.append("measurement")
        elif nm == "M" and shape == "I" and r > 0:
            out.append("moderator")
        else:
            out.append("null")
    return out


def tests_for(family):
    t = ["R1_hc4", "R2_gcm_g0"]
    gs = ("g0",) if family == "main" else ("g0", "g1", "g2")
    if family != "main":
        t += ["R2_gcm_g1", "R2_gcm_g2"]
    if family != "screen":
        for e in ENGINES:
            for g in gs:
                t += [f"{e}_gcm_{g}", f"{e}_pcm_{g}"]
    return t


def enumerate_cells():
    cells = []

    def add(**kw):
        base = dict(error="gaussian", p=5)
        base.update(kw)
        cells.append(base)

    for n in MAIN_N:
        add(family="main", n=n, shape="L", r=0.0, reps=1000)
        add(family="main", n=n, shape="L", r=0.0, reps=500, error="heteroskedastic")
        for shape in SHAPES:
            for r in (0.05, 0.10):
                add(family="main", n=n, shape=shape, r=r, reps=400)
    for n in SMALL_N:
        add(family="small", n=n, shape="L", r=0.0, reps=1000)
        for shape in SHAPES:
            for r in (0.10, 0.20):
                add(family="small", n=n, shape=shape, r=r, reps=400)
    for n in SCREEN_N:
        for p in (5, 10):
            for r in (0.0, 0.10):
                add(family="screen", n=n, shape="L", r=r, p=p, reps=2000 if r == 0 else 1000)

    for i, c in enumerate(cells):
        beta, sigma2, V = coefficients(c["shape"], c["p"], c["r"])
        c["cell_index"] = i
        c["seed_first"] = SEED_BASE + SEED_STRIDE * i
        c["seed_last"] = c["seed_first"] + c["reps"] - 1
        c["names"] = names(c["p"])
        c["roles"] = roles(c["shape"], c["p"], c["r"])
        c["beta"], c["sigma2"], c["unique_variance"] = beta, sigma2, V
        c["moderator_partial_ratio"] = beta if (c["shape"] == "I" and c["r"] > 0) else None
        c["tests"] = tests_for(c["family"])
        c["analytic_power_r1"] = (PA.power_t(c["r"], c["n"], c["p"])
                                  if c["shape"] == "L" and c["r"] > 0 and c["error"] == "gaussian" else None)
    return cells


# ---------------------------------------------------------------- data

def generate(cell, seed, n=None):
    rng = np.random.default_rng(seed)
    n = cell["n"] if n is None else n
    p = cell["p"]
    V = rng.standard_normal((n, p)) @ np.linalg.cholesky(sigma(p)).T
    u = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    X = V.copy()
    if cell["shape"] == "E":
        X[:, 0] = np.sqrt(RELIABILITY) * V[:, 0] + np.sqrt(1 - RELIABILITY) * u
    s = shape_fn(cell["shape"], V[:, 0], V[:, 4])
    if cell["error"] == "heteroskedastic":
        eps = eps * np.sqrt((1.0 + X[:, 3] ** 2 + X[:, 1] ** 2) / 3.0)
    T = cell["beta"] * s + np.sqrt(cell["sigma2"]) * eps
    folds = {}
    for key, k in (("k5", 5), ("k10", 10), ("half", 2)):
        perm = rng.permutation(n)
        f = np.empty(n, dtype=np.int64)
        f[perm] = np.arange(n) % k
        folds[key] = f
    return X, T, folds


# ---------------------------------------------------------------- engines

def _seed(*parts):
    h = hashlib.sha256("|".join(str(x) for x in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


class Rounds:
    def __init__(self):
        self.fits, self.rounds, self.capped = 0, 0, 0

    def add(self, r, cap):
        self.fits += 1
        self.rounds += r
        self.capped += int(cap)


def fit_predict(engine, Xtr, ytr, Xtests, seed, acc=None):
    """Fit one nuisance regression and predict each matrix in Xtests."""
    n_tr = len(ytr)
    if engine == "ols":
        A = np.column_stack([np.ones(n_tr), Xtr])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        return [np.column_stack([np.ones(len(Z)), Z]) @ coef for Z in Xtests]
    if engine == "F1":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=100, min_samples_leaf=max(5, math.ceil(0.005 * n_tr)),
                                  max_features=0.5, n_jobs=1, random_state=seed)
        m.fit(Xtr, ytr)
        if acc is not None:
            acc.add(100, False)
        return [m.predict(Z) for Z in Xtests]
    if engine == "F2":
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(max_depth=2, learning_rate=0.05, subsample=0.8, n_estimators=1000,
                                      n_iter_no_change=20, validation_fraction=0.2, random_state=seed)
        m.fit(Xtr, ytr)
        if acc is not None:
            acc.add(int(m.n_estimators_), m.n_estimators_ >= 1000)
        return [m.predict(Z) for Z in Xtests]
    if engine in ("F3", "F4"):
        import lightgbm as lgb
        kw = dict(num_leaves=15, learning_rate=0.05, min_child_samples=max(20, math.ceil(0.01 * n_tr)),
                  subsample=0.8, subsample_freq=1, reg_lambda=1.0, n_estimators=1000, num_threads=1,
                  deterministic=True, force_col_wise=True, verbose=-1, random_state=seed)
        if engine == "F4":
            kw.update(linear_tree=True, linear_lambda=1.0, num_leaves=7)
        perm = np.random.default_rng(seed).permutation(n_tr)
        n_val = math.ceil(0.2 * n_tr)
        val, fit = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        m = lgb.LGBMRegressor(**kw)
        m.fit(Xtr[fit], ytr[fit], eval_set=[(Xtr[val], ytr[val])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(m.best_iteration_ or 1000)
        if acc is not None:
            acc.add(best, best >= 1000)
        return [m.predict(Z, num_iteration=best) for Z in Xtests]
    raise ValueError(engine)


# ---------------------------------------------------------------- tests

def hc4_p(X, y):
    A = PA._design(X)
    n, k = A.shape
    inv = np.linalg.inv(A.T @ A)
    C = inv @ A.T
    beta = C @ y
    e = y - A @ beta
    h = np.einsum("ij,ji->i", A, C)
    w = e ** 2 / (1 - h) ** np.minimum(4.0, h / (k / n))
    t = beta / np.sqrt((C ** 2) @ w)
    return 2 * stats.t.sf(np.abs(t[1:]), n - k)


def gcm_residuals(engine, X, T, fold, K, dseed, acc=None):
    """Cross-fitted residuals eT, eX (p x n) of T and each Xj on C-j."""
    n, p = X.shape
    eT, eX = np.empty((p, n)), np.empty((p, n))
    for j in range(p):
        C = np.delete(X, j, axis=1)
        for k in range(K):
            test = fold == k
            train = ~test
            fT, = fit_predict(engine, C[train], T[train], [C[test]], _seed(dseed, engine, K, k, "gcmT", j), acc)
            fX, = fit_predict(engine, C[train], X[train, j], [C[test]], _seed(dseed, engine, K, k, "gcmX", j), acc)
            eT[j, test] = T[test] - fT
            eX[j, test] = X[test, j] - fX
    return eT, eX


def stat_normal(R, two_sided=True):
    n = R.shape[-1]
    Z = np.sqrt(n) * R.mean(axis=-1) / R.std(axis=-1, ddof=1)
    return 2 * stats.norm.sf(np.abs(Z)) if two_sided else stats.norm.sf(Z)


def stat_cluster(R, fold, K, two_sided=True):
    """Fold-cluster-robust (CR1) t for the mean of R, referred to t(K - 1)."""
    n = R.shape[-1]
    Rbar = R.mean(axis=-1, keepdims=True)
    sums = np.stack([(R - Rbar)[..., fold == k].sum(axis=-1) for k in range(K)], axis=-1)
    V = K / (K - 1) * (sums ** 2).sum(axis=-1) / n ** 2
    t = Rbar[..., 0] / np.sqrt(V)
    return 2 * stats.t.sf(np.abs(t), K - 1) if two_sided else stats.t.sf(t, K - 1)


def pcm_products(engine, X, T, half, fold, K, dseed, acc=None):
    """For each half h (A = half == h) and candidate j: products R on B's rows, and B's fold labels."""
    n, p = X.shape
    out = []
    for h in (0, 1):
        A, B = half == h, half != h
        gA, gB = fit_predict(engine, X[A], T[A], [X[A], X[B]], _seed(dseed, engine, K, h, "pcm_g", -1), acc)
        fB = fold[B]
        TB = T[B]
        Rh = np.empty((p, int(B.sum())))
        for j in range(p):
            C = np.delete(X, j, axis=1)
            mB, = fit_predict(engine, C[A], gA, [C[B]], _seed(dseed, engine, K, h, "pcm_m", j), acc)
            hB = gB - mB
            CB = C[B]
            for k in range(K):
                test = fB == k
                train = ~test
                fT, = fit_predict(engine, CB[train], TB[train], [CB[test]], _seed(dseed, engine, K, h, f"pcmT{k}", j), acc)
                fH, = fit_predict(engine, CB[train], hB[train], [CB[test]], _seed(dseed, engine, K, h, f"pcmH{k}", j), acc)
                Rh[j, test] = (TB[test] - fT) * (hB[test] - fH)
        out.append((Rh, fB))
    return out


def pcm_p(parts, K, rule):
    ps = []
    for Rh, fB in parts:
        ps.append(stat_normal(Rh, two_sided=False) if rule == "normal" else stat_cluster(Rh, fB, K, two_sided=False))
    return np.minimum(1.0, 2 * np.minimum(ps[0], ps[1]))


def analyze(cell, X, T, folds, dseed):
    """p-values (len(tests) x p) in the order of cell['tests'], and engine round counts."""
    fam = cell["family"]
    res = {"R1_hc4": hc4_p(X, T)}
    eT, eX = gcm_residuals("ols", X, T, folds["k5"], 5, dseed)
    res["R2_gcm_g0"] = stat_normal(eT * eX)
    if fam != "main":
        eT, eX = gcm_residuals("ols", X, T, folds["k10"], 10, dseed)
        res["R2_gcm_g1"] = stat_normal(eT * eX)
        res["R2_gcm_g2"] = stat_cluster(eT * eX, folds["k10"], 10)
    rounds = np.zeros((len(ENGINES), 3))
    if fam != "screen":
        for ei, e in enumerate(ENGINES):
            acc = Rounds()
            eT, eX = gcm_residuals(e, X, T, folds["k5"], 5, dseed, acc)
            res[f"{e}_gcm_g0"] = stat_normal(eT * eX)
            parts = pcm_products(e, X, T, folds["half"], folds["k5"], 5, dseed, acc)
            res[f"{e}_pcm_g0"] = pcm_p(parts, 5, "normal")
            if fam == "small":
                eT, eX = gcm_residuals(e, X, T, folds["k10"], 10, dseed, acc)
                res[f"{e}_gcm_g1"] = stat_normal(eT * eX)
                res[f"{e}_gcm_g2"] = stat_cluster(eT * eX, folds["k10"], 10)
                parts = pcm_products(e, X, T, folds["half"], folds["k10"], 10, dseed, acc)
                res[f"{e}_pcm_g1"] = pcm_p(parts, 10, "normal")
                res[f"{e}_pcm_g2"] = pcm_p(parts, 10, "cluster")
            rounds[ei] = (acc.fits, acc.rounds, acc.capped)
    return np.vstack([res[t] for t in cell["tests"]]), rounds


# ---------------------------------------------------------------- run (chunked)

def chunk_name(cell_index, rep_lo):
    return f"cell_{cell_index:03d}_r{rep_lo:05d}.npz"


def run_chunk(cell, out_dir, rep_lo, rep_hi, seed_first=None):
    seed_first = cell["seed_first"] if seed_first is None else seed_first
    t0 = time.perf_counter()
    m = rep_hi - rep_lo
    P = np.empty((m, len(cell["tests"]), cell["p"]))
    RD = np.empty((m, len(ENGINES), 3))
    for i, r in enumerate(range(rep_lo, rep_hi)):
        X, T, folds = generate(cell, seed_first + r)
        P[i], RD[i] = analyze(cell, X, T, folds, seed_first + r)
    secs = time.perf_counter() - t0
    np.savez_compressed(Path(out_dir) / "pvalues" / chunk_name(cell["cell_index"], rep_lo),
                        p=P, rounds=RD, seed_first=seed_first, rep_lo=rep_lo, rep_hi=rep_hi, seconds=secs)
    return cell["cell_index"], rep_lo, secs


def chunks_for(cell, size):
    return [(lo, min(lo + size, cell["reps"])) for lo in range(0, cell["reps"], size)]


def chunk_size(cell):
    return 250 if cell["family"] == "screen" else (5 if cell["n"] >= 5000 else 10 if cell["n"] >= 1000 else 25)


def run_cells(cells, out_dir, workers, seed_override=None, reps_override=None, size=None):
    (Path(out_dir) / "pvalues").mkdir(parents=True, exist_ok=False)
    timings = {}
    jobs = []
    for c in cells:
        c = dict(c, reps=reps_override or c["reps"])
        seed = None if seed_override is None else seed_override + SEED_STRIDE * c["cell_index"]
        for lo, hi in chunks_for(c, size or chunk_size(c)):
            jobs.append((c, lo, hi, seed))
    jobs.sort(key=lambda j: -(j[0]["n"] * (j[2] - j[1]) * len(j[0]["tests"])))
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_chunk, c, out_dir, lo, hi, seed) for c, lo, hi, seed in jobs]
        for f in cf.as_completed(futs):
            idx, lo, secs = f.result()
            timings[f"{idx}:{lo}"] = secs
    return timings


def load_cell(cell, out_dir):
    files = sorted((Path(out_dir) / "pvalues").glob(f"cell_{cell['cell_index']:03d}_r*.npz"))
    parts = [np.load(f) for f in files]
    parts.sort(key=lambda d: int(d["rep_lo"]))
    lo = 0
    for d in parts:
        if int(d["rep_lo"]) != lo:
            raise SystemExit(f"cell {cell['cell_index']}: chunk gap at rep {lo}")
        lo = int(d["rep_hi"])
    if lo != cell["reps"]:
        raise SystemExit(f"cell {cell['cell_index']}: {lo} of {cell['reps']} reps present")
    return (np.concatenate([d["p"] for d in parts]), np.concatenate([d["rounds"] for d in parts]),
            float(sum(float(d["seconds"]) for d in parts)))


# ---------------------------------------------------------------- summary

def cauchy(p1, p2):
    a = np.clip(p1, 1e-15, 1 - 1e-15)
    b = np.clip(p2, 1e-15, 1 - 1e-15)
    t = (np.tan((0.5 - a) * np.pi) + np.tan((0.5 - b) * np.pi)) / 2
    return 0.5 - np.arctan(t) / np.pi


def derived(cell, P):
    """All reported methods for a cell: name -> (reps x p) p-values."""
    out = {t: P[:, i, :] for i, t in enumerate(cell["tests"])}
    for e in ENGINES:
        for g in CORRECTION_ORDER:
            if f"{e}_gcm_{g}" in out:
                a, b = out[f"{e}_gcm_{g}"], out[f"{e}_pcm_{g}"]
                out[f"{e}_comb_{g}"] = np.minimum(1.0, 2 * np.minimum(a, b))
                out[f"{e}_cauchy_{g}"] = cauchy(a, b)
    return out


def _role_key(name):
    return "W" if name.startswith("W") else name


def cell_stats(cell, P, RD, secs):
    roles_ = np.array(cell["roles"])
    nullm = roles_ == "null"
    falsem = nullm
    truem = np.isin(roles_, ("signal", "moderator", "measurement"))
    R = P.shape[0]
    st = {"R": R, "seconds": secs, "methods": {}, "rounds": {}}
    for ei, e in enumerate(ENGINES):
        f = RD[:, ei, 0].sum()
        if f:
            st["rounds"][e] = {"fits": float(f), "mean_rounds": float(RD[:, ei, 1].sum() / f),
                               "share_capped": float(RD[:, ei, 2].sum() / f)}
    for m, Pm in derived(cell, P).items():
        rej = Pm < ALPHA
        ms = {"null_rates": {}, "rej": {}}
        for j, nm in enumerate(cell["names"]):
            ms["rej"][nm] = int(rej[:, j].sum())
            if roles_[j] == "null":
                ms["null_rates"].setdefault(_role_key(nm), []).append(float(rej[:, j].mean()))
        ms["null_rates"] = {k: float(np.mean(v)) for k, v in ms["null_rates"].items()}
        if cell["r"] > 0:
            ms["power_X1"] = float(rej[:, 0].mean())
            if cell["shape"] == "I":
                ms["power_M"] = float(rej[:, 4].mean())
            if cell["shape"] == "E":
                ms["leakage"] = {nm: float(rej[:, j].mean()) for j, nm in enumerate(cell["names"])
                                 if roles_[j] == "measurement"}
        for q in (0.05, 0.20):
            rj = PA.bh_reject(Pm, q)
            fdp = (rj & falsem).sum(axis=1) / np.maximum(rj.sum(axis=1), 1)
            ms[f"bh{q}"] = {"fdr": float(fdp.mean()), "fdr_sd": float(fdp.std(ddof=1)),
                            "power_X1": float(rj[:, 0].mean()) if cell["r"] > 0 else None}
        st["methods"][m] = ms
    return st


def _rej_matrix(cell, P, method):
    return derived(cell, P)[method] < ALPHA


# ---------------------------------------------------------------- criteria

def allowance(K, pi):
    a = 0
    while stats.binom.sf(a, K, pi) > COUNT_LEVEL:
        a += 1
    return a


def _se(pi, R):
    return math.sqrt(pi * (1 - pi) / R)


def c1(cells, cs, method):
    """Calibration over a set of cells (one N): pooled per role, count allowance, gross bound."""
    pooled, fails, gross = {}, [], []
    for c in cells:
        st = cs[c["cell_index"]]
        R = st["R"]
        rates = st["methods"][method]["null_rates"]
        bad = False
        for role, rate in rates.items():
            k = sum(1 for nm, rl in zip(c["names"], c["roles"]) if rl == "null" and _role_key(nm) == role)
            agg = pooled.setdefault(role, [0.0, 0])
            agg[0] += rate * R * k
            agg[1] += R * k
            if rate > 0.05 + MARGIN + Z99 * _se(0.05 + MARGIN, R):
                bad = True
            if rate > 0.06 + ZG * _se(0.06, R):
                gross.append({"cell_index": c["cell_index"], "role": role, "rate": rate})
        if bad:
            fails.append(c["cell_index"])
    pooled_out = {}
    for role, (num, den) in pooled.items():
        rate = num / den
        # Each W-role or pooled role count is treated as independent Bernoulli trials (conservative
        # for W, which pools five predictors from the same datasets: SE is then understated).
        lim = 0.05 + MARGIN + Z99 * _se(0.05 + MARGIN, den)
        pooled_out[role] = {"rate": rate, "trials": den, "limit": lim, "pass": rate <= lim}
    a = allowance(len(cells), C1_CELL_CHANCE)
    ok = all(v["pass"] for v in pooled_out.values()) and len(fails) <= a and not gross
    return {"pass": bool(ok), "pooled": pooled_out, "failing_cells": fails, "allowance": a,
            "cells": len(cells), "gross": gross}


def c4(cells, cs, method):
    out = {}
    for q in (0.05, 0.20):
        num = den = 0.0
        var_num = 0.0
        fails, gross = [], []
        for c in cells:
            b = cs[c["cell_index"]]["methods"][method][f"bh{q}"]
            R = cs[c["cell_index"]]["R"]
            se = b["fdr_sd"] / math.sqrt(R)
            num += b["fdr"] * R
            den += R
            var_num += b["fdr_sd"] ** 2 * R
            if b["fdr"] > q + MARGIN + Z99 * se:
                fails.append(c["cell_index"])
            if b["fdr"] > q + 0.01 + ZG * se:
                gross.append({"cell_index": c["cell_index"], "fdr": b["fdr"]})
        pooled = num / den
        lim = q + MARGIN + Z99 * math.sqrt(var_num) / den
        a = allowance(len(cells), C4_CELL_CHANCE)
        out[str(q)] = {"pooled_fdr": pooled, "limit": lim, "failing_cells": fails, "allowance": a,
                       "gross": gross, "pass": bool(pooled <= lim and len(fails) <= a and not gross)}
    out["pass"] = all(v["pass"] for k, v in out.items() if k != "pass")
    return out


def _paired(cells, Ps, method, ref, factor=1.0, target=0):
    d = []
    for c in cells:
        P = Ps[c["cell_index"]]
        d.append(_rej_matrix(c, P, method)[:, target].astype(float)
                 - factor * _rej_matrix(c, P, ref)[:, target].astype(float))
    d = np.concatenate(d)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def evaluate(cells, cs, Ps):
    by = lambda **kw: [c for c in cells if all(c[k] == v for k, v in kw.items())]
    res = {"constants": {"z99": Z99, "zg": ZG, "margin": MARGIN, "count_level": COUNT_LEVEL}}

    # Correction screen (R2, OLS nuisances).
    screen = {}
    for g in CORRECTION_ORDER:
        m = f"R2_gcm_{g}"
        per_n = {n: c1(by(family="screen", n=n), cs, m) for n in SCREEN_N}
        power = {}
        for n in SCREEN_N:
            sig = [c for c in by(family="screen", n=n) if c["r"] > 0]
            loss, se = _paired(sig, Ps, "R1_hc4", m)
            power[n] = {"loss_vs_r1": loss, "se": se, "pass": loss <= SCREEN_POWER_LOSS + Z99 * se}
        n0 = None
        for n in SCREEN_N:
            if all(per_n[x]["pass"] for x in SCREEN_N if x >= n) and power[n]["pass"]:
                n0 = n
                break
        mean_power = float(np.mean([cs[c["cell_index"]]["methods"][m]["power_X1"]
                                    for c in by(family="screen") if c["r"] > 0]))
        screen[g] = {"c1_by_n": {str(k): v for k, v in per_n.items()},
                     "power_by_n": {str(k): v for k, v in power.items()},
                     "qualifies_from": n0, "mean_power": mean_power}
    elig = {g: v for g, v in screen.items() if v["qualifies_from"] is not None and v["qualifies_from"] < 1000}
    chosen = None
    if elig:
        best = min(v["qualifies_from"] for v in elig.values())
        pool = [g for g, v in elig.items() if v["qualifies_from"] == best]
        top = max(screen[g]["mean_power"] for g in pool)
        chosen = sorted([g for g in pool if top - screen[g]["mean_power"] <= 0.01], key=CORRECTION_ORDER.index)[0]
    res["correction_screen"] = {"candidates": screen, "chosen": chosen,
                                "chosen_from_n": screen[chosen]["qualifies_from"] if chosen else None}

    # Reference defect checks (protocol 7.6).
    ref_cells = [c for c in cells if c["family"] in ("main", "small") and (c["shape"] in ("L", "Mo") or c["error"] != "gaussian")]
    r1_c1 = {n: c1([c for c in ref_cells if c["n"] == n], cs, "R1_hc4") for n in SMALL_N + MAIN_N}
    r1_c4 = {n: c4([c for c in ref_cells if c["n"] == n], cs, "R1_hc4") for n in SMALL_N + MAIN_N}
    r2_1000 = screen["g0"]["c1_by_n"]["1000"]["pass"]
    defects = [f"R1 C1 at N={n}" for n, v in r1_c1.items() if not v["pass"]] + \
              [f"R1 C4 at N={n}" for n, v in r1_c4.items() if not v["pass"]] + \
              ([] if r2_1000 else ["R2 G0 fails C1 at N=1000 (contradicts Phase A M4)"])
    res["reference"] = {"r1_c1": {str(k): v["pass"] for k, v in r1_c1.items()},
                        "r1_c4": {str(k): v["pass"] for k, v in r1_c4.items()},
                        "r2_g0_c1_n1000": r2_1000, "defects": defects}

    # Flexible configurations.
    configs = {}
    for e in ENGINES:
        m0 = f"{e}_comb_g0"
        steps = {}
        for n in MAIN_N:
            cc = by(family="main", n=n)
            a, b = c1(cc, cs, m0), c4(cc, cs, m0)
            c3 = c1([c for c in cc if c["error"] != "gaussian" or c["shape"] in ("U", "I")], cs, m0)
            steps[n] = {"method": m0, "C1": a, "C3": c3, "C4": b, "pass": a["pass"] and b["pass"]}
        if chosen:
            mg = f"{e}_comb_{chosen}"
            for n in SMALL_N:
                cc = by(family="small", n=n)
                a, b = c1(cc, cs, mg), c4(cc, cs, mg)
                steps[n] = {"method": mg, "C1": a, "C3": None, "C4": b, "pass": a["pass"] and b["pass"]}
        ladder = sorted(steps)
        min_n = next((n0 for n0 in ladder if all(steps[x]["pass"] for x in ladder if x >= n0)), None)
        eff = {}
        for r in (0.05, 0.10):
            cc = by(family="main", n=5000, shape="L", r=r)
            d, se = _paired(cc, Ps, m0, "R1_hc4", factor=EFFICIENCY)
            p_m = cs[cc[0]["cell_index"]]["methods"][m0]["power_X1"]
            p_r = cs[cc[0]["cell_index"]]["methods"]["R1_hc4"]["power_X1"]
            eff[str(r)] = {"power": p_m, "power_r1": p_r, "ratio": p_m / p_r if p_r else None,
                           "d": d, "se": se, "pass": d >= -Z99 * se}
        c2 = all(v["pass"] for v in eff.values())
        nonlin = {}
        for shape in ("Mo", "U", "I"):
            for n in MAIN_N:
                for r in (0.05, 0.10):
                    c = by(family="main", n=n, shape=shape, r=r)[0]
                    ms = cs[c["cell_index"]]["methods"]
                    nonlin[f"{shape}_n{n}_r{r}"] = {"comb": ms[m0]["power_X1"], "r1": ms["R1_hc4"]["power_X1"],
                                                     "gcm": ms[f"{e}_gcm_g0"]["power_X1"],
                                                     "pcm": ms[f"{e}_pcm_g0"]["power_X1"],
                                                     "cauchy": ms[f"{e}_cauchy_g0"]["power_X1"]}
        lin_power = float(np.mean([cs[c["cell_index"]]["methods"][m0]["power_X1"]
                                   for c in by(family="main", shape="L") if c["r"] > 0]))
        configs[e] = {"engine": ENGINE_NAMES[e], "ladder": {str(k): v for k, v in steps.items()},
                      "C2_linear_efficiency": {"by_r": eff, "pass": c2},
                      "supported_min_n": min_n if c2 else None, "supported_min_n_ignoring_c2": min_n,
                      "mean_linear_power": lin_power, "nonlinear_power": nonlin}
    res["configurations"] = configs

    table = []
    for engine, ids in (("random_forest", ["F1"]), ("gradient_boosting", ["F2"]), ("lightgbm", ["F3", "F4"])):
        ok = [i for i in ids if configs[i]["supported_min_n"] is not None]
        if ok:
            best = min(configs[i]["supported_min_n"] for i in ok)
            pool = [i for i in ok if configs[i]["supported_min_n"] == best]
            top = max(configs[i]["mean_linear_power"] for i in pool)
            pick = sorted([i for i in pool if top - configs[i]["mean_linear_power"] <= 1e-12], key=ids.index)[0]
        else:
            pick = ids[0]
        cfg = configs[pick]
        table.append({"engine": engine, "configuration": pick, "supported_min_n": cfg["supported_min_n"],
                      "gcm_correction": chosen if (cfg["supported_min_n"] or 10 ** 9) < 1000 else "g0",
                      "linear_efficiency": {r: v["ratio"] for r, v in cfg["C2_linear_efficiency"]["by_r"].items()},
                      "alternatives": [i for i in ids if i != pick]})
    res["supported_n_table"] = table
    return res


def summarize(cells, out_dir):
    cs, Ps, rows = {}, {}, []
    for c in cells:
        P, RD, secs = load_cell(c, out_dir)
        Ps[c["cell_index"]] = P
        st = cell_stats(c, P, RD, secs)
        cs[c["cell_index"]] = st
        for m, ms in st["methods"].items():
            for j, nm in enumerate(c["names"]):
                k = ms["rej"][nm]
                lo, hi = PA.clopper_pearson(k, st["R"])
                rows.append(dict(cell_index=c["cell_index"], family=c["family"], n=c["n"], p=c["p"],
                                 shape=c["shape"], r=c["r"], error=c["error"], method=m, predictor=nm,
                                 role=c["roles"][j], reps=st["R"], rejections=k, rate=k / st["R"],
                                 ci99_low=lo, ci99_high=hi,
                                 analytic=c["analytic_power_r1"] if (j == 0 and m == "R1_hc4") else None))
    for st in cs.values():
        for ms in st["methods"].values():
            ms.pop("rej")
    return rows, cs, evaluate(cells, cs, Ps)


def write_summary(rows, cs, crit, out_dir):
    out_dir = Path(out_dir)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "cell-stats.json").write_text(json.dumps({str(k): v for k, v in cs.items()}, indent=1,
                                                        default=float), encoding="utf-8")
    (out_dir / "criteria.json").write_text(json.dumps(crit, indent=1, default=float), encoding="utf-8")
    (out_dir / "supported-n.json").write_text(json.dumps(crit["supported_n_table"], indent=1, default=float),
                                              encoding="utf-8")


# ---------------------------------------------------------------- manifest and seed audit

def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifest():
    import statsmodels
    import sklearn
    import lightgbm
    cells = enumerate_cells()
    return {
        "schema": SCHEMA,
        "protocol": "outline/2026-10-18-phase-b-local-detector-protocol.md (main checkout)",
        "source": "examples/phase_b_local_detector.py", "source_sha256": source_sha256(),
        "phase_a_source_sha256": PHASE_A_SOURCE_SHA, "phase_a_prime_source_sha256": PHASE_A_PRIME_SOURCE_SHA,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__, "scikit-learn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__},
        "seeds": {"base": SEED_BASE, "stride": SEED_STRIDE, "fixture_base": FIXTURE_SEED_BASE,
                  "rerun_base": RERUN_SEED_BASE,
                  "run_range": [cells[0]["seed_first"], max(c["seed_last"] for c in cells)],
                  "engine_seed": "first 31 bits of SHA-256('dataset_seed|engine|K|fold_or_half|role|j')"},
        "design": {"sigma_5": SIGMA5.tolist(), "reliability_E": RELIABILITY, "mo_sd": _MO_SD,
                   "shapes": {"L": "x", "Mo": "tanh(1.5x)/sd", "U": "(x^2-1)/sqrt(2)", "I": "x*m",
                              "E": "latent x; observed sqrt(.8)x + sqrt(.2)u"},
                   "effect": "r^2 = beta^2 V / (beta^2 V + sigma2), V = Var(s - E[s|C-1]); Var(T) = 1",
                   "heteroskedastic": "eps *= sqrt((1 + Z2^2 + X0^2)/3)"},
        "engines": ENGINE_CONFIG, "corrections": CORRECTIONS,
        "test_details": {"alpha": ALPHA, "R1": "OLS + HC4, t(n - k)",
                         "gcm": "R = eT*eX over all rows; Z = sqrt(n) mean/sd, two-sided normal (g0/g1); "
                                "CR1 fold-cluster t(K-1) (g2)",
                         "pcm": "per half h: g on A (all candidates), m = E[g|C-j] on A, hB = g(B) - m(B); "
                                "cross-fitted (K folds within B) GCM of T vs hB given C-j, one-sided; "
                                "p = min(1, 2 min(pA, pB))",
                         "combined": "min(1, 2 min(p_gcm, p_pcm)); Cauchy descriptive",
                         "clarification": "PCM's second-half regressions are cross-fitted within the half "
                                          "(protocol 2.3 step 2), so flexible engines are never evaluated in-sample"},
        "generation_order": list(GENERATION_ORDER),
        "criteria": {"z99": Z99, "zg": ZG, "margin": MARGIN, "count_level": COUNT_LEVEL,
                     "c1_cell_chance": C1_CELL_CHANCE, "c4_cell_chance": C4_CELL_CHANCE,
                     "c1_se": "binomial SE at the tested rate (.055 limit; .06 gross)",
                     "c2": f"power >= {EFFICIENCY} power(R1) - z99 SE_paired at N = 5,000, shape L",
                     "screen_power_loss": SCREEN_POWER_LOSS,
                     "ladders": {"main": list(MAIN_N), "small": list(SMALL_N), "screen": list(SCREEN_N)}},
        "cells": cells,
    }


def manifest_sha(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":"), default=float)
                          .encode("utf-8")).hexdigest()


_INT = re.compile(r"(?<![0-9A-Za-z.])3\d{9}(?![0-9A-Za-z.])")


def seed_audit():
    lo, hi = AUDIT_RANGE
    files = sorted(set((ROOT / "examples" / "study_manifests").glob("*.json"))
                   | set((ROOT / "examples" / "results").rglob("*.json"))
                   | set((ROOT / "examples").rglob("*.py")))
    scanned, skipped, overlaps = [], [], []
    for path in files:
        if path.resolve() == Path(__file__).resolve() or "phase-b-local-detector" in path.as_posix():
            continue
        if path.stat().st_size > 50_000_000:
            skipped.append(path.relative_to(ROOT).as_posix())
            continue
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        if path.suffix == ".py":
            text = re.sub(r"(?<=\d)_(?=\d)", "", text)
        hits = sorted({int(m) for m in _INT.findall(text) if lo <= int(m) <= hi})
        blocks = []
        if path.suffix == ".json":
            try:
                blocks = [b for b in PA._blocks(json.loads(text)) if b[0] <= hi and b[1] >= lo]
            except ValueError:
                pass
        if hits or blocks:
            overlaps.append({"path": path.relative_to(ROOT).as_posix(), "values": hits[:20], "blocks": blocks[:20]})
        scanned.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(data).hexdigest()})
    inc_a = any("phase-a-linear-reference" in f["path"] for f in scanned)
    inc_ap = any("phase-a-prime-small-sample" in f["path"] for f in scanned)
    return {"checked_range": [lo, hi], "files_scanned": len(scanned), "skipped": skipped,
            "overlaps": overlaps, "includes_phase_a_outputs": inc_a, "includes_phase_a_prime_outputs": inc_ap,
            "pass": not overlaps and not skipped and inc_a and inc_ap, "files": scanned}


# ---------------------------------------------------------------- check: loop references

def _loop_fit_resid(engine, C, y, train_idx, test_idx, seed, acc=None):
    Ctr = np.array([C[i] for i in train_idx])
    ytr = np.array([y[i] for i in train_idx])
    Cte = np.array([C[i] for i in test_idx])
    f, = fit_predict(engine, Ctr, ytr, [Cte], seed, acc)
    return {i: float(y[i] - f[t]) for t, i in enumerate(test_idx)}


def _loop_stat(Rs, labels, K, rule, two_sided):
    n = len(Rs)
    mean = math.fsum(Rs) / n
    if rule == "normal":
        sd = math.sqrt(math.fsum((r - mean) ** 2 for r in Rs) / (n - 1))
        z = math.sqrt(n) * mean / sd
        return 2 * stats.norm.sf(abs(z)) if two_sided else stats.norm.sf(z)
    sums = [math.fsum(r - mean for r, l in zip(Rs, labels) if l == k) for k in range(K)]
    V = K / (K - 1) * math.fsum(s * s for s in sums) / n ** 2
    t = mean / math.sqrt(V)
    return 2 * stats.t.sf(abs(t), K - 1) if two_sided else stats.t.sf(t, K - 1)


def loop_gcm(engine, X, T, fold, K, dseed, rule):
    n, p = X.shape
    out = []
    for j in range(p):
        C = [[X[i, c] for c in range(p) if c != j] for i in range(n)]
        C = np.array(C)
        eT, eX = {}, {}
        for k in range(K):
            te = [i for i in range(n) if fold[i] == k]
            tr = [i for i in range(n) if fold[i] != k]
            eT.update(_loop_fit_resid(engine, C, T, tr, te, _seed(dseed, engine, K, k, "gcmT", j)))
            eX.update(_loop_fit_resid(engine, C, X[:, j], tr, te, _seed(dseed, engine, K, k, "gcmX", j)))
        Rs = [eT[i] * eX[i] for i in range(n)]
        out.append(_loop_stat(Rs, [fold[i] for i in range(n)], K, rule, True))
    return np.array(out)


def loop_pcm(engine, X, T, half, fold, K, dseed, rule):
    n, p = X.shape
    halves = []
    for h in (0, 1):
        A = [i for i in range(n) if half[i] == h]
        B = [i for i in range(n) if half[i] != h]
        XA = np.array([X[i] for i in A])
        XB = np.array([X[i] for i in B])
        gA, gB = fit_predict(engine, XA, np.array([T[i] for i in A]), [XA, XB],
                             _seed(dseed, engine, K, h, "pcm_g", -1))
        ps = []
        for j in range(p):
            CA = np.array([[X[i, c] for c in range(p) if c != j] for i in A])
            CB = np.array([[X[i, c] for c in range(p) if c != j] for i in B])
            mB, = fit_predict(engine, CA, gA, [CB], _seed(dseed, engine, K, h, "pcm_m", j))
            hB = [gB[t] - mB[t] for t in range(len(B))]
            TB = [T[i] for i in B]
            fl = [fold[i] for i in B]
            eT, eH = {}, {}
            for k in range(K):
                te = [t for t in range(len(B)) if fl[t] == k]
                tr = [t for t in range(len(B)) if fl[t] != k]
                eT.update(_loop_fit_resid(engine, CB, TB, tr, te, _seed(dseed, engine, K, h, f"pcmT{k}", j)))
                eH.update(_loop_fit_resid(engine, CB, hB, tr, te, _seed(dseed, engine, K, h, f"pcmH{k}", j)))
            Rs = [eT[t] * eH[t] for t in range(len(B))]
            ps.append(_loop_stat(Rs, fl, K, rule, False))
        halves.append(ps)
    return np.array([min(1.0, 2 * min(a, b)) for a, b in zip(*halves)])


def _pick(**kw):
    return next(c for c in enumerate_cells() if all(c[k] == v for k, v in kw.items()))


def _loop_task(task):
    engine, cell, seed = task
    X, T, folds = generate(cell, seed)
    worst = 0.0
    for K, key in ((5, "k5"), (10, "k10")):
        eT, eX = gcm_residuals(engine, X, T, folds[key], K, seed)
        parts = pcm_products(engine, X, T, folds["half"], folds[key], K, seed)
        rules = ("normal",) if K == 5 else ("normal", "cluster")
        for rule in rules:
            g = stat_normal(eT * eX) if rule == "normal" else stat_cluster(eT * eX, folds[key], K)
            worst = max(worst, float(np.max(np.abs(g - loop_gcm(engine, X, T, folds[key], K, seed, rule)))))
            pc = pcm_p(parts, K, rule)
            worst = max(worst, float(np.max(np.abs(pc - loop_pcm(engine, X, T, folds["half"], folds[key], K, seed, rule)))))
    return engine, worst


def check_loops(workers, per_engine=20, engines=("ols",) + ENGINES):
    cells = [_pick(family="small", n=300, shape=s, r=0.20) for s in SHAPES] + [_pick(family="small", n=300, shape="L", r=0.0)]
    tasks = [(e, cells[i % len(cells)], FIXTURE_SEED_BASE + 1000 * (ei + 1) + i)
             for ei, e in enumerate(("ols",) + ENGINES) if e in engines for i in range(per_engine)]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(_loop_task, tasks))
    worst = {}
    for e, w in res:
        worst[e] = max(worst.get(e, 0.0), w)
    return {"fixtures_per_engine": per_engine, "n": 300, "K": [5, 10],
            "statistics": "GCM g0/g1/g2 and PCM g0/g1/g2 p-values", "max_abs_p_diff": worst,
            "tolerance": 1e-10, "pass": bool(max(worst.values()) <= 1e-10)}


def check_hc4_identity(n_fixtures=200):
    AP = _load_a_prime()
    fx = AP._fixture_cells()
    worst = 0.0
    for i in range(n_fixtures):
        c = fx[i % len(fx)]
        X, y, _ = AP.generate(c, FIXTURE_SEED_BASE + i, weights=False)
        P, _ = AP.tests(X, y, np.ones((len(y), 3)))
        worst = max(worst, float(np.max(np.abs(hc4_p(X, y) - P[AP.METHODS.index("M6_hc4_t")]))))
    return {"fixtures": n_fixtures, "max_abs_p_diff": worst, "tolerance": 1e-12, "pass": bool(worst <= 1e-12)}


def check_gcm_identity(n_fixtures=60):
    fx = PA._fixture_cells()
    worst = 0.0
    for i in range(n_fixtures):
        c = fx[i % len(fx)]
        X, y, fold = PA.generate(c, FIXTURE_SEED_BASE + 500 + i)
        eT, eX = gcm_residuals("ols", X, y, fold, PA.K_FOLDS, 0)
        worst = max(worst, float(np.max(np.abs(stat_normal(eT * eX) - PA.gcm_crossfit(X, y, fold)))))
    return {"fixtures": n_fixtures, "reference": "Phase A gcm_crossfit (M4)", "max_abs_p_diff": worst,
            "tolerance": 1e-10, "pass": bool(worst <= 1e-10)}


def _determinism_task(task):
    cell, seed = task
    X, T, folds = generate(cell, seed)
    P, RD = analyze(cell, X, T, folds, seed)
    return P.tobytes(), RD.tobytes()


def check_determinism():
    tasks = [(_pick(family="main", n=1000, shape="Mo", r=0.10), FIXTURE_SEED_BASE + 50_000),
             (_pick(family="small", n=300, shape="I", r=0.20), FIXTURE_SEED_BASE + 50_001)]
    runs = []
    for _ in range(2):
        with cf.ProcessPoolExecutor(max_workers=2) as pool:
            runs.append(list(pool.map(_determinism_task, tasks)))
    same = all(a == b for a, b in zip(runs[0], runs[1]))
    return {"fixtures": len(tasks), "separate_process_pools": 2, "identical_bytes": bool(same), "pass": bool(same)}


def _cond_mean_s(shape, X, p):
    """E[s | C-1] under the latent design, for the population check."""
    S = sigma(p)
    a = np.linalg.solve(S[1:, 1:], S[1:, 0])
    mu = X[:, 1:] @ a
    t2 = _tau2(p)
    if shape in ("L", "E"):
        return mu
    if shape == "U":
        return (mu * mu + t2 - 1) / np.sqrt(2)
    if shape == "I":
        return mu * X[:, 4]
    if shape == "Mo":
        return (np.tanh(1.5 * (mu[:, None] + np.sqrt(t2) * _HX[None, :])) / _MO_SD) @ _HW
    raise ValueError(shape)


def check_population(n=1_000_000):
    rows = []
    designs = [("L", 0.10, "gaussian"), ("Mo", 0.10, "gaussian"), ("U", 0.10, "gaussian"),
               ("I", 0.10, "gaussian"), ("E", 0.10, "gaussian"), ("L", 0.0, "heteroskedastic")]
    for i, (shape, r, err) in enumerate(designs):
        c = _pick(family="main", n=1000, shape=shape, r=r, error=err)
        rng_seed = FIXTURE_SEED_BASE + 900_000 + i
        X, T, _ = generate(c, rng_seed, n=n)
        rng = np.random.default_rng(rng_seed)
        Vlat = rng.standard_normal((n, 5)) @ np.linalg.cholesky(sigma(5)).T
        s = shape_fn(shape, Vlat[:, 0], Vlat[:, 4])
        if r > 0:
            resid_s = s - _cond_mean_s(shape, Vlat, 5)
            realized = float(np.sqrt(c["beta"] ** 2 * resid_s.var() /
                                     (c["beta"] ** 2 * resid_s.var() + c["sigma2"])))
        else:
            realized = 0.0
        pc = PA._partial_corrs(X, T)
        nulls = {nm: float(pc[j]) for j, nm in enumerate(c["names"]) if c["roles"][j] == "null"}
        ok_nulls = all(abs(v) <= 4 / np.sqrt(n) for v in nulls.values())
        row = {"shape": shape, "r_target": r, "error": err, "realized_r": realized,
               "observed_linear_partial_X1": float(pc[0]), "null_linear_partials": nulls,
               "var_T": float(T.var()), "pass": bool(ok_nulls and abs(T.var() - 1) <= 0.01
                                                   and abs(realized - r) <= 0.005)}
        if shape == "E":
            row["measurement_leakage_partials"] = {nm: float(pc[j]) for j, nm in enumerate(c["names"])
                                                   if c["roles"][j] == "measurement"}
        if shape == "I":
            row["moderator_partial_ratio_target"] = c["moderator_partial_ratio"]
            row["moderator_linear_partial"] = float(pc[4])
        rows.append(row)
    return {"rows_per_draw": n, "designs": rows,
            "note": "realized r uses the analytic E[s | C-1]; nulls checked by linear partial correlation "
                    "(|pc| <= 4/sqrt(n)); nulls are exact by construction for every shape",
            "pass": all(r["pass"] for r in rows)}


def guard_output_dir(out_dir):
    if any((Path(out_dir) / x).exists() for x in ("pvalues", "summary.csv", "criteria.json")):
        raise SystemExit(f"refusing to overwrite existing results in {out_dir}")


def check_run_guard():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "pvalues").mkdir()
        try:
            guard_output_dir(tmp)
        except SystemExit:
            return {"refuses_existing_output": True, "pass": True}
    return {"refuses_existing_output": False, "pass": False}


def check_smoke(workers, reps=20):
    cells = enumerate_cells()
    chosen = [_pick(family="main", n=1000, shape="L", r=0.0, error="gaussian"),
              _pick(family="main", n=1000, shape="U", r=0.10),
              _pick(family="main", n=1000, shape="L", r=0.0, error="heteroskedastic"),
              _pick(family="small", n=300, shape="I", r=0.20),
              _pick(family="small", n=500, shape="L", r=0.0),
              _pick(family="screen", n=100, p=10, r=0.0)]
    tmp = Path(tempfile.mkdtemp())
    try:
        t0 = time.perf_counter()
        run_cells(chosen, tmp, workers, seed_override=FIXTURE_SEED_BASE + 100_000, reps_override=reps, size=5)
        small = [dict(c, reps=reps) for c in chosen]
        cs, Ps = {}, {}
        rows = []
        for c in small:
            P, RD, secs = load_cell(c, tmp)
            Ps[c["cell_index"]] = P
            cs[c["cell_index"]] = cell_stats(c, P, RD, secs)
        exercised = {
            "c1_main": c1([small[0], small[1], small[2]], cs, "F3_comb_g0")["pass"],
            "c4_main": c4([small[0], small[1], small[2]], cs, "F1_comb_g0")["pass"],
            "c1_small_g2": c1([small[3], small[4]], cs, "F4_comb_g2")["pass"],
            "screen_c1": c1([small[5]], cs, "R2_gcm_g1")["pass"],
            "paired": _paired([small[1]], Ps, "F2_comb_g0", "R1_hc4", factor=EFFICIENCY),
        }
        files = sorted(p.name for p in (tmp / "pvalues").iterdir())
        stats_ok = all(set(st["methods"]) >= {"R1_hc4", "R2_gcm_g0"} for st in cs.values())
        (tmp / "cell-stats.json").write_text(json.dumps({str(k): v for k, v in cs.items()}, default=float))
        rounds = {str(k): v["rounds"] for k, v in cs.items() if v["rounds"]}
        secs = time.perf_counter() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"cells": [c["cell_index"] for c in chosen], "reps_each": reps, "seconds": round(secs, 1),
            "chunk_files": len(files), "criteria_exercised": exercised, "engine_rounds": rounds,
            "note": "fixture seeds; pipeline smoke test only, not evidence. The full evaluate() needs every "
                    "cell and is exercised by the review on real output.",
            "pass": bool(len(files) == sum(math.ceil(reps / 5) for _ in chosen) and stats_ok)}


def _time_task(task):
    cell, seed, k = task
    t0 = time.perf_counter()
    for r in range(k):
        X, T, folds = generate(cell, seed + r)
        analyze(cell, X, T, folds, seed + r)
    return cell["family"], cell["n"], cell["p"], (time.perf_counter() - t0) / k


def estimate_runtime(workers):
    cells = enumerate_cells()
    keys = {}
    for c in cells:
        keys.setdefault((c["family"], c["n"], c["p"]), c)
    tasks = [(c, FIXTURE_SEED_BASE + 990_000 + 10 * i, 1 if c["n"] >= 2000 else 2)
             for i, c in enumerate(keys.values())]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        per = {(f, n, p): s for f, n, p, s in pool.map(_time_task, tasks)}
    total = sum(per[(c["family"], c["n"], c["p"])] * c["reps"] for c in cells)
    return {"seconds_per_dataset": {f"{f}_n{n}_p{p}": round(s, 2) for (f, n, p), s in per.items()},
            "cpu_seconds_estimate": round(total), "local_wall_hours_at_workers": round(total / workers / 3600, 1),
            "workers": workers, "datasets": sum(c["reps"] for c in cells),
            "actions_runner_minutes_rough": round(total * 1.5 / 4 / 60),
            "note": "single dataset per (family, N, p) on this machine; Actions figure assumes 4 workers per "
                    "runner and a 1.5x slowdown. Stage B0 replaces this."}


PINS = {"python": "3.11.9", "numpy": "1.26.4", "scipy": "1.11.2", "statsmodels": "0.14.4",
        "scikit-learn": "1.3.0", "lightgbm": "4.6.0"}
CHECK_PARTS = ("core", "loops_ols", "loops_F1", "loops_F2", "loops_F3", "loops_F4", "determinism", "smoke",
               "runtime")
GATES = ("manifest", "hc4_identity_with_phase_a_prime", "gcm_identity_with_phase_a", "loop_cross_checks",
         "determinism", "population", "run_guard", "smoke_run", "seed_audit")


def run_part(part, out_dir, workers):
    out_dir = Path(out_dir)
    t0 = time.perf_counter()
    if part == "core":
        m1, m2 = build_manifest(), build_manifest()
        cells = m1["cells"]
        audit = seed_audit()
        (out_dir / "seed-audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
        (out_dir / "protocol-draft.json").write_text(json.dumps(m1, indent=1, default=float), encoding="utf-8")
        rep = {
            "environment": {**m1["software"], "pins_match": m1["software"] == PINS},
            "grid": {"cells": len(cells), "datasets": sum(c["reps"] for c in cells),
                     "by_family": {f: sum(c["family"] == f for c in cells) for f in ("main", "small", "screen")},
                     "datasets_by_family": {f: sum(c["reps"] for c in cells if c["family"] == f)
                                            for f in ("main", "small", "screen")},
                     "seed_range": m1["seeds"]["run_range"],
                     "coefficients": [{"shape": c["shape"], "p": c["p"], "r": c["r"], "beta": c["beta"],
                                       "sigma2": c["sigma2"], "V": c["unique_variance"]}
                                      for c in cells if c["family"] == "main" and c["n"] == 1000 and c["r"] > 0]},
            "manifest": {"sha256": manifest_sha(m1), "deterministic": manifest_sha(m1) == manifest_sha(m2),
                         "source_sha256": m1["source_sha256"], "pass": manifest_sha(m1) == manifest_sha(m2)},
            "hc4_identity_with_phase_a_prime": check_hc4_identity(),
            "gcm_identity_with_phase_a": check_gcm_identity(),
            "population": check_population(),
            "run_guard": check_run_guard(),
            "seed_audit": {k: v for k, v in audit.items() if k != "files"},
        }
    elif part.startswith("loops_"):
        rep = {"loop_cross_checks": check_loops(workers, engines=(part[6:],))}
    elif part == "determinism":
        rep = {"determinism": check_determinism()}
    elif part == "smoke":
        rep = {"smoke_run": check_smoke(workers)}
    elif part == "runtime":
        rep = {"runtime_projection": estimate_runtime(workers)}
    else:
        raise SystemExit(f"unknown part {part}")
    rep["_part"] = {"name": part, "seconds": round(time.perf_counter() - t0, 1), "workers": workers,
                    "source_sha256": source_sha256(), "platform": platform.platform()}
    (out_dir / f"part-{part}.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    return rep


def combine_check(parts_dir, out_dir):
    parts_dir, out_dir = Path(parts_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep, meta, loops = {"schema": SCHEMA + ".check"}, {}, []
    for part in CHECK_PARTS:
        f = next(iter(sorted(parts_dir.rglob(f"part-{part}.json"))), None)
        if f is None:
            meta[part] = "missing"
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        meta[part] = d.pop("_part")
        if "loop_cross_checks" in d:
            loops.append(d.pop("loop_cross_checks"))
        rep.update(d)
    if loops:
        worst = {k: v for lp in loops for k, v in lp["max_abs_p_diff"].items()}
        rep["loop_cross_checks"] = {**{k: v for k, v in loops[0].items() if k != "max_abs_p_diff"},
                                    "max_abs_p_diff": worst, "engines": sorted(worst),
                                    "pass": len(loops) == 5 and all(lp["pass"] for lp in loops)}
    rep["parts"] = meta
    sources = {m["source_sha256"] for m in meta.values() if isinstance(m, dict)}
    rep["parts_same_source"] = len(sources) == 1
    rep["all_pass"] = bool(rep.get("environment", {}).get("pins_match") and rep["parts_same_source"]
                           and all(rep.get(g, {}).get("pass") for g in GATES))
    (out_dir / "check.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    for name in ("seed-audit.json", "protocol-draft.json"):
        f = next(iter(sorted(parts_dir.rglob(name))), None)
        if f is not None and f.parent.resolve() != out_dir.resolve():
            shutil.copy2(f, out_dir / name)
    return rep


def run_check(out_dir, workers, part=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ((part,) if part else CHECK_PARTS):
        run_part(name, out_dir, workers)
    if part is None:
        return combine_check(out_dir, out_dir)
    return json.loads((out_dir / f"part-{part}.json").read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--combine-check", metavar="DIR")
    mode.add_argument("--write-manifest", metavar="PATH")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--part", choices=CHECK_PARTS)
    a = ap.parse_args(argv)
    if a.check:
        if not a.out:
            ap.error("--check requires --out")
        r = run_check(a.out, a.workers, a.part)
        print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
                          for k, v in r.items() if k not in ("grid", "runtime_projection", "parts")},
                         indent=1, default=float))
        if a.part:
            return 0 if all(v["pass"] for v in r.values() if isinstance(v, dict) and "pass" in v) else 1
        return 0 if r["all_pass"] else 1
    if a.combine_check:
        if not a.out:
            ap.error("--combine-check requires --out")
        r = combine_check(a.combine_check, a.out)
        print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
                          for k, v in r.items() if k not in ("grid", "runtime_projection", "parts")},
                         indent=1, default=float))
        return 0 if r["all_pass"] else 1
    if a.write_manifest:
        m = build_manifest()
        Path(a.write_manifest).write_text(json.dumps(m, indent=1, default=float), encoding="utf-8")
        print(manifest_sha(m))
        return 0
    if not (a.manifest and a.out):
        ap.error("--run requires --manifest and --out")
    frozen = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    current = json.loads(json.dumps(build_manifest(), default=float))
    if manifest_sha(frozen) != manifest_sha(current):
        raise SystemExit("manifest does not match this source/environment; refusing to run")
    guard_output_dir(a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    timings = run_cells(current["cells"], a.out, a.workers)
    rows, cs, crit = summarize(current["cells"], a.out)
    write_summary(rows, cs, crit, a.out)
    (Path(a.out) / "runtime.json").write_text(json.dumps({"seconds": time.perf_counter() - t0,
                                                          "per_chunk": timings}, indent=1), encoding="utf-8")
    print(json.dumps(crit["supported_n_table"], default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
