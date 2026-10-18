"""Phase A linear-reference calibration; research script, not a package API.

Protocol: outline/2026-10-17-phase-a-linear-reference-protocol.md (main checkout).

Modes:
  --check --out DIR            fixture/population checks, seed audit, manifest
                               determinism, run guard and a fixture-seed smoke run
  --write-manifest PATH        write the canonical manifest (freezing step)
  --run --manifest PATH --out DIR [--workers N]
                               full run; refuses unless PATH equals the manifest
                               regenerated from this source, and DIR has no output
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import concurrent.futures as cf
import hashlib
import json
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

SCHEMA = "open.phase_a_linear_reference.v1"
SEED_BASE = 3_700_000_000
SEED_STRIDE = 10_000
FIXTURE_SEED_BASE = 3_709_000_000      # --check fixtures and smoke run only
RERUN_SEED_BASE = 3_710_000_000        # reserved for a defect rerun (protocol §7.4)
AUDIT_RANGE = (3_700_000_000, 3_719_999_999)
NULL_REPS = 5000
SIGNAL_REPS = 1000
K_FOLDS = 5
ALPHA = 0.05
Z99 = float(stats.norm.ppf(0.995))
LADDER = (100, 200, 385, 500, 1000, 5000)
THRESHOLDS = {"symmetric": (-1.5, -0.5, 0.5, 1.5), "skewed": (-2.0, -1.3, -0.5, 0.4)}
METHODS = ("M1_ols_t", "M2_hc3_t", "M3_gcm_ols", "M4_gcm_ols_crossfit")
CRITERION_FAMILIES = ("primary", "robustness", "coarsened_target", "group")
GENERATION_ORDER = (
    "rng = numpy.random.default_rng(seed)",
    "X = rng.standard_normal((n, p)) @ cholesky(Sigma).T",
    "error draw (gaussian: standard_normal(n); heteroskedastic: standard_normal(n) "
    "scaled by sqrt((1 + X1^2 + X2^2) / 3); skewed: (chisquare(3, n) - 3) / sqrt(6); "
    "heavy_tailed: standard_t(5, n) * sqrt(3/5))",
    "T = X @ beta + sqrt(sigma2) * error",
    "optional coarsening of T and/or X at thresholds into categories 1..5",
    "optional MCAR mask = rng.random((n, p)) < rate, then complete-case or median imputation",
    "cross-fitting folds: perm = rng.permutation(n_used); fold[perm] = arange(n_used) % 5",
)


# ---------------------------------------------------------------- design

def correlation(structure, p):
    if structure == "G":
        sigma = np.eye(4)
        sigma[0, 1] = sigma[1, 0] = 0.8
        return sigma
    if structure == "S1":
        return np.eye(p)
    sigma = np.full((p, p), 0.3)
    np.fill_diagonal(sigma, 1.0)
    if structure == "S3":
        sigma[0, 1] = sigma[1, 0] = 0.6
    return sigma


def coefficients(cell):
    """Return (beta, sigma2) so Var(T) = 1 and the effect matches the protocol."""
    if cell["family"] == "group":
        eta2 = cell["eta2"]
        b = np.sqrt(eta2 / 3.6)
        return np.array([b, b, 0.0, 0.0]), 1.0 - eta2
    sigma = correlation(cell["structure"], cell["p"])
    v = 1.0 / np.linalg.inv(sigma)[0, 0]
    rho = cell["rho"]
    b2 = rho ** 2 / (v * (1 - rho ** 2) + rho ** 2)
    beta = np.zeros(cell["p"])
    beta[0] = np.sqrt(b2)
    return beta, 1.0 - b2


def is_null_cell(cell):
    return (cell["eta2"] if cell["family"] == "group" else cell["rho"]) == 0


def roles(cell):
    """'true', 'null' (criterion-eligible) or 'measurement' (descriptive only)."""
    if cell["family"] == "group":
        signal = "true" if cell["eta2"] > 0 else "null"
        return [signal, signal, "null", "null"]
    out = ["true" if cell["rho"] > 0 else "null"]
    for _ in range(1, cell["p"]):
        if cell["rho"] == 0:
            out.append("null")
        elif cell["family"] == "likert_predictors" and cell["structure"] != "S1":
            # Correlated with latent X1, so informative given coarsened X1.
            out.append("measurement")
        else:
            out.append("null")
    return out


def enumerate_cells():
    cells = []

    def add(**kw):
        base = dict(p=4, rho=None, eta2=None, error="gaussian",
                    predictor_scale="continuous", target_scale="continuous",
                    thresholds=None, missing_rate=None, missing_analysis=None)
        base.update(kw)
        cells.append(base)

    for n in LADDER:
        for s in ("S1", "S2", "S3"):
            for p in (4, 10):
                for rho in (0.0, 0.05, 0.10, 0.20):
                    add(family="primary", n=n, structure=s, p=p, rho=rho)
    for err in ("heteroskedastic", "skewed", "heavy_tailed"):
        for n in (100, 200, 500, 1000, 5000):
            for s in ("S2", "S3"):
                for rho in (0.0, 0.10):
                    add(family="robustness", n=n, structure=s, rho=rho, error=err)
    for th in ("symmetric", "skewed"):
        for n in (200, 1000, 5000):
            for rho in (0.0, 0.10):
                add(family="coarsened_target", n=n, structure="S2", rho=rho,
                    target_scale="likert5", thresholds=th)
    for n in (200, 385, 500, 1000, 5000):
        for eta2 in (0.0, 0.005, 0.01, 0.02):
            add(family="group", n=n, structure="G", eta2=eta2)
    for th in ("symmetric", "skewed"):
        for n in (200, 1000, 5000):
            for s in ("S1", "S3"):
                for rho in (0.0, 0.10):
                    add(family="likert_predictors", n=n, structure=s, rho=rho,
                        predictor_scale="likert5", thresholds=th)
    for n in (500, 1000, 5000):
        for rho in (0.0, 0.10):
            for analysis in ("complete_case", "median_impute"):
                add(family="missing_mcar", n=n, structure="S3", rho=rho,
                    missing_rate=0.10, missing_analysis=analysis)

    for i, cell in enumerate(cells):
        beta, sigma2 = coefficients(cell)
        cell["cell_index"] = i
        cell["reps"] = NULL_REPS if is_null_cell(cell) else SIGNAL_REPS
        cell["seed_first"] = SEED_BASE + SEED_STRIDE * i
        cell["seed_last"] = cell["seed_first"] + cell["reps"] - 1
        cell["roles"] = roles(cell)
        cell["beta"] = [float(b) for b in beta]
        cell["sigma2"] = float(sigma2)
        cell["criterion_bearing"] = cell["family"] in CRITERION_FAMILIES
        cell["analytic_power"] = analytic_power(cell)
    return cells


def analytic_power(cell):
    """Analytic power for the true hypothesis (primary/group) at two-sided .05."""
    if cell["family"] == "primary" and cell["rho"] > 0:
        return power_t(cell["rho"], cell["n"], cell["p"])
    if cell["family"] == "group" and cell["eta2"] > 0:
        df2 = cell["n"] - 4 - 1
        lam = cell["n"] * cell["eta2"] / (1 - cell["eta2"])
        crit = stats.f.ppf(1 - ALPHA, 2, df2)
        return float(stats.ncf.sf(crit, 2, df2, lam))
    return None


def power_t(rho, n, p, alpha=ALPHA):
    df = n - p - 1
    ncp = rho / np.sqrt(1 - rho ** 2) * np.sqrt(df)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


# ---------------------------------------------------------------- data

def cut(values, thresholds):
    return np.searchsorted(np.asarray(thresholds), values).astype(float) + 1.0


def draw_error(kind, rng, n, X):
    if kind == "gaussian":
        return rng.standard_normal(n)
    if kind == "heteroskedastic":
        z = rng.standard_normal(n)
        return z * np.sqrt((1.0 + X[:, 0] ** 2 + X[:, 1] ** 2) / 3.0)
    if kind == "skewed":
        return (rng.chisquare(3, n) - 3.0) / np.sqrt(6.0)
    if kind == "heavy_tailed":
        return rng.standard_t(5, n) * np.sqrt(3.0 / 5.0)
    raise ValueError(kind)


def generate(cell, seed, n=None, folds=True):
    rng = np.random.default_rng(seed)
    n = cell["n"] if n is None else n
    sigma = correlation(cell["structure"], cell["p"])
    X = rng.standard_normal((n, sigma.shape[0])) @ np.linalg.cholesky(sigma).T
    e = draw_error(cell["error"], rng, n, X)
    y = X @ np.asarray(cell["beta"]) + np.sqrt(cell["sigma2"]) * e
    if cell["target_scale"] == "likert5":
        y = cut(y, THRESHOLDS[cell["thresholds"]])
    if cell["predictor_scale"] == "likert5":
        X = cut(X, THRESHOLDS[cell["thresholds"]])
    if cell["missing_rate"]:
        mask = rng.random(X.shape) < cell["missing_rate"]
        if cell["missing_analysis"] == "complete_case":
            keep = ~mask.any(axis=1)
            X, y = X[keep], y[keep]
        else:
            X = X.copy()
            for j in range(X.shape[1]):
                X[mask[:, j], j] = np.median(X[~mask[:, j], j])
    fold = None
    if folds:
        perm = rng.permutation(len(y))
        fold = np.empty(len(y), dtype=np.int64)
        fold[perm] = np.arange(len(y)) % K_FOLDS
    return X, y, fold


# ---------------------------------------------------------------- methods

def _design(X):
    return np.column_stack([np.ones(len(X)), X])


def ols_tests(X, y, group=None):
    """M1 classical t, M2 HC3 t (t reference, df = n - p - 1), optional M5 HC3 Wald F."""
    A = _design(X)
    n, k = A.shape
    df = n - k
    inv = np.linalg.inv(A.T @ A)
    beta = inv @ (A.T @ y)
    e = y - A @ beta
    h = np.einsum("ij,jk,ik->i", A, inv, A)
    se1 = np.sqrt(e @ e / df * np.diag(inv))
    meat = A.T @ (A * (e ** 2 / (1 - h) ** 2)[:, None])
    V = inv @ meat @ inv
    se2 = np.sqrt(np.diag(V))
    t1, t2 = beta / se1, beta / se2
    out = {"beta": beta[1:], "t1": t1[1:], "t2": t2[1:],
           "p1": 2 * stats.t.sf(np.abs(t1[1:]), df),
           "p2": 2 * stats.t.sf(np.abs(t2[1:]), df), "df": df}
    if group is not None:
        idx = np.asarray(group) + 1
        b = beta[idx]
        F = float(b @ np.linalg.solve(V[np.ix_(idx, idx)], b)) / len(idx)
        out["F5"] = F
        out["p5"] = float(stats.f.sf(F, len(idx), df))
    return out


def _resid(A_train, b_train, A_test, b_test):
    coef, *_ = np.linalg.lstsq(A_train, b_train, rcond=None)
    return b_test - A_test @ coef


def _gcm_p(ey, ex):
    R = ey * ex
    Z = np.sqrt(len(R)) * R.mean() / R.std(ddof=1)
    return float(2 * stats.norm.sf(abs(Z)))


def gcm_insample(X, y, return_residuals=False):
    out, res = [], []
    for j in range(X.shape[1]):
        A = _design(np.delete(X, j, axis=1))
        ey = _resid(A, y, A, y)
        ex = _resid(A, X[:, j], A, X[:, j])
        out.append(_gcm_p(ey, ex))
        res.append((ey, ex))
    return (np.array(out), res) if return_residuals else np.array(out)


def gcm_crossfit(X, y, fold):
    p = X.shape[1]
    ey = np.empty((p, len(y)))
    ex = np.empty((p, len(y)))
    for k in range(K_FOLDS):
        test = fold == k
        train = ~test
        for j in range(p):
            A = _design(np.delete(X, j, axis=1))
            ey[j, test] = _resid(A[train], y[train], A[test], y[test])
            ex[j, test] = _resid(A[train], X[train, j], A[test], X[test, j])
    return np.array([_gcm_p(ey[j], ex[j]) for j in range(p)])


def analyze(cell, X, y, fold):
    group = (0, 1) if cell["family"] == "group" else None
    o = ols_tests(X, y, group)
    P = np.vstack([o["p1"], o["p2"], gcm_insample(X, y), gcm_crossfit(X, y, fold)])
    return P, o.get("p5", np.nan)


# ---------------------------------------------------------------- run

def run_cell(cell, out_dir, seed_first=None, reps=None):
    seed_first = cell["seed_first"] if seed_first is None else seed_first
    reps = cell["reps"] if reps is None else reps
    started = time.perf_counter()
    P = np.empty((reps, len(METHODS), len(cell["roles"])))
    G = np.full(reps, np.nan)
    n_used = np.empty(reps, dtype=np.int64)
    for r in range(reps):
        X, y, fold = generate(cell, seed_first + r)
        P[r], G[r] = analyze(cell, X, y, fold)
        n_used[r] = len(y)
    path = Path(out_dir) / "pvalues" / f"cell_{cell['cell_index']:03d}.npz"
    np.savez_compressed(path, p=P, group_p=G, n_used=n_used,
                        seed_first=seed_first, reps=reps)
    return cell["cell_index"], time.perf_counter() - started


def run_cells(cells, out_dir, workers, seed_override=None, reps_override=None):
    (Path(out_dir) / "pvalues").mkdir(parents=True, exist_ok=False)
    timings = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for c in sorted(cells, key=lambda c: -c["n"] * c["reps"] * c["p"]):
            seed = None if seed_override is None else seed_override + SEED_STRIDE * c["cell_index"]
            futures.append(pool.submit(run_cell, c, out_dir, seed, reps_override))
        for f in cf.as_completed(futures):
            idx, secs = f.result()
            timings[idx] = secs
    return timings


# ---------------------------------------------------------------- summary

def clopper_pearson(k, n, level=0.99):
    a = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def bh_reject(P, q):
    """Vectorized Benjamini-Hochberg over each row of P (reps x m)."""
    reps, m = P.shape
    order = np.argsort(P, axis=1)
    sorted_p = np.take_along_axis(P, order, axis=1)
    ok = sorted_p <= q * np.arange(1, m + 1) / m
    kmax = np.where(ok.any(axis=1), m - np.argmax(ok[:, ::-1], axis=1), 0)
    rej_sorted = np.arange(m)[None, :] < kmax[:, None]
    rej = np.zeros_like(rej_sorted)
    np.put_along_axis(rej, order, rej_sorted, axis=1)
    return rej


def null_band(R):
    return Z99 * np.sqrt(ALPHA * (1 - ALPHA) / R)


def hard_band(R):
    return 0.013 if R >= NULL_REPS else 0.025


def summarize(cells, out_dir):
    out_dir = Path(out_dir)
    rows, cellstats = [], {}
    for c in cells:
        d = np.load(out_dir / "pvalues" / f"cell_{c['cell_index']:03d}.npz")
        P, G = d["p"], d["group_p"]
        R = P.shape[0]
        rej = P < ALPHA
        roles_ = np.array(c["roles"])
        st = {"R": R, "rates": rej.mean(axis=0), "n": c["n"]}
        for mi, m in enumerate(METHODS):
            for j, role in enumerate(c["roles"]):
                k = int(rej[:, mi, j].sum())
                lo, hi = clopper_pearson(k, R)
                analytic = c["analytic_power"] if (role == "true" and j == 0 and c["family"] == "primary") else None
                rows.append(dict(cell_index=c["cell_index"], family=c["family"], n=c["n"],
                                 structure=c["structure"], p=c["p"], rho=c["rho"], eta2=c["eta2"],
                                 error=c["error"], thresholds=c["thresholds"],
                                 missing_analysis=c["missing_analysis"], method=m,
                                 predictor=f"X{j + 1}" if c["family"] != "group" else ["G1", "G2", "X3", "X4"][j],
                                 role=role, reps=R, rejections=k, rate=k / R, ci99_low=lo, ci99_high=hi,
                                 analytic=analytic, mean_n_used=float(d["n_used"].mean())))
        # BH on M2 over the cell's family of hypotheses.
        null_mask, true_mask = roles_ == "null", roles_ == "true"
        for q in (0.05, 0.20):
            rj = bh_reject(P[:, 1, :], q)
            false = (rj & null_mask).sum(axis=1)
            fdp = false / np.maximum(rj.sum(axis=1), 1)
            st[f"bh{q}"] = {"fdr": float(fdp.mean()), "fdr_se": float(fdp.std(ddof=1) / np.sqrt(R)),
                            "power": float((rj & true_mask).sum(axis=1).mean() / true_mask.sum()) if true_mask.any() else None}
        if c["family"] == "group":
            grej = G < ALPHA
            members = grej[:, None] & (P[:, 1, :2] < ALPHA)
            st["group"] = {"rate": float(grej.mean()), "analytic": c["analytic_power"],
                           "hier_member_any": float(members.any(axis=1).mean()),
                           "hier_both_members": float(members.all(axis=1).mean()),
                           "fwer_se": float(np.sqrt(grej.mean() * (1 - grej.mean()) / R))}
        st["m4_vs_m2"] = {"decision_agreement": float((rej[:, 1] == rej[:, 3]).mean()),
                          "p_corr": float(np.corrcoef(P[:, 1].ravel(), P[:, 3].ravel())[0, 1])}
        cellstats[c["cell_index"]] = st
    return rows, cellstats, evaluate_criteria(cells, cellstats)


def _a1_rule(entries):
    """entries: list of (rate, R). Band + Q99 + hard-bound rule (protocol A1)."""
    K = len(entries)
    q99 = int(stats.binom.ppf(0.99, K, 0.01)) if K else 0
    outside = sum(abs(r - ALPHA) > null_band(R) for r, R in entries)
    hard = sum(abs(r - ALPHA) > hard_band(R) for r, R in entries)
    return {"K": K, "Q99": q99, "outside_band": int(outside), "outside_hard": int(hard),
            "pass": bool(outside <= q99 and hard == 0)}


def _null_entries(cells, cs, families, method, min_n=0, predicate=None):
    out = []
    for c in cells:
        if c["family"] not in families or c["n"] < min_n or (predicate and not predicate(c)):
            continue
        st = cs[c["cell_index"]]
        for j, role in enumerate(c["roles"]):
            if role == "null":
                out.append((float(st["rates"][method, j]), st["R"]))
    return out


def evaluate_criteria(cells, cs):
    res = {}
    mi2, mi4 = 1, 3

    def ladder(rule):
        for n0 in LADDER:
            if rule(n0)["pass"]:
                return n0
        return None

    a1 = lambda n0: _a1_rule(_null_entries(cells, cs, ("primary",), mi2, n0))
    res["A1"] = {**a1(0), "scope_min_n": ladder(a1)}

    fails = []
    for c in cells:
        if c["family"] == "primary" and c["rho"] > 0:
            pi = c["analytic_power"]
            obs = float(cs[c["cell_index"]]["rates"][mi2, 0])
            tol = Z99 * np.sqrt(pi * (1 - pi) / cs[c["cell_index"]]["R"]) + 0.01
            if abs(obs - pi) > tol:
                fails.append({"cell_index": c["cell_index"], "observed": obs, "analytic": pi, "tol": tol})
    res["A2"] = {"failures": fails, "pass": not fails}

    def a3(n0):
        fam = ("robustness", "coarsened_target")
        big = _a1_rule(_null_entries(cells, cs, fam, mi2, max(n0, 500)))
        small = [(r, R) for r, R in _null_entries(cells, cs, fam, mi2, n0, lambda c: c["n"] < 500)]
        small_ok = all(r <= 0.075 for r, _ in small)
        return {**big, "small_n_entries": len(small), "small_n_pass": small_ok,
                "pass": bool(big["pass"] and small_ok)}
    res["A3"] = {**a3(0), "scope_min_n": ladder(a3)}

    a4f = []
    for c in cells:
        if c["family"] in ("primary", "robustness", "coarsened_target"):
            st = cs[c["cell_index"]]
            for q in (0.05, 0.20):
                b = st[f"bh{q}"]
                if b["fdr"] > q + Z99 * b["fdr_se"]:
                    a4f.append({"cell_index": c["cell_index"], "q": q, **b})
    res["A4"] = {"failures": a4f, "pass": not a4f}

    gnull = [(cs[c["cell_index"]]["group"]["rate"], cs[c["cell_index"]]["R"])
             for c in cells if c["family"] == "group" and c["eta2"] == 0]
    gcal = _a1_rule(gnull)
    gpow = []
    for c in cells:
        if c["family"] == "group" and c["eta2"] > 0:
            g = cs[c["cell_index"]]["group"]
            pi = g["analytic"]
            tol = Z99 * np.sqrt(pi * (1 - pi) / cs[c["cell_index"]]["R"]) + 0.01
            if abs(g["rate"] - pi) > tol:
                gpow.append({"cell_index": c["cell_index"], "observed": g["rate"], "analytic": pi, "tol": tol})
    gfwer = [c["cell_index"] for c in cells if c["family"] == "group" and c["eta2"] == 0
             and cs[c["cell_index"]]["group"]["rate"] > ALPHA + Z99 * cs[c["cell_index"]]["group"]["fwer_se"]]
    res["A5"] = {"calibration": gcal, "power_failures": gpow, "fwer_failures": gfwer,
                 "pass": bool(gcal["pass"] and not gpow and not gfwer)}

    def a6(n0):
        cal = _a1_rule(_null_entries(cells, cs, ("primary",), mi4, n0))
        loss = []
        for c in cells:
            if c["family"] == "primary" and c["rho"] > 0 and c["n"] >= max(n0, 385):
                r = cs[c["cell_index"]]["rates"]
                if r[mi4, 0] < r[mi2, 0] - 0.03:
                    loss.append({"cell_index": c["cell_index"], "m4": float(r[mi4, 0]), "m2": float(r[mi2, 0])})
        return {**cal, "power_losses": loss, "pass": bool(cal["pass"] and not loss)}
    res["A6"] = {**a6(0), "scope_min_n": ladder(a6)}

    hard_fail = [k for k in ("A2", "A4", "A5") if not res[k]["pass"]]
    scoped = [k for k in ("A1", "A3", "A6") if not res[k]["pass"]]
    if hard_fail or any(res[k]["scope_min_n"] is None for k in scoped):
        verdict = "FAIL"
    elif scoped:
        verdict = "SCOPED"
    else:
        verdict = "PASS"
    res["verdict"] = verdict
    res["supported_min_n"] = max([LADDER[0]] + [res[k]["scope_min_n"] for k in scoped
                                                if res[k]["scope_min_n"] is not None])
    return res


def write_summary(rows, cellstats, criteria, out_dir):
    import csv
    out_dir = Path(out_dir)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    extra = {str(k): {kk: vv for kk, vv in v.items() if kk not in ("rates",)} for k, v in cellstats.items()}
    (out_dir / "cell-stats.json").write_text(json.dumps(extra, indent=1, default=float), encoding="utf-8")
    (out_dir / "criteria.json").write_text(json.dumps(criteria, indent=1, default=float), encoding="utf-8")


# ---------------------------------------------------------------- manifest

def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifest():
    import statsmodels
    cells = enumerate_cells()
    return {
        "schema": SCHEMA,
        "protocol": "outline/2026-10-17-phase-a-linear-reference-protocol.md (main checkout)",
        "source": "examples/phase_a_linear_reference.py",
        "source_sha256": source_sha256(),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "seeds": {"base": SEED_BASE, "stride": SEED_STRIDE, "fixture_base": FIXTURE_SEED_BASE,
                  "rerun_base": RERUN_SEED_BASE,
                  "run_range": [cells[0]["seed_first"], max(c["seed_last"] for c in cells)]},
        "replications": {"null": NULL_REPS, "signal": SIGNAL_REPS},
        "methods": list(METHODS) + ["M5_group_hc3_wald (group cells)"],
        "test_details": {"alpha": ALPHA, "two_sided": True, "hc3_reference": "t, df = n - p - 1",
                         "group_reference": "F(q, n - p - 1)", "gcm_reference": "N(0,1)",
                         "crossfit_folds": K_FOLDS},
        "thresholds": {k: list(v) for k, v in THRESHOLDS.items()},
        "generation_order": list(GENERATION_ORDER),
        "criteria": {"z99": Z99, "hard_band": {"R>=5000": 0.013, "R=1000": 0.025},
                     "a2_tolerance": "z99*sqrt(pi(1-pi)/R)+0.01", "a3_small_n_max": 0.075,
                     "a6_power_loss_max": 0.03, "ladder": list(LADDER)},
        "cells": cells,
    }


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)


def manifest_sha(manifest):
    return hashlib.sha256(canonical(manifest).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- seed audit

# Alphanumeric boundaries: digit runs inside hex digests (SHA-256) are not seeds.
_INT = re.compile(r"(?<![0-9A-Za-z.])3\d{9}(?![0-9A-Za-z.])")


def seed_audit():
    lo, hi = AUDIT_RANGE
    own = {Path(__file__).resolve()}
    scanned, skipped, overlaps = [], [], []
    files = sorted(set((ROOT / "examples" / "study_manifests").glob("*.json"))
                   | set((ROOT / "examples" / "results").rglob("*.json"))
                   | set((ROOT / "examples").rglob("*.py")))
    for path in files:
        if path.resolve() in own or "phase-a-linear-reference" in path.as_posix():
            continue
        size = path.stat().st_size
        if size > 50_000_000:
            skipped.append({"path": path.relative_to(ROOT).as_posix(), "bytes": size})
            continue
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        if path.suffix == ".py":
            text = re.sub(r"(?<=\d)_(?=\d)", "", text)  # 3_700_000_000 -> 3700000000
        hits = [int(m) for m in _INT.findall(text) if lo <= int(m) <= hi]
        blocks = []
        if path.suffix == ".json":
            try:
                blocks = _blocks(json.loads(text))
            except ValueError:
                pass
        bad_blocks = [b for b in blocks if b[0] <= hi and b[1] >= lo]
        if hits or bad_blocks:
            overlaps.append({"path": path.relative_to(ROOT).as_posix(),
                             "values": sorted(set(hits))[:20], "blocks": bad_blocks[:20]})
        scanned.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest()})
    return {"checked_range": [lo, hi], "files_scanned": len(scanned), "skipped": skipped,
            "overlaps": overlaps, "pass": not overlaps and not skipped,
            "scope": "every 10-digit integer beginning with 3 in examples/study_manifests/*.json, "
                     "examples/results/**/*.json and examples/**/*.py, plus JSON start/end_inclusive blocks",
            "files": scanned}


def _blocks(obj):
    found = []
    if isinstance(obj, dict):
        if isinstance(obj.get("start"), int) and isinstance(obj.get("end_inclusive"), int):
            found.append((obj["start"], obj["end_inclusive"]))
        for v in obj.values():
            found.extend(_blocks(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_blocks(v))
    return found


# ---------------------------------------------------------------- check

def _fixture_cells():
    cells = {c["cell_index"]: c for c in enumerate_cells()}
    pick = lambda **kw: next(c for c in cells.values() if all(c[k] == v for k, v in kw.items()))
    return [
        pick(family="primary", n=100, structure="S3", p=10, rho=0.20),
        pick(family="primary", n=200, structure="S2", p=4, rho=0.0),
        pick(family="robustness", n=100, structure="S3", rho=0.10, error="heteroskedastic"),
        pick(family="robustness", n=200, structure="S2", rho=0.0, error="heavy_tailed"),
        pick(family="coarsened_target", n=200, thresholds="skewed", rho=0.10),
        pick(family="group", n=200, eta2=0.02),
        pick(family="likert_predictors", n=200, structure="S1", rho=0.10, thresholds="symmetric"),
        pick(family="missing_mcar", n=500, rho=0.10, missing_analysis="complete_case"),
        pick(family="missing_mcar", n=500, rho=0.10, missing_analysis="median_impute"),
    ]


def check_statsmodels(n_fixtures=200):
    import statsmodels.api as sm
    worst = {"p1": 0.0, "p2": 0.0, "p5": 0.0, "t1_rel": 0.0, "t2_rel": 0.0, "F5_rel": 0.0}
    fixtures = _fixture_cells()
    for i in range(n_fixtures):
        c = fixtures[i % len(fixtures)]
        X, y, _ = generate(c, FIXTURE_SEED_BASE + i, folds=False)
        group = (0, 1) if c["family"] == "group" else None
        mine = ols_tests(X, y, group)
        A = _design(X)
        r1 = sm.OLS(y, A).fit()
        r2 = sm.OLS(y, A).fit(cov_type="HC3", use_t=True)
        worst["p1"] = max(worst["p1"], float(np.max(np.abs(mine["p1"] - r1.pvalues[1:]))))
        worst["p2"] = max(worst["p2"], float(np.max(np.abs(mine["p2"] - r2.pvalues[1:]))))
        worst["t1_rel"] = max(worst["t1_rel"], float(np.max(np.abs(mine["t1"] / r1.tvalues[1:] - 1))))
        worst["t2_rel"] = max(worst["t2_rel"], float(np.max(np.abs(mine["t2"] / r2.tvalues[1:] - 1))))
        if group:
            Rm = np.zeros((2, A.shape[1]))
            Rm[0, 1] = Rm[1, 2] = 1
            w = r2.wald_test(Rm, use_f=True, scalar=True)
            worst["p5"] = max(worst["p5"], abs(mine["p5"] - float(w.pvalue)))
            worst["F5_rel"] = max(worst["F5_rel"], abs(mine["F5"] / float(w.statistic) - 1))
    ok = (max(worst["p1"], worst["p2"], worst["p5"]) <= 1e-10
          and max(worst["t1_rel"], worst["t2_rel"], worst["F5_rel"]) <= 1e-9)
    return {"fixtures": n_fixtures, "fixture_cells": [c["cell_index"] for c in fixtures],
            "max_abs_p_diff": {k: worst[k] for k in ("p1", "p2", "p5")},
            "max_rel_stat_diff": {k: worst[k] for k in ("t1_rel", "t2_rel", "F5_rel")},
            "tolerance": "p-values 1e-10 absolute; statistics 1e-9 relative", "pass": bool(ok)}


def check_gcm_identities():
    worst_fwl, fold_ok, repro_ok = 0.0, True, True
    for i, c in enumerate(_fixture_cells()):
        X, y, fold = generate(c, FIXTURE_SEED_BASE + 500 + i)
        X2, y2, fold2 = generate(c, FIXTURE_SEED_BASE + 500 + i)
        repro_ok &= bool(np.array_equal(X, X2) and np.array_equal(y, y2) and np.array_equal(fold, fold2))
        sizes = np.bincount(fold, minlength=K_FOLDS)
        fold_ok &= bool(sizes.max() - sizes.min() <= 1 and sizes.sum() == len(y))
        _, res = gcm_insample(X, y, return_residuals=True)
        beta = ols_tests(X, y)["beta"]
        for j, (ey, ex) in enumerate(res):
            worst_fwl = max(worst_fwl, abs((ey @ ex) / (ex @ ex) / beta[j] - 1))
    ok = worst_fwl <= 1e-9 and fold_ok and repro_ok
    return {"fwl_max_rel_diff": worst_fwl, "fold_partition_balanced": fold_ok,
            "seed_reproducible": repro_ok,
            "note": "M3 is verified through the Frisch-Waugh-Lovell identity "
                    "sum(eT*eX)/sum(eX^2) = OLS coefficient; GCM's Z is not algebraically "
                    "equal to the classical t (protocol §5 item 2 wording corrected)",
            "pass": bool(ok)}


def _partial_corrs(X, y):
    out = []
    for j in range(X.shape[1]):
        A = _design(np.delete(X, j, axis=1))
        ey, ex = _resid(A, y, A, y), _resid(A, X[:, j], A, X[:, j])
        out.append(float(np.corrcoef(ey, ex)[0, 1]))
    return np.array(out)


def check_population(n=1_000_000):
    rows = []
    cells = enumerate_cells()
    pick = lambda **kw: next(c for c in cells if all(c[k] == v for k, v in kw.items()))
    designs = []
    for s in ("S1", "S2", "S3"):
        for p in (4, 10):
            designs.append(pick(family="primary", n=5000, structure=s, p=p, rho=0.20))
    for err in ("heteroskedastic", "skewed", "heavy_tailed"):
        designs.append(pick(family="robustness", n=5000, structure="S3", rho=0.10, error=err))
    for i, c in enumerate(designs):
        X, y, _ = generate(c, FIXTURE_SEED_BASE + 900 + i, n=n, folds=False)
        pc = _partial_corrs(X, y)
        target = np.zeros(c["p"])
        target[0] = c["rho"]
        se_pc = (1 - target ** 2) / np.sqrt(n)
        dev = y.var() - 1
        kurt = stats.kurtosis(y, fisher=False)
        se_var = np.sqrt((kurt - 1) / n)
        tol_pc = np.maximum(0.003, 4 * se_pc)
        tol_var = max(0.003, 4 * se_var)
        ok = bool(np.all(np.abs(pc - target) <= tol_pc) and abs(dev) <= tol_var)
        rows.append({"cell_index": c["cell_index"], "family": c["family"], "structure": c["structure"],
                     "p": c["p"], "error": c["error"], "rho": c["rho"],
                     "max_partial_corr_dev": float(np.max(np.abs(pc - target))),
                     "var_T_dev": float(dev), "var_tol": float(tol_var), "pass": ok})
    g = pick(family="group", n=5000, eta2=0.02)
    X, y, _ = generate(g, FIXTURE_SEED_BASE + 950, n=n, folds=False)
    A = _design(X)
    r2 = 1 - np.var(_resid(A, y, A, y)) / np.var(y)
    rows.append({"cell_index": g["cell_index"], "family": "group", "r2": float(r2), "eta2": g["eta2"],
                 "dev": float(r2 - g["eta2"]), "pass": bool(abs(r2 - g["eta2"]) <= 0.003 and abs(y.var() - 1) <= 0.003)})
    for th, cuts in THRESHOLDS.items():
        rng = np.random.default_rng(FIXTURE_SEED_BASE + 960 + len(th))
        cat = cut(rng.standard_normal(n), cuts)
        props = np.bincount(cat.astype(int), minlength=6)[1:] / n
        expected = np.diff(np.concatenate([[0.0], stats.norm.cdf(cuts), [1.0]]))
        rows.append({"family": "likert_thresholds", "thresholds": th, "proportions": props.round(4).tolist(),
                     "expected": expected.round(4).tolist(),
                     "pass": bool(np.max(np.abs(props - expected)) <= 0.003)})
    return {"rows_per_draw": n, "designs": rows,
            "tolerance": "max(.003, 4 Monte Carlo SE): protocol stated a flat ±.003; the heavy-tailed "
                         "variance has SE ≈ .003 at 10^6 rows, so a flat bound would fail by chance",
            "pass": all(r["pass"] for r in rows)}


def check_run_guard():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "pvalues").mkdir()
        try:
            guard_output_dir(Path(tmp))
        except SystemExit:
            return {"refuses_existing_output": True, "pass": True}
    return {"refuses_existing_output": False, "pass": False}


def check_smoke(workers):
    cells = enumerate_cells()
    pick = lambda **kw: next(c for c in cells if all(c[k] == v for k, v in kw.items()))
    chosen = [pick(family="primary", n=200, structure="S3", p=4, rho=0.0),
              pick(family="primary", n=500, structure="S2", p=10, rho=0.10),
              pick(family="group", n=385, eta2=0.01),
              pick(family="robustness", n=100, structure="S2", rho=0.0, error="skewed")]
    tmp = Path(tempfile.mkdtemp())
    try:
        started = time.perf_counter()
        run_cells(chosen, tmp, workers, seed_override=FIXTURE_SEED_BASE + 100_000, reps_override=60)
        rows, cs, crit = summarize(chosen, tmp)
        write_summary(rows, cs, crit, tmp)
        secs = time.perf_counter() - started
        files = sorted(p.name for p in (tmp / "pvalues").iterdir())
        written = all((tmp / f).exists() for f in ("summary.csv", "cell-stats.json", "criteria.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"cells": [c["cell_index"] for c in chosen], "reps_each": 60, "seconds": round(secs, 2),
            "npz_files": files, "summary_rows": len(rows), "outputs_written": written,
            "subset_verdict_exercised": crit["verdict"],
            "example_rates_M2": {str(k): np.round(v["rates"][1], 3).tolist() for k, v in cs.items()},
            "note": "fixture seeds; pipeline smoke test only, not evidence",
            "pass": bool(len(files) == len(chosen) and len(rows) > 0 and written)}


def estimate_runtime(workers):
    """Time a few datasets per cell shape to project full-run wall time."""
    cells = enumerate_cells()
    total = 0.0
    timed = {}
    for c in cells:
        key = (c["n"], c["p"], c["family"] == "group")
        if key not in timed:
            t0 = time.perf_counter()
            for r in range(5):
                X, y, fold = generate(c, FIXTURE_SEED_BASE + 990_000 + r)
                analyze(c, X, y, fold)
            timed[key] = (time.perf_counter() - t0) / 5
        total += timed[key] * c["reps"]
    return {"cpu_seconds_estimate": round(total, 1), "workers": workers,
            "wall_minutes_estimate": round(total / workers / 60, 1),
            "datasets": sum(c["reps"] for c in cells)}


def guard_output_dir(out_dir):
    out_dir = Path(out_dir)
    if any((out_dir / name).exists() for name in ("pvalues", "summary.csv", "criteria.json")):
        raise SystemExit(f"refusing to overwrite existing results in {out_dir}")


def run_check(out_dir, workers):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    m1, m2 = build_manifest(), build_manifest()
    cells = m1["cells"]
    report = {
        "schema": SCHEMA + ".check",
        "environment": {**m1["software"], "pinned_python": "3.11.9",
                        "python_matches_pin": m1["software"]["python"] == "3.11.9"},
        "grid": {"cells": len(cells),
                 "null_cells": sum(is_null_cell(c) for c in cells),
                 "signal_cells": sum(not is_null_cell(c) for c in cells),
                 "datasets": sum(c["reps"] for c in cells),
                 "by_family": {f: sum(c["family"] == f for c in cells)
                               for f in dict.fromkeys(c["family"] for c in cells)},
                 "seed_range": m1["seeds"]["run_range"]},
        "manifest": {"sha256": manifest_sha(m1), "deterministic": manifest_sha(m1) == manifest_sha(m2),
                     "source_sha256": m1["source_sha256"]},
        "statsmodels_agreement": check_statsmodels(),
        "gcm_identities": check_gcm_identities(),
        "population": check_population(),
        "run_guard": check_run_guard(),
        "smoke_run": check_smoke(workers),
        "runtime_projection": estimate_runtime(workers),
    }
    audit = seed_audit()
    (out_dir / "seed-audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    report["seed_audit"] = {k: audit[k] for k in ("checked_range", "files_scanned", "skipped", "overlaps", "pass")}
    report["manifest"]["pass"] = report["manifest"]["deterministic"]
    gates = ["statsmodels_agreement", "gcm_identities", "population", "run_guard", "smoke_run",
             "seed_audit", "manifest"]
    report["all_pass"] = bool(report["environment"]["python_matches_pin"]
                              and all(report[g]["pass"] for g in gates))
    report["seconds"] = round(time.perf_counter() - started, 1)
    (out_dir / "check.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (out_dir / "protocol-draft.json").write_text(json.dumps(m1, indent=1, default=float), encoding="utf-8")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write-manifest", metavar="PATH")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.check:
        if not a.out:
            ap.error("--check requires --out")
        r = run_check(a.out, a.workers)
        print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
                          for k, v in r.items() if k not in ("grid",)}, indent=1, default=float))
        return 0 if r["all_pass"] else 1
    if a.write_manifest:
        m = build_manifest()
        Path(a.write_manifest).write_text(json.dumps(m, indent=1, default=float), encoding="utf-8")
        print(manifest_sha(m))
        return 0
    if not (a.manifest and a.out):
        ap.error("--run requires --manifest and --out")
    frozen = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    current = json.loads(canonical(build_manifest()))
    if manifest_sha(frozen) != manifest_sha(current):
        raise SystemExit("manifest does not match this source/environment; refusing to run")
    out = Path(a.out)
    guard_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = run_cells(current["cells"], out, a.workers)
    rows, cs, crit = summarize(current["cells"], out)
    write_summary(rows, cs, crit, out)
    (out / "runtime.json").write_text(json.dumps({"seconds": time.perf_counter() - started,
                                                  "per_cell": timings}, indent=1), encoding="utf-8")
    print(crit["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
