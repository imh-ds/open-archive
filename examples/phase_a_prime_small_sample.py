"""Phase A' small-sample variance estimators; research script, not a package API.

Protocol: outline/2026-10-17-phase-a-prime-small-sample-protocol.md (main checkout).
Compares HC3, HC4, HC5 and a restricted wild bootstrap of the HC3 t (B = 999)
at N = 100-385, with classical OLS as the exact control and HC2 as a diagnostic.

Modes:
  --check --out DIR            fixture cross-checks, generation identity with Phase A,
                               population draws, seed audit, manifest determinism,
                               run guard, smoke run and runtime projection
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

SCHEMA = "open.phase_a_prime_small_sample.v1"
SEED_BASE = 3_720_000_000
SEED_STRIDE = 10_000
FIXTURE_SEED_BASE = 3_729_000_000
RERUN_SEED_BASE = 3_730_000_000
AUDIT_RANGE = (3_720_000_000, 3_739_999_999)
NULL_REPS = 5000
SIGNAL_REPS = 1000
B_BOOT = 999
ALPHA = 0.05
Z99 = float(stats.norm.ppf(0.995))
LADDER = (100, 150, 200, 275, 385)
METHODS = ("M1_ols_t", "M2_hc3_t", "M6_hc4_t", "M7_hc5_t", "M8_wild_boot_hc3", "D_hc2_t")
CANDIDATES = ("M2_hc3_t", "M6_hc4_t", "M7_hc5_t", "M8_wild_boot_hc3")
ANALYTIC_ORDER = ("M2_hc3_t", "M6_hc4_t", "M7_hc5_t")   # continuity tie-break order
CRITERION_FAMILIES = ("primary", "robustness", "leverage", "bridge")
GENERATION_ORDER = (
    "rng = numpy.random.default_rng(seed)",
    "Z = rng.standard_normal((n, p)) @ cholesky(Sigma).T  (identical to Phase A's X draw)",
    "leverage cells only: X1, X3 = (exp(s*Z) - exp(s^2/2)) / sqrt((exp(s^2)-1) exp(s^2)); others X = Z",
    "error draw exactly as Phase A (gaussian / heteroskedastic on X1, X2 / skewed / heavy_tailed)",
    "T = X @ beta + sqrt(sigma2) * error",
    "W = 2 * rng.integers(0, 2, (n, 999)) - 1  (Rademacher bootstrap weights)",
)


def _load_phase_a():
    data = PHASE_A_SOURCE.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != PHASE_A_SOURCE_SHA:
        raise SystemExit("Phase A source changed; refusing to reuse it")
    spec = importlib.util.spec_from_file_location("phase_a_ref", PHASE_A_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PA = _load_phase_a()


# ---------------------------------------------------------------- design

def coefficients(structure, p, rho):
    sigma = PA.correlation(structure, p)
    v = 1.0 / np.linalg.inv(sigma)[0, 0]
    b2 = rho ** 2 / (v * (1 - rho ** 2) + rho ** 2)
    beta = np.zeros(p)
    beta[0] = np.sqrt(b2)
    return beta, 1.0 - b2


def enumerate_cells():
    cells = []

    def add(**kw):
        base = dict(error="gaussian", lognormal_shape=None)
        base.update(kw)
        cells.append(base)

    for n in LADDER:
        for s in ("S1", "S2", "S3"):
            for p in (4, 10):
                for rho in (0.0, 0.10, 0.20, 0.30):
                    add(family="primary", n=n, structure=s, p=p, rho=rho)
    for err in ("heteroskedastic", "skewed", "heavy_tailed"):
        for n in (100, 200, 385):
            for s in ("S2", "S3"):
                for p in (4, 10):
                    for rho in (0.0, 0.20):
                        add(family="robustness", n=n, structure=s, p=p, rho=rho, error=err)
    for shape in (0.5, 0.8):
        for err in ("gaussian", "heteroskedastic"):
            for n in (100, 200, 385):
                for p in (4, 10):
                    for rho in (0.0, 0.20):
                        add(family="leverage", n=n, structure="S3", p=p, rho=rho, error=err,
                            lognormal_shape=shape)
    for p in (4, 10):
        for rho in (0.0, 0.10):
            add(family="bridge", n=1000, structure="S3", p=p, rho=rho)
    for n in (100, 150, 200):
        for s in ("S2", "S3"):
            for rho in (0.0, 0.20):
                for err in ("gaussian", "heteroskedastic"):
                    add(family="stress", n=n, structure=s, p=15, rho=rho, error=err)

    for i, c in enumerate(cells):
        beta, sigma2 = coefficients(c["structure"], c["p"], c["rho"])
        c["cell_index"] = i
        c["reps"] = NULL_REPS if c["rho"] == 0 else SIGNAL_REPS
        c["seed_first"] = SEED_BASE + SEED_STRIDE * i
        c["seed_last"] = c["seed_first"] + c["reps"] - 1
        c["roles"] = (["true"] if c["rho"] > 0 else ["null"]) + ["null"] * (c["p"] - 1)
        c["beta"] = [float(b) for b in beta]
        c["sigma2"] = float(sigma2)
        c["criterion_bearing"] = c["family"] in CRITERION_FAMILIES
        c["analytic_power"] = (PA.power_t(c["rho"], c["n"], c["p"])
                               if c["family"] in ("primary", "bridge") and c["rho"] > 0 else None)
    return cells


# ---------------------------------------------------------------- data

def _lognormal_std(z, s):
    mean = np.exp(s * s / 2)
    sd = np.sqrt((np.exp(s * s) - 1) * np.exp(s * s))
    return (np.exp(s * z) - mean) / sd


def generate(cell, seed, n=None, weights=True):
    rng = np.random.default_rng(seed)
    n = cell["n"] if n is None else n
    sigma = PA.correlation(cell["structure"], cell["p"])
    X = rng.standard_normal((n, cell["p"])) @ np.linalg.cholesky(sigma).T
    if cell["lognormal_shape"]:
        X = X.copy()
        for j in (0, 2):
            X[:, j] = _lognormal_std(X[:, j], cell["lognormal_shape"])
    e = PA.draw_error(cell["error"], rng, n, X)
    y = X @ np.asarray(cell["beta"]) + np.sqrt(cell["sigma2"]) * e
    W = (2.0 * rng.integers(0, 2, (n, B_BOOT)) - 1.0) if weights else None
    return X, y, W


# ---------------------------------------------------------------- methods

def hc_weights(kind, e, h, k):
    n = len(e)
    hbar = k / n
    if kind == "HC2":
        return e ** 2 / (1 - h)
    if kind == "HC3":
        return e ** 2 / (1 - h) ** 2
    if kind == "HC4":
        delta = np.minimum(4.0, h / hbar)
        return e ** 2 / (1 - h) ** delta
    if kind == "HC5":
        alpha = np.minimum(h / hbar, np.maximum(4.0, 0.7 * h.max() / hbar))
        return e ** 2 / (1 - h) ** (alpha / 2)
    raise ValueError(kind)


def tests(X, y, W):
    """p-values (len(METHODS) x p) and the per-method t statistics (for checks)."""
    A = PA._design(X)
    n, k = A.shape
    df = n - k
    inv = np.linalg.inv(A.T @ A)
    C = inv @ A.T
    beta = C @ y
    e = y - A @ beta
    h = np.einsum("ij,ji->i", A, C)
    out_p, out_t = {}, {}
    se1 = np.sqrt(e @ e / df * np.diag(inv))
    out_t["M1_ols_t"] = beta / se1
    for name, kind in (("M2_hc3_t", "HC3"), ("M6_hc4_t", "HC4"), ("M7_hc5_t", "HC5"), ("D_hc2_t", "HC2")):
        w = hc_weights(kind, e, h, k)
        out_t[name] = beta / np.sqrt((C ** 2) @ w)
    for name, t in out_t.items():
        out_p[name] = 2 * stats.t.sf(np.abs(t[1:]), df)
    out_p["M8_wild_boot_hc3"] = wild_bootstrap(A, y, W, C, h, np.abs(out_t["M2_hc3_t"]))
    P = np.vstack([out_p[m] for m in METHODS])
    return P, {m: t[1:] for m, t in out_t.items()}


def wild_bootstrap(A, y, W, C, h, abs_t_obs):
    """Restricted (null-imposed) wild bootstrap of the HC3 t, Rademacher weights."""
    k = A.shape[1]
    lev = 1.0 / (1 - h) ** 2
    pvals = np.empty(k - 1)
    for j in range(1, k):
        Ar = np.delete(A, j, axis=1)
        coef, *_ = np.linalg.lstsq(Ar, y, rcond=None)
        fitted = Ar @ coef
        resid = y - fitted
        Ystar = fitted[:, None] + resid[:, None] * W
        Bstar = C @ Ystar
        Estar = Ystar - A @ Bstar
        se = np.sqrt(((C[j] ** 2) * lev) @ (Estar ** 2))
        tstar = np.abs(Bstar[j] / se)
        pvals[j - 1] = (1 + np.count_nonzero(tstar >= abs_t_obs[j])) / (W.shape[1] + 1)
    return pvals


# ---------------------------------------------------------------- run

def run_cell(cell, out_dir, seed_first=None, reps=None):
    seed_first = cell["seed_first"] if seed_first is None else seed_first
    reps = cell["reps"] if reps is None else reps
    t0 = time.perf_counter()
    P = np.empty((reps, len(METHODS), cell["p"]))
    for r in range(reps):
        X, y, W = generate(cell, seed_first + r)
        P[r], _ = tests(X, y, W)
    np.savez_compressed(Path(out_dir) / "pvalues" / f"cell_{cell['cell_index']:03d}.npz",
                        p=P, seed_first=seed_first, reps=reps)
    return cell["cell_index"], time.perf_counter() - t0


def run_cells(cells, out_dir, workers, seed_override=None, reps_override=None):
    (Path(out_dir) / "pvalues").mkdir(parents=True, exist_ok=False)
    timings = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = []
        for c in sorted(cells, key=lambda c: -c["n"] * c["reps"] * c["p"]):
            seed = None if seed_override is None else seed_override + SEED_STRIDE * c["cell_index"]
            futs.append(pool.submit(run_cell, c, out_dir, seed, reps_override))
        for f in cf.as_completed(futs):
            idx, secs = f.result()
            timings[idx] = secs
    return timings


# ---------------------------------------------------------------- summary and criteria

def summarize(cells, out_dir):
    out_dir = Path(out_dir)
    rows, cs = [], {}
    for c in cells:
        d = np.load(out_dir / "pvalues" / f"cell_{c['cell_index']:03d}.npz")
        P = d["p"]
        R = P.shape[0]
        rej = P < ALPHA
        roles = np.array(c["roles"])
        nullm, truem = roles == "null", roles == "true"
        st = {"R": R, "methods": {}}
        for mi, m in enumerate(METHODS):
            per_ds = rej[:, mi, nullm].mean(axis=1)
            ms = {"pooled_null": float(per_ds.mean()),
                  "pooled_null_se": float(per_ds.std(ddof=1) / np.sqrt(R)),
                  "null_rates": rej[:, mi, nullm].mean(axis=0).tolist()}
            if truem.any():
                ms["power"] = float(rej[:, mi, 0].mean())
                diff = rej[:, 0, 0].astype(float) - rej[:, mi, 0].astype(float)
                ms["loss_vs_m1"] = float(diff.mean())
                ms["loss_vs_m1_se"] = float(diff.std(ddof=1) / np.sqrt(R))
            for q in (0.05, 0.20):
                rj = PA.bh_reject(P[:, mi, :], q)
                fdp = (rj & nullm).sum(axis=1) / np.maximum(rj.sum(axis=1), 1)
                ms[f"bh{q}"] = {"fdr": float(fdp.mean()), "fdr_se": float(fdp.std(ddof=1) / np.sqrt(R)),
                                "power": float((rj & truem).sum(axis=1).mean() / truem.sum()) if truem.any() else None}
            if m == "M8_wild_boot_hc3":
                ms["share_p_min"] = float((P[:, mi, :] <= 1 / (B_BOOT + 1) + 1e-15).mean())
            st["methods"][m] = ms
            for j, role in enumerate(c["roles"]):
                k = int(rej[:, mi, j].sum())
                lo, hi = PA.clopper_pearson(k, R)
                rows.append(dict(cell_index=c["cell_index"], family=c["family"], n=c["n"],
                                 structure=c["structure"], p=c["p"], rho=c["rho"], error=c["error"],
                                 lognormal_shape=c["lognormal_shape"], method=m, predictor=f"X{j + 1}",
                                 role=role, reps=R, rejections=k, rate=k / R, ci99_low=lo, ci99_high=hi,
                                 analytic=c["analytic_power"] if (j == 0 and role == "true") else None))
        cs[c["cell_index"]] = st
    return rows, cs, evaluate(cells, cs)


def _c1_entry(c, ms, R):
    pooled_lo, pooled_hi = (0.040, 0.060) if R >= NULL_REPS else (0.035, 0.065)
    one_lo, one_hi = (0.030, 0.070) if R >= NULL_REPS else (0.025, 0.075)
    ok_pooled = pooled_lo <= ms["pooled_null"] <= pooled_hi
    ok_each = all(one_lo <= r <= one_hi for r in ms["null_rates"])
    return ok_pooled and ok_each


def evaluate(cells, cs):
    failures = {m: [] for m in CANDIDATES}
    defects = []
    for c in cells:
        if not c["criterion_bearing"]:
            continue
        st = cs[c["cell_index"]]
        R = st["R"]
        m1 = st["methods"]["M1_ols_t"]
        m1_defect = False
        if c["family"] == "primary" and c["rho"] > 0:
            pi = c["analytic_power"]
            tol = Z99 * np.sqrt(pi * (1 - pi) / R) + 0.01
            m1_defect = abs(m1["power"] - pi) > tol
            if m1_defect:
                defects.append({"cell_index": c["cell_index"], "m1_power": m1["power"], "analytic": pi})
        for m in CANDIDATES:
            ms = st["methods"][m]
            crit = "C1" if c["family"] in ("primary", "bridge") else "C3"
            if not _c1_entry(c, ms, R):
                failures[m].append({"criterion": crit, "cell_index": c["cell_index"], "n": c["n"],
                                    "pooled_null": ms["pooled_null"]})
            if c["family"] == "primary" and c["rho"] > 0 and not m1_defect:
                pi = c["analytic_power"]
                tol = Z99 * np.sqrt(pi * (1 - pi) / R) + 0.01
                loss_ok = ms["loss_vs_m1"] <= 0.03 + Z99 * ms["loss_vs_m1_se"]
                if not loss_ok or abs(ms["power"] - pi) > tol:
                    failures[m].append({"criterion": "C2", "cell_index": c["cell_index"], "n": c["n"],
                                        "power": ms["power"], "analytic": pi, "loss_vs_m1": ms["loss_vs_m1"]})
            for q in (0.05, 0.20):
                b = ms[f"bh{q}"]
                if b["fdr"] > q + Z99 * b["fdr_se"]:
                    failures[m].append({"criterion": "C4", "cell_index": c["cell_index"], "n": c["n"],
                                        "q": q, "fdr": b["fdr"]})
    result = {"candidates": {}, "simulation_defects": defects}
    for m in CANDIDATES:
        fails = failures[m]
        # The ladder covers N <= 385; bridge (N = 1,000) failures decide only the
        # "all N" versus "N < 385, HC3 above" scope in select().
        small = [f for f in fails if f["n"] <= LADDER[-1]]
        min_n = next((n0 for n0 in LADDER if not [f for f in small if f["n"] >= n0]), None)
        bridge_ok = not [f for f in fails if f["n"] > LADDER[-1]]
        power = [cs[c["cell_index"]]["methods"][m]["power"] for c in cells
                 if c["family"] == "primary" and c["rho"] > 0]
        result["candidates"][m] = {"failures": fails, "n_failures": len(fails),
                                   "by_criterion": {k: sum(f["criterion"] == k for f in fails)
                                                    for k in ("C1", "C2", "C3", "C4")},
                                   "supported_min_n": min_n, "passes_bridge": bridge_ok,
                                   "mean_primary_power": float(np.mean(power))}
    result["selection"] = select(result["candidates"])
    return result


def select(cand):
    eligible = {m: v for m, v in cand.items() if v["supported_min_n"] is not None}
    if not eligible:
        return {"winner": None, "outcome": "NO_CANDIDATE: stop and investigate (contradicts Phase A)"}
    best_n = min(v["supported_min_n"] for v in eligible.values())
    pool = {m: v for m, v in eligible.items() if v["supported_min_n"] == best_n}
    top = max(v["mean_primary_power"] for v in pool.values())
    tied = [m for m, v in pool.items() if top - v["mean_primary_power"] <= 0.01]
    order = list(ANALYTIC_ORDER) + ["M8_wild_boot_hc3"]
    winner = sorted(tied, key=order.index)[0]
    scope = "all N" if pool[winner]["passes_bridge"] else "N < 385, HC3 at N >= 385"
    return {"winner": winner, "supported_min_n": best_n, "applies": scope,
            "tied_within_01": sorted(tied, key=order.index),
            "outcome": "ADOPT" if best_n == LADDER[0] else f"SCOPED: supported from N >= {best_n}"}


def write_summary(rows, cs, crit, out_dir):
    out_dir = Path(out_dir)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "cell-stats.json").write_text(json.dumps({str(k): v for k, v in cs.items()}, indent=1,
                                                        default=float), encoding="utf-8")
    (out_dir / "criteria.json").write_text(json.dumps(crit, indent=1, default=float), encoding="utf-8")


# ---------------------------------------------------------------- manifest and seed audit

def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifest():
    import statsmodels
    cells = enumerate_cells()
    return {
        "schema": SCHEMA,
        "protocol": "outline/2026-10-17-phase-a-prime-small-sample-protocol.md (main checkout)",
        "source": "examples/phase_a_prime_small_sample.py", "source_sha256": source_sha256(),
        "phase_a_source_sha256": PHASE_A_SOURCE_SHA,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "seeds": {"base": SEED_BASE, "stride": SEED_STRIDE, "fixture_base": FIXTURE_SEED_BASE,
                  "rerun_base": RERUN_SEED_BASE,
                  "run_range": [cells[0]["seed_first"], max(c["seed_last"] for c in cells)]},
        "replications": {"null": NULL_REPS, "signal": SIGNAL_REPS, "bootstrap_B": B_BOOT},
        "methods": list(METHODS), "candidates": list(CANDIDATES),
        "test_details": {"alpha": ALPHA, "reference": "t(n - k) for analytic estimators",
                         "bootstrap": "restricted wild, Rademacher, symmetric p = (1 + #|t*|>=|t|)/(B+1)",
                         "hc4": "delta = min(4, h/hbar)",
                         "hc5": "alpha = min(h/hbar, max(4, 0.7 h_max/hbar)); omega = e^2/(1-h)^(alpha/2)"},
        "generation_order": list(GENERATION_ORDER),
        "criteria": {"C1_pooled": {"R5000": [0.040, 0.060], "R1000": [0.035, 0.065]},
                     "C1_each": {"R5000": [0.030, 0.070], "R1000": [0.025, 0.075]},
                     "C2": "loss_vs_m1 <= .03 + z99*SE_paired and |power - analytic| <= z99*sqrt(pi(1-pi)/R)+.01",
                     "C3": "C1 rule on robustness and leverage cells at every N",
                     "C4": "BH FDR <= q + z99*SE, q in {.05, .20}", "z99": Z99,
                     "ladder": list(LADDER), "tie_window": 0.01,
                     "tie_order": list(ANALYTIC_ORDER) + ["M8_wild_boot_hc3"]},
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
        if path.resolve() == Path(__file__).resolve() or "phase-a-prime-small-sample" in path.as_posix():
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
    includes_phase_a = any("phase-a-linear-reference" in f["path"] for f in scanned)
    return {"checked_range": [lo, hi], "files_scanned": len(scanned), "skipped": skipped,
            "overlaps": overlaps, "includes_phase_a_outputs": includes_phase_a,
            "pass": not overlaps and not skipped and includes_phase_a, "files": scanned}


# ---------------------------------------------------------------- check

def _fixture_cells():
    cells = enumerate_cells()
    pick = lambda **kw: next(c for c in cells if all(c[k] == v for k, v in kw.items()))
    return [pick(family="primary", n=100, structure="S3", p=10, rho=0.20),
            pick(family="primary", n=150, structure="S1", p=4, rho=0.0),
            pick(family="robustness", n=100, structure="S3", p=10, rho=0.0, error="heteroskedastic"),
            pick(family="robustness", n=200, structure="S2", p=4, rho=0.20, error="heavy_tailed"),
            pick(family="leverage", n=100, p=10, rho=0.0, error="heteroskedastic", lognormal_shape=0.8),
            pick(family="leverage", n=100, p=4, rho=0.20, error="gaussian", lognormal_shape=0.5),
            pick(family="stress", n=100, structure="S3", rho=0.20, error="heteroskedastic")]


def _loop_hc(X, y, kind):
    """Deliberately naive per-observation implementation of the published formulas."""
    A = PA._design(X)
    n, k = A.shape
    XtX_inv = np.linalg.inv(A.T @ A)
    beta = XtX_inv @ A.T @ y
    meat = np.zeros((k, k))
    hs = [float(A[i] @ XtX_inv @ A[i]) for i in range(n)]
    hbar, hmax = k / n, max(hs)
    for i in range(n):
        e = y[i] - A[i] @ beta
        hi = hs[i]
        if kind == "HC4":
            w = e * e / (1 - hi) ** min(4.0, hi / hbar)
        else:
            a = min(hi / hbar, max(4.0, 0.7 * hmax / hbar))
            w = e * e / np.sqrt((1 - hi) ** a)
        meat += w * np.outer(A[i], A[i])
    V = XtX_inv @ meat @ XtX_inv
    return (beta / np.sqrt(np.diag(V)))[1:]


def check_estimators(n_fixtures=200):
    import statsmodels.api as sm
    worst = {"M1_p": 0.0, "M2_p": 0.0, "D_hc2_p": 0.0, "HC4_t_rel": 0.0, "HC5_t_rel": 0.0}
    fx = _fixture_cells()
    for i in range(n_fixtures):
        c = fx[i % len(fx)]
        X, y, W = generate(c, FIXTURE_SEED_BASE + i)
        P, T = tests(X, y, W)
        A = PA._design(X)
        worst["M1_p"] = max(worst["M1_p"], float(np.max(np.abs(P[0] - sm.OLS(y, A).fit().pvalues[1:]))))
        worst["M2_p"] = max(worst["M2_p"], float(np.max(np.abs(P[1] - sm.OLS(y, A).fit(cov_type="HC3", use_t=True).pvalues[1:]))))
        worst["D_hc2_p"] = max(worst["D_hc2_p"], float(np.max(np.abs(P[5] - sm.OLS(y, A).fit(cov_type="HC2", use_t=True).pvalues[1:]))))
        worst["HC4_t_rel"] = max(worst["HC4_t_rel"], float(np.max(np.abs(T["M6_hc4_t"] / _loop_hc(X, y, "HC4") - 1))))
        worst["HC5_t_rel"] = max(worst["HC5_t_rel"], float(np.max(np.abs(T["M7_hc5_t"] / _loop_hc(X, y, "HC5") - 1))))
    # Constant-leverage property: balanced +/-1 full factorial (p = 4, 16 runs x 6).
    import itertools
    import statsmodels.api as sm2
    Xf = np.array(list(itertools.product((-1.0, 1.0), repeat=4)) * 6)
    rng = np.random.default_rng(FIXTURE_SEED_BASE + 777)
    yf = Xf @ np.array([0.2, 0.0, 0.1, 0.0]) + rng.standard_normal(len(Xf)) * (1 + 0.5 * (Xf[:, 0] > 0))
    _, Tf = tests(Xf, yf, np.ones((len(yf), 3)))
    A = PA._design(Xf)
    h = np.einsum("ij,ji->i", A, np.linalg.inv(A.T @ A) @ A.T)
    hc2 = sm2.OLS(yf, A).fit(cov_type="HC2").tvalues[1:]
    hc0 = sm2.OLS(yf, A).fit(cov_type="HC0").tvalues[1:]
    prop_hc4 = float(np.max(np.abs(Tf["M6_hc4_t"] / hc2 - 1)))
    prop_hc5 = float(np.max(np.abs(Tf["M7_hc5_t"] / (hc0 * (1 - h[0]) ** 0.25) - 1)))
    ok = (max(worst["M1_p"], worst["M2_p"], worst["D_hc2_p"]) <= 1e-10
          and max(worst["HC4_t_rel"], worst["HC5_t_rel"]) <= 1e-12
          and float(np.ptp(h)) < 1e-12 and prop_hc4 <= 1e-12 and prop_hc5 <= 1e-12)
    return {"fixtures": n_fixtures, "fixture_cells": [c["cell_index"] for c in fx], "max_diffs": worst,
            "constant_leverage": {"leverage_range": float(np.ptp(h)), "hc4_equals_hc2_rel": prop_hc4,
                                  "hc5_equals_hc0_over_sqrt_1mh_rel": prop_hc5},
            "tolerance": "p 1e-10 abs vs statsmodels; HC4/HC5 t 1e-12 rel vs loop and properties",
            "pass": bool(ok)}


def _brute_boot(task):
    import statsmodels.api as sm
    c, seed = task
    X, y, W = generate(c, seed)
    P, T = tests(X, y, W)
    A = PA._design(X)
    k = A.shape[1]
    t_obs = np.abs(T["M2_hc3_t"])
    brute = []
    for j in range(1, k):
        Ar = np.delete(A, j, axis=1)
        r = sm.OLS(y, Ar).fit()
        cnt = 0
        for b in range(W.shape[1]):
            ys = r.fittedvalues + r.resid * W[:, b]
            tb = sm.OLS(ys, A).fit(cov_type="HC3").tvalues[j]
            cnt += abs(tb) >= t_obs[j - 1] - 1e-12 * max(1.0, t_obs[j - 1])
        brute.append((1 + cnt) / (W.shape[1] + 1))
    return float(np.max(np.abs(np.array(brute) - P[METHODS.index("M8_wild_boot_hc3")])))


def check_bootstrap(workers, n_fixtures=20):
    fx = [c for c in _fixture_cells() if c["p"] <= 10 and c["n"] <= 150]
    tasks = [(fx[i % len(fx)], FIXTURE_SEED_BASE + 300 + i) for i in range(n_fixtures)]
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        diffs = list(pool.map(_brute_boot, tasks))
    worst = max(diffs)
    return {"fixtures": n_fixtures, "max_abs_p_diff": worst, "tolerance": 1e-10,
            "note": "brute force refits statsmodels HC3 for each of 999 replicates with identical weights",
            "pass": bool(worst <= 1e-10)}


def check_generation_identity():
    worst, n_checked = 0.0, 0
    for c in enumerate_cells():
        if c["family"] in ("leverage",) or c["cell_index"] % 7:
            continue
        pa_cell = dict(family=c["family"], n=c["n"], structure=c["structure"], p=c["p"], error=c["error"],
                       beta=c["beta"], sigma2=c["sigma2"], target_scale="continuous",
                       predictor_scale="continuous", thresholds=None, missing_rate=None, missing_analysis=None)
        seed = FIXTURE_SEED_BASE + 400 + c["cell_index"]
        X, y, _ = generate(c, seed)
        Xa, ya, _ = PA.generate(pa_cell, seed, folds=False)
        worst = max(worst, float(np.max(np.abs(X - Xa))), float(np.max(np.abs(y - ya))))
        n_checked += 1
    return {"cells_checked": n_checked, "max_abs_diff": worst, "pass": worst == 0.0}


def check_population(n=1_000_000):
    rows = []
    cells = enumerate_cells()
    for shape in (0.5, 0.8):
        for p in (4, 10):
            c = next(x for x in cells if x["family"] == "leverage" and x["lognormal_shape"] == shape
                     and x["p"] == p and x["rho"] == 0.20 and x["error"] == "gaussian" and x["n"] == 385)
            X, y, _ = generate(c, FIXTURE_SEED_BASE + 900 + int(shape * 10) + p, n=n, weights=False)
            pc = PA._partial_corrs(X, y)
            ok_nulls = bool(np.all(np.abs(pc[1:]) <= 4 / np.sqrt(n)))
            hmax = []
            for r in range(200):
                Xs, _, _ = generate(c, FIXTURE_SEED_BASE + 1000 + r, n=100, weights=False)
                A = PA._design(Xs)
                h = np.einsum("ij,ji->i", A, np.linalg.inv(A.T @ A) @ A.T)
                hmax.append(h.max() / (A.shape[1] / 100))
            rows.append({"lognormal_shape": shape, "p": p, "realized_partial_corr_X1": float(pc[0]),
                         "target_gaussian_rho": 0.20, "max_null_partial_corr": float(np.max(np.abs(pc[1:]))),
                         "var_T": float(y.var()), "skewness_X1": float(stats.skew(X[:, 0])),
                         "median_max_leverage_ratio_n100": float(np.median(hmax)),
                         "pass": bool(ok_nulls and abs(y.var() - 1) <= 0.02)})
    return {"rows_per_draw": n, "designs": rows, "pass": all(r["pass"] for r in rows)}


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


def check_smoke(workers):
    cells = enumerate_cells()
    pick = lambda **kw: next(c for c in cells if all(c[k] == v for k, v in kw.items()))
    chosen = [pick(family="primary", n=100, structure="S3", p=10, rho=0.0),
              pick(family="primary", n=200, structure="S2", p=4, rho=0.20),
              pick(family="leverage", n=100, p=4, rho=0.0, error="heteroskedastic", lognormal_shape=0.8),
              pick(family="stress", n=100, structure="S2", rho=0.0, error="gaussian")]
    tmp = Path(tempfile.mkdtemp())
    try:
        t0 = time.perf_counter()
        run_cells(chosen, tmp, workers, seed_override=FIXTURE_SEED_BASE + 100_000, reps_override=60)
        rows, cs, crit = summarize(chosen, tmp)
        write_summary(rows, cs, crit, tmp)
        written = all((tmp / f).exists() for f in ("summary.csv", "cell-stats.json", "criteria.json"))
        files = sorted(p.name for p in (tmp / "pvalues").iterdir())
        secs = time.perf_counter() - t0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"cells": [c["cell_index"] for c in chosen], "reps_each": 60, "seconds": round(secs, 2),
            "npz_files": files, "outputs_written": written,
            "selection_exercised": crit["selection"],
            "note": "fixture seeds; pipeline smoke test only, not evidence",
            "pass": bool(written and len(files) == len(chosen))}


def estimate_runtime(workers):
    cells = enumerate_cells()
    timed, total = {}, 0.0
    for c in cells:
        key = (c["n"], c["p"])
        if key not in timed:
            t0 = time.perf_counter()
            for r in range(3):
                X, y, W = generate(c, FIXTURE_SEED_BASE + 990_000 + r)
                tests(X, y, W)
            timed[key] = (time.perf_counter() - t0) / 3
        total += timed[key] * c["reps"]
    wall = total / workers / 60
    return {"cpu_seconds_estimate": round(total, 1), "workers": workers,
            "wall_minutes_estimate": round(wall, 1), "datasets": sum(c["reps"] for c in cells),
            "split_into_two_batches": wall > 75}


def run_check(out_dir, workers):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    m1, m2 = build_manifest(), build_manifest()
    cells = m1["cells"]
    rep = {
        "schema": SCHEMA + ".check",
        "environment": {**m1["software"], "python_matches_pin": m1["software"]["python"] == "3.11.9",
                        "phase_a_source_verified": True},
        "grid": {"cells": len(cells), "criterion_cells": sum(c["criterion_bearing"] for c in cells),
                 "null_cells": sum(c["rho"] == 0 for c in cells),
                 "signal_cells": sum(c["rho"] > 0 for c in cells),
                 "datasets": sum(c["reps"] for c in cells),
                 "by_family": {f: sum(c["family"] == f for c in cells) for f in dict.fromkeys(c["family"] for c in cells)},
                 "seed_range": m1["seeds"]["run_range"]},
        "manifest": {"sha256": manifest_sha(m1), "deterministic": manifest_sha(m1) == manifest_sha(m2),
                     "source_sha256": m1["source_sha256"]},
        "estimators": check_estimators(),
        "bootstrap_brute_force": check_bootstrap(workers),
        "generation_identity_with_phase_a": check_generation_identity(),
        "leverage_population": check_population(),
        "run_guard": check_run_guard(),
        "smoke_run": check_smoke(workers),
        "runtime_projection": estimate_runtime(workers),
    }
    audit = seed_audit()
    (out_dir / "seed-audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    rep["seed_audit"] = {k: v for k, v in audit.items() if k != "files"}
    rep["manifest"]["pass"] = rep["manifest"]["deterministic"]
    gates = ("manifest", "estimators", "bootstrap_brute_force", "generation_identity_with_phase_a",
             "leverage_population", "run_guard", "smoke_run", "seed_audit")
    rep["all_pass"] = bool(rep["environment"]["python_matches_pin"] and all(rep[g]["pass"] for g in gates))
    rep["seconds"] = round(time.perf_counter() - t0, 1)
    (out_dir / "check.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    (out_dir / "protocol-draft.json").write_text(json.dumps(m1, indent=1, default=float), encoding="utf-8")
    return rep


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
                          for k, v in r.items() if k != "grid"}, indent=1, default=float))
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
                                                          "per_cell": timings}, indent=1), encoding="utf-8")
    print(json.dumps(crit["selection"], default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
