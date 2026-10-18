"""Independent review of the Phase A' run (does not reuse its summary/criteria code).

1. Integrity: manifest/source hashes, one npz per cell, shapes, seeds, p-value ranges.
2. Recount every rejection in summary.csv from the saved p-values.
3. Recompute pooled null rates, power, paired loss vs M1 and BH (statsmodels).
4. Re-derive C1-C4, the ladder and the selection with separately written code.
5. Regenerate 2% of datasets: M1/M2/HC2 with statsmodels, HC4/HC5 with a naive loop,
   M8 with a separately written bootstrap; plus brute-force statsmodels bootstrap
   refits on 40 datasets.

Usage (repository root, pinned Python 3.11.9):
  python -I examples/results/2026-10-17-phase-a-prime-small-sample/review.py [--workers 16]
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
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = ROOT / "examples" / "study_manifests" / "phase-a-prime-small-sample.json"
SCRIPT = ROOT / "examples" / "phase_a_prime_small_sample.py"
EXPECTED_MANIFEST_SHA = "044bf3863d3e60d2bc8bd2899d4c83b245ed0f50f8b98da6913f452174d8cb01"
EXPECTED_SOURCE_SHA = "8329c91ee91460460e0787551dfdd3bdf9953e6e59a697e016f4d335642c984b"
METHODS = ("M1_ols_t", "M2_hc3_t", "M6_hc4_t", "M7_hc5_t", "M8_wild_boot_hc3", "D_hc2_t")
CANDIDATES = ("M2_hc3_t", "M6_hc4_t", "M7_hc5_t", "M8_wild_boot_hc3")
LADDER = (100, 150, 200, 275, 385)
A = 0.05
Z = stats.norm.ppf(0.995)


def sha_canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
                          .encode("utf-8")).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("phase_a_prime_src", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def integrity(manifest):
    problems, data = [], {}
    if sha_canonical(manifest) != EXPECTED_MANIFEST_SHA:
        problems.append("manifest sha mismatch")
    src = hashlib.sha256(SCRIPT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if src != EXPECTED_SOURCE_SHA:
        problems.append("source sha mismatch")
    for c in manifest["cells"]:
        path = HERE / "pvalues" / f"cell_{c['cell_index']:03d}.npz"
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        d = np.load(path)
        P = d["p"]
        if P.shape != (c["reps"], len(METHODS), c["p"]):
            problems.append(f"{path.name} shape")
        if int(d["seed_first"]) != c["seed_first"] or int(d["reps"]) != c["reps"]:
            problems.append(f"{path.name} seeds")
        if not np.isfinite(P).all() or (P <= 0).any() or (P > 1).any():
            problems.append(f"{path.name} p range")
        data[c["cell_index"]] = P
    return problems, data


def recount(manifest, data):
    rows = list(csv.DictReader(open(HERE / "summary.csv", encoding="utf-8")))
    want = {(c["cell_index"], m, f"X{j + 1}"): int((data[c["cell_index"]][:, mi, j] < A).sum())
            for c in manifest["cells"] for mi, m in enumerate(METHODS) for j in range(c["p"])}
    bad = [r for r in rows if want.get((int(r["cell_index"]), r["method"], r["predictor"])) != int(r["rejections"])]
    return {"rows": len(rows), "expected": len(want), "mismatches": len(bad),
            "pass": not bad and len(rows) == len(want)}


_Q = {0.05: (1, 20), 0.20: (1, 5)}


def exact_bh_grid(row, q):
    """BH in exact integer arithmetic for bootstrap p-values on the 1/1,000 grid.

    Bootstrap p-values can sit exactly on a BH threshold q*k/m; floating-point
    implementations may then round the comparison either way (statsmodels did, in
    one dataset each of three cells on the first Actions run). p = j/1000 and
    q = a/b, so p_(k) <= q*k/m  <=>  j_(k) * m * b <= a * k * 1000.
    """
    j = np.rint(np.asarray(row) * 1000).astype(np.int64)
    if not np.array_equal(j / 1000, np.asarray(row)):
        raise ValueError("bootstrap p-value off the 1/1000 grid")
    a, b = _Q[q]
    m = len(j)
    js = np.sort(j)
    ok = js * m * b <= a * np.arange(1, m + 1) * 1000
    if not ok.any():
        return np.zeros(m, dtype=bool)
    kmax = int(np.nonzero(ok)[0].max()) + 1
    return j <= js[kmax - 1]


def recompute_stats(manifest, data):
    from statsmodels.stats.multitest import multipletests
    saved = json.loads((HERE / "cell-stats.json").read_text(encoding="utf-8"))
    mine, worst = {}, 0.0
    for c in manifest["cells"]:
        P = data[c["cell_index"]]
        roles = np.array(c["roles"])
        nul, tru = roles == "null", roles == "true"
        R = P.shape[0]
        cell = {}
        for mi, m in enumerate(METHODS):
            rej = P[:, mi, :] < A
            per = rej[:, nul].mean(axis=1)
            st = {"pooled_null": per.mean(), "null_rates": rej[:, nul].mean(axis=0)}
            if tru.any():
                st["power"] = rej[:, 0].mean()
                d = (P[:, 0, 0] < A).astype(float) - rej[:, 0].astype(float)
                st["loss"], st["loss_se"] = d.mean(), d.std(ddof=1) / math.sqrt(R)
            for q in (0.05, 0.20):
                fdp = []
                for row in P[:, mi, :]:
                    if m == "M8_wild_boot_hc3":
                        rj = exact_bh_grid(row, q)
                    else:
                        rj = multipletests(row, q, "fdr_bh")[0]
                    fdp.append((rj & nul).sum() / max(rj.sum(), 1))
                fdp = np.array(fdp)
                st[q] = (fdp.mean(), fdp.std(ddof=1) / math.sqrt(R))
            s = saved[str(c["cell_index"])]["methods"][m]
            worst = max(worst, abs(s["pooled_null"] - st["pooled_null"]), abs(s["bh0.05"]["fdr"] - st[0.05][0]),
                        abs(s["bh0.2"]["fdr"] - st[0.20][0]))
            if tru.any():
                worst = max(worst, abs(s["power"] - st["power"]), abs(s["loss_vs_m1"] - st["loss"]))
            cell[m] = st
        mine[c["cell_index"]] = cell
    return {"max_abs_diff": worst, "pass": worst <= 1e-12}, mine


def criteria(manifest, mine):
    out = {}
    defects = set()
    for c in manifest["cells"]:
        if c["family"] == "primary" and c["rho"] > 0:
            pi = c["analytic_power"]
            if abs(mine[c["cell_index"]]["M1_ols_t"]["power"] - pi) > Z * math.sqrt(pi * (1 - pi) / c["reps"]) + 0.01:
                defects.add(c["cell_index"])
    for m in CANDIDATES:
        fail_n = []
        for c in manifest["cells"]:
            if c["family"] not in ("primary", "robustness", "leverage", "bridge"):
                continue
            st = mine[c["cell_index"]][m]
            big = c["reps"] >= 5000
            lo, hi = (0.04, 0.06) if big else (0.035, 0.065)
            lo1, hi1 = (0.03, 0.07) if big else (0.025, 0.075)
            bad = not (lo <= st["pooled_null"] <= hi) or any(not (lo1 <= r <= hi1) for r in st["null_rates"])
            if c["family"] == "primary" and c["rho"] > 0 and c["cell_index"] not in defects:
                pi = c["analytic_power"]
                bad |= st["loss"] > 0.03 + Z * st["loss_se"]
                bad |= abs(st["power"] - pi) > Z * math.sqrt(pi * (1 - pi) / c["reps"]) + 0.01
            for q in (0.05, 0.20):
                bad |= st[q][0] > q + Z * st[q][1]
            if bad:
                fail_n.append(c["n"])
        small = [n for n in fail_n if n <= 385]
        min_n = next((n0 for n0 in LADDER if not [n for n in small if n >= n0]), None)
        power = [mine[c["cell_index"]][m]["power"] for c in manifest["cells"]
                 if c["family"] == "primary" and c["rho"] > 0]
        out[m] = {"supported_min_n": min_n, "passes_bridge": not [n for n in fail_n if n > 385],
                  "mean_primary_power": float(np.mean(power)), "n_failing_cells": len(fail_n)}
    elig = {m: v for m, v in out.items() if v["supported_min_n"] is not None}
    if not elig:
        sel = {"winner": None}
    else:
        bn = min(v["supported_min_n"] for v in elig.values())
        pool = {m: v for m, v in elig.items() if v["supported_min_n"] == bn}
        top = max(v["mean_primary_power"] for v in pool.values())
        order = ["M2_hc3_t", "M6_hc4_t", "M7_hc5_t", "M8_wild_boot_hc3"]
        win = sorted([m for m, v in pool.items() if top - v["mean_primary_power"] <= 0.01], key=order.index)[0]
        sel = {"winner": win, "supported_min_n": bn, "applies": "all N" if pool[win]["passes_bridge"] else "N < 385, HC3 at N >= 385"}
    return out, sel, sorted(defects)


def compare_criteria(out, sel):
    saved = json.loads((HERE / "criteria.json").read_text(encoding="utf-8"))
    diffs = []
    for m, v in out.items():
        s = saved["candidates"][m]
        for k in ("supported_min_n", "passes_bridge"):
            if s[k] != v[k]:
                diffs.append(f"{m}.{k}")
        if abs(s["mean_primary_power"] - v["mean_primary_power"]) > 1e-12:
            diffs.append(f"{m}.mean_primary_power")
    for k in ("winner", "supported_min_n", "applies"):
        if saved["selection"].get(k) != sel.get(k):
            diffs.append(f"selection.{k}")
    return {"independent": out, "independent_selection": sel, "differences": diffs, "pass": not diffs}


# ------------------------------------------------------------ regeneration

def _naive_hc_t(A_, y, kind):
    n, k = A_.shape
    inv = np.linalg.inv(A_.T @ A_)
    b = inv @ A_.T @ y
    e = y - A_ @ b
    h = np.array([A_[i] @ inv @ A_[i] for i in range(n)])
    hbar = k / n
    if kind == "HC4":
        w = e ** 2 / (1 - h) ** np.minimum(4, h / hbar)
    else:
        w = e ** 2 / np.sqrt((1 - h) ** np.minimum(h / hbar, np.maximum(4, 0.7 * h.max() / hbar)))
    V = inv @ (A_.T * w) @ A_ @ inv
    return b / np.sqrt(np.diag(V)), df_of(A_)


def df_of(A_):
    return A_.shape[0] - A_.shape[1]


def _boot_separate(A_, y, W, t_abs):
    """Separately written bootstrap: QR solve per replicate block."""
    n, k = A_.shape
    Q, Rm = np.linalg.qr(A_)
    h = (Q ** 2).sum(axis=1)
    inv = np.linalg.inv(Rm.T @ Rm)
    out = []
    for j in range(1, k):
        Ar = np.delete(A_, j, axis=1)
        Qr, _ = np.linalg.qr(Ar)
        fitted = Qr @ (Qr.T @ y)
        Ys = fitted[:, None] + (y - fitted)[:, None] * W
        Bs = np.linalg.solve(Rm, Q.T @ Ys)
        Es = Ys - A_ @ Bs
        row = (inv @ A_.T)[j]
        se = np.sqrt((row ** 2 / (1 - h) ** 2) @ Es ** 2)
        out.append((1 + np.sum(np.abs(Bs[j] / se) >= t_abs[j - 1] * (1 - 1e-12))) / (W.shape[1] + 1))
    return np.array(out)


def regenerate(task):
    import statsmodels.api as sm
    cell, reps, brute = task
    src = load_source()
    P = np.load(HERE / "pvalues" / f"cell_{cell['cell_index']:03d}.npz")["p"]
    worst = 0.0
    for r in reps:
        X, y, W = src.generate(cell, cell["seed_first"] + int(r))
        A_ = sm.add_constant(X, has_constant="add")
        f1 = sm.OLS(y, A_).fit()
        f3 = sm.OLS(y, A_).fit(cov_type="HC3", use_t=True)
        f2 = sm.OLS(y, A_).fit(cov_type="HC2", use_t=True)
        df = df_of(A_)
        t4, _ = _naive_hc_t(A_, y, "HC4")
        t5, _ = _naive_hc_t(A_, y, "HC5")
        p4 = 2 * stats.t.sf(np.abs(t4[1:]), df)
        p5 = 2 * stats.t.sf(np.abs(t5[1:]), df)
        p8 = _boot_separate(A_, y, W, np.abs(f3.tvalues[1:]))
        ref = np.vstack([f1.pvalues[1:], f3.pvalues[1:], p4, p5, p8, f2.pvalues[1:]])
        worst = max(worst, float(np.max(np.abs(ref - P[r]))))
    brute_worst = 0.0
    for r in brute:
        X, y, W = src.generate(cell, cell["seed_first"] + int(r))
        A_ = sm.add_constant(X, has_constant="add")
        t_obs = np.abs(sm.OLS(y, A_).fit(cov_type="HC3").tvalues[1:])
        pb = []
        for j in range(1, A_.shape[1]):
            rf = sm.OLS(y, np.delete(A_, j, axis=1)).fit()
            cnt = sum(abs(sm.OLS(rf.fittedvalues + rf.resid * W[:, b], A_).fit(cov_type="HC3").tvalues[j])
                      >= t_obs[j - 1] * (1 - 1e-12) for b in range(W.shape[1]))
            pb.append((1 + cnt) / (W.shape[1] + 1))
        brute_worst = max(brute_worst, float(np.max(np.abs(np.array(pb) - P[r, 4]))))
    return len(reps), worst, len(brute), brute_worst


def check_regeneration(manifest, workers):
    rng = np.random.default_rng(20261017)
    tasks, brute_left = [], 40
    cells = manifest["cells"]
    brute_cells = set(rng.choice([c["cell_index"] for c in cells if c["n"] <= 200 and c["p"] <= 10], 40, replace=False))
    for c in cells:
        k = max(1, math.ceil(0.02 * c["reps"]))
        reps = np.sort(rng.choice(c["reps"], k, replace=False))
        brute = reps[:1] if c["cell_index"] in brute_cells else reps[:0]
        tasks.append((c, reps, brute))
    n = worst = nb = bworst = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for k, w, kb, bw in pool.map(regenerate, tasks):
            n, worst, nb, bworst = n + k, max(worst, w), nb + kb, max(bworst, bw)
    return {"datasets_regenerated": n, "max_abs_p_diff": worst, "brute_force_datasets": nb,
            "brute_force_max_abs_p_diff": bworst, "tolerance": 1e-9,
            "pass": worst <= 1e-9 and bworst <= 1e-9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems, data = integrity(manifest)
    rep = {"integrity": {"problems": problems, "pass": not problems}}
    if problems:
        print(json.dumps(rep, indent=1))
        return 1
    rep["recount"] = recount(manifest, data)
    rep["cell_stats"], mine = recompute_stats(manifest, data)
    out, sel, defects = criteria(manifest, mine)
    rep["criteria"] = compare_criteria(out, sel)
    rep["simulation_defects_independent"] = defects
    rep["regeneration"] = check_regeneration(manifest, a.workers)
    rep["all_pass"] = all(v["pass"] for v in rep.values() if isinstance(v, dict) and "pass" in v)
    rep["sha256"] = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(HERE.rglob("*")) if p.is_file() and p.name != "verification.json"
                     and "__pycache__" not in p.parts}
    (HERE / "verification.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
                      for k, v in rep.items() if k != "sha256"}, indent=1, default=float))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
