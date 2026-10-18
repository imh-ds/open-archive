"""Independent review of the Phase A linear-reference run (no reuse of its summary code).

1. Integrity: manifest/source hashes, one npz per cell, shapes, seeds, finiteness.
2. Recompute every rejection count in summary.csv from the saved p-values.
3. Recompute BH (statsmodels multipletests) and group/hierarchical statistics.
4. Re-derive criteria A1-A6 with separately written code and compare verdicts.
5. Regenerate a 2% sample of datasets from their seeds; recompute M1/M2/M5 with
   statsmodels and M3/M4 with statsmodels-residual GCM; compare stored p-values.

Usage (repository root, pinned Python 3.11.9):
  python -I examples/results/2026-10-17-phase-a-linear-reference/review.py [--workers 16]
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
MANIFEST = ROOT / "examples" / "study_manifests" / "phase-a-linear-reference.json"
SCRIPT = ROOT / "examples" / "phase_a_linear_reference.py"
EXPECTED_MANIFEST_SHA = "cd2f459b1bbeb4618fb4ee89403ee362ffa2d1d6c298f39a1a14c6191695c107"
EXPECTED_SOURCE_SHA = "d621a6e55bb4697296bb7c29f8f8dd61faba3a1709dccd7c19aa6c0657128bba"
METHODS = ("M1_ols_t", "M2_hc3_t", "M3_gcm_ols", "M4_gcm_ols_crossfit")
A = 0.05
Z = stats.norm.ppf(0.995)
SAMPLE_FRACTION = 0.02
SAMPLE_SEED = 20261017


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
                          .encode("utf-8")).hexdigest()


def load_generator():
    spec = importlib.util.spec_from_file_location("phase_a_src", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ 1-3

def integrity(manifest, cells):
    problems = []
    if canonical_sha(manifest) != EXPECTED_MANIFEST_SHA:
        problems.append("manifest canonical sha mismatch")
    src = hashlib.sha256(SCRIPT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if src != EXPECTED_SOURCE_SHA or manifest["source_sha256"] != EXPECTED_SOURCE_SHA:
        problems.append("source sha mismatch")
    files = sorted((HERE / "pvalues").glob("cell_*.npz"))
    if len(files) != len(cells):
        problems.append(f"expected {len(cells)} npz files, found {len(files)}")
    data = {}
    for c in cells:
        path = HERE / "pvalues" / f"cell_{c['cell_index']:03d}.npz"
        if not path.exists():
            problems.append(f"missing {path.name}")
            continue
        d = np.load(path)
        P = d["p"]
        if P.shape != (c["reps"], len(METHODS), len(c["roles"])):
            problems.append(f"{path.name} shape {P.shape}")
        if int(d["seed_first"]) != c["seed_first"] or int(d["reps"]) != c["reps"]:
            problems.append(f"{path.name} seed/reps mismatch")
        if not np.isfinite(P).all() or (P < 0).any() or (P > 1).any():
            problems.append(f"{path.name} invalid p-values")
        if c["family"] == "group" and not np.isfinite(d["group_p"]).all():
            problems.append(f"{path.name} invalid group p-values")
        data[c["cell_index"]] = {k: d[k] for k in ("p", "group_p", "n_used")}
    return problems, data


def check_summary(cells, data):
    with open(HERE / "summary.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names_std = lambda c: [f"X{j + 1}" for j in range(len(c["roles"]))] if c["family"] != "group" else ["G1", "G2", "X3", "X4"]
    want = {}
    for c in cells:
        P = data[c["cell_index"]]["p"]
        for mi, m in enumerate(METHODS):
            for j, name in enumerate(names_std(c)):
                want[(c["cell_index"], m, name)] = int((P[:, mi, j] < A).sum())
    seen, mism = set(), []
    for r in rows:
        key = (int(r["cell_index"]), r["method"], r["predictor"])
        seen.add(key)
        if key not in want or int(r["rejections"]) != want[key]:
            mism.append(key)
    missing = sorted(set(want) - seen)
    return {"rows": len(rows), "expected_rows": len(want), "mismatches": len(mism),
            "missing": len(missing), "pass": not mism and not missing and len(rows) == len(want)}


def check_cellstats(cells, data):
    from statsmodels.stats.multitest import multipletests
    saved = json.loads((HERE / "cell-stats.json").read_text(encoding="utf-8"))
    worst = 0.0
    mine = {}
    for c in cells:
        P = data[c["cell_index"]]["p"][:, 1, :]
        roles = np.array(c["roles"])
        st = {}
        for q in (0.05, 0.20):
            fdp, pw = [], []
            for row in P:
                rej = multipletests(row, q, "fdr_bh")[0]
                fdp.append((rej & (roles == "null")).sum() / max(rej.sum(), 1))
                if (roles == "true").any():
                    pw.append((rej & (roles == "true")).sum() / (roles == "true").sum())
            fdp = np.array(fdp)
            st[q] = {"fdr": fdp.mean(), "fdr_se": fdp.std(ddof=1) / math.sqrt(len(fdp)),
                     "power": float(np.mean(pw)) if pw else None}
            s = saved[str(c["cell_index"])][f"bh{q}"]
            worst = max(worst, abs(s["fdr"] - st[q]["fdr"]))
            if st[q]["power"] is not None:
                worst = max(worst, abs(s["power"] - st[q]["power"]))
        if c["family"] == "group":
            G = data[c["cell_index"]]["group_p"]
            g = (G < A)
            st["group"] = {"rate": g.mean()}
            worst = max(worst, abs(saved[str(c["cell_index"])]["group"]["rate"] - g.mean()))
        mine[c["cell_index"]] = st
    return {"max_abs_diff": worst, "pass": worst <= 1e-12}, mine


# ------------------------------------------------------------ 4: criteria, rewritten

def rate(data, c, mi, j):
    return float((data[c["cell_index"]]["p"][:, mi, j] < A).mean())


def band_rule(pairs):
    k = len(pairs)
    allowed = int(stats.binom.ppf(0.99, k, 0.01)) if k else 0
    soft = sum(abs(r - A) > Z * math.sqrt(A * (1 - A) / R) for r, R in pairs)
    hard = sum(abs(r - A) > (0.013 if R >= 5000 else 0.025) for r, R in pairs)
    return soft <= allowed and hard == 0


def nulls(cells, data, fams, mi, n_min=0, n_max=10 ** 9):
    return [(rate(data, c, mi, j), c["reps"]) for c in cells
            if c["family"] in fams and n_min <= c["n"] <= n_max
            for j, role in enumerate(c["roles"]) if role == "null"]


def criteria(cells, data, mine):
    ladder = (100, 200, 385, 500, 1000, 5000)
    out = {}
    out["A1"] = band_rule(nulls(cells, data, ("primary",), 1))
    out["A1_scope"] = next((n for n in ladder if band_rule(nulls(cells, data, ("primary",), 1, n))), None)
    a2 = True
    for c in cells:
        if c["family"] == "primary" and c["rho"] > 0:
            pi = c["analytic_power"]
            a2 &= abs(rate(data, c, 1, 0) - pi) <= Z * math.sqrt(pi * (1 - pi) / c["reps"]) + 0.01
    out["A2"] = bool(a2)

    def a3(n0):
        fams = ("robustness", "coarsened_target")
        big = band_rule(nulls(cells, data, fams, 1, max(n0, 500)))
        small = all(r <= 0.075 for r, _ in nulls(cells, data, fams, 1, n0, 499))
        return big and small
    out["A3"] = a3(0)
    out["A3_scope"] = next((n for n in ladder if a3(n)), None)
    a4 = True
    for c in cells:
        if c["family"] in ("primary", "robustness", "coarsened_target"):
            for q in (0.05, 0.20):
                s = mine[c["cell_index"]][q]
                a4 &= s["fdr"] <= q + Z * s["fdr_se"]
    out["A4"] = bool(a4)
    g0 = [(mine[c["cell_index"]]["group"]["rate"], c["reps"]) for c in cells
          if c["family"] == "group" and c["eta2"] == 0]
    a5 = band_rule(g0)
    for c in cells:
        if c["family"] != "group":
            continue
        r = mine[c["cell_index"]]["group"]["rate"]
        if c["eta2"] == 0:
            a5 &= r <= A + Z * math.sqrt(r * (1 - r) / c["reps"])
        else:
            pi = c["analytic_power"]
            a5 &= abs(r - pi) <= Z * math.sqrt(pi * (1 - pi) / c["reps"]) + 0.01
    out["A5"] = bool(a5)

    def a6(n0):
        ok = band_rule(nulls(cells, data, ("primary",), 3, n0))
        for c in cells:
            if c["family"] == "primary" and c["rho"] > 0 and c["n"] >= max(n0, 385):
                ok &= rate(data, c, 3, 0) >= rate(data, c, 1, 0) - 0.03
        return ok
    out["A6"] = a6(0)
    out["A6_scope"] = next((n for n in ladder if a6(n)), None)
    scoped = [k for k in ("A1", "A3", "A6") if not out[k]]
    if not (out["A2"] and out["A4"] and out["A5"]) or any(out[k + "_scope"] is None for k in scoped):
        out["verdict"] = "FAIL"
    else:
        out["verdict"] = "SCOPED" if scoped else "PASS"
    return out


def compare_criteria(mine):
    saved = json.loads((HERE / "criteria.json").read_text(encoding="utf-8"))
    diffs = []
    for k in ("A1", "A2", "A3", "A4", "A5", "A6"):
        if bool(saved[k]["pass"]) != bool(mine[k]):
            diffs.append(k)
    for k in ("A1", "A3", "A6"):
        if saved[k]["scope_min_n"] != mine[k + "_scope"]:
            diffs.append(k + "_scope")
    if saved["verdict"] != mine["verdict"]:
        diffs.append("verdict")
    return {"independent": mine, "saved_verdict": saved["verdict"], "differences": diffs,
            "pass": not diffs}


# ------------------------------------------------------------ 5: regeneration

def _sm_resid(y, X):
    import statsmodels.api as sm
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit().resid


def _gcm(ey, ex):
    R = ey * ex
    return 2 * stats.norm.sf(abs(math.sqrt(len(R)) * R.mean() / R.std(ddof=1)))


def regenerate(task):
    import statsmodels.api as sm
    cell, reps_idx = task
    gen = load_generator()
    stored = np.load(HERE / "pvalues" / f"cell_{cell['cell_index']:03d}.npz")
    worst = 0.0
    for r in reps_idx:
        X, y, fold = gen.generate(cell, cell["seed_first"] + int(r))
        Xc = sm.add_constant(X, has_constant="add")
        p1 = sm.OLS(y, Xc).fit().pvalues[1:]
        fit3 = sm.OLS(y, Xc).fit(cov_type="HC3", use_t=True)
        p2 = fit3.pvalues[1:]
        p3, p4 = [], []
        for j in range(X.shape[1]):
            others = np.delete(X, j, axis=1)
            p3.append(_gcm(_sm_resid(y, others), _sm_resid(X[:, j], others)))
            ey, ex = np.empty(len(y)), np.empty(len(y))
            for k in range(5):
                te, tr = fold == k, fold != k
                oc = sm.add_constant(others, has_constant="add")
                ey[te] = y[te] - oc[te] @ sm.OLS(y[tr], oc[tr]).fit().params
                ex[te] = X[te, j] - oc[te] @ sm.OLS(X[tr, j], oc[tr]).fit().params
            p4.append(_gcm(ey, ex))
        ref = np.vstack([p1, p2, p3, p4])
        worst = max(worst, float(np.max(np.abs(ref - stored["p"][r]))))
        if cell["family"] == "group":
            Rm = np.zeros((2, Xc.shape[1]))
            Rm[0, 1] = Rm[1, 2] = 1
            p5 = float(fit3.wald_test(Rm, use_f=True, scalar=True).pvalue)
            worst = max(worst, abs(p5 - float(stored["group_p"][r])))
        if int(stored["n_used"][r]) != len(y):
            worst = float("inf")
    return cell["cell_index"], len(reps_idx), worst


def check_regeneration(cells, workers):
    rng = np.random.default_rng(SAMPLE_SEED)
    tasks = []
    for c in cells:
        k = max(1, math.ceil(SAMPLE_FRACTION * c["reps"]))
        tasks.append((c, np.sort(rng.choice(c["reps"], k, replace=False))))
    worst, n = 0.0, 0
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for _, k, w in pool.map(regenerate, tasks):
            worst, n = max(worst, w), n + k
    return {"datasets_regenerated": n, "max_abs_p_diff": worst, "tolerance": 1e-9,
            "pass": worst <= 1e-9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cells = manifest["cells"]
    problems, data = integrity(manifest, cells)
    report = {"integrity": {"problems": problems, "pass": not problems}}
    if problems:
        print(json.dumps(report, indent=1))
        return 1
    report["summary_counts"] = check_summary(cells, data)
    report["cell_stats"], mine = check_cellstats(cells, data)
    report["criteria"] = compare_criteria(criteria(cells, data, mine))
    report["regeneration"] = check_regeneration(cells, a.workers)
    report["all_pass"] = all(v["pass"] for v in report.values() if isinstance(v, dict))
    hashes = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(HERE.rglob("*")) if p.is_file() and p.name != "verification.json"
              and "__pycache__" not in p.parts}
    report["sha256"] = hashes
    (HERE / "verification.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
                      for k, v in report.items() if k != "sha256"}, indent=1, default=float))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
