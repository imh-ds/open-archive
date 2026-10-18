"""Independent review of the Phase B results (does not import the study script).

Run from the results directory's parent or anywhere: paths are relative to this file.
Checks:
  1. integrity: every cell's chunk files cover its replications contiguously with the
     frozen seeds and shapes;
  2. recomputation: every per-cell statistic in cell-stats.json, recomputed from the raw
     p-values with separately written BH, Bonferroni and Cauchy code (abs 1e-12);
  3. summary.csv row count;
  4. criteria: correction choice, ladder steps, linear efficiency, supported-N table and
     reference defects re-derived from the recomputed statistics with separately
     written rules (protocol section 7), compared with criteria.json;
  5. regeneration: regen-*.json from the Actions regen jobs cover the 1% stratified
     sample (frozen code) and the 20 loop-implementation datasets, all within 1e-10.
Writes verification.json; exits 1 on any failure.
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
TOL = 1e-12
REGEN_TOL = 1e-10
ENGINES = ("F1", "F2", "F3", "F4")
GS = ("g0", "g1", "g2")


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def cell_arrays(c):
    files = sorted((HERE / "pvalues").glob(f"cell_{c['cell_index']:03d}_r*.npz"))
    parts, problems = [], []
    for f in files:
        z = np.load(f)
        parts.append((int(z["rep_lo"]), int(z["rep_hi"]), z["p"], int(z["seed_first"])))
    parts.sort()
    lo = 0
    for a, b, P, s in parts:
        if a != lo:
            problems.append(f"cell {c['cell_index']}: gap/overlap at {lo}")
        if s != c["seed_first"]:
            problems.append(f"cell {c['cell_index']}: seed_first {s}")
        if P.shape != (b - a, len(c["tests"]), c["p"]):
            problems.append(f"cell {c['cell_index']}: shape {P.shape}")
        lo = b
    if lo != c["reps"]:
        problems.append(f"cell {c['cell_index']}: {lo}/{c['reps']} reps")
    P = np.concatenate([x[2] for x in parts]) if parts else None
    return P, problems


def bh(row, q):
    """Benjamini-Hochberg on one family, written as a plain step-up loop."""
    m = len(row)
    order = sorted(range(m), key=lambda i: row[i])
    k = 0
    for rank, i in enumerate(order, start=1):
        if row[i] <= q * rank / m:
            k = rank
    rej = [False] * m
    for i in order[:k]:
        rej[i] = True
    return rej


def methods(c, P):
    out = {t: P[:, i, :] for i, t in enumerate(c["tests"])}
    for e in ENGINES:
        for g in GS:
            if f"{e}_gcm_{g}" in out:
                a, b = out[f"{e}_gcm_{g}"], out[f"{e}_pcm_{g}"]
                out[f"{e}_comb_{g}"] = np.minimum(1.0, 2.0 * np.minimum(a, b))
                ca, cb = np.clip(a, 1e-15, 1 - 1e-15), np.clip(b, 1e-15, 1 - 1e-15)
                stat = 0.5 * (np.tan(np.pi * (0.5 - ca)) + np.tan(np.pi * (0.5 - cb)))
                out[f"{e}_cauchy_{g}"] = 0.5 - np.arctan(stat) / np.pi
    return out


def recompute(c, P):
    roles = c["roles"]
    keys = ["W" if nm.startswith("W") else nm for nm in c["names"]]
    st = {}
    for m, Pm in methods(c, P).items():
        rej = Pm < 0.05
        R = len(Pm)
        nr = {}
        for j, r in enumerate(roles):
            if r == "null":
                nr.setdefault(keys[j], []).append(rej[:, j].sum() / R)
        d = {"null_rates": {k: float(np.mean(v)) for k, v in nr.items()}, "rej_X1": rej[:, 0]}
        if c["r"] > 0:
            d["power_X1"] = float(rej[:, 0].mean())
            if c["shape"] == "I":
                d["power_M"] = float(rej[:, 4].mean())
        for q in (0.05, 0.20):
            fdp = []
            for row in Pm:
                rj = bh(list(row), q)
                false = sum(1 for j, x in enumerate(rj) if x and roles[j] == "null")
                fdp.append(false / max(sum(rj), 1))
            fdp = np.array(fdp)
            d[f"bh{q}"] = {"fdr": float(fdp.mean()), "fdr_sd": float(fdp.std(ddof=1))}
        st[m] = d
    return st


def compare_stats(c, mine, theirs):
    bad = []
    for m, d in mine.items():
        t = theirs["methods"].get(m)
        if t is None:
            bad.append(f"cell {c['cell_index']} {m}: missing")
            continue
        pairs = [(k, d["null_rates"][k], t["null_rates"].get(k, math.nan)) for k in d["null_rates"]]
        for k in ("power_X1", "power_M"):
            if k in d:
                pairs.append((k, d[k], t.get(k, math.nan)))
        for q in ("0.05", "0.2"):
            for k in ("fdr", "fdr_sd"):
                pairs.append((f"bh{q}.{k}", d[f"bh{float(q)}"][k], t[f"bh{q}"][k]))
        for k, x, y in pairs:
            if not abs(x - y) <= TOL:
                bad.append(f"cell {c['cell_index']} {m} {k}: {x} vs {y}")
    return bad


# ---------------------------------------------------------------- criteria (protocol 7)

def allowance(K, pi, level):
    a = 0
    while stats.binom.sf(a, K, pi) > level:
        a += 1
    return a


def c1(cells, S, m, cr):
    z99, zg, mg = cr["z99"], cr["zg"], cr["margin"]
    tot = {}
    fails = gross = 0
    for c in cells:
        R = c["reps"]
        keys = ["W" if nm.startswith("W") else nm for nm, r in zip(c["names"], c["roles"]) if r == "null"]
        bad = False
        for role, rate in S[c["cell_index"]][m]["null_rates"].items():
            k = keys.count(role)
            t = tot.setdefault(role, [0.0, 0])
            t[0] += rate * R * k
            t[1] += R * k
            if rate > 0.05 + mg + z99 * math.sqrt((0.05 + mg) * (0.95 - mg) / R):
                bad = True
            if rate > 0.06 + zg * math.sqrt(0.06 * 0.94 / R):
                gross += 1
        fails += bad
    pooled_ok = all(n / d <= 0.05 + mg + z99 * math.sqrt((0.05 + mg) * (0.95 - mg) / d) for n, d in tot.values())
    return pooled_ok and fails <= allowance(len(cells), cr["c1_cell_chance"], cr["count_level"]) and gross == 0


def c4(cells, S, m, cr):
    z99, zg, mg = cr["z99"], cr["zg"], cr["margin"]
    for q, key in ((0.05, "bh0.05"), (0.20, "bh0.2")):
        num = den = var = 0.0
        fails = gross = 0
        for c in cells:
            b = S[c["cell_index"]][m][key]
            R = c["reps"]
            se = b["fdr_sd"] / math.sqrt(R)
            num += b["fdr"] * R
            den += R
            var += b["fdr_sd"] ** 2 * R
            fails += b["fdr"] > q + mg + z99 * se
            gross += b["fdr"] > q + 0.01 + zg * se
        if not (num / den <= q + mg + z99 * math.sqrt(var) / den
                and fails <= allowance(len(cells), cr["c4_cell_chance"], cr["count_level"]) and gross == 0):
            return False
    return True


def paired_ok(cells, S, a, b, factor, bound, sign):
    d = np.concatenate([S[c["cell_index"]][a]["rej_X1"].astype(float)
                        - factor * S[c["cell_index"]][b]["rej_X1"].astype(float) for c in cells])
    se = d.std(ddof=1) / math.sqrt(len(d))
    return (d.mean() <= bound + cr_z99 * se) if sign > 0 else (d.mean() >= -cr_z99 * se)


def derive(cells, S, cr):
    lad = cr["ladders"]
    sel = lambda **kw: [c for c in cells if all(c[k] == v for k, v in kw.items())]
    screen = {}
    for g in GS:
        m = f"R2_gcm_{g}"
        ok = {n: c1(sel(family="screen", n=n), S, m, cr) for n in lad["screen"]}
        pw = {n: paired_ok([c for c in sel(family="screen", n=n) if c["r"] > 0], S, "R1_hc4", m, 1.0,
                           cr["screen_power_loss"], +1) for n in lad["screen"]}
        n0 = next((n for n in lad["screen"] if all(ok[x] for x in lad["screen"] if x >= n) and pw[n]), None)
        mp = float(np.mean([S[c["cell_index"]][m]["power_X1"] for c in sel(family="screen") if c["r"] > 0]))
        screen[g] = (n0, mp)
    el = {g: v for g, v in screen.items() if v[0] is not None and v[0] < 1000}
    chosen = None
    if el:
        b = min(v[0] for v in el.values())
        pool = [g for g in GS if g in el and el[g][0] == b]
        top = max(screen[g][1] for g in pool)
        chosen = [g for g in pool if top - screen[g][1] <= 0.01][0]
    configs = {}
    for e in ENGINES:
        m0 = f"{e}_comb_g0"
        steps = {n: c1(sel(family="main", n=n), S, m0, cr) and c4(sel(family="main", n=n), S, m0, cr)
                 for n in lad["main"]}
        if chosen:
            mg = f"{e}_comb_{chosen}"
            steps.update({n: c1(sel(family="small", n=n), S, mg, cr) and c4(sel(family="small", n=n), S, mg, cr)
                          for n in lad["small"]})
        ns = sorted(steps)
        min_n = next((n for n in ns if all(steps[x] for x in ns if x >= n)), None)
        c2 = all(paired_ok(sel(family="main", n=5000, shape="L", r=r), S, m0, "R1_hc4", 0.9, 0.0, -1)
                 for r in (0.05, 0.10))
        lp = float(np.mean([S[c["cell_index"]][m0]["power_X1"] for c in sel(family="main", shape="L") if c["r"] > 0]))
        configs[e] = {"steps": {str(k): v for k, v in steps.items()}, "c2": c2,
                      "supported_min_n": min_n if c2 else None, "linear_power": lp}
    table = {}
    for name, ids in (("random_forest", ["F1"]), ("gradient_boosting", ["F2"]), ("lightgbm", ["F3", "F4"])):
        ok = [i for i in ids if configs[i]["supported_min_n"] is not None]
        if ok:
            b = min(configs[i]["supported_min_n"] for i in ok)
            pool = [i for i in ok if configs[i]["supported_min_n"] == b]
            top = max(configs[i]["linear_power"] for i in pool)
            pick = [i for i in ids if i in pool and top - configs[i]["linear_power"] <= 1e-12][0]
        else:
            pick = ids[0]
        table[name] = (pick, configs[pick]["supported_min_n"])
    ref = [c for c in cells if c["family"] in ("main", "small") and (c["shape"] in ("L", "Mo") or c["error"] != "gaussian")]
    ref_ok = all(c1([c for c in ref if c["n"] == n], S, "R1_hc4", cr) and c4([c for c in ref if c["n"] == n], S, "R1_hc4", cr)
                 for n in lad["small"] + lad["main"])
    return {"chosen": chosen, "screen": {g: v[0] for g, v in screen.items()}, "configs": configs,
            "table": table, "r1_ok": ref_ok}


cr_z99 = 0.0


def main():
    global cr_z99
    man = load("manifest.json")
    cells, cr = man["cells"], man["criteria"]
    cr_z99 = cr["z99"]
    cs, crit = load("cell-stats.json"), load("criteria.json")
    out = {"integrity": [], "stat_mismatches": [], "checks": {}}
    S = {}
    for c in cells:
        P, prob = cell_arrays(c)
        out["integrity"] += prob
        if P is None:
            continue
        S[c["cell_index"]] = recompute(c, P)
        out["stat_mismatches"] += compare_stats(c, S[c["cell_index"]], cs[str(c["cell_index"])])
    out["checks"]["integrity"] = not out["integrity"]
    out["checks"]["recomputation"] = not out["stat_mismatches"] and len(S) == len(cells)
    with open(HERE / "summary.csv", encoding="utf-8") as f:
        n_rows = sum(1 for _ in csv.DictReader(f))
    exp_rows = sum(len(S[c["cell_index"]]) * c["p"] for c in cells if c["cell_index"] in S)
    out["summary_rows"] = {"found": n_rows, "expected": exp_rows}
    out["checks"]["summary_rows"] = n_rows == exp_rows
    if out["checks"]["recomputation"]:
        d = derive(cells, S, cr)
        theirs_table = {r["engine"]: (r["configuration"], r["supported_min_n"]) for r in crit["supported_n_table"]}
        agree = {
            "correction_chosen": d["chosen"] == crit["correction_screen"]["chosen"],
            "screen_qualifies_from": all(d["screen"][g] == crit["correction_screen"]["candidates"][g]["qualifies_from"]
                                         for g in GS),
            "ladder_steps": all(d["configs"][e]["steps"] == {k: v["pass"] for k, v in crit["configurations"][e]["ladder"].items()}
                                for e in ENGINES),
            "linear_efficiency": all(d["configs"][e]["c2"] == crit["configurations"][e]["C2_linear_efficiency"]["pass"]
                                     for e in ENGINES),
            "supported_min_n": all(d["configs"][e]["supported_min_n"] == crit["configurations"][e]["supported_min_n"]
                                   for e in ENGINES),
            "supported_n_table": d["table"] == theirs_table,
            "reference_defects": d["r1_ok"] == (not [x for x in crit["reference"]["defects"] if x.startswith("R1")]),
        }
        out["criteria_rederived"] = {"agree": agree, "independent": {
            "chosen": d["chosen"], "screen": d["screen"], "table": d["table"], "r1_ok": d["r1_ok"],
            "configs": {e: {k: v for k, v in c.items()} for e, c in d["configs"].items()}}}
        out["checks"]["criteria"] = all(agree.values())
    else:
        out["checks"]["criteria"] = False
    rp = load("regen-plan.json")
    regen = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(HERE.glob("regen/**/regen-*.json"))]
    items = [r for g in regen for r in g["results"]]
    want = len(rp["sample"]) + len(rp["loop"])
    worst = max((r["max_abs_diff"] for r in items), default=math.inf)
    out["regeneration"] = {"parts": len(regen), "items": len(items), "expected": want,
                           "frozen_items": sum(r["kind"] == "frozen" for r in items),
                           "loop_items": sum(r["kind"] == "loop" for r in items), "max_abs_diff": worst,
                           "tolerance": REGEN_TOL}
    out["checks"]["regeneration"] = len(items) == want and worst <= REGEN_TOL
    out["integrity"] = out["integrity"][:100]
    out["stat_mismatches"] = out["stat_mismatches"][:100]
    out["pass"] = all(out["checks"].values())
    (HERE / "verification.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"checks": out["checks"], "pass": out["pass"]}, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
