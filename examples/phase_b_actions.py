"""GitHub Actions harness for the frozen Phase B study (execution only; no design changes).

The scientific code and design are the frozen `phase_b_local_detector.py` and its
manifest. This harness splits each cell's replications into chunks, runs them on
parallel runners with the frozen `run_chunk()`, and reassembles them. Per-dataset seeds
are unchanged, so a reassembled cell is identical to a single local run on the same
platform. Chunk sizes and the shard plan are execution details, not design.

Commands:
  prepare   --manifest M --out D [--shards 120] [--carry DIR]
  shard     --manifest M --plan P --shard I --out D [--deadline-seconds S] [--limit K]
  aggregate --manifest M --plan P --download D --results R
  regen     --manifest M --results R --part I --parts K --out D
  publish   --results R --tag T --target SHA

`--carry DIR` (continuation): previously completed chunk files found under DIR are
verified and excluded from the new plan; aggregate then merges them with new output.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "phase_b_local_detector.py"
FROZEN_SOURCE_SHA = "7f2da249796593144ba51d17b99cfc860e393b79060e726c81fb8115e1165af6"
FROZEN_MANIFEST_SHA = "73aef5e6ca31a963a7a3d9af2cdd869e15448c0f0074c85e97472fa08b04febd"
PROCS = 4
# Seconds per dataset measured on windows-2022 with 4 concurrent workers by the passing
# check (Actions run 39113988219, part "runtime"). Used only to size chunks and balance shards.
MEASURED = {("main", 1000): 70.37, ("main", 2000): 123.77, ("main", 5000): 237.77,
            ("small", 300): 105.39, ("small", 500): 97.0}
SCREEN_SECONDS = 0.1
CHUNK_REPS = {("main", 1000): 15, ("main", 2000): 10, ("main", 5000): 5, ("small", 300): 10,
              ("small", 500): 10}
SCREEN_CHUNK = 250
REGEN_SHARE = 0.01
REGEN_SELECT_SEED = 20261018
LOOP_DATASETS = 20
_CHUNK = re.compile(r"cell_(\d{3})_r(\d{5})\.npz$")


def frozen():
    data = FROZEN.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != FROZEN_SOURCE_SHA:
        raise SystemExit("frozen Phase B source changed; refusing")
    # Import under its real name so worker processes can unpickle its functions.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import phase_b_local_detector as mod
    return mod


def load_manifest(path, mod):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if mod.manifest_sha(m) != FROZEN_MANIFEST_SHA:
        raise SystemExit("manifest is not the frozen Phase B manifest")
    # Runs read beta, sigma2 and roles from the frozen file. Regenerating only confirms this
    # runtime reproduces the design: non-floats exactly, floats to 1e-12 relative.
    current = json.loads(json.dumps(mod.build_manifest(), default=float))
    diffs = _compare(m, current, "")
    if diffs:
        print(json.dumps({"manifest_regeneration_differences": diffs[:50]}, indent=1), file=sys.stderr)
        raise SystemExit("this runtime does not reproduce the frozen design (see differences above)")
    return m


def _compare(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}.{k}: missing" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += _compare(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _compare(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and not isinstance(a, bool):
        return [] if abs(a - b) <= 1e-12 * max(1.0, abs(a)) else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _chunk_reps(c):
    return SCREEN_CHUNK if c["family"] == "screen" else CHUNK_REPS[(c["family"], c["n"])]


def _seconds(c):
    return SCREEN_SECONDS if c["family"] == "screen" else MEASURED[(c["family"], c["n"])]


def all_chunks(cells):
    out = []
    for c in cells:
        size = _chunk_reps(c)
        for lo in range(0, c["reps"], size):
            hi = min(lo + size, c["reps"])
            out.append({"cell_index": c["cell_index"], "rep_lo": lo, "rep_hi": hi,
                        "cost": round((hi - lo) * _seconds(c), 2)})
    return out


def verify_chunk(path, cells):
    """Return (cell_index, rep_lo, rep_hi) for a valid chunk file, else None."""
    mt = _CHUNK.search(Path(path).name)
    if not mt:
        return None
    try:
        z = np.load(path)
        ci, lo, hi = int(mt.group(1)), int(z["rep_lo"]), int(z["rep_hi"])
        c = cells[ci]
        P = z["p"]
        ok = (lo == int(mt.group(2)) and 0 <= lo < hi <= c["reps"]
              and int(z["seed_first"]) == c["seed_first"]
              and P.shape == (hi - lo, len(c["tests"]), c["p"])
              and z["rounds"].shape == (hi - lo, 4, 3)
              and bool(np.all(np.isfinite(P))) and bool(np.all((P >= 0) & (P <= 1))))
        return (ci, lo, hi) if ok else None
    except Exception:
        return None


def collect(root, cells):
    """Valid chunk files under root, keyed by (cell, rep_lo, rep_hi); first copy wins."""
    found, bad = {}, []
    for p in sorted(Path(root).rglob("cell_*_r*.npz")):
        key = verify_chunk(p, cells)
        if key is None:
            bad.append(p.name)
        else:
            found.setdefault(key, p)
    return found, bad


def make_plan(chunks, shards):
    load = [0.0] * shards
    assign = [[] for _ in range(shards)]
    for ch in sorted(chunks, key=lambda x: (-x["cost"], x["cell_index"], x["rep_lo"])):  # LPT
        i = min(range(shards), key=load.__getitem__)
        assign[i].append(ch)
        load[i] += ch["cost"]
    return assign, load


def cmd_prepare(a):
    mod = frozen()
    m = load_manifest(a.manifest, mod)
    cells = m["cells"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {"hc4_identity_with_phase_a_prime": mod.check_hc4_identity(),
              "gcm_identity_with_phase_a": mod.check_gcm_identity(),
              "determinism": mod.check_determinism(), "run_guard": mod.check_run_guard()}
    chunks = all_chunks(cells)
    carried = {}
    bad = []
    if a.carry and Path(a.carry).exists():
        found, bad = collect(a.carry, cells)
        planned = {(c["cell_index"], c["rep_lo"], c["rep_hi"]) for c in chunks}
        carried = {k: p for k, p in found.items() if k in planned}
        (out / "carried").mkdir(exist_ok=True)
        for k, p in carried.items():
            shutil.copy2(p, out / "carried" / p.name)
    todo = [c for c in chunks if (c["cell_index"], c["rep_lo"], c["rep_hi"]) not in carried]
    shards = max(1, min(a.shards, len(todo))) if todo else 0
    assign, load = make_plan(todo, shards) if todo else ([], [])
    plan = {"chunks_total": len(chunks), "chunks_carried": len(carried), "chunks_to_run": len(todo),
            "carried_invalid_files": bad, "shards": shards, "procs_per_shard": PROCS,
            "measured_seconds_per_dataset": {f"{k[0]}_{k[1]}": v for k, v in MEASURED.items()},
            "chunk_reps": {f"{k[0]}_{k[1]}": v for k, v in CHUNK_REPS.items()} | {"screen": SCREEN_CHUNK},
            "measured_thread_seconds_to_run": round(sum(c["cost"] for c in todo)),
            "max_shard_measured_minutes_at_4_procs": round(max(load, default=0) / PROCS / 60, 1),
            "all_chunks": chunks, "assignment": assign}
    report = {"manifest_sha256": FROZEN_MANIFEST_SHA, "source_sha256": FROZEN_SOURCE_SHA, "checks": checks,
              "all_pass": all(v["pass"] for v in checks.values()),
              "note": "full --check (all parts) passed on this runner image for this exact source: "
                      "Actions run 39113988219, examples/results/2026-10-18-phase-b-local-detector-check/"}
    (out / "prepare.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    shutil.copy(a.manifest, out / "manifest.json")
    if not report["all_pass"]:
        raise SystemExit("cross-platform checks failed")
    print(json.dumps({"shard": list(range(shards)) or [-1]}))
    return 0


def _run(task):
    cell, out_dir, lo, hi = task
    mod = frozen()
    _, _, secs = mod.run_chunk(cell, out_dir, lo, hi)
    return secs


def cmd_shard(a):
    mod = frozen()
    m = load_manifest(a.manifest, mod)
    cells = m["cells"]
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    out = Path(a.out)
    (out / "pvalues").mkdir(parents=True, exist_ok=True)
    if a.shard < 0:
        (out / "control.json").write_text(json.dumps({"shard": a.shard, "assigned": 0, "complete": True}),
                                          encoding="utf-8")
        return 0
    mine = plan["assignment"][a.shard]
    mine = mine[: a.limit] if a.limit else mine
    t0 = time.time()
    done, skipped = [], []
    with cf.ProcessPoolExecutor(max_workers=PROCS) as pool:
        pending, queue = {}, list(mine)
        while queue or pending:
            while queue and len(pending) < PROCS:
                if time.time() - t0 > a.deadline_seconds:
                    skipped.extend(queue)
                    queue = []
                    break
                ch = queue.pop(0)
                f = pool.submit(_run, (cells[ch["cell_index"]], out, ch["rep_lo"], ch["rep_hi"]))
                pending[f] = ch
            if not pending:
                break
            finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                ch = pending.pop(f)
                done.append({**ch, "seconds": round(f.result(), 1)})
    control = {"shard": a.shard, "assigned": len(mine), "completed": len(done), "not_started": skipped,
               "seconds": round(time.time() - t0, 1),
               "measured_vs_planned": round(sum(d["seconds"] for d in done) / max(sum(d["cost"] for d in done), 1e-9), 3),
               "complete": not skipped and len(done) == len(mine), "done": done}
    (out / "control.json").write_text(json.dumps(control, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in control.items() if k not in ("not_started", "done")}))
    return 0


def regen_sample(cells):
    rng = np.random.default_rng(REGEN_SELECT_SEED)
    out = []
    for c in cells:
        k = max(1, math.ceil(REGEN_SHARE * c["reps"]))
        out += [{"cell_index": c["cell_index"], "rep": int(r)} for r in sorted(rng.choice(c["reps"], k, replace=False))]
    loops = [c for c in cells if c["family"] == "main" and c["n"] == 1000 and c["r"] > 0]
    loop = [{"cell_index": loops[i % len(loops)]["cell_index"], "rep": int(rng.integers(0, 400))}
            for i in range(LOOP_DATASETS)]
    return out, loop


def cmd_aggregate(a):
    mod = frozen()
    m = load_manifest(a.manifest, mod)
    cells = m["cells"]
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    results = Path(a.results)
    (results / "pvalues").mkdir(parents=True, exist_ok=False)
    found, bad = collect(a.download, cells)
    expected = [(c["cell_index"], c["rep_lo"], c["rep_hi"]) for c in plan["all_chunks"]]
    missing = [k for k in expected if k not in found]
    for k in expected:
        if k in found:
            shutil.copy2(found[k], results / "pvalues" / found[k].name)
    controls = [json.loads(p.read_text(encoding="utf-8")) for p in Path(a.download).rglob("control.json")]
    done_cells = {ci for ci, _, _ in expected} - {ci for ci, _, _ in missing}
    report = {"cells_planned": len(cells), "cells_complete": len(done_cells), "chunks_planned": len(expected),
              "chunks_missing": len(missing), "missing": missing[:200], "invalid_files": bad,
              "shard_controls": sorted(({k: v for k, v in c.items() if k != "done"} for c in controls),
                                       key=lambda c: c["shard"]),
              "runner_seconds_max": max((c.get("seconds", 0) for c in controls), default=None),
              "complete": not missing and len(done_cells) == len(cells)}
    if report["complete"]:
        rows, cs, crit = mod.summarize(cells, results)
        mod.write_summary(rows, cs, crit, results)
        report["supported_n_table"] = crit["supported_n_table"]
        report["correction_screen_chosen"] = crit["correction_screen"]["chosen"]
        report["reference_defects"] = crit["reference"]["defects"]
        sample, loop = regen_sample(cells)
        (results / "regen-plan.json").write_text(json.dumps({"share": REGEN_SHARE, "select_seed": REGEN_SELECT_SEED,
                                                             "sample": sample, "loop": loop}, indent=1),
                                                 encoding="utf-8")
    else:
        report["supported_n_table"] = ("INCOMPLETE: dispatch again with continue_from = this run id; "
                                       "no criteria are computed on partial data")
    (results / "aggregate.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("shard_controls", "missing")}, default=float))
    return 0 if report["complete"] else 1


def _regen_task(task):
    kind, cell, rep, stored = task
    mod = frozen()
    seed = cell["seed_first"] + rep
    X, T, folds = mod.generate(cell, seed)
    if kind == "frozen":
        P, _ = mod.analyze(cell, X, T, folds, seed)
        return {"cell_index": cell["cell_index"], "rep": rep, "kind": kind,
                "max_abs_diff": float(np.max(np.abs(P - stored)))}
    worst = 0.0
    tests = cell["tests"]
    for e in mod.ENGINES:
        g = mod.loop_gcm(e, X, T, folds["k5"], 5, seed, "normal")
        p = mod.loop_pcm(e, X, T, folds["half"], folds["k5"], 5, seed, "normal")
        worst = max(worst, float(np.max(np.abs(g - stored[tests.index(f"{e}_gcm_g0")]))),
                    float(np.max(np.abs(p - stored[tests.index(f"{e}_pcm_g0")]))))
    return {"cell_index": cell["cell_index"], "rep": rep, "kind": kind, "max_abs_diff": worst}


def cmd_regen(a):
    mod = frozen()
    m = load_manifest(a.manifest, mod)
    cells = m["cells"]
    rp = json.loads((Path(a.results) / "regen-plan.json").read_text(encoding="utf-8"))
    items = [("frozen", x) for x in rp["sample"]] + [("loop", x) for x in rp["loop"]]
    mine = items[a.part::a.parts]
    cache = {}
    tasks = []
    for kind, x in mine:
        c = cells[x["cell_index"]]
        if c["cell_index"] not in cache:
            cache[c["cell_index"]] = mod.load_cell(c, a.results)[0]
        tasks.append((kind, c, x["rep"], cache[c["cell_index"]][x["rep"]]))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=PROCS) as pool:
        res = list(pool.map(_regen_task, tasks))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"part": a.part, "parts": a.parts, "items": len(res), "seconds": round(time.time() - t0, 1),
           "max_abs_diff": max((r["max_abs_diff"] for r in res), default=0.0), "results": res}
    (out / f"regen-{a.part}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "results"}))
    return 0


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_publish(a):
    results = Path(a.results)
    stage = results.parent / (results.name + "-release")
    stage.mkdir(exist_ok=True)
    archive = stage / "phase-b-pvalues.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted((results / "pvalues").glob("*.npz")):
            zf.write(p, f"pvalues/{p.name}")
    assets = [archive] + [results / n for n in ("summary.csv", "criteria.json", "cell-stats.json",
                                                "supported-n.json", "aggregate.json", "verification.json",
                                                "prepare.json", "plan.json", "manifest.json", "regen-plan.json")
                          if (results / n).exists()]
    local = {p.name: _sha(p) for p in assets}
    notes = ("Phase B local detector per engine (frozen manifest "
             f"{FROZEN_MANIFEST_SHA[:12]}). Raw per-dataset p-values and reports. "
             "Interpretation is committed separately after review.")
    subprocess.run(["gh", "release", "create", a.tag, "--prerelease", "--target", a.target,
                    "--title", a.tag, "--notes", notes, *map(str, assets)], check=True)
    check = stage / "download"
    subprocess.run(["gh", "release", "download", a.tag, "--dir", str(check)], check=True)
    remote = {p.name: _sha(p) for p in check.iterdir()}
    ok = remote == local
    (results / "publication-verification.json").write_text(json.dumps(
        {"tag": a.tag, "target": a.target, "local_sha256": local, "downloaded_sha256": remote,
         "verified": ok}, indent=1), encoding="utf-8")
    if not ok:
        raise SystemExit("downloaded release assets do not match")
    print(json.dumps({"tag": a.tag, "verified": ok, "assets": len(local)}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--shards", type=int, default=125); p.add_argument("--carry")
    p = sub.add_parser("shard"); p.add_argument("--manifest", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--shard", type=int, required=True); p.add_argument("--out", required=True)
    p.add_argument("--deadline-seconds", type=int, default=280 * 60)
    p.add_argument("--limit", type=int, default=0, help="testing only: run the first K assigned chunks")
    p = sub.add_parser("aggregate"); p.add_argument("--manifest", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--download", required=True); p.add_argument("--results", required=True)
    p = sub.add_parser("regen"); p.add_argument("--manifest", required=True); p.add_argument("--results", required=True)
    p.add_argument("--part", type=int, required=True); p.add_argument("--parts", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("publish"); p.add_argument("--results", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--target", required=True)
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "shard": cmd_shard, "aggregate": cmd_aggregate, "regen": cmd_regen,
            "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
