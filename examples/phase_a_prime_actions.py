"""GitHub Actions harness for the frozen Phase A' study (execution only; no design changes).

The scientific code and design are the frozen `phase_a_prime_small_sample.py` and its
manifest. This harness only splits the frozen datasets into chunks of replications,
runs them on parallel runners, reassembles each cell's p-value array in replication
order, and publishes the evidence. Per-dataset seeds are unchanged, so a reassembled
cell is identical in content to `run_cell()` on the same platform.

Commands:
  prepare   --manifest M --out D [--shards 20]   verify manifest, cross-platform checks, chunk plan
  shard     --manifest M --plan P --shard I --out D [--deadline-seconds S] [--limit K]
  aggregate --manifest M --plan P --download D --results R
  publish   --results R --tag T --target SHA
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "phase_a_prime_small_sample.py"
FROZEN_SOURCE_SHA = "8329c91ee91460460e0787551dfdd3bdf9953e6e59a697e016f4d335642c984b"
FROZEN_MANIFEST_SHA = "044bf3863d3e60d2bc8bd2899d4c83b245ed0f50f8b98da6913f452174d8cb01"
CHUNK_REPS = 250
PROCS = 4


def frozen():
    data = FROZEN.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != FROZEN_SOURCE_SHA:
        raise SystemExit("frozen Phase A' source changed; refusing")
    # Import under its real name so worker processes can unpickle its functions.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import phase_a_prime_small_sample as mod
    return mod


def load_manifest(path, mod):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if mod.manifest_sha(m) != FROZEN_MANIFEST_SHA:
        raise SystemExit("manifest is not the frozen Phase A' manifest")
    current = json.loads(json.dumps(mod.build_manifest(), default=float))
    if mod.manifest_sha(current) != FROZEN_MANIFEST_SHA:
        raise SystemExit("this runtime does not regenerate the frozen manifest (versions/source differ)")
    return m


def make_plan(cells, shards):
    chunks = []
    for c in cells:
        for start in range(0, c["reps"], CHUNK_REPS):
            count = min(CHUNK_REPS, c["reps"] - start)
            chunks.append({"cell_index": c["cell_index"], "start": start, "count": count,
                           "cost": count * c["n"] * (c["p"] + 1)})
    load = [0] * shards
    assign = [[] for _ in range(shards)]
    for ch in sorted(chunks, key=lambda x: -x["cost"]):       # longest-processing-time first
        i = min(range(shards), key=load.__getitem__)
        assign[i].append(ch)
        load[i] += ch["cost"]
    return {"chunk_reps": CHUNK_REPS, "shards": shards, "chunks": len(chunks),
            "relative_load": [round(x / max(load), 3) for x in load], "assignment": assign}


def cmd_prepare(a):
    mod = frozen()
    m = load_manifest(a.manifest, mod)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = {"estimators": mod.check_estimators(), "bootstrap_brute_force": mod.check_bootstrap(PROCS),
              "generation_identity_with_phase_a": mod.check_generation_identity(),
              "run_guard": mod.check_run_guard(), "smoke_run": mod.check_smoke(PROCS)}
    report = {"manifest_sha256": FROZEN_MANIFEST_SHA, "source_sha256": FROZEN_SOURCE_SHA,
              "checks": checks, "all_pass": all(v["pass"] for v in checks.values()),
              "note": "seed audit and 10^6-row population draws ran locally before freezing "
                      "(examples/results/2026-10-17-phase-a-prime-small-sample-check/)"}
    (out / "prepare.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    if not report["all_pass"]:
        raise SystemExit("cross-platform checks failed")
    plan = make_plan(m["cells"], a.shards)
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    shutil.copy(a.manifest, out / "manifest.json")
    print(json.dumps({"shard": list(range(a.shards))}))
    return 0


def _run_chunk(task):
    cell, start, count, out_dir = task
    mod = frozen()
    P = np.empty((count, len(mod.METHODS), cell["p"]))
    for i in range(count):
        X, y, W = mod.generate(cell, cell["seed_first"] + start + i)
        P[i], _ = mod.tests(X, y, W)
    path = Path(out_dir) / f"chunk_{cell['cell_index']:03d}_{start:05d}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, p=P, seed_first=cell["seed_first"] + start, reps=count,
                        cell_index=cell["cell_index"], start=start)
    tmp.replace(path)
    return path.name


def cmd_shard(a):
    mod = frozen()
    m = load_manifest(a.manifest, mod)
    cells = {c["cell_index"]: c for c in m["cells"]}
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    mine = plan["assignment"][a.shard][: a.limit] if a.limit else plan["assignment"][a.shard]
    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done, skipped = [], []
    with cf.ProcessPoolExecutor(max_workers=PROCS) as pool:
        pending = {}
        queue = list(mine)
        while queue or pending:
            while queue and len(pending) < PROCS * 2:
                if time.time() - t0 > a.deadline_seconds:
                    skipped.extend(queue)
                    queue = []
                    break
                ch = queue.pop(0)
                f = pool.submit(_run_chunk, (cells[ch["cell_index"]], ch["start"], ch["count"], out / "chunks"))
                pending[f] = ch
            if not pending:
                break
            finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                ch = pending.pop(f)
                done.append({**ch, "file": f.result()})
    control = {"shard": a.shard, "assigned": len(mine), "completed": len(done), "not_started": skipped,
               "seconds": round(time.time() - t0, 1), "complete": not skipped and len(done) == len(mine)}
    (out / "control.json").write_text(json.dumps(control, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in control.items() if k != "not_started"}))
    return 0


def cmd_aggregate(a):
    mod = frozen()
    m = load_manifest(a.manifest, mod)
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    results = Path(a.results)
    (results / "pvalues").mkdir(parents=True, exist_ok=False)
    found = {}
    for p in Path(a.download).rglob("chunk_*.npz"):
        z = np.load(p)
        found[(int(z["cell_index"]), int(z["start"]))] = z
    controls = [json.loads(p.read_text(encoding="utf-8")) for p in Path(a.download).rglob("control.json")]
    expected = [ch for shard in plan["assignment"] for ch in shard]
    missing, problems = [], []
    by_cell = {}
    for ch in expected:
        key = (ch["cell_index"], ch["start"])
        if key not in found:
            missing.append(key)
            continue
        z = found[key]
        cell = m["cells"][ch["cell_index"]]
        if int(z["reps"]) != ch["count"] or int(z["seed_first"]) != cell["seed_first"] + ch["start"]:
            problems.append(f"chunk {key} seeds/reps mismatch")
        by_cell.setdefault(ch["cell_index"], []).append((ch["start"], z["p"]))
    complete_cells = []
    for c in m["cells"]:
        parts = sorted(by_cell.get(c["cell_index"], []), key=lambda t: t[0])
        if sum(len(p) for _, p in parts) == c["reps"] and not any(k[0] == c["cell_index"] for k in missing):
            P = np.concatenate([p for _, p in parts])
            np.savez_compressed(results / "pvalues" / f"cell_{c['cell_index']:03d}.npz",
                                p=P, seed_first=c["seed_first"], reps=c["reps"])
            complete_cells.append(c["cell_index"])
    complete = len(complete_cells) == len(m["cells"]) and not problems
    report = {"cells_planned": len(m["cells"]), "cells_complete": len(complete_cells),
              "chunks_planned": len(expected), "chunks_missing": len(missing), "problems": problems,
              "shard_controls": sorted(controls, key=lambda c: c["shard"]),
              "runner_seconds_max": max((c["seconds"] for c in controls), default=None),
              "complete": complete}
    if complete:
        rows, cs, crit = mod.summarize(m["cells"], results)
        mod.write_summary(rows, cs, crit, results)
        report["selection"] = crit["selection"]
    else:
        report["selection"] = "INCOMPLETE: unknown cells remain in denominators; no selection"
    (results / "aggregate.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "shard_controls"}, default=float))
    return 0 if complete else 1


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmd_publish(a):
    results = Path(a.results)
    stage = results.parent / (results.name + "-release")
    stage.mkdir(exist_ok=True)
    archive = stage / "phase-a-prime-pvalues.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted((results / "pvalues").glob("*.npz")):
            zf.write(p, f"pvalues/{p.name}")
    assets = [archive] + [results / n for n in ("summary.csv", "criteria.json", "cell-stats.json",
                                                "aggregate.json", "verification.json", "prepare.json",
                                                "plan.json", "manifest.json") if (results / n).exists()]
    local = {p.name: _sha(p) for p in assets}
    notes = ("Phase A' small-sample variance estimators (frozen manifest "
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
    p.add_argument("--shards", type=int, default=20)
    p = sub.add_parser("shard"); p.add_argument("--manifest", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--shard", type=int, required=True); p.add_argument("--out", required=True)
    p.add_argument("--deadline-seconds", type=int, default=45 * 60)
    p.add_argument("--limit", type=int, default=0, help="testing only: run the first K assigned chunks")
    p = sub.add_parser("aggregate"); p.add_argument("--manifest", required=True); p.add_argument("--plan", required=True)
    p.add_argument("--download", required=True); p.add_argument("--results", required=True)
    p = sub.add_parser("publish"); p.add_argument("--results", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--target", required=True)
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "shard": cmd_shard, "aggregate": cmd_aggregate, "publish": cmd_publish}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
