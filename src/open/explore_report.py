"""Tables and graph for exploratory-mode results (C3, C6).

``write_explore`` saves the result tables, settings, a text summary and the
network graph. ``plot_explore`` draws the graph:

- outcome on the right, one column per depth going left;
- solid lines for kept links, dotted lines for flagged links, widths scaled
  by held-out R² lost;
- shared (substitute) groups as a box around their members, with one line
  to the target;
- interactions as a × marker joining the two predictors, with a line to the
  target;
- predictors that are only flagged, and never kept, as grey candidate nodes.

Relations hidden by ``hide_below`` are left out of the graph but stay in the tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from open.explore import ExploreResult


def write_explore(result: ExploreResult, out_dir, *, plot: bool = True, show_flagged: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = result.nodes.assign(parents=result.nodes.parents.map(";".join))
    nodes.to_csv(out / "nodes.csv", index=False)
    if len(result.links):
        result.links.assign(members=result.links.members.map(";".join)).to_csv(out / "links.csv", index=False)
    result.pairs.to_csv(out / "pairs.csv", index=False)
    result.folds.to_csv(out / "folds.csv", index=False)
    result.bars.to_csv(out / "noise-bars.csv", index=False)
    (out / "settings.json").write_text(json.dumps(dict(result.settings, stops=result.stops), indent=2) + "\n")
    (out / "summary.txt").write_text(result.summary() + "\n", encoding="utf-8")
    if plot:
        plot_explore(result, out / "network.png", show_flagged=show_flagged)
    return out


def _r2(v):
    return f"{v:.3f}" if abs(v) >= .001 else ("<.001" if v > 0 else f"{v:.3f}")


def _width(r2):
    return float(np.clip(1 + 40 * max(r2, 0), 0.8, 6))


def _layout(result: ExploreResult, links: pd.DataFrame):
    column = {r.variable: int(r.depth) for r in result.nodes.itertuples()}
    candidate = set()
    for r in links.itertuples():
        for m in r.members:
            if m not in column:
                column[m] = int(r.depth)
                candidate.add(m)
    order = {}
    for r in links.itertuples():   # keep group and interaction members next to each other
        for m in r.members:
            order.setdefault(m, len(order))
    pos = {}
    for d in sorted(set(column.values())):
        names = sorted((v for v, c in column.items() if c == d), key=lambda v: order.get(v, -1))
        for i, v in enumerate(names):
            pos[v] = (-d * 3.0, (len(names) - 1) / 2 - i)
    return pos, candidate


def plot_explore(result: ExploreResult, path=None, *, show_flagged: bool = True):
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    links = result.links
    if len(links):
        keep = ["kept", "flagged"] if show_flagged else ["kept"]
        links = links[links.tier.isin(keep)]
    pos, candidate = _layout(result, links)
    depth_count = max([0] + [int(d) for d in result.nodes.depth] + [int(d) for d in links.depth])
    tall = max([1] + [sum(1 for p in pos.values() if p[0] == x) for x in {p[0] for p in pos.values()}])
    fig, ax = plt.subplots(figsize=(3 + 3 * depth_count, 1.5 + 0.9 * tall))

    def arrow(a, b, tier_, r2, label, color="#333333"):
        style = "-" if tier_ == "kept" else (0, (2, 2))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, linewidth=_width(r2),
                                     linestyle=style, color=color, shrinkA=18, shrinkB=18, zorder=1))
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        ax.text(mx, my + 0.12, label, fontsize=7, ha="center", color=color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    for r in links.itertuples():
        target = pos[r.target]
        label = f"{r.pass_rate:.0%} | R² {_r2(r.r2_median)}"
        if r.kind == "unique":
            arrow(pos[r.source], target, r.tier, r.r2_median, label)
        elif r.kind == "shared":
            xs = [pos[m][0] for m in r.members]
            ys = [pos[m][1] for m in r.members]
            ax.add_patch(FancyBboxPatch((min(xs) - 0.55, min(ys) - 0.3), max(xs) - min(xs) + 1.1,
                                        max(ys) - min(ys) + 0.6, boxstyle="round,pad=0.05",
                                        fill=False, ec="#2b6cb0", lw=1.2,
                                        ls="-" if r.tier == "kept" else (0, (2, 2)), zorder=0))
            start = (max(xs) + 0.55, (min(ys) + max(ys)) / 2)
            arrow(start, target, r.tier, r.r2_median, "shared " + label, color="#2b6cb0")
        else:
            (ax_, ay), (bx, by) = pos[r.members[0]], pos[r.members[1]]
            mx, my = (ax_ + bx) / 2, (ay + by) / 2
            mark = (mx + .4 * (target[0] - mx), my + .4 * (target[1] - my))
            ls = "-" if r.tier == "kept" else (0, (2, 2))
            for m in r.members:
                ax.plot([pos[m][0], mark[0]], [pos[m][1], mark[1]], color="#c05621", lw=0.8, ls=ls, zorder=1)
            ax.text(*mark, "×", fontsize=14, ha="center", va="center", color="#c05621", zorder=3)
            arrow(mark, target, r.tier, r.r2_median, label, color="#c05621")

    for v, (x, y) in pos.items():
        grey = v in candidate
        ax.text(x, y, v, ha="center", va="center", fontsize=9, zorder=4,
                color="#777777" if grey else "black",
                bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2" if grey else "white",
                          ec="#999999" if grey else "black", ls="--" if grey else "-"))
    for d in range(depth_count + 1):
        ax.text(-d * 3.0, max(p[1] for p in pos.values()) + 0.8 if pos else 1, "Outcome" if d == 0 else
                f"Depth {d}", ha="center", fontsize=9, color="#555555")
    s = result.settings
    ax.set_title(f"{s['outcome']}: {s['engine']}, N = {s['n']}. Solid: kept (≥ {s['keep']:.0%} of folds); "
                 f"dotted: flagged (≥ {s['flag']:.0%}).\nLabel: folds above the noise bar | held-out R² lost "
                 f"(gained, for × interactions). Blue box: shared group. Grey: flagged only.", fontsize=9)
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1, max(ys) + 1.3)
    ax.axis("off")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return Path(path)
    return fig
