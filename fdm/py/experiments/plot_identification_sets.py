#!/usr/bin/env python
"""
Claim 3: A_k = A(xbar).

One panel per problem.  Instead of plotting the SIZE of the estimate (equal
cardinality does not imply equal sets), each panel shows the two set
differences:

    above the axis   |A_k \\ A(xbar)|   constraints the estimate carries that are
                                       not active at xbar   (surplus)
    below the axis  -|A(xbar) \\ A_k|   constraints active at xbar that the
                                       estimate omits        (deficit)

A_k = A(xbar) exactly when both are zero, so identification appears as the band
collapsing onto the axis and staying there.

    python3 plot_identification_sets.py results_c1_10.pkl --outdir figs_c1_10
    python3 plot_identification_sets.py results_c1_10.pkl --examples HATFLDH,HS93,GIGOMEZ2,AVGASA
    python3 plot_identification_sets.py results_c1_10.pkl --list
"""
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    "xtick.direction": "out", "ytick.direction": "out", "figure.dpi": 150,
})


def reached_xstar(run, ref, tol):
    x, xs = np.asarray(run["x"], float), np.asarray(ref["x"], float)
    return np.linalg.norm(x - xs, np.inf) / (1.0 + np.linalg.norm(xs, np.inf)) <= tol


def series(history, A_star):
    A_star = set(A_star)
    extra = np.array([len(set(r["A_initial"]) - A_star) for r in history])
    miss = np.array([len(A_star - set(r["A_initial"])) for r in history])
    acc = np.array([bool(r["accepted"]) for r in history])
    return extra, miss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--solver", default="TV-AS-lazy")
    ap.add_argument("--examples", default=None, help="comma-separated problem names")
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--dist-tol", type=float, default=1e-2)
    ap.add_argument("--require-sc", action="store_true", default=True)
    ap.add_argument("--list", action="store_true", help="print candidates and exit")
    ap.add_argument("--name", default="fig_identification_sets.pdf")
    ap.add_argument("--no-legend", action="store_true",
                    help="omit the legend; the caption can carry it instead")
    ap.add_argument("--layout", default="auto", choices=["auto", "row", "grid"],
                    help="auto: a 2x2 grid for four panels, one row otherwise")
    ap.add_argument("--draw", default="bars", choices=["bars", "steps", "dots"],
                    help="bars: filled counts; steps: thin outlines; "
                         "dots: only the tip of each count")
    a = ap.parse_args()

    d = pickle.load(open(a.pickle, "rb"))
    results, refs = d["results"], d.get("references", {})
    pinfo = {e["name"]: e for e in d["meta"]["problems"]}

    cand = {}
    for name, runs in results.items():
        r, ref = runs.get(a.solver), refs.get(name)
        if not r or not r.get("history") or not ref:
            continue
        if not reached_xstar(r, ref, a.dist_tol):
            continue
        if a.require_sc and not ref.get("strict_compl"):
            continue
        extra, miss, acc = series(r["history"], ref["A"])
        n = len(extra)
        tail = slice(int(.75 * n), n)
        cand[name] = dict(extra=extra, miss=miss, acc=acc, p=pinfo[name]["p"],
                          n=pinfo[name]["n"], astar=len(ref["A"]), iters=n,
                          tail_extra=extra[tail].mean(), tail_miss=miss[tail].mean(),
                          tail_exact=float(np.mean((extra[tail] == 0) & (miss[tail] == 0))),
                          start_mismatch=float((extra[:max(1, n // 10)] + miss[:max(1, n // 10)]).mean()))
    if not cand:
        raise SystemExit("no problems pass the filter")

    exact = {k: v for k, v in cand.items() if v["tail_exact"] >= 0.9}
    print(f"{len(cand)} problems pass the filter; {len(exact)} have A_k = A(xbar) "
          f"throughout the last quarter of the run")

    if a.list:
        print(f"\n{'problem':12s} {'n':>3s} {'p':>4s} {'|A*|':>5s} {'iters':>6s} "
              f"{'tail_exact':>11s} {'tail_extra':>11s} {'tail_miss':>10s} {'start_mism':>11s}")
        for k, v in sorted(cand.items(), key=lambda t: (-t[1]["tail_exact"], -t[1]["start_mismatch"])):
            print(f"{k:12s} {v['n']:3d} {v['p']:4d} {v['astar']:5d} {v['iters']:6d} "
                  f"{v['tail_exact']:11.2f} {v['tail_extra']:11.2f} {v['tail_miss']:10.2f} "
                  f"{v['start_mismatch']:11.2f}")
        return

    if a.examples:
        names = [x for x in a.examples.split(",") if x in cand]
        missing = [x for x in a.examples.split(",") if x not in cand]
        if missing:
            print(f"  (not available / filtered out: {', '.join(missing)})")
    else:
        # illustrative: identifies in the tail, and started away from it
        ranked = sorted(exact.items(), key=lambda t: -(t[1]["start_mismatch"] + .1 * t[1]["p"]))
        names = [k for k, _ in ranked[:a.n_examples]]
    if not names:
        raise SystemExit("no example problems selected")

    grid = a.layout == "grid" or (a.layout == "auto" and len(names) == 4)
    if grid:
        nr, nc = 2, 2
        fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.5 * nr), squeeze=False,
                                 sharex=False)
    else:
        nr, nc = 1, len(names)
        fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.4), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(names):]:
        ax.axis("off")
    h_surplus = h_omit = h_eq = None
    for ax, name in zip(flat, names):
        v = cand[name]
        k = np.arange(v["iters"])
        if a.draw == "bars":
            h_surplus = ax.bar(k, v["extra"], width=1.0, color="C0", alpha=.85, linewidth=0)
            h_omit = ax.bar(k, -v["miss"], width=1.0, color="C3", alpha=.85, linewidth=0)
        elif a.draw == "dots":
            # only the tip of each count, so a long run of identical values reads
            # as a line of markers rather than a block of colour
            h_surplus, = ax.plot(k, v["extra"], ".", ms=2.6, color="C0", mec="none")
            h_omit, = ax.plot(k, -v["miss"], ".", ms=2.6, color="C3", mec="none")
        else:
            h_surplus, = ax.step(k, v["extra"], where="mid", color="C0", lw=1.25)
            h_omit, = ax.step(k, -v["miss"], where="mid", color="C3", lw=1.25)
        ax.axhline(0, color="k", lw=1)

        both0 = (v["extra"] == 0) & (v["miss"] == 0)
        lo = min(-v["miss"].max(), 0)
        hi = max(v["extra"].max(), 0)
        span = max(hi - lo, 1)
        if both0.any():                              # markers ON the zero line
            h_eq = ax.plot(k[both0], np.zeros(both0.sum()), ".", ms=3.2, color="C3",
                           mec="none", zorder=5)[0]
            stable = np.flatnonzero(np.cumsum(~both0[::-1])[::-1] == 0)
            if len(stable):
                ax.axvline(stable[0], color="grey", ls="--", lw=.9)
        ax.set_ylim(lo - 0.10 * span, hi + 0.12 * span)
        ax.set_title(f"{name} ($n$={v['n']}, $p$={v['p']}, "
                     r"$|\mathcal{A}(\bar{x})|$=" f"{v['astar']})", fontsize=9)
        ax.set_xlabel("iteration $k$")
        ax.grid(alpha=.25, axis="y", lw=.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for r in range(nr):
        axes[r][0].set_ylabel("constraints in one set\nbut not the other")

    if not a.no_legend:
        handles = [h for h in (h_surplus, h_omit) if h is not None]
        labels = [r"$|\mathcal{A}_k\setminus\mathcal{A}(\bar{x})|$  (surplus)",
                  r"$|\mathcal{A}(\bar{x})\setminus\mathcal{A}_k|$  (omitted)"][:len(handles)]
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))

    bottom = 0.0 if a.no_legend else (0.10 if grid else 0.08)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, a.name)
    fig.savefig(out, bbox_inches="tight")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
