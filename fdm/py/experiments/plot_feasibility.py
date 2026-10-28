#!/usr/bin/env python
r"""
Fraction of trial points that are feasible, per solver.

For the model-based variants a trial point is the solution of the step
subproblem, and it is feasible when every constraint is satisfied there.

    python3 plot_feasibility.py results_figures.pkl --outdir figs
    python3 plot_feasibility.py results_figures.pkl --solvers TV-AS-lazy,SQP-ALL
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

# The names on the left are the ones stored in the data; the labels are what the
# figure prints.  Comment a line out to drop that solver from the figure, or
# uncomment it to bring it back.  Order here is the order of the boxes.
STYLE = {
    "TV-AS-lazy":         dict(color="C0",   label="FDM-AS\n(lazy)"),
    "TV-AS-full":         dict(color="C1",   label="FDM-AS\n(all models)"),
    "TV-ALL":             dict(color="C2",   label="FDM\n(all constraints)"),
    "SQP-ALL":            dict(color="C3",   label="SQP\n(all constraints)"),
    "SQP-AS-lazy":        dict(color="C4",   label="SQP-AS\n(lazy)"),
}

ap = argparse.ArgumentParser()
ap.add_argument("pickle")
ap.add_argument("--outdir", default="figs")
ap.add_argument("--name", default="fig_feasibility.pdf")
ap.add_argument("--solvers", default=None, help="comma-separated; default all present")
a = ap.parse_args()

d = pickle.load(open(a.pickle, "rb"))
results = {k: v for k, v in d["results"].items() if v}
present = [s for s in STYLE if any(s in r for r in results.values())]
solvers = [s for s in (a.solvers.split(",") if a.solvers else present) if s in present]

data = []
for s in solvers:
    vals = []
    for runs in results.values():
        r = runs.get(s)
        if not r:
            continue
        h = r.get("history") or []
        v = np.mean([bool(x["trial_feasible"]) for x in h]) if h else np.nan
        if np.isfinite(v):
            vals.append(float(v))
    data.append(np.array(vals))
    print(f"  {STYLE[s]['label'].replace(chr(10), ' '):28s} median {np.median(vals):.3f}  "
          f"mean {np.mean(vals):.3f}  below 0.5 on {np.mean(np.array(vals) < .5):.2f}  "
          f"({len(vals)} problems)")

fig, ax = plt.subplots(figsize=(6.6, 3.5))
kw = dict(patch_artist=True, widths=.58,
          flierprops=dict(marker=".", markersize=3.5, markerfacecolor="0.4",
                          markeredgecolor="none"),
          medianprops=dict(color="black", lw=1.6),
          whiskerprops=dict(color="0.35", lw=1.0),
          capprops=dict(color="0.35", lw=1.0),
          boxprops=dict(lw=.8, edgecolor="0.35"))
labels = [STYLE[s]["label"] for s in solvers]
try:
    bp = ax.boxplot(data, tick_labels=labels, **kw)
except TypeError:
    bp = ax.boxplot(data, labels=labels, **kw)
for patch, s in zip(bp["boxes"], solvers):
    patch.set_facecolor(STYLE[s]["color"])
    patch.set_alpha(.30)
ax.set_ylabel("fraction of feasible trial points")
ax.set_ylim(-0.03, 1.03)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(alpha=.25, axis="y", lw=.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.setp(ax.get_xticklabels(), fontsize=8)
fig.tight_layout()
os.makedirs(a.outdir, exist_ok=True)
out = os.path.join(a.outdir, a.name)
fig.savefig(out)
print(f"\n-> {out}")
