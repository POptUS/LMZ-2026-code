#!/usr/bin/env python
r"""
Data profiles in the oracle-query accounting.

Implements

    d_s(alpha) = |{ j : t_{j,s} / ((p_j+1)(n_j+1)) <= alpha }| / |P|,

where t_{j,s} is the cumulative number of scalar oracle queries the run of
solver s on problem j had issued when it first held a FEASIBLE point x with

    f(x) <= f_L^j + eps_tau ( f(x_0^j) - f_L^j ).

A point earns credit only once all of f, c_1, ..., c_p have been queried there
and c(x) <= 0; the queries are charged either way, so a point abandoned as
infeasible costs whatever was spent discovering that.  One unit on the axis is
one evaluation of the whole vector (f, c_1, ..., c_p) per simplex gradient, so
the budget of 100(n+1)(p+1) queries sits at alpha = 100.

    python3 plot_data_profiles.py results_q100.pkl --outdir figs_q100
    python3 plot_data_profiles.py results_q100.pkl --solvers TV-AS-lazy,TV-AS-full,TV-ALL
    python3 plot_data_profiles.py results_q100.pkl --eps-taus 1e-1,1e-3,1e-5,1e-7
    python3 plot_data_profiles.py results_q100.pkl --fL-solvers TV-AS-lazy,TV-AS-full,TV-ALL,SQP-ALL,SQP-AS-lazy
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

# Comment a line out to drop that solver from the figure, or uncomment it to
# bring it back.  Order here is the order of the curves and of the legend.
# f_L is computed over this same selection (see f0_and_fL below), so toggling
# a line here moves both the picture and the difficulty targets together.
STYLE = {
    "TV-AS-lazy":  dict(color="C0", ls="-",  lw=1.4, label="FDM-AS (lazy)"),
    "TV-AS-full":  dict(color="C1", ls="--", lw=1.1, label="FDM-AS (all models)"),
    "TV-ALL":      dict(color="C2", ls="-.", lw=1.1, label="FDM (all constraints)"),
    "SQP-ALL":     dict(color="C3", ls=":",  lw=1.1, label="SQP (all constraints)"),
    "SQP-AS-lazy": dict(color="C4", ls=(0, (3, 1, 1, 1)), lw=1.0, label="SQP-AS (lazy)"),
    # ---- the two COBYLA variants: uncomment to include them ------------------
    # "COBYLA":      dict(color="0.35", ls="-", lw=1.1, label="COBYLA"),
    # "COBYLA-unrelaxable": dict(color="0.35", ls=(0, (4, 2)), lw=1.1,
    #                            label="COBYLA (NaN at infeasible $f$)"),
    # ---- CONORBIT (ORBITc.m via MATLAB): uncomment to include it -------------
    "CONORBIT":    dict(color="0.6", ls=(0, (1, 1)), lw=1.1, label="CONORBIT"),
}

ap = argparse.ArgumentParser()
ap.add_argument("pickle")
ap.add_argument("--outdir", default="figs")
ap.add_argument("--name", default="fig_data_profiles.pdf")
ap.add_argument("--solvers", default=None, help="comma-separated; default all present")
ap.add_argument("--eps-taus", "--taus", dest="eps_taus", default="1e-1,1e-3,1e-5,1e-7",
                help="comma-separated accuracy levels eps_tau")
ap.add_argument("--alpha-max", type=float, default=100.0)
ap.add_argument("--fL-solvers", default=None,
                help="comma-separated solvers defining f_L; default: same as "
                     "--solvers (everything drawn). Pass this to make the target "
                     "stricter/looser than, or independent of, what's plotted")
a = ap.parse_args()

d = pickle.load(open(a.pickle, "rb"))
results = {k: v for k, v in d["results"].items() if v}
pinfo = {e["name"]: e for e in d["meta"]["problems"]}
present = [s for s in STYLE if any(s in r for r in results.values())]
solvers = [s for s in (a.solvers.split(",") if a.solvers else present) if s in present]
fL_solvers = a.fL_solvers.split(",") if a.fL_solvers else solvers
eps_taus = [float(t) for t in a.eps_taus.split(",")]
print(f"{len(results)} problems, solvers: {', '.join(solvers)}")
print(f"f_L defined over: {', '.join(fL_solvers)}")

def f0_and_fL(name, runs):
    """f(x_0) and f_L: the first credited value (identical for all solvers, since
    x_0 is common) and the best credited value over `fL_solvers` -- which is the
    drawn solver set by default, but can be set independently via --fL-solvers."""
    scope = [s for s in fL_solvers if s in runs]
    f0, fL = np.inf, np.inf
    for s in scope:
        tr = runs[s].get("trace") or []
        vals = [rec[2] for rec in tr if np.isfinite(rec[2])]
        if not vals:
            continue
        f0 = min(f0, vals[0])      # the starting point is feasible and common
        fL = min(fL, vals[-1])
    return f0, fL

def t_js(run, thresh):
    """Cumulative oracle queries when the run first held a credited point at or
    below thresh; inf if never."""
    for n_points, n_queries, best in run.get("trace") or []:
        if np.isfinite(best) and best <= thresh:
            return n_queries
    return np.inf

ncol = 2 if len(eps_taus) == 4 else len(eps_taus)
nrow = int(np.ceil(len(eps_taus) / ncol))
fig, axg = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow),
                        sharex=True, sharey=True)
axes = np.atleast_1d(axg).ravel()
for ax in axes[len(eps_taus):]:
    ax.axis("off")
grid = np.linspace(0, a.alpha_max, 400)

# f0_and_fL(name, runs) doesn't depend on eps_tau or the solver being plotted,
# so compute it once per problem instead of once per (eps_tau, solver, problem).
f0_fL_by_name = {name: f0_and_fL(name, runs) for name, runs in results.items()}

for j, (ax, eps_tau) in enumerate(zip(axes, eps_taus)):
    for s in solvers:
        ratio = []
        for name, runs in results.items():
            if s not in runs:
                continue
            f0, fL = f0_fL_by_name[name]
            if not np.isfinite(f0) or not np.isfinite(fL):
                continue
            t = t_js(runs[s], fL + eps_tau * (f0 - fL))
            unit = (pinfo[name]["p"] + 1) * (pinfo[name]["n"] + 1)
            ratio.append(t / unit)
        ratio = np.array(ratio, dtype=float)
        if not len(ratio):
            continue
        ax.plot(grid, [(ratio <= g).mean() for g in grid], **STYLE[s])
    ax.set_title(rf"$\varepsilon_\tau = 10^{{{int(round(np.log10(eps_tau)))}}}$", pad=4)
    ax.set_xlim(0, a.alpha_max)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if j >= len(eps_taus) - ncol:
        ax.set_xlabel(r"$\alpha$")

fig.supylabel("fraction of problems solved", fontsize=9)
fig.supxlabel(r"$\alpha$: evaluations of $(f,c_1,\dots,c_p)$ per simplex gradient",
              fontsize=9, y=-0.01)
for ax in axes[:len(eps_taus)]:
    ax.set_xlabel("")

handles = [plt.Line2D([], [], **{k: v for k, v in STYLE[s].items() if k != "label"})
           for s in solvers]
# matplotlib fills legend columns top to bottom, so the entries have to be
# permuted for the rows to read left to right in the order of `solvers`
ncol_leg = 4 if len(solvers) > 6 else 3
labels = [STYLE[s]["label"] for s in solvers]
nrow_leg = int(np.ceil(len(labels) / ncol_leg))
order = [r * ncol_leg + c for c in range(ncol_leg) for r in range(nrow_leg)]
order = [i for i in order if i < len(labels)]
fig.legend([handles[i] for i in order], [labels[i] for i in order],
           loc="upper center", ncol=ncol_leg, frameon=False,
           bbox_to_anchor=(0.5, 0.988), handlelength=2.2, columnspacing=1.1,
           fontsize=7.5)
top = 0.975 - 0.06 * (nrow_leg - 1)   # extra legend rows push the subplots down
fig.tight_layout(rect=(0.02, 0.0, 1, top), pad=0.35)
os.makedirs(a.outdir, exist_ok=True)
out = os.path.join(a.outdir, a.name)
fig.savefig(out, bbox_inches="tight")
print(f"\nfraction solved at alpha = {a.alpha_max:g}")
for s in solvers:
    row = []
    for eps_tau in eps_taus:
        r = []
        for name, runs in results.items():
            if s not in runs:
                continue
            f0, fL = f0_fL_by_name[name]
            if not np.isfinite(f0) or not np.isfinite(fL):
                continue
            unit = (pinfo[name]["p"] + 1) * (pinfo[name]["n"] + 1)
            r.append(t_js(runs[s], fL + eps_tau * (f0 - fL)) / unit)
        row.append(np.mean(np.array(r) <= a.alpha_max))
    print(f"  {s:13s} " + "  ".join(f"{v:.2f}" for v in row))
print(f"\n-> {out}")
