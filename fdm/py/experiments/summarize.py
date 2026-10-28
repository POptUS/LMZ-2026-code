#!/usr/bin/env python
"""
Print a compact digest of the headline numbers for the experiments section.

    python summarize.py results_cutest.pkl
    python summarize.py results_cutest.pkl --csv digest.csv   # also dump per-problem rows
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from fdm.fdm.problems import Problem
from fdm.fdm.profiles import collect_costs, feasibility_stats, identification_stats

ORDER = ["TV-AS-lazy", "TV-AS-full", "TV-ALL", "SQP-ALL", "SQP-AS-lazy"]


def _apply_subset(results, meta, only=None, natural=False, cache="problems_cache.json"):
    """Restrict the analysis to a subset of problems chosen by a criterion that
    does not depend on any solver's performance."""
    import json
    import os
    keep = None
    if natural:
        if not os.path.exists(cache):
            raise SystemExit(f"--natural-x0 needs {cache} (it records whether x0 was shifted)")
        with open(cache) as fh:
            entries = json.load(fh)
        keep = {e["name"] for e in entries if not e.get("meta", {}).get("x0_shifted")}
        print(f"--natural-x0: {len(keep)} of {len(entries)} problems start feasible at the SIF point")
    if only:
        names = open(only[1:]).read().split() if only.startswith("@") else only.split(",")
        sel = {n.strip() for n in names if n.strip()}
        keep = sel if keep is None else (keep & sel)
    if keep is None:
        return results, meta
    results = {k: v for k, v in results.items() if k in keep}
    meta = dict(meta)
    meta["problems"] = [e for e in meta["problems"] if e["name"] in keep]
    print(f"analysing {len(results)} problems after subsetting")
    return results, meta


def stubs(meta, results):
    probs = []
    for e in meta["problems"]:
        runs = results.get(e["name"], {})
        f0 = None
        for r in runs.values():
            for rec in r["trace"]:
                if np.isfinite(rec[2]):
                    f0 = rec[2]
                    break
            if f0 is not None:
                break
        probs.append(Problem(e["name"], e["n"], e["p"], e["x0"], lambda x, v=(f0 or 0.0): v,
                             lambda x, q=e["p"]: np.zeros(q)))
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--only", default=None, help="comma-separated problem names, or @file")
    ap.add_argument("--natural-x0", action="store_true",
                    help="keep only problems whose SIF starting point is already feasible")
    ap.add_argument("--cache", default="problems_cache.json")
    args = ap.parse_args()
    with open(args.pickle, "rb") as fh:
        data = pickle.load(fh)
    results = {k: v for k, v in data["results"].items() if v}
    refs, meta = data.get("references", {}), data["meta"]
    results, meta = _apply_subset(results, meta, args.only, args.natural_x0, args.cache)
    pinfo = {e["name"]: e for e in meta["problems"]}
    probs = stubs(meta, results)
    solvers = [s for s in ORDER if any(s in r for r in results.values())]
    N = len(results)
    print(f"\n=== {N} problems, solvers: {', '.join(solvers)} ===")
    print(f"n in [{min(e['n'] for e in pinfo.values())}, {max(e['n'] for e in pinfo.values())}], "
          f"p in [{min(e['p'] for e in pinfo.values())}, {max(e['p'] for e in pinfo.values())}]")
    refs = {k: v for k, v in refs.items() if k in results}
    nref = sum(1 for v in refs.values() if v)
    nsc = sum(1 for v in refs.values() if v and v.get("strict_compl"))
    print(f"reference solution available: {nref}/{N}; of those, strict complementarity: {nsc}")

    # ---------------- 1. feasibility ----------------
    print("\n--- (1) FEASIBILITY of trial points ---")
    print(f"{'solver':14s} {'median':>8s} {'mean':>8s} {'<0.5':>6s} {'>0.8':>6s}  {'subQP infeas':>13s}")
    feas_tbl = {}
    for s in solvers:
        vals, subinf = [], []
        for name, runs in results.items():
            if s not in runs:
                continue
            r = runs[s]
            st = feasibility_stats(r)
            v = st["frac_feasible_trials"]
            subinf.append(st["frac_sub_infeasible"])
            if np.isfinite(v):
                vals.append(v)
        vals = np.array(vals)
        feas_tbl[s] = vals
        si = f"{np.mean(subinf):.3f}" if subinf else "--"
        print(f"{s:14s} {np.median(vals):8.3f} {np.mean(vals):8.3f} "
              f"{np.mean(vals < .5):6.2f} {np.mean(vals > .8):6.2f}  {si:>13s}")

    # ---------------- 2. accuracy / solved counts ----------------
    print("\n--- (2a) PROBLEMS EVENTUALLY SOLVED (within the whole budget) ---")
    print("    " + f"{'solver':14s}" + "".join(f"{f'tau=1e-{k}':>10s}" for k in (1, 3, 5, 7)))
    Tcache = {}
    for cost in ["points", "components"]:
        for tau in (1e-1, 1e-3, 1e-5, 1e-7):
            Tcache[(cost, tau)] = collect_costs(results, probs, tau, cost=cost)
    for s_ in solvers:
        row = f"    {s_:14s}"
        for tau in (1e-1, 1e-3, 1e-5, 1e-7):
            sv, T = Tcache[("points", tau)]
            row += f"{np.mean(np.isfinite(T[:, sv.index(s_)])):10.2f}"
        print(row)
    print("    (independent of the cost convention: a run either reaches tau or not)")

    print("\n--- (2b) SOLVED WITHIN A FIXED BUDGET (this is what the data profiles show) ---")
    for cost, levels in [("points", (5, 10, 25, 50, 100)), ("components", (1, 2, 5, 10, 25))]:
        unit = "(n+1)" if cost == "points" else "(n+1)(p+1)"
        for tau in (1e-3, 1e-5):
            print(f"  cost = {cost:11s} tau = {tau:g}   budget in multiples of {unit}")
            print("    " + f"{'solver':14s}" + "".join(f"{f'<={b}':>8s}" for b in levels))
            for s_ in solvers:
                sv, T = Tcache[(cost, tau)]
                col = T[:, sv.index(s_)]
                print(f"    {s_:14s}" + "".join(f"{np.mean(col <= b):8.2f}" for b in levels))

    # ---------------- 3. cost of the constraint models ----------------
    print("\n--- (3) CONSTRAINT EVALUATIONS (lazy vs rebuilt-every-iteration) ---")
    rows = []
    for name, runs in results.items():
        if not {"TV-AS-lazy", "TV-AS-full"} <= set(runs):
            continue
        p = pinfo[name]["p"]
        a = len(refs[name]["A"]) / p if refs.get(name) else np.nan
        rows.append(dict(problem=name, p=p, astar_frac=a,
                         ratio_full=runs["TV-AS-lazy"]["oracle"]["n_c"] / max(1, runs["TV-AS-full"]["oracle"]["n_c"]),
                         ratio_all=(runs["TV-AS-lazy"]["oracle"]["n_c"] / max(1, runs["TV-ALL"]["oracle"]["n_c"]))
                         if "TV-ALL" in runs else np.nan))
    df = pd.DataFrame(rows)
    if len(df):
        print(f"  lazy/full  : median {df.ratio_full.median():.3f}  mean {df.ratio_full.mean():.3f}")
        print(f"  lazy/ALL   : median {df.ratio_all.median():.3f}")
        for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
            m = df.astar_frac.between(lo, hi, inclusive="left")
            if m.any():
                print(f"    |A*|/p in [{lo:.2f},{hi:.2f}): {m.sum():3d} problems, "
                      f"median lazy/full {df.loc[m, 'ratio_full'].median():.3f}")

    # ---------------- 4. identification ----------------
    print("\n--- (4) ACTIVE-SET IDENTIFICATION (TV-AS-lazy) ---")
    idr = []
    for name, runs in results.items():
        if "TV-AS-lazy" not in runs or not refs.get(name):
            continue
        st = identification_stats(runs["TV-AS-lazy"], refs[name]["A"], pinfo[name]["p"])
        if not st:
            continue
        # did the run actually converge? (within 1% of the best feasible value seen)
        fbest = min(r["trace"][-1][2] for r in runs.values() if r["trace"])
        f = runs["TV-AS-lazy"]["f"]
        conv = abs(f - fbest) <= 1e-2 * (1 + abs(fbest))
        idr.append(dict(problem=name, p=pinfo[name]["p"], sc=refs[name].get("strict_compl", False),
                        converged=conv, exact=st["frac_exact_after_accept"], superset=st["frac_superset"],
                        size_ratio=st["mean_size_ratio"], astar_frac=len(refs[name]["A"]) / max(1, pinfo[name]["p"])))
    di = pd.DataFrame(idr)
    if len(di):
        for label, m in [("all problems", np.ones(len(di), bool)),
                         ("converged runs", di.converged.values),
                         ("converged + strict compl.", (di.converged & di.sc).values)]:
            d = di[m]
            if not len(d):
                continue
            print(f"  {label} ({len(d)}): median exact-after-accept {d.exact.median():.2f}, "
                  f">0.9 on {np.mean(d.exact > .9):.2f}, >0.5 on {np.mean(d.exact > .5):.2f}; "
                  f"median superset {d.superset.median():.2f}; median |A_k|/p {d.size_ratio.median():.2f} "
                  f"(|A*|/p {d.astar_frac.median():.2f})")

    # ---------------- 5. termination ----------------
    print("\n--- (5) TERMINATION ---")
    for s in solvers:
        st = [runs[s]["status"] for runs in results.values() if s in runs]
        c = pd.Series(st).value_counts()
        print(f"  {s:14s} " + "  ".join(f"{k}={v}" for k, v in c.items()))

    # ---------------- 6. where we lose ----------------
    print("\n--- (6) WORST GAPS: TV-AS-lazy vs best other solver (relative) ---")
    gaps = []
    for name, runs in results.items():
        if "TV-AS-lazy" not in runs:
            continue
        f = runs["TV-AS-lazy"]["f"]
        others = {s: r["f"] for s, r in runs.items() if s != "TV-AS-lazy" and np.all(r["c"] <= 1e-6)}
        if not others:
            continue
        bs = min(others, key=others.get)
        gap = (f - others[bs]) / (1 + abs(others[bs]))
        gaps.append(dict(problem=name, n=pinfo[name]["n"], p=pinfo[name]["p"], ours=f,
                         best_other=others[bs], by=bs, rel_gap=gap))
    dg = pd.DataFrame(gaps).sort_values("rel_gap", ascending=False)
    print(f"  TV-AS-lazy is best or tied (gap<=1e-6) on {np.mean(dg.rel_gap <= 1e-6):.2f} of problems")
    print(dg.head(12).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    if args.csv:
        out = dg.merge(df, on="problem", how="outer").merge(di, on="problem", how="outer", suffixes=("", "_id"))
        out.to_csv(args.csv, index=False)
        print(f"\nper-problem rows -> {args.csv}")


if __name__ == "__main__":
    main()
