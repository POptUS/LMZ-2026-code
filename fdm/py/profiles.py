"""
Data profiles (More & Wild, 2009) and performance profiles (Dolan & More, 2002)
computed from the oracle traces.

Convergence test:   f(x) <= f_L + tau * (f(x0) - f_L)
where f_L is the best *feasible* value found by any solver on that problem
(optionally replaced by the derivative-based reference f*).

Two cost measures are supported:
  cost='points'      : number of distinct points at which anything was evaluated
                       (one simulation gives f and all c_i)      -> unit: n+1
  cost='components'  : total number of scalar evaluations f + sum_i c_i
                       (constraints are separate oracles)        -> unit: (n+1)(p+1)
"""
from __future__ import annotations

import numpy as np

TAUS = (1e-1, 1e-3, 1e-5, 1e-7)


def cost_to_converge(trace, f0, fL, tau, cost="points"):
    """Smallest cost at which the run's best feasible f satisfies the test; inf if never."""
    thr = fL + tau * (f0 - fL)
    col = 0 if cost == "points" else 1
    for rec in trace:
        if rec[2] <= thr:
            return rec[col]
    return np.inf


def collect_costs(results, problems, tau, cost="points", use_reference=False):
    """results[problem_name][solver] = run dict.  Returns (solvers, T) with
    T[i, j] = cost for solver j on problem i (inf = failure), already
    normalized by the unit (n+1) or (n+1)(p+1)."""
    solvers = sorted({s for d in results.values() for s in d})
    T = np.full((len(problems), len(solvers)), np.inf)
    for i, prob in enumerate(problems):
        runs = results.get(prob.name, {})
        if not runs:
            continue
        f0 = prob.f(prob.x0)
        fL = min(r["trace"][-1][2] for r in runs.values() if r["trace"])
        if use_reference and runs and "reference_f" in next(iter(runs.values())):
            fL = min(fL, next(iter(runs.values()))["reference_f"])
        unit = (prob.n + 1) if cost == "points" else (prob.n + 1) * (prob.p + 1)
        for j, s in enumerate(solvers):
            if s in runs:
                T[i, j] = cost_to_converge(runs[s]["trace"], f0, fL, tau, cost) / unit
    return solvers, T


def _empty(T):
    return T.size == 0 or T.shape[0] == 0 or T.shape[1] == 0


def data_profile(T, grid=None):
    """Return (grid, D) where D[j, k] = fraction of problems solved by solver j with budget <= grid[k]."""
    if _empty(T):
        return np.linspace(0, 1, 2), np.zeros((max(T.shape[1] if T.ndim == 2 else 0, 0), 2))
    if grid is None:
        m = np.nanmax(np.where(np.isfinite(T), T, np.nan)) if np.any(np.isfinite(T)) else 1.0
        grid = np.linspace(0, m, 400)
    D = np.zeros((T.shape[1], len(grid)))
    for j in range(T.shape[1]):
        for k, a in enumerate(grid):
            D[j, k] = np.mean(T[:, j] <= a)
    return grid, D


def performance_profile(T, grid=None):
    """Ratios r_ij = T_ij / min_j T_ij;  P[j, k] = fraction with r_ij <= grid[k] (log2 grid)."""
    if _empty(T):
        return np.array([1.0, 2.0]), np.zeros((T.shape[1] if T.ndim == 2 else 0, 2))
    best = np.min(T, axis=1, keepdims=True)
    best[~np.isfinite(best)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        R = T / best
    R[~np.isfinite(R)] = np.inf
    if grid is None:
        finite = R[np.isfinite(R)]
        rmax = max(2.0, float(np.max(finite))) if finite.size else 2.0
        grid = np.logspace(0, np.log2(rmax) + 0.1, 300, base=2)
    P = np.zeros((T.shape[1], len(grid)))
    for j in range(T.shape[1]):
        for k, a in enumerate(grid):
            P[j, k] = np.mean(R[:, j] <= a)
    return grid, P


# --------------------------------------------------------------------------- #
# Aggregate metrics for the "feasibility" and "identification" experiments
# --------------------------------------------------------------------------- #
def feasibility_stats(run):
    h = run.get("history", [])
    if not h:
        return dict(frac_feasible_trials=np.nan, frac_sub_infeasible=np.nan)
    n_tr = len(h)
    return dict(frac_feasible_trials=np.mean([r["trial_feasible"] for r in h]),
                frac_sub_infeasible=np.mean([r["sub_infeasible"] for r in h]),
                frac_accepted=np.mean([r["accepted"] for r in h]), n_iter=n_tr)


def identification_stats(run, A_star, p):
    """A_k statistics relative to the reference active set A*."""
    h = run.get("history", [])
    if not h or A_star is None:
        return None
    A_star = set(A_star)
    exact = np.array([set(r["A_initial"]) == A_star for r in h])
    superset = np.array([A_star <= set(r["A_initial"]) for r in h])
    size = np.array([len(r["A_initial"]) for r in h])
    after_accept = np.array([False] + [h[i - 1]["accepted"] for i in range(1, len(h))])
    # first iteration after which the estimate is exact at every post-acceptance iteration
    first_stable = None
    for i in range(len(h)):
        idx = np.flatnonzero(after_accept[i:]) + i
        if len(idx) and np.all(exact[idx]):
            first_stable = i
            break
    return dict(frac_exact=float(exact.mean()), frac_superset=float(superset.mean()),
                frac_exact_after_accept=float(exact[after_accept].mean()) if after_accept.any() else np.nan,
                mean_size_ratio=float(size.mean() / max(p, 1)), size_star=len(A_star),
                first_stable_iter=first_stable, first_stable_frac=None if first_stable is None else first_stable / len(h),
                n_iter=len(h))