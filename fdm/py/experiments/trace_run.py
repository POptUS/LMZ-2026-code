#!/usr/bin/env python
"""
A line-by-line account of where one run spent its budget.

Writes a log with one row per model build and one per iteration, recording the
trust-region radius, the criticality measure, how many points and queries the
build cost, and the running totals.  Intended for explaining a run that
exhausted its budget without making progress.

    python3 trace_run.py HAIFAS
    python3 trace_run.py HAIFAS --solver TV-AS-full --out haifas.log
"""
from __future__ import annotations

import argparse

import numpy as np

import fdm.fdm.algorithm as A
from fdm.fdm.algorithm import CDFO, Options
from fdm.fdm.problems import load_cutest_testset, hs_problems

VARIANTS = {
    "TV-AS-lazy":  dict(subproblem="tv",  active_set=True,  lazy=True),
    "TV-AS-full":  dict(subproblem="tv",  active_set=True,  lazy=False),
    "TV-ALL":      dict(subproblem="tv",  active_set=False, lazy=False),
    "SQP-ALL":     dict(subproblem="sqp", active_set=False, lazy=False),
    "SQP-AS-lazy": dict(subproblem="sqp", active_set=True,  lazy=True),
}

ap = argparse.ArgumentParser()
ap.add_argument("problem")
ap.add_argument("--solver", default="TV-AS-lazy", choices=list(VARIANTS))
ap.add_argument("--cache", default="problems_cache.json")
ap.add_argument("--testset", default="cutest", choices=["cutest", "hs"])
ap.add_argument("--budget", type=float, default=100.0)
ap.add_argument("--par-c1-mult", type=float, default=10.0)
ap.add_argument("--out", default=None,
                help="write the log here (default: <problem>_<solver>.log)")
ap.add_argument("--verbose-steps", action="store_true",
                help="also log every constraint evaluation at a trial point")
a = ap.parse_args()

probs = hs_problems() if a.testset == "hs" else load_cutest_testset(a.cache)
prob = {p.name: p for p in probs}[a.problem]
alg = CDFO(prob, Options(budget_factor=a.budget, par_c1_mult=a.par_c1_mult,
                         **VARIANTS[a.solver]))
cap = int(a.budget * (prob.n + 1) * (prob.p + 1))

lines = []
def say(s=""):
    lines.append(s)
    print(s)

say(f"{a.problem}: n={prob.n} p={prob.p} solver={a.solver}")
say(f"budget = {a.budget:g}(n+1)(p+1) = {cap} oracle queries")
say("")
say("One row per event, in the order they happened.  An iteration may contain")
say("several rows: the criticality loop rebuilds the models at each reduced")
say("radius before any step is computed.")
say("")
say("  event     what happened.  'build models' is one pass of model building;")
say("            'step |s|' is a solve of the step subproblem, with the")
say("            infinity norm of the step it returned.")
say("  k         iteration number.")
say("  Delta     trust-region radius in force for this event.")
say("  chi       criticality measure most recently computed.  The criticality")
say("            loop runs while Delta > mu*chi, with mu = 1 here.")
say("  |A|       size of the active-set estimate.")
say("  pts+      points newly evaluated by this event.")
say("  queries+  scalar oracle queries charged for this event: one per")
say("            evaluation of f or of an individual c_i.")
say("  queries   running total of queries.")
say("  % cap     that total as a share of the budget.")
say("")
say(f"{'event':<26} {'k':>3} {'Delta':>11} {'chi':>11} {'|A|':>4} "
    f"{'pts+':>5} {'queries+':>9} {'queries':>8} {'% cap':>6}")
say("-" * 94)

state = {"pts": 0, "q": 0}
def totals():
    s = alg.oracle.summary()
    dp = s["n_points"] - state["pts"]
    dq = s["n_components"] - state["q"]
    state["pts"], state["q"] = s["n_points"], s["n_components"]
    return dp, dq, s["n_components"]

build_orig, chi_orig = A.CDFO._build_and_estimate, A.CDFO._chi
solve_orig = A.CDFO._solve
last_chi = {"v": float("nan")}
cur = {"k": -1, "delta": float("nan")}

def build(self, xk, delta, k, A_prev):
    out = build_orig(self, xk, delta, k, A_prev)
    cur["k"], cur["delta"] = k, delta
    dp, dq, q = totals()
    say(f"{'build models':<26} {k:>3} {delta:>11.3e} {last_chi['v']:>11.3e} "
        f"{len(out[0]):>4} {dp:>5} {dq:>9} {q:>8} {100*q/cap:>5.1f}%")
    return out


def solve(self, g_f, G, c, Aset, H, delta):
    """Log the subproblem solve and the feasibility verdict on its trial point."""
    out = solve_orig(self, g_f, G, c, Aset, H, delta)
    cur["delta"] = delta
    dp, dq, q = totals()
    st = out.get("s")
    if st is None:
        what = "subproblem infeasible"
    else:
        nrm = float(np.max(np.abs(st)))
        what = f"step |s|={nrm:.2e}" + ("  (zero)" if nrm <= 1e-8 * delta else "")
    say(f"{what:<26} {cur['k']:>3} {delta:>11.3e} {last_chi['v']:>11.3e} "
        f"{len(Aset):>4} {dp:>5} {dq:>9} {q:>8} {100*q/cap:>5.1f}%")
    return out

def chi(self, *args, **kw):
    v = chi_orig(self, *args, **kw)
    last_chi["v"] = float(v)
    return v

# the trial point's fate is visible through the oracle: log each constraint
# evaluation charged at a point that is not the incumbent
eval_c_orig = type(alg.oracle).eval_c
def eval_c(self, idx, i):
    before = self.n_c
    v = eval_c_orig(self, idx, i)
    if self.n_c > before and cur["k"] >= 0:
        dp, dq, q = totals()
        if dq:
            say(f"{'  eval c_%d -> %s' % (i, 'violated' if v > 0 else 'ok'):<26} "
                f"{cur['k']:>3} {cur['delta']:>11.3e} {last_chi['v']:>11.3e} "
                f"{'':>4} {dp:>5} {dq:>9} {q:>8} {100*q/cap:>5.1f}%")
    return v

A.CDFO._build_and_estimate, A.CDFO._chi, A.CDFO._solve = build, chi, solve
if a.verbose_steps:
    type(alg.oracle).eval_c = eval_c
try:
    res = alg.run()
finally:
    A.CDFO._build_and_estimate, A.CDFO._chi = build_orig, chi_orig
    A.CDFO._solve = solve_orig
    type(alg.oracle).eval_c = eval_c_orig

say("-" * 94)
say("")
say("Per-iteration record.  'crit' counts the criticality-loop reductions in")
say("that iteration, 'feas' whether its trial point was feasible, 'acc' whether")
say("the step was accepted.  An iteration that runs out of budget inside the")
say("criticality loop never completes and so is absent from this table.")
say("")
say(f"{'k':>3} {'Delta':>11} {'chi':>11} {'crit':>5} {'|A|':>4} {'feas':>5} "
    f"{'acc':>4} {'queries':>8}")
for h in res["history"]:
    say(f"{h['k']:>3} {h['delta']:>11.3e} {h['chi']:>11.3e} {h['n_crit_shrinks']:>5} "
        f"{len(h['A_initial']):>4} {int(h['trial_feasible']):>5} {int(h['accepted']):>4} "
        f"{h['n_components']:>8}")

s = alg.oracle.summary()
geo = res.get("n_geometry_points", 0)
say("")
say(f"status            : {res['status']}")
say(f"f                 : {res['f']:.8g}")
say(f"iterations logged : {len(res['history'])}")
say(f"points evaluated  : {s['n_points']}  (geometry {geo}, "
    f"{100*geo/max(s['n_points'],1):.1f}%)")
say(f"oracle queries    : {s['n_components']} of {cap}")
say(f"formquad calls    : {res.get('n_formquad_calls')}")
say("")
say("An iteration that exhausts the budget inside the criticality loop leaves no")
say("history record, so the last logged iteration is not the last one attempted.")
if any(h["n_crit_shrinks"] for h in res["history"]) or res["status"] == "budget":
    say("")
    say("Note: the loop exits when Delta <= mu*chi.  A chi of exactly zero makes")
    say("that condition unsatisfiable for any positive Delta, so the loop then")
    say("runs until the budget or max_crit_shrinks stops it.")

out = a.out or f"{a.problem}_{a.solver}.log"
open(out, "w").write("\n".join(lines) + "\n")
print(f"\n-> {out}")
