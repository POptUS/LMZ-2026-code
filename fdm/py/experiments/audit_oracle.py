#!/usr/bin/env python
"""
Independent audit of the oracle counts.

Every solver reaches f and the constraints only through the Oracle class, which
does its own counting -- no solver reports its own usage.  With
``log_queries=True`` the oracle additionally keeps one ledger per oracle (f and
each c_i), recording the point and value of every scalar query.  This script
re-runs a sample of problems with the ledgers on and checks:

  * the ledger lengths agree with the counters n_f and n_c;
  * the proposed method's queries per point, which should be well below p+1.

    python3 audit_oracle.py --n-sample 12
    python3 audit_oracle.py --only HS113,AVGASA,OPTPRLOC
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fdm.fdm.algorithm import CDFO, Options
from fdm.fdm.problems import load_cutest_testset, hs_problems

ap = argparse.ArgumentParser()
ap.add_argument("--only", default=None)
ap.add_argument("--n-sample", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cache", default="problems_cache.json")
ap.add_argument("--testset", default="cutest", choices=["cutest", "hs"])
ap.add_argument("--budget", type=float, default=100.0)
ap.add_argument("--par-c1-mult", type=float, default=10.0)
a = ap.parse_args()

probs = hs_problems() if a.testset == "hs" else load_cutest_testset(a.cache)
if a.only:
    keep = set(a.only.split(","))
    probs = [p for p in probs if p.name in keep]
elif a.n_sample and a.n_sample < len(probs):
    rng = np.random.default_rng(a.seed)
    probs = [probs[i] for i in rng.choice(len(probs), a.n_sample, replace=False)]
print(f"{len(probs)} problems\n")

rows = []
for prob in probs:
    # the proposed method
    alg = CDFO(prob, Options(budget_factor=a.budget, par_c1_mult=a.par_c1_mult,
                             log_queries=True))
    try:
        alg.run()
    except Exception as e:  # noqa
        print(f"  {prob.name}: {e!r}")
        continue
    au, su = alg.oracle.audit(), alg.oracle.summary()
    rows.append(dict(problem=prob.name, p=prob.p, solver="TV-AS-lazy",
                     n_f=au["n_f"], n_f_logged=au["n_f_logged"],
                     n_c=au["n_c"], n_c_logged=au["n_c_logged"], agrees=au["agrees"],
                     points=su["n_points"], queries=su["n_components"],
                     q_per_pt=su["n_components"] / max(su["n_points"], 1), expected=prob.p + 1))
    print(f"  {prob.name}: done")

df = pd.DataFrame(rows)
print("\n" + df[["problem", "p", "solver", "n_f", "n_f_logged", "n_c", "n_c_logged",
                 "agrees", "points", "queries", "q_per_pt", "expected"]]
      .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

ours = df[df.solver == "TV-AS-lazy"]
print(f"\nledger agrees with the counters on {int(ours.agrees.sum())}/{len(ours)} runs")
print(f"median queries per point: proposed {ours.q_per_pt.median():.2f} "
      f"(of p+1 = {ours.expected.median():.0f})")
