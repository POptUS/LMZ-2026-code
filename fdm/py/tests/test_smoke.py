"""Smoke tests: run without pycutest/ibcdfo/clarabel via the fallbacks."""
import numpy as np
from fdm.fdm.problems import hs_problems, make_feasible
from fdm.fdm.algorithm import run_cdfo
from fdm.fdm.baselines import reference_solution

FSTAR = {"HS12": -30.0, "HS22": 1.0, "HS29": -16 * np.sqrt(2), "HS43": -44.0, "HS100": 680.6300573, "HS113": 24.3062091}


def test_tv_as_lazy_reaches_optima():
    for P in hs_problems():
        r = run_cdfo(P, budget_factor=100)
        assert np.all(r["c"] <= 1e-10), P.name
        assert abs(r["f"] - FSTAR[P.name]) <= 1e-4 * (1 + abs(FSTAR[P.name])), (P.name, r["f"])


def test_variants_run():
    P = hs_problems()[3]
    for kw in [dict(lazy=False), dict(active_set=False), dict(subproblem="sqp"), dict(hess="lagrangian"), dict(hess="identity")]:
        r = run_cdfo(P, budget_factor=20, **kw)
        assert np.all(r["c"] <= 1e-10)


def test_reference_solution():
    P = hs_problems()[0]
    r = run_cdfo(P, budget_factor=50)
    ref = reference_solution(P, r["x"])
    assert abs(ref["f"] - FSTAR["HS12"]) < 1e-6 and ref["A"] == [0]


def test_make_feasible():
    P = hs_problems()[1].with_x0(np.array([2.0, 2.0]))   # HS22's original infeasible x0
    Q = make_feasible(P)
    assert Q is not None and np.all(Q.c(Q.x0) <= 0)
