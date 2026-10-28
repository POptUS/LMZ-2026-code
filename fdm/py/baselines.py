"""
Reference solutions.

* ``reference_solution`` -- derivative-based polish (SLSQP with exact CUTEst
  derivatives) started from the best point any DFO solver found; gives f*
  and the "true" active set A* used in the identification experiments.
"""
from __future__ import annotations

import numpy as np


def reference_solution(problem, x_start, starts=None, tol_active=1e-6, feas_tol=1e-7, verbose=False):
    """Reference solution: polish with derivative-based solvers and report
    (x*, f*, A*, multipliers).

    Robustness matters here: a failed reference costs the problem its |A*| and
    excludes it from the identification statistics.  So we
      * try SLSQP and trust-constr from several starting points,
      * judge feasibility with a RELATIVE tolerance (an absolute 1e-8 rejects
        numerically fine points on badly scaled problems such as CRESC4), and
      * fall back to the best feasible point the DFO solvers found, flagged as
        ``source='dfo'``, rather than returning None.
    """
    from scipy.optimize import minimize, NonlinearConstraint

    def viol(x):
        c = problem.c(x)
        return float(np.max(c / (1.0 + np.abs(c)))) if len(c) else 0.0

    cand = []           # (f, x, source)
    pts = [np.asarray(x_start, float)] + [np.asarray(s, float) for s in (starts or [])]
    pts.append(np.asarray(problem.x0, float))
    seen = []
    for x0 in pts:
        if any(np.allclose(x0, s) for s in seen):
            continue
        seen.append(x0)
        if viol(x0) <= feas_tol:
            cand.append((problem.f(x0), x0.copy(), "start"))
        # SLSQP
        try:
            r = minimize(problem.f, x0, jac=problem.grad_f, method="SLSQP",
                         constraints=[{"type": "ineq", "fun": lambda x: -problem.c(x),
                                       "jac": lambda x: -problem.jac_c(x)}],
                         options={"maxiter": 2000, "ftol": 1e-14})
            if np.all(np.isfinite(r.x)) and viol(r.x) <= feas_tol:
                cand.append((float(problem.f(r.x)), r.x.copy(), "slsqp"))
        except Exception:
            pass
        # trust-constr: succeeds on several problems where SLSQP stalls
        try:
            nlc = NonlinearConstraint(problem.c, -np.inf, 0.0, jac=problem.jac_c)
            r = minimize(problem.f, x0, jac=problem.grad_f, method="trust-constr",
                         constraints=[nlc], options={"maxiter": 3000, "gtol": 1e-12, "xtol": 1e-14})
            if np.all(np.isfinite(r.x)) and viol(r.x) <= feas_tol:
                cand.append((float(problem.f(r.x)), r.x.copy(), "trust-constr"))
        except Exception:
            pass
    if not cand:
        if verbose:
            print(f"  {problem.name}: no feasible reference found")
        return None

    f, x, source = min(cand, key=lambda t: t[0])
    c = problem.c(x)
    A = [i for i in range(problem.p) if c[i] >= -tol_active * (1 + abs(c[i]))]
    J = problem.jac_c(x)[A] if A else np.zeros((0, problem.n))
    g = problem.grad_f(x)
    if len(A):
        lam, *_ = np.linalg.lstsq(J.T, -g, rcond=None)
        resid = float(np.linalg.norm(g + J.T @ lam))
    else:
        lam, resid = np.zeros(0), float(np.linalg.norm(g))
    return dict(x=x, f=float(f), A=A, multipliers=lam, kkt_residual=resid,
                strict_compl=bool(np.all(lam > 1e-8)) if len(A) else True,
                source=source, max_viol=viol(x))
