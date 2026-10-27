"""
Baselines and reference solutions.

* ``run_cobyla`` -- SciPy's COBYLA through the same counting ``Oracle`` so the
  evaluation traces are directly comparable.  COBYLA evaluates f and every
  constraint at every point, and its iterates are not guaranteed feasible;
  we therefore also report the fraction of infeasible points it visited.

* ``reference_solution`` -- derivative-based polish (SLSQP with exact CUTEst
  derivatives) started from the best point any DFO solver found; gives f*
  and the "true" active set A* used in the identification experiments.
"""
from __future__ import annotations

import time

import numpy as np

from .oracle import Oracle



# --------------------------------------------------------------------------- #
# Telling COBYLA's trial points from its geometry points
# --------------------------------------------------------------------------- #
def _cobyla_eval_sites():
    """Locate the evaluation call sites inside PRIMA's COBYLA and say what each
    one is.

    SciPy >= 1.16 ships PRIMA's COBYLA as pure Python.  Its main loop evaluates
    the functions in two places: the trust-region branch (a trial point, the
    counterpart of x_k + s_k here) and the ``improve_geo`` branch (a point taken
    only to repair the geometry of the simplex).  The initial simplex is
    evaluated in a third module.  We find the call sites by parsing the source,
    rather than hard-coding line numbers, so this survives minor version
    changes; if the layout ever changes beyond recognition the caller falls back
    to counting every point.

    Returns (cobylb_module, initialize_module, {lineno: kind}) or None.
    """
    import ast
    import importlib
    import inspect
    try:
        cb = importlib.import_module("scipy._lib.pyprima.cobyla.cobylb")
        ini = importlib.import_module("scipy._lib.pyprima.cobyla.initialize")
    except Exception:
        return None
    sites = {}

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_If(self, node):
            self.stack.append(ast.unparse(node.test))
            for k in node.body:
                self.visit(k)
            self.stack.pop()
            for k in node.orelse:
                self.visit(k)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "evaluate":
                sites[node.lineno] = ("geometry" if any("improve_geo" in c for c in self.stack)
                                      else "trial")
            self.generic_visit(node)

    try:
        V().visit(ast.parse(inspect.getsource(cb)))
    except Exception:
        return None
    if not sites or "trial" not in sites.values():
        return None
    return cb, ini, sites


def run_cobyla(problem, max_points=None, budget_factor=100.0, rhobeg=1.0, tol=1e-12,
               classify_points=True, budget_in="queries", max_queries=None,
               log_queries=False, unrelaxable=False):
    """COBYLA through the same counting Oracle.

    The budget is enforced WITHOUT raising: SciPy's COBYLA calls back from
    Fortran through f2py, and an exception crossing that boundary aborts the
    interpreter ("Fatal Python error: F2PySwapThreadLocalCallbackPtr").  So once
    the budget is spent we stop charging the oracle and return frozen sentinel
    values, and we also cap ``maxiter`` (for COBYLA this bounds the number of
    function evaluations), which is what normally ends the run."""
    from scipy.optimize import minimize

    n, p = problem.n, problem.p
    if budget_in == "queries":
        max_queries = max_queries or int(budget_factor * (n + 1) * (p + 1))
        oracle = Oracle(problem, max_queries=max_queries, log_queries=log_queries)
        # COBYLA queries all p+1 oracles at every point, so this many points at most
        max_points = max(1, max_queries // (p + 1))
    else:
        max_points = max_points or int(budget_factor * (n + 1))
        oracle = Oracle(problem, max_points=max_points, log_queries=log_queries)
    t0 = time.time()
    state = {"exhausted": False, "f": 0.0, "c": np.zeros(p)}
    kinds = {}          # oracle index -> "initial" | "trial" | "geometry"
    pending = {"kind": None}

    def point(x):
        """Oracle index for x, or None once the budget is spent."""
        key = x.tobytes()
        if key in point.cache:
            return point.cache[key]
        if oracle.n_points >= max_points or oracle.budget_exhausted():
            state["exhausted"] = True
            return None
        point.cache[key] = oracle.add_point(x)
        if pending["kind"] is not None:
            kinds[point.cache[key]] = pending["kind"]
        return point.cache[key]
    point.cache = {}

    def f(x):
        idx = point(np.asarray(x, float))
        if idx is None:
            return state["f"]           # frozen: no progress, no charge
        if unrelaxable:
            # The setting of the paper: the objective does not exist at an
            # infeasible point.  Evaluate the constraints first and hand back a
            # NaN without querying f if any is violated, so such a point is
            # charged for the constraints alone.  COBYLA has no provision for
            # this; what it does with the NaN is the point of the variant.
            cv = np.array([oracle.eval_c(idx, i) for i in range(p)], float)
            state["c"] = cv
            if np.any(cv > 0) or not np.all(np.isfinite(cv)):
                state["f"] = np.nan
                return np.nan
            state["f"] = oracle.eval_f(idx)
            return state["f"]
        oracle.eval_all(idx)            # COBYLA needs f and all constraints here
        row = oracle.row(idx)
        state["f"], state["c"] = row[0], row[1:].copy()
        return row[0]

    def con(x, i):
        idx = point(np.asarray(x, float))
        if idx is None:
            return -abs(state["c"][i])  # frozen, reported feasible
        return -oracle.eval_c(idx, i)

    cons = [{"type": "ineq", "fun": lambda x, i=i: con(x, i)} for i in range(p)]

    sites = _cobyla_eval_sites() if classify_points else None
    patched = []
    if sites is not None:
        import inspect as _inspect
        cb, ini, site_kind = sites

        def _wrap(mod, orig, default):
            def g(calcfc, x, *args, **kw):
                ln = _inspect.currentframe().f_back.f_lineno
                pending["kind"] = site_kind.get(ln, default)
                try:
                    return orig(calcfc, x, *args, **kw)
                finally:
                    pending["kind"] = None
            patched.append((mod, orig))
            return g

        cb.evaluate = _wrap(cb, cb.evaluate, "trial")
        ini.evaluate = _wrap(ini, ini.evaluate, "initial")

    try:
        minimize(f, problem.x0, method="COBYLA", constraints=cons, tol=tol,
                 options={"rhobeg": rhobeg, "maxiter": max_points, "catol": 0.0})
    finally:
        for mod, orig in patched:
            mod.evaluate = orig
    status = "budget" if state["exhausted"] else "done"

    F = oracle.F
    known_all = ~np.isnan(F[:, 1:]).any(axis=1)
    feas_pt = known_all & ~(F[:, 1:] > 0).any(axis=1)
    infeas = np.sum(known_all & ~feas_pt)
    xb = oracle.best_x if oracle.best_x is not None else problem.x0
    out = dict(solver="COBYLA-unrelaxable" if unrelaxable else "COBYLA", status=status, x=np.asarray(xb), f=float(oracle.best_f),
               c=problem.c(xb), wall=time.time() - t0, trace=list(oracle.trace), history=[],
               oracle=oracle.summary(),
               frac_infeasible_points=float(infeas / max(1, known_all.sum())))
    if kinds:
        counts, feas = {}, {}
        for kind in ("initial", "trial", "geometry"):
            idx = [i for i, k in kinds.items() if k == kind and known_all[i]]
            counts[kind] = len(idx)
            feas[kind] = float(np.mean([feas_pt[i] for i in idx])) if idx else float("nan")
        out["point_kind_counts"] = counts
        out["frac_feasible_by_kind"] = feas
        # the like-for-like counterpart of the model-based solvers' trial points
        out["frac_feasible_trials"] = feas["trial"]
    return out


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
