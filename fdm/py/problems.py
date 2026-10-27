"""
Problem interface + PyCUTEst test-set construction.

A ``Problem`` exposes
    n, p, x0, name
    f(x)  -> float
    c(x)  -> array (p,)      constraints in the form  c_i(x) <= 0
    grad_f(x), jac_c(x)      (derivatives; used ONLY for phase-1 feasibility
                              and for computing reference solutions / A*,
                              never by the DFO solvers)

CUTEst constraints come as  cl <= C(x) <= cu  (with equality if cl == cu) and
variable bounds bl <= x <= bu.  We keep problems with inequality constraints
only, converting every finite side into one c_i(x) <= 0.  Bounds are either
excluded (default) or converted to linear inequality constraints
(``bounds='as_constraints'``).
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

BIG = 1e19


class Problem:
    def __init__(self, name, n, p, x0, f, c, grad_f=None, jac_c=None, meta=None):
        self.name, self.n, self.p = name, int(n), int(p)
        self.x0 = np.asarray(x0, dtype=float)
        self._f, self._c, self._gf, self._jc = f, c, grad_f, jac_c
        self.meta = meta or {}

    def f(self, x):
        return float(self._f(np.asarray(x, dtype=float)))

    def c(self, x):
        return np.atleast_1d(np.asarray(self._c(np.asarray(x, dtype=float)), dtype=float))

    def grad_f(self, x):
        if self._gf is None:
            return _fd_grad(self.f, x)
        return np.asarray(self._gf(np.asarray(x, dtype=float)), dtype=float)

    def jac_c(self, x):
        if self._jc is None:
            return _fd_jac(self.c, x, self.p)
        return np.atleast_2d(np.asarray(self._jc(np.asarray(x, dtype=float)), dtype=float))

    def with_x0(self, x0):
        return Problem(self.name, self.n, self.p, x0, self._f, self._c, self._gf, self._jc, dict(self.meta))


def _fd_grad(f, x, h=1e-7):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    f0 = f(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h * max(1.0, abs(x[i]))
        g[i] = (f(x + e) - f0) / e[i]
    return g


def _fd_jac(c, x, p, h=1e-7):
    x = np.asarray(x, float)
    J = np.zeros((p, len(x)))
    c0 = c(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h * max(1.0, abs(x[i]))
        J[:, i] = (c(x + e) - c0) / e[i]
    return J


# --------------------------------------------------------------------------- #
# Picklable problem specification (what we send to worker processes)
# --------------------------------------------------------------------------- #
class ProblemSpec:
    """Everything needed to re-create a Problem inside a worker process.
    Problems wrap closures around the pycutest handle and cannot be pickled,
    so the driver ships specs and each worker calls ``spec.load()``."""

    def __init__(self, kind, name, x0=None, bounds="exclude", meta=None):
        self.kind, self.name, self.bounds = kind, name, bounds
        self.x0 = None if x0 is None else np.asarray(x0, float)
        self.meta = meta or {}

    @staticmethod
    def from_problem(prob, kind, bounds="exclude"):
        return ProblemSpec(kind, prob.name, prob.x0, bounds, dict(prob.meta))

    def load(self):
        if self.kind == "hs":
            prob = {q.name: q for q in hs_problems()}[self.name]
        else:
            prob = cutest_to_problem(import_cutest(self.name), bounds=self.bounds)
        if self.x0 is not None:
            prob = prob.with_x0(self.x0)
        prob.meta.update(self.meta)
        return prob


# --------------------------------------------------------------------------- #
# Phase 1: feasible starting point
# --------------------------------------------------------------------------- #
def make_feasible(prob, margin=1e-3, maxiter=500):
    """If c(x0) <= 0 return prob unchanged.  Otherwise solve
        min ||x - x0||^2  s.t.  c_i(x) <= -margin*(1 + |c_i(x0)|)
    with SLSQP (using derivatives; this is problem *preparation*, not part of
    the DFO run) and return a copy of the problem with the feasible x0.
    Returns None if no feasible point is found."""
    from scipy.optimize import minimize

    x0 = prob.x0
    c0 = prob.c(x0)
    if np.all(c0 <= 0):
        return prob
    tgt = -margin * (1.0 + np.abs(c0))
    cons = [{"type": "ineq", "fun": lambda x: tgt - prob.c(x), "jac": lambda x: -prob.jac_c(x)}]
    res = minimize(lambda x: 0.5 * np.sum((x - x0) ** 2), x0, jac=lambda x: x - x0,
                   constraints=cons, method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-12})
    x = res.x
    if not np.all(prob.c(x) <= 0):
        # second attempt: feasibility-only objective
        res = minimize(lambda x: 0.5 * np.sum(np.maximum(prob.c(x), 0) ** 2), x0,
                       constraints=cons, method="SLSQP", options={"maxiter": maxiter})
        x = res.x
        if not np.all(prob.c(x) <= 0):
            return None
    q = prob.with_x0(x)
    q.meta["x0_shifted"] = True
    q.meta["x0_shift_norm"] = float(np.linalg.norm(x - x0))
    return q


# --------------------------------------------------------------------------- #
# PyCUTEst
# --------------------------------------------------------------------------- #
def import_cutest(name, retries=4, pause=0.25):
    """``pycutest.import_problem`` made robust to the "No module named
    'pycutest_cache_holder.NAME'" failure.

    PyCUTEst compiles each problem into a new subpackage of the cache holder and
    then imports it.  The import system caches the holder's directory listing,
    so a package created moments earlier can be invisible, even though the build
    succeeded and the files are on disk.  The cure is to invalidate the import
    caches (and reload the holder package) and try again.  Crucially we do NOT
    clear the build cache on failure: the artifacts are already correct, and
    rebuilding merely re-triggers the same race."""
    import importlib
    import sys
    import time

    def _refresh():
        importlib.invalidate_caches()
        holder = sys.modules.get("pycutest_cache_holder")
        if holder is not None:
            try:
                importlib.reload(holder)
            except Exception:
                pass

    import pycutest
    last = None
    for attempt in range(retries):
        _refresh()
        try:
            return pycutest.import_problem(name)
        except (ImportError, ModuleNotFoundError) as e:
            last = e
            if "pycutest_cache_holder" not in str(e):
                raise
            time.sleep(pause * (attempt + 1))
    _refresh()
    return pycutest.import_problem(name)   # final attempt; let the error surface


def cutest_to_problem(p, bounds="exclude"):
    """Wrap a pycutest problem object as a ``Problem`` (or return None if it
    has equality constraints / bounds that we do not handle)."""
    n = p.n
    if p.m == 0:
        return None
    if np.any(p.is_eq_cons):
        return None
    has_bounds = np.any(p.bl > -BIG) or np.any(p.bu < BIG)
    if has_bounds and bounds == "exclude":
        return None
    cl, cu = np.asarray(p.cl, float), np.asarray(p.cu, float)
    up = np.flatnonzero(cu < BIG)     # C_j(x) - cu_j <= 0
    lo = np.flatnonzero(cl > -BIG)    # cl_j - C_j(x) <= 0
    lb_idx = np.flatnonzero(p.bl > -BIG) if (has_bounds and bounds == "as_constraints") else np.array([], int)
    ub_idx = np.flatnonzero(p.bu < BIG) if (has_bounds and bounds == "as_constraints") else np.array([], int)
    pc = len(up) + len(lo) + len(lb_idx) + len(ub_idx)
    if pc == 0:
        return None
    bl, bu = np.asarray(p.bl, float), np.asarray(p.bu, float)

    def c(x):
        C = p.cons(x)
        parts = [C[up] - cu[up], cl[lo] - C[lo]]
        if len(lb_idx):
            parts.append(bl[lb_idx] - x[lb_idx])
        if len(ub_idx):
            parts.append(x[ub_idx] - bu[ub_idx])
        return np.concatenate(parts)

    def jac(x):
        _, J = p.cons(x, gradient=True)
        J = np.atleast_2d(J)
        parts = [J[up], -J[lo]]
        if len(lb_idx):
            E = np.zeros((len(lb_idx), n)); E[np.arange(len(lb_idx)), lb_idx] = -1.0; parts.append(E)
        if len(ub_idx):
            E = np.zeros((len(ub_idx), n)); E[np.arange(len(ub_idx)), ub_idx] = 1.0; parts.append(E)
        return np.vstack(parts)

    def f(x):
        return p.obj(x)

    def gf(x):
        return p.obj(x, gradient=True)[1]

    meta = dict(cutest_m=int(p.m), has_bounds=bool(has_bounds), n_bound_cons=int(len(lb_idx) + len(ub_idx)))
    return Problem(p.name, n, pc, np.asarray(p.x0, float), f, c, gf, jac, meta)


# Inequality-constrained CUTEst problems known to be small; used as a fallback
# candidate list if the classification query returns nothing.  (Bounds are
# common among these -> use --bounds as_constraints.)
KNOWN_INEQ_PROBLEMS = [
    "HS10", "HS11", "HS12", "HS13", "HS15", "HS16", "HS17", "HS18", "HS19", "HS20", "HS21", "HS22", "HS23",
    "HS24", "HS29", "HS30", "HS31", "HS33", "HS34", "HS35", "HS36", "HS37", "HS43", "HS44", "HS57", "HS59",
    "HS64", "HS65", "HS66", "HS67", "HS70", "HS72", "HS76", "HS83", "HS84", "HS85", "HS86", "HS88", "HS89",
    "HS90", "HS91", "HS92", "HS93", "HS95", "HS96", "HS97", "HS98", "HS100", "HS101", "HS102", "HS103",
    "HS104", "HS105", "HS106", "HS108", "HS113", "HS116", "HS117", "HS118", "HS268",
    "CB2", "CB3", "CHACONN1", "CHACONN2", "CONGIGMZ", "DEMYMALO", "DIPIGRI", "GIGOMEZ1", "GIGOMEZ2",
    "GIGOMEZ3", "HALDMADS", "KIWCRESC", "MADSEN", "MAKELA1", "MAKELA2", "MAKELA3", "MAKELA4", "MATRIX2",
    "MIFFLIN1", "MIFFLIN2", "MINMAXBD", "MINMAXRB", "OET1", "OET2", "OET3", "OET7", "POLAK1", "POLAK2",
    "POLAK3", "POLAK4", "POLAK5", "POLAK6", "ROSENMMX", "SNAKE", "SPIRAL", "WOMFLET", "ZECEVIC2",
    "ZECEVIC3", "ZECEVIC4", "ZY2", "CANTILVR", "SYNTHES1", "SYNTHES2", "SYNTHES3", "TENBARS4", "LOOTSMA",
    "HIMMELP2", "HIMMELP3", "HIMMELP4", "HIMMELP5", "HIMMELP6", "PENTAGON", "TWOBARS", "DEMBO7",
    "AVGASA", "AVGASB", "BIGGSC4", "EXPFITA", "EXPFITB", "EXPFITC", "HATFLDH", "SIMPLLPA", "SIMPLLPB",
    "SIPOW1", "SIPOW2", "SIPOW3", "SIPOW4", "LSQFIT", "S268", "STANCMIN", "DUALC1", "DUALC2", "DUALC5",
    "DUALC8", "DUAL1", "DUAL2", "DUAL4", "PRIMAL1", "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
    "HS118", "OPTPRLOC", "OPTCNTRL", "SVANBERG", "READING4", "DECONVC", "DNIEPER", "ERRINBAR",
]


def _props(name):
    import pycutest
    try:
        return pycutest.problem_properties(name)
    except Exception:
        return {}


def find_cutest_candidates(n_max=30, m_max=100, verbose=True):
    """Names of regular CUTEst problems with general constraints (classes L, Q,
    O) whose classified size fits (variable-size problems are kept and checked
    after import).  Falls back to KNOWN_INEQ_PROBLEMS if the query is empty."""
    import pycutest
    names = set()
    # pycutest.find_problems matches the *word* form of the classification
    # ('linear', 'quadratic', 'other'), not the single-letter SIF codes.
    for cls in ["linear", "quadratic", "other"]:
        try:
            names |= set(pycutest.find_problems(constraints=cls, regular=True))
        except Exception as e:  # noqa
            if verbose:
                print(f"find_problems(constraints={cls!r}) failed: {e}")
    if not names:
        try:
            names = {nm for nm in pycutest.find_problems()
                     if str(_props(nm).get("constraints", "")).lower() in ("linear", "quadratic", "other", "l", "q", "o")}
        except Exception:
            names = set()
    if not names:
        if verbose:
            print("pycutest.find_problems returned nothing; falling back to KNOWN_INEQ_PROBLEMS")
        names = set(KNOWN_INEQ_PROBLEMS)
    out = []
    for name in sorted(names):
        pr = _props(name)
        n, m = pr.get("n"), pr.get("m")
        if isinstance(n, (int, np.integer)) and not (1 <= n <= n_max):
            continue
        if isinstance(m, (int, np.integer)) and not (1 <= m <= m_max):
            continue
        if pr.get("regular") is False:
            continue
        out.append(name)
    if verbose:
        print(f"{len(out)} candidate problems (of {len(names)} with general constraints)")
    return out


def build_cutest_testset(n_max=30, m_max=100, p_max=100, bounds="exclude", cache="problems_cache.json",
                         margin=1e-3, verbose=True, names=None, require_reference=True, n_ref_starts=5,
                         ref_seed=0):
    """Select and prepare inequality-constrained CUTEst problems.

    A problem is kept only if (i) it is inequality-constrained and small enough,
    (ii) a feasible starting point exists, and (iii) -- when
    ``require_reference`` -- a derivative-based polish produces a reference
    solution.  The reference (f*, A*, multipliers) is computed ONCE here and
    stored in the cache, so experiments never recompute it and f* is identical
    across runs.  All three criteria are independent of how any solver performs."""
    import pycutest
    if names is None:
        names = find_cutest_candidates(n_max, m_max, verbose=verbose)
    problems, log = [], []
    for name in names:
        try:
            cp = import_cutest(name)
        except Exception as e:  # noqa
            log.append((name, f"import failed ({type(e).__name__}: {str(e)[:60]})"))
            continue
        if cp.n > n_max or cp.n < 1 or cp.m > m_max:
            log.append((name, f"skipped (size n={cp.n} m={cp.m})"))
            continue
        prob = cutest_to_problem(cp, bounds=bounds)
        if prob is None:
            log.append((name, "skipped (equalities/bounds/no inequalities)"))
            continue
        if prob.p > p_max:
            log.append((name, f"skipped (p={prob.p} > p_max)"))
            continue
        try:
            fx = prob.f(prob.x0)
            if not np.isfinite(fx) or not np.all(np.isfinite(prob.c(prob.x0))):  # noqa
                log.append((name, "skipped (non-finite at x0)"))
                continue
            q = make_feasible(prob, margin=margin)
        except Exception as e:  # noqa
            log.append((name, f"evaluation failed: {e}"))
            continue
        if q is None:
            log.append((name, "skipped (no feasible point found)"))
            continue
        ref = None
        if require_reference:
            from .baselines import reference_solution
            rng = np.random.default_rng(ref_seed)
            scale = 1.0 + np.abs(q.x0)
            extra = [q.x0 + 0.1 * scale * rng.standard_normal(q.n) for _ in range(max(0, n_ref_starts - 1))]
            try:
                ref = reference_solution(q, q.x0, starts=extra)
            except Exception as e:  # noqa
                ref = None
                log.append((name, f"skipped (reference failed: {e})"))
                continue
            if ref is None or ref.get("source") == "start":
                log.append((name, "skipped (no polished reference solution)"))
                continue
            q.meta["reference"] = dict(x=ref["x"].tolist(), f=ref["f"], A=list(ref["A"]),
                                       multipliers=np.asarray(ref["multipliers"]).tolist(),
                                       kkt_residual=ref["kkt_residual"],
                                       strict_compl=bool(ref["strict_compl"]), source=ref["source"],
                                       max_viol=ref["max_viol"])
        problems.append(q)
        rtxt = "" if ref is None else f" f*={ref['f']:.6g} |A*|={len(ref['A'])} sc={ref['strict_compl']}"
        log.append((name, f"OK n={q.n} p={q.p} shifted={q.meta.get('x0_shifted', False)}{rtxt}"))
        if verbose:
            print(log[-1])
    with open(cache, "w") as fh:
        json.dump([dict(name=q.name, n=q.n, p=q.p, x0=q.x0.tolist(), meta=q.meta, bounds=bounds)
                   for q in problems], fh, indent=1)
    with open(os.path.splitext(cache)[0] + "_log.txt", "w") as fh:
        for name, msg in log:
            fh.write(f"{name:12s} {msg}\n")
    if verbose:
        from collections import Counter
        def _reason(m):
            if m.startswith("OK"):
                return "OK"
            inner = m[m.find("(") + 1:m.rfind(")")] if "(" in m else m
            # collapse per-problem specifics (sizes, module names) into one bucket
            for key in ("size", "pycutest_cache_holder", "p="):
                if key in inner:
                    return m.split("(")[0].strip() + " (" + key.rstrip("=") + ")"
            return m.split("(")[0].strip() + " (" + inner + ")"
        tally = Counter(_reason(m) for _, m in log)
        print(f"selected {len(problems)} of {len(names)} candidates -> {cache}")
        for k, v in tally.most_common():
            print(f"  {v:4d}  {k}")
        if bounds == "exclude" and tally.get("skipped", 0) > len(problems):
            print("  hint: most small inequality-constrained CUTEst problems have variable bounds; "
                  "rerun with  --bounds as_constraints --rebuild-cache  to keep them.")
    return problems, log


def load_cutest_testset(cache="problems_cache.json"):
    """Re-create the problems from the cache (skips filtering, keeps x0)."""
    import pycutest
    with open(cache) as fh:
        entries = json.load(fh)
    problems = []
    for e in entries:
        cp = import_cutest(e["name"])
        prob = cutest_to_problem(cp, bounds=e.get("bounds", "exclude"))
        prob = prob.with_x0(np.array(e["x0"]))
        prob.meta.update(e.get("meta", {}))     # includes the cached "reference" dict, if any
        problems.append(prob)
    return problems


# --------------------------------------------------------------------------- #
# Hand-coded Hock-Schittkowski problems (inequality-only, no bounds) for
# testing without pycutest.
# --------------------------------------------------------------------------- #
def hs_problems():
    P = []
    # HS12: f* = -30 at (2,3); one constraint, active
    P.append(Problem("HS12", 2, 1, [0.0, 0.0],
                     lambda x: 0.5 * x[0] ** 2 + x[1] ** 2 - x[0] * x[1] - 7 * x[0] - 7 * x[1],
                     lambda x: [4 * x[0] ** 2 + x[1] ** 2 - 25],
                     lambda x: [x[0] - x[1] - 7, 2 * x[1] - x[0] - 7],
                     lambda x: [[8 * x[0], 2 * x[1]]]))
    # HS22: f* = 1 at (1,1); both constraints active
    P.append(Problem("HS22", 2, 2, [0.0, 0.0],
                     lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
                     lambda x: [x[0] + x[1] - 2, x[0] ** 2 - x[1]],
                     lambda x: [2 * (x[0] - 2), 2 * (x[1] - 1)],
                     lambda x: [[1, 1], [2 * x[0], -1]]))
    # HS29: f* = -16 sqrt(2); one constraint
    P.append(Problem("HS29", 3, 1, [1.0, 1.0, 1.0],
                     lambda x: -x[0] * x[1] * x[2],
                     lambda x: [x[0] ** 2 + 2 * x[1] ** 2 + 4 * x[2] ** 2 - 48],
                     lambda x: [-x[1] * x[2], -x[0] * x[2], -x[0] * x[1]],
                     lambda x: [[2 * x[0], 4 * x[1], 8 * x[2]]]))
    # HS43 (Rosen-Suzuki): f* = -44 at (0,1,2,-1); constraints 1 and 3 active
    def f43(x):
        return x[0] ** 2 + x[1] ** 2 + 2 * x[2] ** 2 + x[3] ** 2 - 5 * x[0] - 5 * x[1] - 21 * x[2] + 7 * x[3]

    def c43(x):
        return [x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[0] - x[1] + x[2] - x[3] - 8,
                x[0] ** 2 + 2 * x[1] ** 2 + x[2] ** 2 + 2 * x[3] ** 2 - x[0] - x[3] - 10,
                2 * x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + 2 * x[0] - x[1] - x[3] - 5]
    P.append(Problem("HS43", 4, 3, [0.0, 0.0, 0.0, 0.0], f43, c43))
    # HS100: f* = 680.6300573; constraints 1,4 active (of 4)
    def f100(x):
        return ((x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 + 3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6
                + 7 * x[5] ** 2 + x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6])

    def c100(x):
        return [-(127 - 2 * x[0] ** 2 - 3 * x[1] ** 4 - x[2] - 4 * x[3] ** 2 - 5 * x[4]),
                -(282 - 7 * x[0] - 3 * x[1] - 10 * x[2] ** 2 - x[3] + x[4]),
                -(196 - 23 * x[0] - x[1] ** 2 - 6 * x[5] ** 2 + 8 * x[6]),
                -(-4 * x[0] ** 2 - x[1] ** 2 + 3 * x[0] * x[1] - 2 * x[2] ** 2 - 5 * x[5] + 11 * x[6])]
    P.append(Problem("HS100", 7, 4, [1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0], f100, c100))
    # HS113: n=10, p=8, f* = 24.3062091; x0 feasible
    def f113(x):
        return (x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] + (x[2] - 10) ** 2 + 4 * (x[3] - 5) ** 2
                + (x[4] - 3) ** 2 + 2 * (x[5] - 1) ** 2 + 5 * x[6] ** 2 + 7 * (x[7] - 11) ** 2 + 2 * (x[8] - 10) ** 2
                + (x[9] - 7) ** 2 + 45)

    def c113(x):
        return [-(105 - 4 * x[0] - 5 * x[1] + 3 * x[6] - 9 * x[7]),
                -(-10 * x[0] + 8 * x[1] + 17 * x[6] - 2 * x[7]),
                -(8 * x[0] - 2 * x[1] - 5 * x[8] + 2 * x[9] + 12),
                -(-3 * (x[0] - 2) ** 2 - 4 * (x[1] - 3) ** 2 - 2 * x[2] ** 2 + 7 * x[3] + 120),
                -(-5 * x[0] ** 2 - 8 * x[1] - (x[2] - 6) ** 2 + 2 * x[3] + 40),
                -(-0.5 * (x[0] - 8) ** 2 - 2 * (x[1] - 4) ** 2 - 3 * x[4] ** 2 + x[5] + 30),
                -(-x[0] ** 2 - 2 * (x[1] - 2) ** 2 + 2 * x[0] * x[1] - 14 * x[4] + 6 * x[5]),
                -(3 * x[0] - 6 * x[1] - 12 * (x[8] - 8) ** 2 + 7 * x[9])]
    P.append(Problem("HS113", 10, 8, [2, 3, 5, 5, 1, 2, 7, 3, 6, 10], f113, c113))
    return P