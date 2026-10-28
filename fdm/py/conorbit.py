"""
CONORBIT (ORBITc.m) through the same counting Oracle every other solver uses.

The CONORBIT source code is NOT included in this repository, at the request
of its developers. To obtain it, please contact the authors of CONORBIT.
Place it in a `conorbit/` directory (containing an `ORBITc/` subdirectory)
at the top level of this repo, or set the CONORBIT_DIR environment variable
to its location. This module never reads or embeds any of it -- it only
points MATLAB's path at that directory by location. Running it still needs a local MATLAB install plus
the MATLAB Engine API for Python installed into this same Python
environment (``import matlab.engine`` must work) -- that part is not
shipped with the repo and can't be.

Two interop directions are in play, both through the MATLAB Engine API:
  * Python -> MATLAB to start a session and call ORBITc.
  * MATLAB -> Python, per evaluation, via MATLAB's own `pyenv`/`py.*` calling
    back into fdm.fdm.matlab_bridge.  That embedded Python interpreter does
    NOT share memory with this process, so the Oracle it drives is rebuilt
    inside it from a ProblemSpec rather than passed in by reference.

ORBITc requires a finite box [Low, Upp]; the 127-problem CUTEst set (built
with --bounds exclude) has none, so a generous box centered on x0 is
synthesized -- see `bounds_radius` below.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.realpath(__file__))  # resolve the fdm_pypkg/src symlink back to fdm/py
CONORBIT_DIR = os.environ.get("CONORBIT_DIR", os.path.join(_HERE, "..", "..", "conorbit"))
MATLAB_GLUE_DIR = os.path.join(_HERE, "..", "matlab")

_engine = None


def _quit_engine():
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        _engine = None


def _get_engine():
    """Lazily start (and cache) one MATLAB engine per process: startup takes
    several seconds to tens of seconds, so it is reused across every problem
    this process runs CONORBIT on, not restarted per problem."""
    global _engine
    if _engine is not None:
        return _engine
    import matlab.engine

    orbitc_dir = os.path.abspath(os.path.join(CONORBIT_DIR, "ORBITc"))
    if not os.path.isdir(orbitc_dir):
        raise RuntimeError(f"CONORBIT source not found at {orbitc_dir}. CONORBIT is not "
                           "distributed with this repo; contact the authors of CONORBIT to "
                           "obtain it, then place it in a conorbit/ directory at the repo top "
                           "level or set CONORBIT_DIR to a directory that has an ORBITc/ "
                           "subdirectory.")
    import fdm as _fdm_top   # fdm.__file__ sits in the editable install's sys.path entry
    src_dir = os.path.dirname(os.path.dirname(_fdm_top.__file__))

    eng = matlab.engine.start_matlab()
    eng.pyenv("Version", sys.executable, nargout=0)
    eng.eval(f"py.sys.path().append('{src_dir}');", nargout=0)
    eng.eval("py.importlib.import_module('fdm.fdm.matlab_bridge');", nargout=0)
    eng.addpath(orbitc_dir, nargout=0)
    eng.addpath(os.path.abspath(MATLAB_GLUE_DIR), nargout=0)
    _engine = eng
    atexit.register(_quit_engine)
    return eng


def run_conorbit(problem, budget_factor=100.0, budget_in="queries", max_queries=None,
                 spec_kind="cutest", spec_bounds="exclude",
                 rbftype="cubic", gamma=0.5, trnorm=2.0, margin=0.0, bounds_radius=20.0,
                 delta0=1.0):
    import matlab

    eng = _get_engine()
    n, p = problem.n, problem.p
    x0 = np.asarray(problem.x0, dtype=float)

    if budget_in == "queries":
        max_queries = int(max_queries) if max_queries else int(budget_factor * (n + 1) * (p + 1))
    else:
        max_queries = int(budget_factor * (n + 1)) * (p + 1)
    max_points_total = max(1, max_queries // (p + 1))
    nfmax = max(1, max_points_total - 1)   # the other point is the initial x0 evaluation (nfs=1)

    R = bounds_radius * (1.0 + np.abs(x0))
    Low, Upp = x0 - R, x0 + R
    minxrange = float(np.min(Upp - Low))
    npmax = 2 * n + 1
    maxdelta = 0.5 * minxrange
    # Same starting trust-region radius as the other solvers being compared
    # (FDM's Options.delta0 defaults to 1.0, unscaled
    # by problem/box size), not a fraction of the synthetic CONORBIT-only box.
    delta = min(delta0, maxdelta)
    gtol = 1e-4 * minxrange
    constrtol = 1e-6 * minxrange

    t0 = time.time()
    eng.workspace["mb_X0row"] = matlab.double(x0.tolist())
    eng.workspace["mb_maxq"] = float(max_queries)
    eng.eval(f"py.fdm.fdm.matlab_bridge.mb_init('{spec_kind}', '{problem.name}', mb_X0row, "
             f"'{spec_bounds}', mb_maxq);", nargout=0)
    # evaluate the starting point once, through the oracle, before ORBITc runs (nfs=1)
    eng.eval("mb_F0 = double(py.fdm.fdm.matlab_bridge.mb_eval_row(mb_X0row));", nargout=0)

    eng.workspace["mb_rbftype"] = str(rbftype)
    eng.workspace["mb_gamma"] = float(gamma)
    eng.workspace["mb_n"] = float(n)
    eng.workspace["mb_nc"] = float(p)
    eng.workspace["mb_nfmax"] = float(nfmax)
    eng.workspace["mb_npmax"] = float(npmax)
    eng.workspace["mb_delta"] = float(delta)
    eng.workspace["mb_maxdelta"] = float(maxdelta)
    eng.workspace["mb_trnorm"] = float(trnorm)
    eng.workspace["mb_gtol"] = float(gtol)
    eng.workspace["mb_Low"] = matlab.double(Low.tolist())
    eng.workspace["mb_Upp"] = matlab.double(Upp.tolist())
    eng.workspace["mb_nfs"] = 1.0
    eng.workspace["mb_xkin"] = 1.0
    eng.workspace["mb_margin"] = float(margin)
    eng.workspace["mb_constrtol"] = float(constrtol)

    eng.eval("[mb_X,mb_F,mb_xkin,mb_nf] = ORBITc('mb_func', mb_rbftype, mb_gamma, mb_n, mb_nc, "
             "mb_nfmax, mb_npmax, mb_delta, mb_maxdelta, mb_trnorm, mb_gtol, mb_Low, mb_Upp, "
             "mb_nfs, mb_X0row, mb_F0, mb_xkin, mb_margin, mb_constrtol);", nargout=0)

    results_path = tempfile.mktemp(prefix="conorbit_", suffix=".json")
    try:
        eng.eval(f"py.fdm.fdm.matlab_bridge.mb_write_results('{results_path}');", nargout=0)
        with open(results_path) as fh:
            res = json.load(fh)
    finally:
        if os.path.exists(results_path):
            os.remove(results_path)

    summary = res["summary"]
    best_x = np.asarray(res["best_x"]) if res["best_x"] is not None else x0
    status = "budget" if summary["n_f"] + summary["n_c"] >= max_queries else "done"
    return dict(solver="CONORBIT", status=status, x=best_x, f=float(res["best_f"]),
               c=problem.c(best_x), wall=time.time() - t0,
               trace=[tuple(t) for t in res["trace"]], oracle=summary, history=[])
