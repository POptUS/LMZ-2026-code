"""
NOWPAC (via Sandia's DAKOTA toolkit) through the same counting Oracle every
other solver uses.

Unlike CONORBIT (one shared, slow-to-start MATLAB engine per process, with an
in-process embedded-interpreter callback via ``matlab_bridge.py``), DAKOTA
runs as its own OS process and forks a brand-new subprocess per evaluation --
there is no shared memory to put a module-level Oracle in. So the live
Oracle, built directly from the already-loaded ``problem`` (no ProblemSpec
round-trip needed), is made reachable over a short-lived Unix domain socket
by the per-evaluation driver subprocess (``nowpac_driver.py``).

NOWPAC requires a feasible starting point (no feasibility-restoration phase)
and supports only one-sided inequality constraints; this project's own
``c_i(x) <= 0`` convention needs no sign flip (confirmed against
NOWPACOptimizer.cpp's allocate_constraints(): lower_bounds=-1e50,
upper_bounds=0.0 yields exactly that).

NOWPAC internally rescales the design box to [-1,1]^n *per axis*, and its
trust_region.initial_size/minimum_size are a single scalar fraction of that
normalized half-width -- unlike CONORBIT's per-axis box, an isotropic cube is
used here so that one scalar fraction stays isotropic in absolute problem
units, matching the delta0/delta_lower convention shared by every other
solver in this comparison.

Needs a DAKOTA build with -DHAVE_NOWPAC=ON (not in DAKOTA's official
precompiled binaries); set DAKOTA_DIR (default ~/dakota) or DAKOTA_EXE if
installed elsewhere.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from .oracle import Oracle
from .problems import make_feasible

_HERE = os.path.dirname(os.path.realpath(__file__))   # resolve the fdm_pypkg/src symlink back to fdm/py
_DRIVER = os.path.join(_HERE, "nowpac_driver.py")
DAKOTA_DIR = os.environ.get("DAKOTA_DIR", os.path.join(os.path.expanduser("~"), "dakota"))

_INPUT_TEMPLATE = """\
method
  nowpac
    max_function_evaluations = {nfmax}
    trust_region
      initial_size = {init_size:.10g}
      minimum_size = {min_size:.10g}

variables
  continuous_design = {n}
    initial_point   {x0}
    lower_bounds    {lo}
    upper_bounds    {up}

interface
  analysis_drivers = '{driver_cmd}'
    fork
      parameters_file = 'params.in'
      results_file    = 'results.out'

responses
  objective_functions = 1
  nonlinear_inequality_constraints = {p}
  nonlinear_inequality_lower_bounds = {nlb}
  nonlinear_inequality_upper_bounds = {nub}
  no_gradients
  no_hessians
"""


class _OracleServer:
    """Runs the shared Oracle on a background thread, reachable over a Unix
    domain socket by the driver subprocess DAKOTA forks per evaluation (see
    nowpac_driver.py). One request/response pair at a time -- DAKOTA's fork
    interface here is strictly serial -- so no multiplexing is needed. The
    socket path is unique per call (tempfile.mktemp), so concurrent
    run_nowpac calls under --jobs N never collide."""

    def __init__(self, oracle):
        self.oracle = oracle
        self.sock_path = tempfile.mktemp(prefix="nowpac_", suffix=".sock")
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        self._srv.listen(1)
        self._srv.settimeout(0.5)
        self._stop = threading.Event()
        self._last_row = oracle.row(0).tolist()   # x0's row; same "nothing happened" freeze as run_cobyla
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    buf = b""
                    while not buf.endswith(b"\n"):
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        buf += chunk
                    x = np.asarray(json.loads(buf.decode())["x"], dtype=float)
                    if self.oracle.budget_exhausted():
                        row = self._last_row
                    else:
                        idx = self.oracle.add_point(x)
                        self.oracle.eval_all(idx)
                        row = self.oracle.row(idx).tolist()
                        self._last_row = row
                    conn.sendall((json.dumps({"row": row}) + "\n").encode())
                except Exception as e:  # noqa: BLE001 -- report back to the driver, never crash the server
                    try:
                        conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
                    except Exception:
                        pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._srv.close()
        if os.path.exists(self.sock_path):
            os.remove(self.sock_path)


def run_nowpac(problem, budget_factor=100.0, budget_in="queries", max_queries=None,
               delta0=1.0, delta_lower=1e-3, bounds_radius=20.0, dakota_exe=None):
    name = problem.name
    problem = make_feasible(problem)
    if problem is None:
        raise RuntimeError(f"NOWPAC requires a feasible starting point; "
                           f"make_feasible could not find one for {name!r}")
    n, p = problem.n, problem.p
    x0 = np.asarray(problem.x0, dtype=float)

    if budget_in == "queries":
        max_queries = int(max_queries) if max_queries else int(budget_factor * (n + 1) * (p + 1))
    else:
        max_queries = int(budget_factor * (n + 1)) * (p + 1)
    max_points_total = max(1, max_queries // (p + 1))
    nfmax = max(1, max_points_total - 1)   # the other point is the x0 evaluation below

    oracle = Oracle(problem, max_queries=max_queries)
    oracle.eval_all(oracle.add_point(x0))   # evaluate x0 once up front, same as CONORBIT's nfs=1 point

    # A single scalar radius (a cube), not CONORBIT's per-axis box: NOWPAC's
    # trust-region size is one scalar fraction of DAKOTA's OWN per-axis
    # [-1,1] rescaling, so an unequal per-axis box here would turn that one
    # fraction into an anisotropic absolute-unit trust region -- silently
    # breaking the isotropic-radius convention delta0/delta_lower have
    # everywhere else in this comparison.
    Rmax = bounds_radius * (1.0 + float(np.max(np.abs(x0))))
    Low, Upp = x0 - Rmax, x0 + Rmax
    init_size = min(1.0, delta0 / Rmax)
    min_size = min(1.0, delta_lower / Rmax)

    dakota_exe = dakota_exe or os.environ.get("DAKOTA_EXE", os.path.join(DAKOTA_DIR, "bin", "dakota"))
    if not os.path.exists(dakota_exe):
        raise RuntimeError(f"dakota executable not found at {dakota_exe!r}; build it per "
                           f"run_experiments.py's NOWPAC docstring entry, or set DAKOTA_DIR/DAKOTA_EXE.")

    workdir = tempfile.mkdtemp(prefix="nowpac_run_")
    server = _OracleServer(oracle)
    try:
        with open(os.path.join(workdir, "dakota.in"), "w") as fh:
            fh.write(_INPUT_TEMPLATE.format(
                nfmax=nfmax, init_size=init_size, min_size=min_size, n=n, p=p,
                x0=" ".join(f"{v:.17g}" for v in x0),
                lo=" ".join(f"{v:.17g}" for v in Low),
                up=" ".join(f"{v:.17g}" for v in Upp),
                driver_cmd=f"{sys.executable} {_DRIVER}",
                nlb=" ".join(["-1.e+50"] * p),
                nub=" ".join(["0.0"] * p),
            ))
        env = dict(os.environ, NOWPAC_ORACLE_SOCKET=server.sock_path, DAKOTA_DIR=DAKOTA_DIR)
        t0 = time.time()
        proc = subprocess.run([dakota_exe, "-i", "dakota.in"], cwd=workdir, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wall = time.time() - t0
    finally:
        server.close()   # closed before cleanup: a driver blocked mid-request gets a clean socket
                         # error instead of a hang, even if `dakota` itself crashed
        shutil.rmtree(workdir, ignore_errors=True)

    if oracle.budget_exhausted():
        status = "budget"
    elif proc.returncode != 0:
        status = "crashed"
    else:
        status = "done"
    best_x = oracle.best_x if oracle.best_x is not None else x0
    return dict(solver="NOWPAC", status=status, x=best_x, f=float(oracle.best_f),
               c=problem.c(best_x), wall=wall, trace=list(oracle.trace), oracle=oracle.summary(),
               history=[])
