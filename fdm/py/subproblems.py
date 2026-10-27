"""
Subproblems, solved with Clarabel (QP / LP, nonnegative cone) with a SciPy
fallback if clarabel is not installed.

Variables for the Topkis-Veinott (TV) subproblem (3) of the paper:
    z = (s, tau) in R^{n+1}
    min  tau + 1/2 s'H s
    s.t. g_f' s - tau         <= 0
         c_i + g_i' s - tau   <= 0    i in A_k
         -Delta <= s <= Delta         (inf-norm trust region)

The criticality measure chi_k (eq. chik_def) is the same problem with H = 0 and
Delta = 1, and chi_k = -tau*.

The standard SQP subproblem (2), used as a comparison variant:
    min  g_f' s + 1/2 s'H s
    s.t. c_i + g_i' s <= 0   i in A_k,   -Delta <= s <= Delta
which may be infeasible.
"""
from __future__ import annotations

import numpy as np

try:
    import clarabel  # type: ignore
    import scipy.sparse as sp
    HAVE_CLARABEL = True
except Exception:  # pragma: no cover
    HAVE_CLARABEL = False

_SETTINGS = None


def _clarabel_settings():
    global _SETTINGS
    if _SETTINGS is None:
        s = clarabel.DefaultSettings()
        s.verbose = False
        s.tol_gap_abs = 1e-10
        s.tol_gap_rel = 1e-10
        s.tol_feas = 1e-10
        s.max_iter = 200
        _SETTINGS = s
    return _SETTINGS


def solve_qp(P, q, A, b):
    """min 1/2 z'Pz + q'z  s.t. A z <= b.  Returns (z, duals, status) with
    status in {'solved', 'infeasible', 'failed'}."""
    if HAVE_CLARABEL:
        Psp = sp.triu(sp.csc_matrix(P)).tocsc()
        Asp = sp.csc_matrix(A)
        cones = [clarabel.NonnegativeConeT(A.shape[0])]
        solver = clarabel.DefaultSolver(Psp, np.asarray(q, float), Asp, np.asarray(b, float), cones,
                                        _clarabel_settings())
        sol = solver.solve()
        st = str(sol.status)
        if "Solved" in st or "AlmostSolved" in st:
            return np.asarray(sol.x), np.asarray(sol.z), "solved"
        if "Infeasible" in st:
            return None, None, "infeasible"
        return None, None, "failed"
    from .fallbacks import solve_qp_scipy
    return solve_qp_scipy(P, q, A, b)


def _tv_matrices(g_f, G_A, c_A, H, Delta, margin=0.0):
    n = len(g_f)
    mA = len(c_A)
    rows = 1 + mA + 2 * n
    A = np.zeros((rows, n + 1))
    b = np.zeros(rows)
    A[0, :n] = g_f
    A[0, n] = -1.0
    if mA:
        A[1:1 + mA, :n] = G_A
        A[1:1 + mA, n] = -1.0
        b[1:1 + mA] = -c_A - margin      # inner-boundary offset (NOWPAC-style)
    A[1 + mA:1 + mA + n, :n] = np.eye(n)
    b[1 + mA:1 + mA + n] = Delta
    A[1 + mA + n:, :n] = -np.eye(n)
    b[1 + mA + n:] = Delta
    P = np.zeros((n + 1, n + 1))
    if H is not None:
        P[:n, :n] = H
    q = np.zeros(n + 1)
    q[n] = 1.0
    return P, q, A, b


def tv_step(g_f, G_A, c_A, H, Delta, margin=0.0):
    """Solve the TV subproblem.  Returns dict(s, tau, yf, yc, status).

    ``margin`` >= 0 tightens the model constraints to
    ``m^{c_i}(x_k+s) <= tau - margin``, an inner-boundary offset that
    compensates for the O(Delta^2) error of the linear constraint models, so
    that model-feasible steps are more often truly feasible.  (s,tau)=(0,margin)
    remains feasible, so the subproblem is still solvable."""
    n = len(g_f)
    P, q, A, b = _tv_matrices(g_f, G_A, c_A, H, Delta, margin)
    z, duals, status = solve_qp(P, q, A, b)
    if z is None:
        return dict(s=None, tau=None, status=status)
    out = dict(s=z[:n], tau=float(z[n]), status=status, yf=None, yc=None)
    if duals is not None:
        out["yf"] = float(duals[0])
        out["yc"] = np.asarray(duals[1:1 + len(c_A)])
    return out


def criticality_measure(g_f, G_A, c_A):
    """chi_k(x_k, A) = -min_{s in B(0,1)} max{g_f's, max_i (c_i + g_i's)}."""
    n = len(g_f)
    if len(c_A) == 0:
        return float(np.sum(np.abs(g_f)))          # closed form (Lemma: chi_k(x, empty) = ||g||_1)
    P, q, A, b = _tv_matrices(g_f, G_A, c_A, None, 1.0)
    z, _, status = solve_qp(P, q, A, b)
    if z is None:
        # (s,tau)=(0,0) is always feasible; a failure here is numerical
        return 0.0
    return max(0.0, -float(z[n]))


def sqp_step(g_f, G_A, c_A, H, Delta, margin=0.0):
    """Standard TR-SQP subproblem (may be infeasible)."""
    n = len(g_f)
    mA = len(c_A)
    A = np.zeros((mA + 2 * n, n))
    b = np.zeros(mA + 2 * n)
    if mA:
        A[:mA] = G_A
        b[:mA] = -c_A - margin
    A[mA:mA + n] = np.eye(n)
    b[mA:mA + n] = Delta
    A[mA + n:] = -np.eye(n)
    b[mA + n:] = Delta
    z, duals, status = solve_qp(H, g_f, A, b)
    if z is None:
        return dict(s=None, status=status)
    out = dict(s=z, status=status, yf=1.0, yc=None)
    if duals is not None:
        out["yc"] = np.asarray(duals[:mA])
    return out