"""
Fallbacks used only when the real dependencies are missing.

* ``formquad_fallback``  -- a formquad-compatible model builder (same call
  signature and return values as ``ibcdfo.pounders.formquad``) implementing
  affine-independence pivoting + linear interpolation + minimum-Frobenius-norm
  quadratic correction.  It is a *test stub*: the experiments in the paper
  should use the real ``formquad`` (see ``fdm.fdm.models.get_formquad``).

* ``solve_qp_scipy`` -- solves  min 1/2 z'Pz + q'z  s.t.  A z <= b  with
  SciPy (HiGHS for LPs, SLSQP for QPs).  Used only if ``clarabel`` is absent.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# formquad-compatible fallback
# --------------------------------------------------------------------------- #
def formquad_fallback(X, F, delta, xk_in, np_max, Par, vf):
    """
    Parameters mirror ``ibcdfo.pounders.formquad``:
      X      (nf x n)  evaluated points
      F      (nf x m)  values (or residuals) of the m components at X
      delta  trust-region radius
      xk_in  index of the center point in X
      np_max maximum number of interpolation points (>= n+1)
      Par    [c1, c2, theta1, theta2, flag]  (only c1, c2, theta1 used here)
      vf     if True only check validity (Mdir/mp/valid meaningful)

    Returns
      Mdir    ((n-mp) x n) unit directions that would complete the affinely
              independent set; empty if mp == n.  Unscaled -- matches
              ibcdfo.pounders.formquad's convention that the caller scales by
              delta before evaluating a new point.
      mp      number of affinely independent directions found (<= n)
      valid   mp == n
      Gres    (n x m) model gradients at the center
      Hresdel (n x n x m) model Hessians
      Mind    indices (into X) of the interpolation points used
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    nf, n = X.shape
    m = F.shape[1]
    c1, c2, theta1 = Par[0], Par[1], Par[2]

    D = (X - X[xk_in]) / max(delta, 1e-300)
    dist = np.linalg.norm(D, axis=1)
    order = np.argsort(dist)
    cand = [i for i in order if i != xk_in and dist[i] <= c1]

    # greedy affine-independence pivoting (Gram-Schmidt with threshold)
    Q = np.zeros((0, n))
    Mind = [xk_in]
    for i in cand:
        d = D[i]
        r = d - Q.T @ (Q @ d) if Q.shape[0] else d.copy()
        if np.linalg.norm(r) >= theta1:
            Q = np.vstack([Q, r / np.linalg.norm(r)])
            Mind.append(i)
        if Q.shape[0] == n:
            break
    mp = Q.shape[0]
    valid = mp == n

    Gres = np.zeros((n, m))
    Hresdel = np.zeros((n, n, m))
    if not valid:
        # orthonormal complement of span(Q)
        if mp == 0:
            comp = np.eye(n)
        else:
            _, _, Vt = np.linalg.svd(Q, full_matrices=True)
            comp = Vt[mp:]
        Mdir = comp
        return Mdir, mp, valid, Gres, Hresdel, np.array(Mind)

    Mdir = np.zeros((0, n))
    if vf:
        return Mdir, mp, valid, Gres, Hresdel, np.array(Mind)

    # additional points for MFN quadratic (nearest first, within c2*delta)
    extra = [i for i in order if i not in Mind and dist[i] <= c2]
    extra = extra[: max(0, int(np_max) - (n + 1))]
    idx = Mind + extra
    Dsel = D[idx] * delta                  # unscaled displacements
    Fsel = F[idx] - F[xk_in]
    npts = len(idx)

    if npts == n + 1:
        # pure linear interpolation
        Dl = Dsel[1:]
        try:
            G = np.linalg.solve(Dl, Fsel[1:])
        except np.linalg.LinAlgError:
            G = np.linalg.lstsq(Dl, Fsel[1:], rcond=None)[0]
        Gres = G
    else:
        # minimum Frobenius norm quadratic: m(d) = g'd + 1/2 d'Hd,
        # H = sum_i lam_i d_i d_i', with KKT system [Q L; L' 0]
        Qm = 0.5 * (Dsel @ Dsel.T) ** 2
        L = Dsel
        K = np.block([[Qm, L], [L.T, np.zeros((n, n))]])
        rhs = np.vstack([Fsel, np.zeros((n, m))])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        lam = sol[:npts]
        Gres = sol[npts:]
        for j in range(m):
            Hresdel[:, :, j] = (Dsel.T * lam[:, j]) @ Dsel
    return Mdir, mp, valid, Gres, Hresdel, np.array(idx)


# --------------------------------------------------------------------------- #
# QP fallback (SciPy)
# --------------------------------------------------------------------------- #
def solve_qp_scipy(P, q, A, b, z0=None):
    """min 1/2 z'Pz + q'z  s.t.  A z <= b.  Returns (z, duals, status)."""
    from scipy.optimize import linprog, minimize

    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    nz = len(q)
    if not np.any(P):
        res = linprog(q, A_ub=A, b_ub=b, bounds=[(None, None)] * nz, method="highs")
        if res.status != 0:
            return None, None, "infeasible" if res.status == 2 else "failed"
        duals = -res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(len(b))
        return res.x, duals, "solved"
    z0 = np.zeros(nz) if z0 is None else z0
    cons = [{"type": "ineq", "fun": lambda z: b - A @ z, "jac": lambda z: -A}]
    res = minimize(lambda z: 0.5 * z @ P @ z + q @ z, z0, jac=lambda z: P @ z + q,
                   constraints=cons, method="SLSQP",
                   options={"ftol": 1e-14, "maxiter": 500})
    if not res.success and np.max(A @ res.x - b) > 1e-8:
        return None, None, "infeasible"
    return res.x, None, "solved"
