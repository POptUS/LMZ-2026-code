"""
Model management on top of ``formquad``.

For every component j (0 = f, 1+i = c_i) we keep a quadratic model centered at
the point where it was last built,

    m_j(x_c + d) = F[x_c, j] + g_j' d + 1/2 d' H_j d,

built exactly as in ``pounders``: formquad is applied to the residuals
F - F[center] - 1/2 d'H_j d, and H_j accumulates ``Hresdel``.  The algorithm
in the paper only uses the *linear* part (g_j, plus the exact value at the
incumbent); H_0 (the f-Hessian) is what we feed the subproblem as H_k.

Lazy updates: ``update(comps, ...)`` makes the models of the requested
components fully linear (formquad-``valid``) on B(x_k, delta) and evaluates
geometry points **only for those components**.  Models of other components
are left untouched (stale gradient, exact value at x_k), which is the
"refresh models outside A_k only occasionally" mechanism of the paper.
"""
from __future__ import annotations

import importlib
import warnings

import numpy as np

_FORMQUAD = None


def get_formquad():
    """The model builder in use.  Resolution order:
       1. local_formquad.py next to the driver (drop your own copy there)
       2. ibcdfo.pounders.formquad
       3. the stub in fdm.fdm.fallbacks (smoke tests only)"""
    global _FORMQUAD
    if _FORMQUAD is not None:
        return _FORMQUAD
    for mod, attr in [("local_formquad", "formquad"),
                      ("ibcdfo.pounders.formquad", "formquad"),
                      ("ibcdfo.pounders", "formquad")]:
        try:
            _FORMQUAD = getattr(importlib.import_module(mod), attr)
            return _FORMQUAD
        except Exception:
            continue
    from .fallbacks import formquad_fallback
    warnings.warn("formquad not importable (tried local_formquad.py and ibcdfo): using the stub in "
                  "fdm.fdm.fallbacks -- fine for smoke tests, NOT for the paper's experiments.")
    _FORMQUAD = formquad_fallback
    return _FORMQUAD


def using_fallback():
    """True if the model builder in use is the stub from fdm.fdm.fallbacks."""
    from .fallbacks import formquad_fallback
    return get_formquad() is formquad_fallback


def set_formquad(fn):
    """Point the model manager at your own formquad (e.g. a modified local copy)."""
    global _FORMQUAD
    _FORMQUAD = fn


def default_par(n):
    # pounders defaults: [sqrt(n), max(10, sqrt(n)), 1e-3, 1e-3, 0]
    return np.array([np.sqrt(n), max(10.0, np.sqrt(n)), 1e-3, 1e-3, 0.0])


class ModelManager:
    def __init__(self, oracle, np_max=None, Par=None, formquad=None, geometry_feasibility_flip=True,
                 par_c1_mult=1.0, par_c2_mult=1.0, max_geom=-1, linear_constraints=False,
                 max_sweeps=4, geom_delta_floor=0.0):
        self.oracle = oracle
        self.n = oracle.n
        self.m = 1 + oracle.p
        self.np_max = int(np_max) if np_max is not None else 2 * self.n + 1
        self.Par = np.array(Par, dtype=float) if Par is not None else default_par(self.n)
        self.Par[0] *= par_c1_mult      # reuse radius: points within Par[0]*delta are candidates
        self.Par[1] *= par_c2_mult
        self.max_geom = int(max_geom)
        # floor on the delta used to build models / admit reuse candidates, decoupling the
        # model-building aperture from the live (possibly tiny) trust-region radius.  0.0
        # recovers the old behavior (aperture == live delta).
        self.geom_delta_floor = float(geom_delta_floor)
        self._want_linear = bool(linear_constraints)
        self.max_sweeps = int(max_sweeps)
        self.formquad = formquad or get_formquad()
        self.flip = geometry_feasibility_flip
        self.g = np.zeros((self.n, self.m))
        self.H = np.zeros((self.n, self.n, self.m))
        self.center = np.full(self.m, -1, dtype=int)      # point index where model j was built
        self.built_iter = np.full(self.m, -1, dtype=int)  # iteration at which model j was last built
        self.built_delta = np.full(self.m, np.nan)
        self.valid = np.zeros(self.m, dtype=bool)
        # True: fit constraint gradients by pooled least squares over several
        # affinely independent sets, instead of taking the gradient of
        # formquad's minimum-Frobenius-norm quadratic model
        self.linear_constraints = self._want_linear
        self.n_geometry_points = 0
        self.n_formquad_calls = 0
        self.n_linear_sweeps = 0

    # ------------------------------------------------------------------ #

    def _delta_eff(self, delta):
        """Effective delta used to build models / admit reuse candidates: the live
        trust-region radius floored at ``geom_delta_floor``, so model-building
        reuse doesn't collapse when the criticality loop shrinks delta."""
        return max(delta, self.geom_delta_floor)

    def _linear_gradient(self, rows, j, xk, delta, max_sweeps=None):
        """Matt Menickelly's suggestion: a best-fit LINEAR model for component j.

        ``formquad`` returns the gradient of a minimum-Frobenius-norm QUADRATIC
        fit, so the quadratic features compete with the linear ones for the same
        data.  Only the linear part of a constraint model enters
        the step subproblem, so we instead collect several affinely
        independent sets by repeated phase-1 sweeps of ``formquad`` (vf=True, so
        no model is built), pool their displacements, and solve the resulting
        overdetermined system by least squares.

        Returns (g, ok): the gradient and whether at least one affinely
        independent set was found.
        """
        max_sweeps = self.max_sweeps if max_sweeps is None else max_sweeps
        delta_eff = self._delta_eff(delta)
        X = self.oracle.X
        F = self.oracle.F
        avail = list(rows)
        used = []
        for _ in range(max_sweeps):
            if len(avail) < self.n + 1:
                break
            sub = np.array(avail)
            pos = int(np.flatnonzero(sub == xk)[0])
            self.n_formquad_calls += 1
            _, mp, valid, _, _, Mind = self.formquad(
                X[sub], F[sub][:, [j]] - F[xk, j], delta_eff, pos, self.np_max, self.Par, True)
            if not valid or mp < self.n:
                break
            picked = [int(sub[i]) for i in np.atleast_1d(np.asarray(Mind)).ravel().astype(int)]
            picked = [i for i in picked if i != xk]
            if not picked:
                break
            used.extend(picked)
            avail = [i for i in avail if i not in set(picked)]
            self.n_linear_sweeps += 1
        if not used:
            return None, False
        D = X[used] - X[xk]
        y = F[used, j] - F[xk, j]
        g, *_ = np.linalg.lstsq(D, y, rcond=None)
        return (g, True) if np.all(np.isfinite(g)) else (None, False)

    def _residuals(self, rows, comps, xk):
        X = self.oracle.X
        F = self.oracle.F
        D = X[rows] - X[xk]
        Res = np.empty((len(rows), len(comps)))
        for a, j in enumerate(comps):
            quad = 0.5 * np.einsum("ij,jk,ik->i", D, self.H[:, :, j], D)
            Res[:, a] = F[rows, j] - F[xk, j] - quad
        return Res

    def _groups(self, comps, xk):
        """Group components by identical known-row masks (so f and the active
        constraints, which are co-evaluated, share one formquad call)."""
        F = self.oracle.F
        groups = {}
        for j in comps:
            mask = np.isfinite(F[:, j])       # non-finite evaluations are excluded from model building
            if not mask[xk]:
                raise RuntimeError(f"component {j} unknown at center {xk}")
            groups.setdefault(mask.tobytes(), []).append(j)
        return [(np.flatnonzero(np.frombuffer(k, dtype=bool)), js) for k, js in groups.items()]

    def _predicted_infeasible(self, xk, d):
        """Heuristic: does the current linear constraint model predict x_k+d infeasible?"""
        F = self.oracle.F
        c = F[xk, 1:]
        G = self.g[:, 1:]
        pred = c + G.T @ d
        return np.any(pred > 0)

    def update(self, comps, xk, delta, iteration, max_rounds=8):
        """(see class docstring) ``self.max_geom`` caps geometry points per call."""
        """Make models of ``comps`` valid on B(x_k, delta); evaluate geometry
        points (charged only for ``comps``) as needed.  Returns True if all
        requested models ended up formquad-valid."""
        comps = sorted(set(int(j) for j in comps))
        if not comps:
            return True
        delta_eff = self._delta_eff(delta)
        X0 = self.oracle.x(xk)
        spent = 0
        for _round in range(max_rounds):
            pending_dirs = None
            all_valid = True
            for rows, js in self._groups(comps, xk):
                Res = self._residuals(rows, js, xk)
                pos = int(np.flatnonzero(rows == xk)[0])
                self.n_formquad_calls += 1
                Mdir, mp, valid, Gres, Hresdel, Mind = self.formquad(
                    self.oracle.X[rows], Res, delta_eff, pos, self.np_max, self.Par, False)
                if mp < self.n or not valid:
                    all_valid = False
                    if pending_dirs is None:
                        Mdir = np.atleast_2d(np.asarray(Mdir))
                        pending_dirs = Mdir[: self.n - mp]
                    continue
                Gres = np.asarray(Gres, dtype=float).reshape(self.n, len(js))
                Hresdel = np.asarray(Hresdel, dtype=float).reshape(self.n, self.n, len(js))
                if not (np.all(np.isfinite(Gres)) and np.all(np.isfinite(Hresdel))):
                    Gres = np.nan_to_num(Gres, nan=0.0, posinf=0.0, neginf=0.0)
                    Hresdel = np.zeros_like(Hresdel)
                for a, j in enumerate(js):
                    if self.linear_constraints and j > 0:
                        gl, ok = self._linear_gradient(rows, j, xk, delta)
                        self.g[:, j] = gl if ok else Gres[:, a]
                        # the quadratic part is not used for constraints
                    else:
                        self.g[:, j] = Gres[:, a]
                        self.H[:, :, j] = self.H[:, :, j] + Hresdel[:, :, a]
                    self.center[j] = xk
                    self.built_iter[j] = iteration
                    self.built_delta[j] = delta
                    self.valid[j] = True
            if all_valid:
                return True
            if pending_dirs is None or len(pending_dirs) == 0:
                break
            # evaluate geometry points for the requested components only
            if self.max_geom >= 0:
                pending_dirs = pending_dirs[:max(0, self.max_geom - spent)]
                if len(pending_dirs) == 0:
                    break
            for d in pending_dirs:
                d = delta_eff * np.asarray(d, dtype=float)
                if self.flip and self._predicted_infeasible(xk, d) and not self._predicted_infeasible(xk, -d):
                    d = -d
                idx = self.oracle.add_point(X0 + d)
                self.oracle.eval_components(idx, comps)
                self.n_geometry_points += 1
                spent += 1
                if self.oracle.budget_exhausted():
                    return False
        # last resort: least-squares linear fit on nearby points, marked invalid
        self._lstsq_fallback(comps, xk, delta, iteration)
        return False

    def _lstsq_fallback(self, comps, xk, delta, iteration):
        X = self.oracle.X
        F = self.oracle.F
        D = X - X[xk]
        near = np.linalg.norm(D, axis=1) <= self.Par[1] * self._delta_eff(delta)
        for j in comps:
            if self.valid[j] and self.center[j] == xk and self.built_iter[j] == iteration:
                continue
            rows = np.flatnonzero(near & np.isfinite(F[:, j]))
            rows = rows[rows != xk]
            if len(rows) == 0:
                continue
            g = np.linalg.lstsq(D[rows], F[rows, j] - F[xk, j], rcond=None)[0]
            self.g[:, j] = g
            self.center[j] = xk
            self.built_iter[j] = iteration
            self.built_delta[j] = delta
            self.valid[j] = False

    # ------------------------------------------------------------------ #
    def linear_data(self, xk):
        """Return (g_f, G_c (p x n), c(x_k)) for the current incumbent.  Values
        at x_k are exact (all components are known at every incumbent)."""
        row = self.oracle.row(xk)
        return self.g[:, 0].copy(), self.g[:, 1:].T.copy(), row[1:].copy()

    def hessian(self, mode="fmodel", multipliers=None, zeta_low=1e-8, zeta_high=1e8):
        """H_k for the subproblem: PD projection of the f-model Hessian (optionally
        plus multiplier-weighted constraint Hessians)."""
        if mode == "identity":
            return np.eye(self.n) * zeta_low
        H = 0.5 * (self.H[:, :, 0] + self.H[:, :, 0].T)
        if mode == "lagrangian" and multipliers is not None:
            for i, lam in multipliers.items():
                Hi = self.H[:, :, 1 + i]
                H = H + lam * 0.5 * (Hi + Hi.T)
        if not np.all(np.isfinite(H)) or np.max(np.abs(H)) > 1e100:
            self.H[:, :, 0] = 0.0            # reset a blown-up accumulated Hessian
            return np.eye(self.n) * zeta_low
        try:
            w, V = np.linalg.eigh(H)
        except np.linalg.LinAlgError:
            self.H[:, :, 0] = 0.0
            return np.eye(self.n) * zeta_low
        w = np.clip(w, zeta_low, zeta_high)
        return (V * w) @ V.T
