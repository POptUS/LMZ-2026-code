"""
Algorithm 1 (Active-set TR DFO with criticality measure) and its comparison
variants.

Variants (see ``Options``):
  subproblem : 'tv'  -> Topkis-Veinott subproblem (3)      [proposed]
               'sqp' -> standard TR-SQP subproblem (2)      [comparison]
  active_set : True  -> A_k from eq. (active_subproblem)    [proposed]
               False -> A_k = {1..p} always
  lazy       : True  -> models of constraints outside A_k are NOT rebuilt
                        (refreshed every ``refresh_every`` iterations)
               False -> all p constraint models rebuilt every iteration
                        (the algorithm exactly as analyzed)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .models import ModelManager
from .oracle import Oracle
from .subproblems import criticality_measure, sqp_step, tv_step


@dataclass
class Options:
    subproblem: str = "tv"          # 'tv' | 'sqp'
    active_set: bool = True
    lazy: bool = True
    refresh_every: int = 10         # lazy mode: rebuild an inactive constraint model at least this often
    refresh_rule: str = "counter"   # "counter" = the fixed counter above;
                                    # "margin"  = rebuild only when the stale gradient could
                                    #             plausibly have brought the constraint into A_k,
                                    #             with refresh_every as a backstop
    refresh_kappa: float = 1.0      # margin rule: assumed growth of ||g_i||_1 per unit of
                                    # iterate movement since the model was built
    delta0: float = 1.0
    delta_lower: float = 1e-3       # \underline{Delta}: post-acceptance reset (theory requires it;
                                    # gated by use_delta_lower below so it can be toggled off to test
                                    # whether it's needed in practice)
    use_delta_lower: bool = True    # Changed by Matt: toggle for the delta_lower floor/reset on
                                    # acceptance; CONORBIT has no equivalent of this
    delta_min: float = 1e-10        # stop when Delta falls below this
    mu: float = 1.0                 # criticality loop: shrink while Delta > mu * chi_k
    sigma: float = 0.5              # active-set radius exponent
    gamma_dec: float = 0.5
    gamma_inc: float = 2.0
    eta1: float = 0.2               # Changed by Matt: match CONORBIT's single accept/reject rho threshold
    full_step_rel: float = 0.5      # Changed by Matt: CONORBIT-style growth gate (stepsize >= 0.5)
    min_step_rel: float = 5e-4      # Changed by Matt: CONORBIT-style shrink-despite-acceptance gate
    delta_max: float | None = None  # Changed by Matt: cap on Delta's growth (CONORBIT's maxdelta);
                                    # None => computed per-problem in CDFO.__init__
    hess: str = "fmodel"            # 'identity' | 'fmodel' | 'lagrangian'
    zeta_low: float = 1e-6
    zeta_high: float = 1e8
    np_max: int | None = None       # formquad np_max (default 2n+1)
    formquad_par: list | None = None
    par_c1_mult: float = 1.0        # formquad Par[0] = par_c1_mult * sqrt(n): radius (in units of
                                    # Delta) within which EXISTING points may be reused.  Larger =>
                                    # more reuse => far fewer geometry evaluations.
    par_c2_mult: float = 1.0        # formquad Par[1] multiplier (points for the quadratic term)
    geom_delta_floor: float | None = None  # floor on the delta used to build models / admit
                                    # reuse candidates, decoupling model-building reuse from the
                                    # live (possibly tiny) trust-region radius; the true TR step
                                    # bound and criticality loop are unaffected.  None => 0.0 (old
                                    # behavior: aperture tied to live delta).  A large floor (e.g.
                                    # delta0) is available for experimentation but empirically hurt
                                    # final solution quality on the HAIFAS/HS117 test problems, by
                                    # admitting points too far away for the local model to stay
                                    # accurate, so it is not the default.
    linear_constraints: bool = False  # fit constraint gradients by pooled least squares over
                                      # several affinely independent sets, instead of taking the
                                      # gradient of formquad's quadratic model
    max_sweeps: int = 4               # linear_constraints: how many phase-1 sweeps to pool.
                                      # 1 = just the first (best-conditioned) affinely
                                      # independent set, i.e. no pooling
    max_geom_per_iter: int = -1     # cap on geometry points per model update (-1 = unlimited);
                                    # when the cap bites, a least-squares model is used instead
    max_queries: int | None = None  # budget in scalar oracle queries (default
                                    # budget_factor * (n+1) * (p+1))
    max_points: int | None = None   # older per-point budget; used only when
                                    # budget_in == "points"
    budget_factor: float = 100.0    # multiplier for whichever unit budget_in selects
    log_queries: bool = False       # keep a per-oracle ledger of every scalar query,
                                    # for auditing the counts independently
    budget_in: str = "queries"      # "queries": budget_factor*(n+1)*(p+1) scalar queries,
                                    #            i.e. budget_factor evaluations of the whole
                                    #            vector (f, c_1, ..., c_p) per simplex gradient
                                    # "points":  budget_factor*(n+1) evaluated points
    max_iter: int = 100000
    chi_tol: float = 1e-8           # stop if chi_k <= chi_tol and Delta <= delta_tol
    delta_tol: float = 1e-8
    margin_coef: float = 0.0        # inner-boundary offset: margin = margin_coef * Delta^2
    compare_full_step: bool = False # also solve the subproblem with ALL p constraints and record
                                    # how far that step is from the one computed with A_k.  Costs
                                    # extra QP solves only -- no extra function evaluations.
    eval_order: str = "active_first"  # 'natural' | 'active_first'
    zero_step_rel: float = 1e-8
    max_crit_shrinks: int = 200
    verbose: int = 0


@dataclass
class IterationRecord:
    k: int
    delta: float
    chi: float
    A_initial: list
    A_final: list
    n_augment: int
    n_crit_shrinks: int
    f: float
    accepted: bool
    trial_feasible: bool
    sub_infeasible: bool
    rho: float
    step_norm: float
    n_points: int
    n_components: int
    n_c_evals: int
    n_f_evals: int
    x_norm_step: float = 0.0
    step_diff: float = np.nan       # ||s(A_k) - s(all p)||_inf / Delta_k
    tau_diff: float = np.nan        # |tau(A_k) - tau(all p)| / |tau(all p)|
    size_A: int = 0


def _active_set(c, G, delta, sigma, p):
    """A_k = { i : c_i(x_k) + delta^sigma ||g_i||_1 >= 0 }  (eq. Ak_explicit)."""
    r = delta ** sigma
    return [i for i in range(p) if c[i] + r * np.sum(np.abs(G[i])) >= 0]


class CDFO:
    def __init__(self, problem, opts: Options | None = None):
        self.prob = problem
        self.opts = opts or Options()
        n = problem.n
        if self.opts.budget_in == "queries":
            max_queries = self.opts.max_queries or int(
                self.opts.budget_factor * (n + 1) * (problem.p + 1))
            self.oracle = Oracle(problem, max_queries=max_queries,
                                 log_queries=self.opts.log_queries)
        else:
            max_points = self.opts.max_points or int(self.opts.budget_factor * (n + 1))
            self.oracle = Oracle(problem, max_points=max_points,
                                 log_queries=self.opts.log_queries)
        geom_delta_floor = (self.opts.geom_delta_floor if self.opts.geom_delta_floor is not None
                           else 0.0)
        self.models = ModelManager(self.oracle, np_max=self.opts.np_max, Par=self.opts.formquad_par,
                                   par_c1_mult=self.opts.par_c1_mult, par_c2_mult=self.opts.par_c2_mult,
                                   max_geom=self.opts.max_geom_per_iter,
                                   linear_constraints=self.opts.linear_constraints,
                                   max_sweeps=self.opts.max_sweeps,
                                   geom_delta_floor=geom_delta_floor)
        self.history: list[IterationRecord] = []
        self.status = "running"
        self.multipliers = None
        # Changed by Matt: mirrors conorbit.py's run_conorbit bounds_radius / maxdelta computation
        bounds_radius = 20.0  # Changed by Matt
        self.delta_max = (self.opts.delta_max if self.opts.delta_max is not None
                          else bounds_radius * (1.0 + float(np.min(np.abs(problem.x0)))))  # Changed by Matt

    # ------------------------------------------------------------------ #
    def _update_set(self, A, k, xk=None, delta=None, c=None, G=None):
        """Components whose models we (re)build at this pass.

        The exact values c_i(x_k) are always available (the feasibility loop
        evaluates them), so only the gradient in the test
        c_i(x_k) + Delta^sigma ||g_i||_1 >= 0 is uncertain.  The margin

            m_i = -c_i(x_k) - Delta^sigma ||g_i||_1  > 0   (i currently excluded)

        says how much the gradient term would have to grow before i would
        qualify.  Bounding that growth by kappa * (distance the incumbent has
        moved since the model was built) gives a rebuild test that fires only
        when the stale model could actually change the decision; the counter is
        kept as a backstop, since a badly wrong gradient understates its own
        error."""
        p = self.prob.p
        if not self.opts.lazy or not self.opts.active_set:
            return list(range(1 + p))
        U = {0} | {1 + i for i in A}
        o = self.opts
        margin_rule = (o.refresh_rule == "margin" and c is not None and G is not None
                       and delta is not None and xk is not None)
        r = delta ** o.sigma if delta is not None else 0.0
        for i in range(p):
            age = k - self.models.built_iter[1 + i]
            if i in A:
                continue
            if not margin_rule:
                if age >= o.refresh_every:
                    U.add(1 + i)
                continue
            if age >= max(o.refresh_every, 1) * 5:        # backstop
                U.add(1 + i)
                continue
            gi = float(np.sum(np.abs(G[i])))
            margin = -c[i] - r * gi                        # > 0 while i is excluded
            if margin <= 0:
                U.add(1 + i)
                continue
            centre = self.models.center[1 + i]
            moved = 0.0 if centre < 0 else float(np.max(np.abs(
                self.oracle.x(xk) - self.oracle.x(centre))))
            if r * o.refresh_kappa * moved >= margin:      # the stale gradient could matter
                U.add(1 + i)
        return sorted(U)

    def _build_and_estimate(self, xk, delta, k, A_prev):
        """Build/update models, then form A_k (lines 'models' and 'active_subproblem')."""
        p = self.prob.p
        if not self.opts.active_set:
            self.models.update(list(range(1 + p)), xk, delta, k)
            g_f, G, c = self.models.linear_data(xk)
            return list(range(p)), g_f, G, c
        # pre-estimate with (possibly stale) gradients, exact values
        g_f, G, c = self.models.linear_data(xk)
        A_pre = _active_set(c, G, delta, self.opts.sigma, p)
        U = self._update_set(set(A_pre) | set(A_prev), k, xk=xk, delta=delta, c=c, G=G)
        self.models.update(U, xk, delta, k)
        g_f, G, c = self.models.linear_data(xk)
        A = _active_set(c, G, delta, self.opts.sigma, p)
        # anything that entered A only now gets a fresh model too
        new = [1 + i for i in A if (1 + i) not in U]
        if new:
            self.models.update(new, xk, delta, k)
            g_f, G, c = self.models.linear_data(xk)
            A = _active_set(c, G, delta, self.opts.sigma, p)
        return A, g_f, G, c

    def _chi(self, g_f, G, c, A):
        val = criticality_measure(g_f, G[A], c[A])
        return val if np.isfinite(val) else 0.0

    def _solve(self, g_f, G, c, A, H, delta):
        margin = self.opts.margin_coef * delta ** 2
        if self.opts.subproblem == "tv":
            out = tv_step(g_f, G[A], c[A], H, delta, margin)
        else:
            out = sqp_step(g_f, G[A], c[A], H, delta, margin)
        return out

    def _eval_order(self, A):
        p = self.prob.p
        if self.opts.eval_order == "active_first":
            return list(A) + [i for i in range(p) if i not in set(A)]
        return list(range(p))

    # ------------------------------------------------------------------ #
    def run(self):
        o = self.opts
        n, p = self.prob.n, self.prob.p
        t0 = time.time()
        xk = self.oracle.add_point(self.prob.x0)
        self.oracle.eval_all(xk)
        if not self.oracle.is_feasible(xk):
            raise ValueError("x0 is infeasible; the algorithm requires c(x0) <= 0")
        delta = o.delta0
        A_prev = []
        k = 0
        while k < o.max_iter:
            if self.oracle.budget_exhausted():
                self.status = "budget"
                break
            # ---- models + active set estimate ---------------------------
            A, g_f, G, c = self._build_and_estimate(xk, delta, k, A_prev)
            chi = self._chi(g_f, G, c, A)
            # ---- criticality loop ---------------------------------------
            n_shrink = 0
            while delta > o.mu * chi:
                delta *= o.gamma_dec
                n_shrink += 1
                if delta < o.delta_min:
                    self.status = "delta_min"
                    break
                if n_shrink > o.max_crit_shrinks or self.oracle.budget_exhausted():
                    self.status = "budget" if self.oracle.budget_exhausted() else "crit_loop"
                    break
                A, g_f, G, c = self._build_and_estimate(xk, delta, k, A)
                chi = self._chi(g_f, G, c, A)
            if self.status != "running":
                break
            if chi <= o.chi_tol and delta <= o.delta_tol:
                self.status = "stationary"
                break
            A_initial = list(A)
            # ---- step loop (lines 'mainwhile') --------------------------
            n_aug = 0
            step_diff = np.nan
            tau_diff = np.nan
            accepted = False
            trial_feasible = False
            sub_infeasible = False
            rho = np.nan
            step_norm = 0.0
            delta_used = delta
            while True:
                H = self.models.hessian(o.hess, self.multipliers, o.zeta_low, o.zeta_high)
                out = self._solve(g_f, G, c, A, H, delta)
                if out["s"] is None:
                    sub_infeasible = True
                    delta = max(o.delta_min, o.gamma_dec * delta)  # Changed by Matt: floor every shrink, like CONORBIT's mindelta
                    break
                s = out["s"]
                if o.compare_full_step and len(A) < p:
                    full = self._solve(g_f, G, c, list(range(p)), H, delta)
                    if full["s"] is not None:
                        step_diff = float(np.max(np.abs(s - full["s"]))) / max(delta, 1e-300)
                        if out.get("tau") is not None and full.get("tau") is not None:
                            tau_diff = abs(out["tau"] - full["tau"]) / max(abs(full["tau"]), 1e-12)
                elif o.compare_full_step:
                    step_diff, tau_diff = 0.0, 0.0     # A_k is already all of them
                step_norm = float(np.max(np.abs(s)))
                nonzero = step_norm > o.zero_step_rel * delta
                pred = -float(g_f @ s)             # m^f(x_k) - m^f(x_k + s)
                trial = self.oracle.add_point(self.oracle.x(xk) + s)
                i_viol = -1
                for i in self._eval_order(A):
                    cv = self.oracle.eval_c(trial, i)
                    if (not np.isfinite(cv)) or cv > 0:
                        i_viol = i
                        break
                    if self.oracle.budget_exhausted():
                        break
                if i_viol < 0 and nonzero and not self.oracle.budget_exhausted():
                    trial_feasible = True
                    f_trial = self.oracle.eval_f(trial)
                    f_k = self.oracle.row(xk)[0]
                    rho = (f_k - f_trial) / pred if (pred > 0 and np.isfinite(f_trial)) else -np.inf
                    if rho >= o.eta1:
                        # Changed by Matt: CONORBIT-style accept-then-gate-on-step-length TR update
                        accepted = True
                        step_ratio = step_norm / delta  # Changed by Matt
                        if step_ratio >= o.full_step_rel:  # Changed by Matt
                            delta_new = min(self.delta_max, o.gamma_inc * delta)  # Changed by Matt
                        elif step_ratio < o.min_step_rel:  # Changed by Matt
                            delta_new = max(o.delta_min, o.gamma_dec * delta)  # Changed by Matt
                        else:  # Changed by Matt
                            delta_new = delta  # Changed by Matt
                        if o.use_delta_lower:  # Changed by Matt
                            delta_new = max(o.delta_lower, delta_new)  # Changed by Matt
                        xk_new = trial  # Changed by Matt
                    else:
                        xk_new, delta_new = xk, max(o.delta_min, o.gamma_dec * delta)  # Changed by Matt
                    # store multipliers for the Lagrangian Hessian option
                    if out.get("yc") is not None and out.get("yf") is not None and out["yf"] > 1e-8:
                        self.multipliers = {A[a]: float(out["yc"][a] / out["yf"]) for a in range(len(A))}
                    xk, delta = xk_new, delta_new
                    break
                elif i_viol >= 0 and (i_viol not in A) and nonzero and o.active_set:
                    A_t = sorted(A + [i_viol])
                    self.models.update([1 + i_viol], xk, delta, k)
                    g_f, G, c = self.models.linear_data(xk)
                    chi_t = self._chi(g_f, G, c, A_t)
                    n_aug += 1
                    if delta <= o.mu * chi_t:
                        A = A_t
                        continue
                    delta = max(o.delta_min, o.gamma_dec * delta)  # Changed by Matt: floor every shrink, like CONORBIT's mindelta
                    break
                else:
                    delta = max(o.delta_min, o.gamma_dec * delta)  # Changed by Matt: floor every shrink, like CONORBIT's mindelta
                    break
            summ = self.oracle.summary()
            self.history.append(IterationRecord(
                k=k, delta=delta_used, chi=chi, A_initial=A_initial, A_final=list(A), n_augment=n_aug,
                n_crit_shrinks=n_shrink, f=float(self.oracle.row(xk)[0]), accepted=accepted,
                trial_feasible=trial_feasible, sub_infeasible=sub_infeasible, rho=float(rho),
                step_norm=step_norm, n_points=summ["n_points"], n_components=summ["n_components"],
                n_c_evals=summ["n_c"], n_f_evals=summ["n_f"],
                step_diff=float(step_diff), tau_diff=float(tau_diff), size_A=len(A_initial)))
            if o.verbose and (k % o.verbose == 0):
                print(f"k={k:5d} f={self.history[-1].f:.6e} delta={delta_used:.2e} chi={chi:.2e} "
                      f"|A|={len(A)}/{p} acc={int(accepted)} feas={int(trial_feasible)} "
                      f"pts={summ['n_points']} comp={summ['n_components']}")
            A_prev = A
            k += 1
            if delta < o.delta_min:
                self.status = "delta_min"
                break
        if self.status == "running":
            self.status = "max_iter"
        return self.result(xk, time.time() - t0)

    def result(self, xk, wall):
        summ = self.oracle.summary()
        return dict(
            solver=self.name(), status=self.status, x=self.oracle.x(xk).copy(),
            f=float(self.oracle.row(xk)[0]), c=self.oracle.row(xk)[1:].copy(),
            n_iter=len(self.history), wall=wall, trace=list(self.oracle.trace),
            history=[asdict(h) for h in self.history], oracle=summ,
            n_geometry_points=self.models.n_geometry_points,
            n_formquad_calls=self.models.n_formquad_calls,
            n_linear_sweeps=self.models.n_linear_sweeps, options=asdict(self.opts))

    def name(self):
        o = self.opts
        tag = o.subproblem.upper()
        tag += "-AS" if o.active_set else "-ALL"
        if o.margin_coef:
            tag += f"-m{o.margin_coef:g}"
        if o.active_set:
            tag += "-lazy" if o.lazy else "-full"
        if o.lazy and o.refresh_rule != "counter":
            tag += f"-{o.refresh_rule}"
        return tag


def run_cdfo(problem, **kwargs):
    return CDFO(problem, Options(**kwargs)).run()
