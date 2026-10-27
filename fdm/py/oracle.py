"""
Counting oracle.

Every evaluated point gets a row in ``F`` with ``1+p`` columns
(column 0 = f, column 1+i = c_i).  Unknown entries are NaN.  Each scalar
evaluation is charged once; the *point* count (number of distinct x where
anything was evaluated) is also tracked, because in a setting where f and all
c_i come out of a single simulation the point count is the relevant cost,
whereas in the "computationally independent constraints" setting of the paper
the component count is the relevant cost.

``trace`` records (n_points, n_component_evals, best_feasible_f) after every
charged evaluation, which is all the data/performance profiles need.
"""
from __future__ import annotations

import numpy as np


class Oracle:
    def __init__(self, problem, feas_tol=0.0, max_points=None, max_queries=None,
                 log_queries=False):
        self.prob = problem
        self.n = problem.n
        self.p = problem.p
        self.feas_tol = feas_tol
        self.max_points = max_points
        self.max_queries = max_queries   # budget in scalar oracle queries (f and each c_i
                                         # counted separately); takes precedence when set
        # One ledger per oracle: component 0 is f, component 1+i is c_i.  Each entry
        # is (point index, value, cumulative queries after this one).  The solvers
        # never touch these -- every call reaches a function through this class, so
        # the ledgers are an independent record of what was actually asked for, and
        # the counters n_f and n_c can be audited against them.
        self.log_queries = bool(log_queries)
        self.ledger = [[] for _ in range(1 + problem.p)] if log_queries else None
        self._X = []
        self._F = []
        self._cons_cache = {}          # idx -> full constraint vector (compute once, charge per component)
        self.n_points = 0
        self.n_f = 0
        self.n_c = 0
        self.n_c_by_index = np.zeros(self.p, dtype=int)
        self.best_f = np.inf
        self.best_x = None
        self.trace = []                # (n_points, n_components, best_f)
        self.n_feasible_points = 0     # points where all c known and feasible
        self.n_infeasible_points = 0   # points where some c known and violated

    # ---- database -------------------------------------------------------
    @property
    def X(self):
        return np.array(self._X).reshape(-1, self.n)

    @property
    def F(self):
        return np.array(self._F).reshape(-1, 1 + self.p)

    def add_point(self, x):
        x = np.asarray(x, dtype=float).copy()
        self._X.append(x)
        self._F.append(np.full(1 + self.p, np.nan))
        return len(self._X) - 1

    def known(self, idx, j):
        return not np.isnan(self._F[idx][j])   # +/-inf counts as known (and as a failed evaluation)

    def row(self, idx):
        return self._F[idx]

    def x(self, idx):
        return self._X[idx]

    def budget_exhausted(self):
        """True once the run has spent its budget.

        The budget is a number of scalar oracle queries: one unit is one query of
        one of f, c_1, ..., c_p at one point.  A method that settles the fate of a
        point after two constraint queries is charged two, not p.  ``max_points``
        is kept for the older per-point accounting."""
        if self.max_queries is not None:
            return (self.n_f + self.n_c) >= self.max_queries
        return self.max_points is not None and self.n_points >= self.max_points

    # ---- evaluations ----------------------------------------------------
    def _touch(self, idx):
        if not hasattr(self, "_touched"):
            self._touched = set()
        if idx not in self._touched:
            self._touched.add(idx)
            self.n_points += 1

    def _record(self, idx):
        row = self._F[idx]
        cvals = row[1:]
        if np.all(np.isfinite(cvals)):
            if np.all(cvals <= self.feas_tol):
                if np.isfinite(row[0]) and row[0] < self.best_f:
                    self.best_f = row[0]
                    self.best_x = self._X[idx].copy()
        self.trace.append((self.n_points, self.n_f + self.n_c, self.best_f))

    def _log(self, comp, idx, value):
        if self.ledger is not None:
            self.ledger[comp].append((int(idx), float(value), self.n_f + self.n_c))

    def eval_f(self, idx):
        if self.known(idx, 0):
            return self._F[idx][0]
        self._touch(idx)
        try:
            val = float(self.prob.f(self._X[idx]))
        except Exception:
            val = np.inf
        if np.isnan(val):
            val = np.inf
        self._F[idx][0] = val
        self.n_f += 1
        self._log(0, idx, val)
        self._record(idx)
        return val

    def eval_c(self, idx, i):
        """Evaluate constraint i (0-based) at point idx."""
        if self.known(idx, 1 + i):
            return self._F[idx][1 + i]
        self._touch(idx)
        if idx not in self._cons_cache:
            try:
                cv = np.asarray(self.prob.c(self._X[idx]), dtype=float)
            except Exception:
                cv = np.full(self.p, np.inf)
            cv[np.isnan(cv)] = np.inf          # a failed evaluation counts as violated
            self._cons_cache[idx] = cv
        val = float(self._cons_cache[idx][i])
        self._F[idx][1 + i] = val
        self.n_c += 1
        self.n_c_by_index[i] += 1
        self._log(1 + i, idx, val)
        self._record(idx)
        return val

    def eval_components(self, idx, comps):
        """comps: iterable of column indices (0 = f, 1+i = c_i)."""
        for j in comps:
            if j == 0:
                self.eval_f(idx)
            else:
                self.eval_c(idx, j - 1)

    def eval_all(self, idx):
        self.eval_components(idx, range(1 + self.p))

    def is_feasible(self, idx):
        c = self._F[idx][1:]
        return np.all(np.isfinite(c)) and np.all(c <= self.feas_tol)

    # ---- summary --------------------------------------------------------
    def audit(self):
        """Recount the queries from the ledgers and compare with the counters.

        Returns a dict with the per-component counts actually logged, the totals,
        and whether they agree with n_f and n_c.  Raises if logging was off."""
        if self.ledger is None:
            raise RuntimeError("the oracle was built without log_queries=True")
        per = [len(rows) for rows in self.ledger]
        n_f_logged, n_c_logged = per[0], int(sum(per[1:]))
        return dict(per_component=per, n_f_logged=n_f_logged, n_c_logged=n_c_logged,
                    n_f=self.n_f, n_c=self.n_c,
                    agrees=(n_f_logged == self.n_f and n_c_logged == self.n_c),
                    distinct_points=len({i for rows in self.ledger for i, _, _ in rows}))

    def summary(self):
        return dict(n_points=self.n_points, n_f=self.n_f, n_c=self.n_c,
                    n_components=self.n_f + self.n_c,
                    n_c_by_index=self.n_c_by_index.tolist(),
                    best_f=self.best_f)
