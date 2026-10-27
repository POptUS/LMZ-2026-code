"""
Bridge for external MATLAB solvers that call back into Python per evaluation.

A MATLAB session started via ``matlab.engine`` from Python embeds its OWN
separate Python interpreter for ``pyenv``/``py.*`` calls -- it does NOT share
memory with the controlling Python process, even though that process started
the MATLAB session.  So the live ``Oracle`` a MATLAB solver's callback needs
to reach has to be built *inside* that embedded interpreter, from a
``ProblemSpec`` (the same cross-process re-creation mechanism already used to
ship problems to multiprocessing workers), not passed in by reference.

Usage from the MATLAB side (through ``pyenv``):
    py.matlab_bridge.mb_init(kind, name, x0, bounds, max_queries)
    row = double(py.matlab_bridge.mb_eval_row(x))   # called once per point
    py.matlab_bridge.mb_write_results(path)          # once, after the solver returns
"""
from __future__ import annotations

import json

import numpy as np

from .oracle import Oracle
from .problems import ProblemSpec

_oracle = None
_problem = None


def mb_init(kind, name, x0, bounds, max_queries):
    global _oracle, _problem
    spec = ProblemSpec(str(kind), str(name), x0=np.asarray(x0, dtype=float), bounds=str(bounds))
    _problem = spec.load()
    _oracle = Oracle(_problem, max_queries=int(round(float(max_queries))))
    return "ok"


def mb_eval_row(x):
    """Evaluate f and every constraint at x; return [f(x), c_1(x), ..., c_p(x)]."""
    idx = _oracle.add_point(np.asarray(x, dtype=float))
    _oracle.eval_all(idx)
    return _oracle.row(idx).tolist()


def mb_write_results(path):
    best_x = None if _oracle.best_x is None else _oracle.best_x.tolist()
    out = dict(trace=[list(t) for t in _oracle.trace], summary=_oracle.summary(),
               best_f=float(_oracle.best_f), best_x=best_x)
    with open(str(path), "w") as fh:
        json.dump(out, fh)
    return "ok"
