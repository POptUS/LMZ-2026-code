#!/usr/bin/env python
"""
Run all solver variants on the test set and save everything needed for the
plots to a pickle.

Every (problem, solver) run is cached as its own file under
``<results-dir>/<testset>/<config-hash>/<problem>__<solver>.pkl`` as soon as
it completes.  Re-running the same command later skips any run whose file is
already there, so an interrupted or incremental run just picks up where it
left off.  The hash folds in everything that changes what a run computes
(budget, bounds, and the algorithm parameters below), so a parameter sweep
against the same --results-dir lands each configuration in its own bucket
instead of silently mixing results.

Examples
    # smoke test on the hand-coded HS problems (no pycutest needed)
    python run_experiments.py --testset hs --out results_hs.pkl

    # full CUTEst run (first call builds and caches the problem list),
    # single machine, multiple worker processes
    python run_experiments.py --testset cutest --n-max 30 --p-max 100 --budget 100 \
        --out results_cutest.pkl --jobs 8

    # same, distributed over MPI ranks (one dispatcher + workers; needs N>=2)
    mpiexec -n 16 python run_experiments.py --testset cutest --n-max 30 --p-max 100 \
        --budget 100 --out results_cutest.pkl --mpi

Solver variants (--solvers):
    TV-AS-lazy   proposed method (TV subproblem, active set, lazy constraint models)
    TV-AS-full   as analyzed (TV subproblem, active set, all models rebuilt each iteration)
    TV-ALL       TV subproblem, all constraints in every subproblem
    SQP-ALL      standard TR-SQP subproblem, feasible-only acceptance
    SQP-AS-lazy  standard TR-SQP subproblem with the active-set machinery
    COBYLA       SciPy COBYLA
    COBYLA-unrelaxable
                 the same, but the objective is not returned at an infeasible
                 point: f(x) comes back as NaN there, as it would when the
                 constraints are unrelaxable.  COBYLA has no provision for this
    CONORBIT     ORBITc.m, called through the MATLAB Engine for Python (see
                 fdm/py/conorbit.py).  Not in the --solvers default: it needs a
                 local MATLAB install plus the MATLAB Engine API for Python
                 installed into this same Python environment; pass --solvers
                 explicitly to include it, and --jobs 1 with no --mpi (one
                 MATLAB engine per process, slow to start)
    NOWPAC       Via Sandia's DAKOTA toolkit (fdm/py/nowpac.py). Needs a DAKOTA
                 build with -DHAVE_NOWPAC=ON (not in DAKOTA's precompiled
                 binaries) and a feasible starting point (no feasibility-
                 restoration phase); not in the --solvers default -- pass
                 --solvers explicitly to include it. Unlike CONORBIT, each run
                 starts its own `dakota` subprocess and socket server, so
                 --jobs > 1 is safe; set DAKOTA_DIR (default ~/dakota) if
                 installed elsewhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from fdm.fdm.algorithm import Options, CDFO
from fdm.fdm.baselines import run_cobyla, reference_solution
from fdm.fdm.problems import build_cutest_testset, load_cutest_testset, hs_problems, ProblemSpec

try:
    from fdm.fdm.conorbit import run_conorbit
except ImportError:
    run_conorbit = None

try:
    from fdm.fdm.nowpac import run_nowpac
except ImportError:
    run_nowpac = None

VARIANTS = {
    "TV-AS-lazy": dict(subproblem="tv", active_set=True, lazy=True),
    "TV-AS-full": dict(subproblem="tv", active_set=True, lazy=False),
    "TV-ALL": dict(subproblem="tv", active_set=False),
    "SQP-ALL": dict(subproblem="sqp", active_set=False),
    "SQP-AS-lazy": dict(subproblem="sqp", active_set=True, lazy=True),
}


def run_one(args):
    spec, solver, budget, common = args
    try:
        prob = spec.load()          # re-create the (unpicklable) problem inside the worker
        if solver.startswith("COBYLA"):
            r = run_cobyla(prob, budget_factor=budget,
                           budget_in=common.get("budget_in", "queries"),
                           unrelaxable=solver.endswith("-unrelaxable"))
        elif solver == "CONORBIT":
            # spec.kind/spec.bounds must match how `prob` above was just loaded,
            # since ORBITc's callback rebuilds the Oracle's problem from scratch
            # inside MATLAB's embedded interpreter via the same ProblemSpec recipe.
            r = run_conorbit(prob, budget_factor=budget,
                             budget_in=common.get("budget_in", "queries"),
                             spec_kind=spec.kind, spec_bounds=spec.bounds,
                             delta0=common.get("delta0", 1.0))
        elif solver == "NOWPAC":
            r = run_nowpac(prob, budget_factor=budget,
                           budget_in=common.get("budget_in", "queries"),
                           delta0=common.get("delta0", 1.0),
                           delta_lower=common.get("delta_lower", 1e-3))
        else:
            opts = Options(budget_factor=budget, **VARIANTS[solver], **common)
            r = CDFO(prob, opts).run()
        r["solver"] = solver
        return spec.name, solver, r, None
    except Exception:  # noqa
        return spec.name, solver, None, traceback.format_exc()


# --------------------------------------------------------------------------- #
# Per-(problem, solver) result cache
# --------------------------------------------------------------------------- #
def _config_key(args, common, have_ibcdfo):
    """Short hash of everything that changes what a run computes, so a
    different parameter set never silently reuses another configuration's
    cached results."""
    cfg = dict(budget=args.budget, budget_in="queries", bounds=args.bounds,
               have_ibcdfo=have_ibcdfo, **common)
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _write_config_json(bucket_dir, args, common, have_ibcdfo):
    path = os.path.join(bucket_dir, "config.json")
    if os.path.exists(path):
        return
    cfg = dict(testset=args.testset, budget=args.budget, budget_in="queries", bounds=args.bounds,
               have_ibcdfo=have_ibcdfo, **common)
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True, default=str)


def _result_path(bucket_dir, name, solver):
    return os.path.join(bucket_dir, f"{name}__{solver}.pkl")


def _atomic_write(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)    # atomic on the same filesystem: a crash never leaves a bad "done" file


def run_and_save(job, bucket_dir):
    """Run one (problem, solver) job and, on success, persist it immediately
    so a later invocation can skip it."""
    name, solver, r, err = run_one(job)
    if r is not None:
        _atomic_write(_result_path(bucket_dir, name, solver), r)
    return name, solver, r, err


def _log_result(errors, name, solver, r, err, k, total, t0, tag=""):
    if r is None:
        errors.append((name, solver, err))
        print(f"[{k + 1}/{total}] {name:12s} {solver:12s} FAILED{tag}", file=sys.stderr)
        return
    print(f"[{k + 1}/{total}] {name:12s} {solver:12s} f={r['f']:.6e} pts={r['oracle']['n_points']:5d} "
          f"comp={r['oracle']['n_components']:6d} {r['status']:10s} ({time.time() - t0:.0f}s){tag}")


# --------------------------------------------------------------------------- #
# MPI dispatch: rank 0 hands out jobs dynamically; workers run_and_save and
# report back only a small status (the full result, including its oracle
# trace, is already safely on disk by the time rank 0 hears about it).
# --------------------------------------------------------------------------- #
def _mpi_worker(comm, bucket_dir):
    while True:
        job = comm.recv(source=0, tag=0)
        if job is None:
            break
        t0 = time.time()
        name, solver, r, err = run_and_save(job, bucket_dir)
        info = None
        if r is not None:
            info = dict(f=r["f"], n_points=r["oracle"]["n_points"],
                        n_components=r["oracle"]["n_components"], status=r["status"])
        comm.send((name, solver, r is not None, err, info, time.time() - t0), dest=0, tag=0)


def _mpi_master(pending, comm, size, errors, t0):
    from mpi4py import MPI
    total = len(pending)
    next_idx = 0
    in_flight = 0
    for w in range(1, size):
        job = pending[next_idx] if next_idx < total else None
        comm.send(job, dest=w, tag=0)
        if job is not None:
            next_idx += 1
            in_flight += 1
    done = 0
    while in_flight > 0:
        status = MPI.Status()
        name, solver, ok, err, info, elapsed = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
        w = status.Get_source()
        in_flight -= 1
        done += 1
        r = info and dict(info, oracle=dict(n_points=info["n_points"], n_components=info["n_components"]))
        _log_result(errors, name, solver, r, err if not ok else None, done - 1, total, t0, tag=f" (rank {w})")
        job = pending[next_idx] if next_idx < total else None
        comm.send(job, dest=w, tag=0)
        if job is not None:
            next_idx += 1
            in_flight += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="hs", choices=["hs", "cutest"])
    ap.add_argument("--cache", default="problems_cache.json")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--recompute-references", action="store_true",
                    help="re-solve the reference problems instead of using the ones cached at filtering time")
    ap.add_argument("--no-require-reference", action="store_true",
                    help="when rebuilding the cache, keep problems without a polished reference solution")
    ap.add_argument("--n-max", type=int, default=30)
    ap.add_argument("--m-max", type=int, default=100)
    ap.add_argument("--p-max", type=int, default=100)
    ap.add_argument("--bounds", default="exclude", choices=["exclude", "as_constraints"])
    ap.add_argument("--budget", type=float, default=100.0, help="budget in multiples of (n+1) points")
    ap.add_argument("--solvers",
                    default=",".join(list(VARIANTS) + ["COBYLA", "COBYLA-unrelaxable"]))
    ap.add_argument("--only", default=None, help="comma-separated problem names")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated problem names to drop (e.g. known stragglers)")
    ap.add_argument("--allow-fallback-formquad", action="store_true",
                    help="permit running with the stub formquad (smoke tests only; NOT for the paper)")
    ap.add_argument("--names-file", default=None,
                    help="text file with one CUTEst problem name per line; bypasses the classification query")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (ignored under --mpi)")
    ap.add_argument("--mpi", action="store_true",
                    help="distribute jobs across MPI ranks: mpiexec -n N python run_experiments.py --mpi ... "
                         "(N>=2: one dispatcher + at least one worker)")
    ap.add_argument("--results-dir", default="benchmark_results",
                    help="per-(problem,solver) result cache; a rerun skips anything already saved here")
    ap.add_argument("--out", default="results.pkl")
    # algorithm parameters shared by all CDFO variants
    ap.add_argument("--delta0", type=float, default=1.0)
    ap.add_argument("--delta-lower", type=float, default=1e-3)
    ap.add_argument("--disable-delta-lower", action="store_true",
                    help="turn off the delta_lower floor/reset on acceptance, to test whether it's "
                         "needed in practice (CONORBIT has no equivalent of it)")  # Changed by Matt
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--hess", default="fmodel", choices=["identity", "fmodel", "lagrangian"])
    ap.add_argument("--refresh-every", type=int, default=10)
    ap.add_argument("--par-c1-mult", type=float, default=1.0,
                    help="formquad reuse radius multiplier (Par[0] = mult*sqrt(n)); larger = more point reuse")
    ap.add_argument("--max-geom-per-iter", type=int, default=-1)
    ap.add_argument("--margin-coef", type=float, default=0.0,
                    help="inner-boundary offset margin = coef * Delta^2 (0 = off)")
    args = ap.parse_args()

    comm, rank, size = None, 0, 1
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
        if size < 2:
            sys.exit("--mpi needs at least 2 ranks (1 dispatcher + >=1 worker); "
                      "run with e.g. `mpiexec -n 4 python run_experiments.py --mpi ...`")
    is_root = rank == 0
    if args.mpi and args.jobs > 1 and is_root:
        print("note: --jobs is ignored under --mpi (one job in flight per rank)")

    # refuse to produce paper numbers with the stub model builder
    from fdm.fdm.models import using_fallback
    have_ibcdfo = not using_fallback()
    if not have_ibcdfo and not args.allow_fallback_formquad:
        if is_root:
            sys.exit("formquad is not importable, so models would be built by the stub in fdm/py/fallbacks.py\n"
                     "and the results would NOT be those of the paper's algorithm.  Fix one of:\n"
                     "  pip install ibcdfo\n"
                     "  or drop your own formquad.py next to run_experiments.py as local_formquad.py\n"
                     "  or call fdm.fdm.models.set_formquad(<your formquad>) before running.\n"
                     "Pass --allow-fallback-formquad to override for a smoke test.")
        sys.exit(1)
    if not have_ibcdfo and is_root:
        print("WARNING: running with the stub formquad; results are not paper-quality.")

    if args.testset == "hs":
        problems = hs_problems()
    else:
        if is_root and (args.rebuild_cache or not os.path.exists(args.cache)):
            names = None
            if args.names_file:
                names = [l.strip() for l in open(args.names_file) if l.strip() and not l.startswith("#")]
            build_cutest_testset(n_max=args.n_max, m_max=args.m_max, p_max=args.p_max,
                                 bounds=args.bounds, cache=args.cache, names=names,
                                 require_reference=not args.no_require_reference)
        if comm is not None:
            comm.Barrier()    # wait for rank 0 to finish building/writing the cache before anyone reads it
        problems = load_cutest_testset(args.cache)
        with open(args.cache) as fh:
            cached = json.load(fh)
        if cached and cached[0].get("bounds", "exclude") != args.bounds:
            args.bounds = cached[0]["bounds"]
            if is_root:
                print(f"using bounds='{args.bounds}' from the cache (pass --rebuild-cache to change)")
    if args.only:
        keep = set(args.only.split(","))
        problems = [p for p in problems if p.name in keep]
    if args.exclude:
        drop = set(args.exclude.split(","))
        problems = [p for p in problems if p.name not in drop]
    if is_root:
        print(f"{len(problems)} problems")

    solvers = args.solvers.split(",")
    common = dict(delta0=args.delta0, delta_lower=args.delta_lower,
                  use_delta_lower=not args.disable_delta_lower,  # Changed by Matt
                  mu=args.mu, sigma=args.sigma,
                  hess=args.hess, refresh_every=args.refresh_every, margin_coef=args.margin_coef,
                  par_c1_mult=args.par_c1_mult, max_geom_per_iter=args.max_geom_per_iter)

    bucket_dir = os.path.join(args.results_dir, args.testset, _config_key(args, common, have_ibcdfo))
    os.makedirs(bucket_dir, exist_ok=True)
    if is_root:
        _write_config_json(bucket_dir, args, common, have_ibcdfo)

    kind = "hs" if args.testset == "hs" else "cutest"
    specs = [ProblemSpec.from_problem(p, kind, args.bounds) for p in problems]
    all_jobs = [(sp, s) for sp in specs for s in solvers]
    pending = []
    n_cached = 0
    for sp, s in all_jobs:
        if os.path.exists(_result_path(bucket_dir, sp.name, s)):
            n_cached += 1
        else:
            pending.append((sp, s, args.budget, common))
    if is_root:
        print(f"{len(all_jobs)} (problem, solver) runs in {bucket_dir}: "
              f"{n_cached} cached, {len(pending)} to run")

    if args.mpi and not is_root:
        try:
            _mpi_worker(comm, bucket_dir)
        except Exception as e:  # noqa
            print(f"worker rank {rank} failed: {e!r}", file=sys.stderr)
        return

    errors = []
    t0 = time.time()
    try:
        if args.mpi:
            _mpi_master(pending, comm, size, errors, t0)
        elif args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                futs = {ex.submit(run_and_save, j, bucket_dir): j for j in pending}
                for k, fut in enumerate(as_completed(futs)):
                    try:
                        name, solver, r, err = fut.result()
                    except Exception as e:  # worker died (e.g. native crash)
                        spec, solver, _, _ = futs[fut]
                        name, r, err = spec.name, None, f"worker died: {e!r}"
                    _log_result(errors, name, solver, r, err, k, len(pending), t0)
        else:
            for k, j in enumerate(pending):
                name, solver, r, err = run_and_save(j, bucket_dir)
                _log_result(errors, name, solver, r, err, k, len(pending), t0)
    except (KeyboardInterrupt, Exception) as e:  # noqa
        print(f"\ninterrupted ({e!r}); completed runs are saved under {bucket_dir}", file=sys.stderr)

    # Assemble the combined pickle from disk: this picks up both the runs
    # just completed and the ones that were already cached, uniformly.
    results = {p.name: {} for p in problems}
    for sp, s in all_jobs:
        path = _result_path(bucket_dir, sp.name, s)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                results[sp.name][s] = pickle.load(fh)

    # reference solutions: use the ones cached at filtering time (identical across
    # runs), and only re-solve if asked to or if the cache has none
    refs = {}
    n_cached_refs = 0
    for p in problems:
        runs = results[p.name]
        if not runs:
            continue
        cached = p.meta.get("reference")
        if cached and not args.recompute_references:
            ref = dict(cached)
            ref["x"] = np.asarray(ref["x"], float)
            ref["multipliers"] = np.asarray(ref["multipliers"], float)
            refs[p.name] = ref
            for r in runs.values():
                r["reference_f"] = ref["f"]
            n_cached_refs += 1
            continue
        feasible = [r for r in runs.values() if np.all(r["c"] <= 1e-8)]
        pool = feasible or list(runs.values())
        best = min(pool, key=lambda r: r["f"])
        others = [r["x"] for r in pool if r is not best]
        try:
            refs[p.name] = reference_solution(p, best["x"], starts=others)
        except Exception:  # noqa
            refs[p.name] = None
        if refs[p.name] is None:
            print(f"  WARNING: no reference solution for {p.name}")
    if n_cached_refs:
        print(f"using {n_cached_refs} reference solutions cached at filtering time "
              f"(pass --recompute-references to re-solve)")
        if refs[p.name] is not None:
            for r in runs.values():
                r["reference_f"] = refs[p.name]["f"]

    meta = dict(problems=[dict(name=p.name, n=p.n, p=p.p, x0=p.x0.tolist(), meta=p.meta) for p in problems],
                solvers=solvers, budget=args.budget, common=common, testset=args.testset, errors=errors)
    with open(args.out, "wb") as fh:
        pickle.dump(dict(results=results, references=refs, meta=meta), fh)
    print(f"saved {args.out}  ({len(errors)} errors, {time.time() - t0:.0f}s)")
    for name, solver, err in errors[:5]:
        print("ERROR", name, solver, err.splitlines()[-1])


if __name__ == "__main__":
    main()
