fdm
===
This is software for reproducing the numerical results in
<https://arxiv.org/abs/2610.17101>. The CONORBIT code used for comparison is
not included; please contact the authors of CONORBIT to obtain it (see the
top-level [README](../README.md)).

This tool has a Python implementation in [py](py), following the structure
conventions of [template_repo](https://github.com/POptUS/template_repo). It is
distributed to users via the Python package
[fdm_pypkg](../fdm_pypkg).

Testing Python Implementation
==============================
`fdm` has two kinds of tests under [py/tests](py/tests) and
[py/experiments](py/experiments):

## `unittest`-based tests
The `py` implementation is included in
[fdm_pypkg](../fdm_pypkg/src/fdm) via symlink, so its `unittest`-style tests
(files matching `tests/Test*.py`) are intended to be run through that
package's testing facilities, e.g. `python -m unittest fdm.fdm`. See
[load_tests.py](py/load_tests.py), which follows the `unittest`
[load_tests protocol](https://docs.python.org/3/library/unittest.html?highlight=load_tests#load-tests-protocol).

## `pytest`-based smoke tests
[py/tests/test_smoke.py](py/tests/test_smoke.py) is written in plain
`pytest` style (functions, not `unittest.TestCase`), so it is **not**
discovered by the `unittest`-based `load_tests` protocol above. Run it
directly once `fdm_pypkg` is installed (e.g. `pip install -e ../fdm_pypkg`).
`--import-mode=importlib` avoids a pytest/`py`-directory module-naming clash:
```
$ pytest py/tests --import-mode=importlib
```

## Experiments
[py/experiments](py/experiments) holds the research driver scripts used to
produce and analyze results (`run_experiments.py`, `audit_oracle.py`,
`trace_run.py`, `summarize.py`, and the `plot_*.py` scripts). These are not
part of the CI-tested "examples" convention: they require `fdm_pypkg` to be
installed with its `experiments` extra (`pip install -e
"../fdm_pypkg[experiments]"`), which adds `clarabel`, `matplotlib`, and
`pandas`. See [../README.md](../README.md#installation--requirements) for
the full dependency list, including the optional, not-on-PyPI `ibcdfo`/
`pycutest` packages needed only for full-fidelity (non-fallback) runs. A
quick smoke run that avoids all optional dependencies:
```
$ cd py/experiments
$ python run_experiments.py --testset hs --budget 20 --allow-fallback-formquad --out /tmp/smoke_results.pkl
```

Every `(problem, solver)` run is cached as its own file under
`py/experiments/benchmark_results/<testset>/<config-hash>/` as soon as it
finishes; the directory is created on first use. Re-running the same command
later skips anything already on disk, so an interrupted or incremental run
just picks up where it left off. The hash folds in the budget, bounds, and
every algorithm parameter, so pointing a different parameter set at the same
`--results-dir` lands in its own bucket instead of mixing with earlier
results. Pass `--results-dir` to use a different location; delete a bucket
under `benchmark_results/` to force those runs to be redone.

To spread the same run across more cores or nodes than `--jobs` can reach on
one machine, use `mpi4py` instead (needs `mpi4py` installed; `--jobs` is
ignored when `--mpi` is passed):
```
$ mpiexec -n 16 python run_experiments.py --testset cutest --n-max 30 --p-max 100 \
    --budget 100 --mpi --out results_cutest.pkl
```
This needs at least 2 ranks (rank 0 dispatches jobs; the rest run them) and
assumes all ranks share the working directory (for `problems_cache.json` and
`benchmark_results/`), as is typical on an MPI cluster.
