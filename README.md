# FDM

This is software for reproducing the numerical results in the paper
<https://arxiv.org/abs/2610.17101>.

**CONORBIT is not included.** At the request of its developers, the source code
of CONORBIT is not distributed in this repository. The comparisons against
CONORBIT (see [fdm/py/conorbit.py](fdm/py/conorbit.py)) require that you
contact the authors of CONORBIT to obtain their code. Once you have it, place
it in a `conorbit/` directory at the top level of this repo (it must contain an
`ORBITc/` subdirectory), or point the `CONORBIT_DIR` environment variable at
it. Running CONORBIT also requires a local MATLAB install and the MATLAB Engine
API for Python. All other solvers run without it.

## Reproducing the paper's experiments
All commands are run from [fdm/py/experiments](fdm/py/experiments) after
installing the package with its experiments extra (see
[Installation / Requirements](#installation--requirements)). Full-fidelity runs
need `ibcdfo` (`formquad`) and `pycutest` (the CUTEst test set); without them
`fdm` falls back to stubs and the results will not match the paper.

* **Solver names.** The code names the paper's variants `TV-AS-lazy`,
  `TV-AS-full`, `TV-ALL`, `SQP-ALL`, `SQP-AS-lazy`; these are the paper's
  `FDM-AS-lazy`, `FDM-AS-full`, `FDM-ALL`, `SQP-ALL`, `SQP-AS-lazy`.
* **Budget.** `--budget B` means `B*(n+1)*(p+1)` scalar oracle queries; the
  paper uses `--budget 200`.
* **Run, summarize, plot.**
  ```
  $ python run_experiments.py --testset cutest --n-max 30 --p-max 100 --budget 200 \
        --out results.pkl --jobs 8
  $ python summarize.py results.pkl
  $ python plot_data_profiles.py results.pkl --outdir figs
  $ python plot_feasibility.py results.pkl --outdir figs
  $ python plot_identification_sets.py results.pkl --outdir figs
  ```
  Every (problem, solver) run is cached on disk, so interrupted runs resume;
  see [fdm/README.md](fdm/README.md). The CUTEst problem list and reference
  solutions are cached in `problems_cache.json`.
* **CONORBIT** is optional and is not in the default `--solvers` list. It
  must be obtained from its authors (see above). Pass it explicitly, e.g.
  `--solvers CONORBIT` (use `--jobs 1`).
* Runs are deterministic given the same package versions, but small
  numerical differences across platforms and solver versions (e.g. Clarabel,
  SciPy) may change individual runs.

## Repository layout
This repository follows the structure conventions laid out by
[POptUS/template_repo](https://github.com/POptUS/template_repo):

* [fdm](fdm) -- the `fdm` tool, with its Python implementation in
  [fdm/py](fdm/py). See [fdm/README.md](fdm/README.md) for details on running
  its tests and experiments.
* [fdm_pypkg](fdm_pypkg) -- the installable Python package that distributes
  `fdm`'s Python implementation (included via symlink). See
  [fdm_pypkg/README.md](fdm_pypkg/README.md) for installation instructions.

## Installation / Requirements
`fdm` is distributed via [fdm_pypkg](fdm_pypkg). Installing it (e.g.
`pip install -e fdm_pypkg`) pulls in its core dependencies automatically:
* `numpy>=1.24`
* `scipy>=1.11`

With only these installed, `fdm` falls back to built-in SciPy-based
implementations wherever an optional dependency below is unavailable (see
[fdm/py/fallbacks.py](fdm/py/fallbacks.py)).

To run the research scripts in [fdm/py/experiments](fdm/py/experiments)
(`run_experiments.py`, `summarize.py`, the `plot_*.py` scripts, etc.), also
install the `experiments` extra:
```
$ pip install -e "fdm_pypkg[experiments]"
```
which adds:
* `clarabel>=0.9` -- QP/LP solver (falls back to SciPy if absent)
* `matplotlib>=3.7` -- plotting
* `pandas>=2.0` -- results post-processing

Two further dependencies are **not on PyPI** and must be installed from
source if you want full-fidelity (non-fallback) runs rather than the
built-in stubs:
* [ibcdfo](https://github.com/POptUS/IBCDFO) -- provides the real `formquad`
  model builder (`fdm` otherwise uses a stub, see
  [fdm/py/fallbacks.py](fdm/py/fallbacks.py))
* [pycutest](https://jfowkes.github.io/pycutest/) -- provides access to the
  CUTEst problem set (not needed for the built-in `hs` test set)

## Developer Information
The `fdm_pypkg` python package and the management of coverage reports for the
full repository are managed with [tox](https://tox.wiki/en/latest/index.html).
See [fdm_pypkg/tox.ini](fdm_pypkg/tox.ini) for package-level tasks and
[tox.ini](tox.ini) (this directory) for global coverage-aggregation tasks.

CI workflows (GitHub Actions) for `fdm` have not yet been set up; this is
tracked as follow-up work.
