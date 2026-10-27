fdm
===
TODO: one or two sentence description of the `fdm` package, to be presented to
users on PyPI, with a pointer to the package's online documentation once it
exists.

This package distributes the `fdm` tool's Python implementation, developed in
[../fdm/py](../fdm/py) and included here via symlink
([src/fdm/fdm](src/fdm/fdm)), following the conventions of
[template_repo](https://github.com/POptUS/template_repo).

## Installation
```
$ pip install -e .
```
installs the core dependencies only (`numpy`, `scipy`). To also run the
research scripts in [../fdm/py/experiments](../fdm/py/experiments), install
the `experiments` extra instead:
```
$ pip install -e ".[experiments]"
```
See the top-level [../README.md](../README.md#installation--requirements)
for the full list of dependencies, including two (`ibcdfo`, `pycutest`) that
are optional, not on PyPI, and needed only for full-fidelity (non-fallback)
runs.

## Running tests
See [../fdm/README.md](../fdm/README.md) for the full testing story
(`unittest`-based tests vs. `pytest`-based smoke tests vs. experiments).
```
$ python -m unittest fdm -v
```
