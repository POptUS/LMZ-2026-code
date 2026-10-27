import codecs

from pathlib import Path
from setuptools import setup

_PKG_ROOT = Path(__file__).resolve().parent


def readme_md():
    fname = _PKG_ROOT.joinpath("README.md")
    with codecs.open(fname, encoding="utf8") as fptr:
        return fptr.read()


def version():
    fname = _PKG_ROOT.joinpath("VERSION")
    with open(fname, "r") as fptr:
        return fptr.read().strip()


def requirements():
    fname = _PKG_ROOT.joinpath("oldest_reqs.txt")
    with open(fname, "r") as fptr:
        return [line.replace("==", ">=") for line in fptr.readlines()]


# Changes made to python_requires should be propagated to all tox.ini and all
# GitHub Action config files.
python_requires = ">=3.8"

# Optional dependencies only needed by fdm.fdm.experiments scripts, not by the
# core fdm package (which falls back to scipy-only implementations).
extras_require = {
    "experiments": ["clarabel", "matplotlib", "pandas"],
}

project_urls = {
    "Source": "https://github.com/POptUS/lara-fdm",
    "Tracker": "https://github.com/POptUS/lara-fdm/issues",
}

setup(
    name="fdm",
    version=version(),
    author="POptUS",
    maintainer="POptUS",
    package_dir={"": "src"},
    url="https://github.com/POptUS/lara-fdm",
    project_urls=project_urls,
    license="MIT",
    description="TODO: one-line description of the fdm tool",
    long_description=readme_md(),
    long_description_content_type="text/markdown",
    python_requires=python_requires,
    install_requires=requirements(),
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
    ],
)
