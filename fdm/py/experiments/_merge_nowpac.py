#!/usr/bin/env python
"""One-off: merge a NOWPAC-only results pickle into the existing combined
FDM+CONORBIT pickle, by (problem, solver) name. Does not re-run anything."""
import pickle

BASE = "results_q100_partial_with_conorbit.pkl"
NOWPAC_PKL = "results_nowpac_only.pkl"
OUT = "results_q100_partial_with_conorbit_and_nowpac.pkl"

with open(BASE, "rb") as fh:
    base = pickle.load(fh)
with open(NOWPAC_PKL, "rb") as fh:
    nowpac = pickle.load(fh)

n_merged = 0
for name, runs in nowpac["results"].items():
    if "NOWPAC" in runs:
        base["results"].setdefault(name, {})["NOWPAC"] = runs["NOWPAC"]
        n_merged += 1

if "NOWPAC" not in base["meta"]["solvers"]:
    base["meta"]["solvers"] = list(base["meta"]["solvers"]) + ["NOWPAC"]

with open(OUT, "wb") as fh:
    pickle.dump(base, fh)
print(f"merged {n_merged} NOWPAC results into {OUT}")
