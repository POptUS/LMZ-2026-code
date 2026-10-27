#!/usr/bin/env python3
"""Fork-interface driver DAKOTA spawns once per NOWPAC evaluation.

Dependency-free w.r.t. `fdm` -- unlike matlab_bridge.py, nothing here rebuilds
a Problem; the live Oracle already lives in the long-running `run_nowpac`
process, so this is just a socket client that hands x over and gets back the
row (f and all p constraints, in problem.c(x) order) computed there.
"""
import json
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.environ["DAKOTA_DIR"], "share", "dakota", "Python"))
import dakota.interfacing as di

params, results = di.read_parameters_file()
x = [params[i] for i in range(params.num_variables)]

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(os.environ["NOWPAC_ORACLE_SOCKET"])
sock.sendall((json.dumps({"x": x}) + "\n").encode())
buf = b""
while not buf.endswith(b"\n"):
    chunk = sock.recv(4096)
    if not chunk:
        break
    buf += chunk
sock.close()
reply = json.loads(buf.decode())
if "error" in reply:
    raise RuntimeError(f"oracle server error: {reply['error']}")
row = reply["row"]

results[0].function = row[0]            # objective
for i in range(1, len(row)):
    results[i].function = row[i]        # constraints, same order as problem.c(x)
results.write()
