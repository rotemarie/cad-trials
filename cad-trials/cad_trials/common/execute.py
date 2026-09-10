"""Sandboxed execution of a model-generated CadQuery program into an STL.

The program is run in a **fresh child interpreter** via :mod:`subprocess` with a
self-contained stdlib-only driver string, so the child imports only ``cadquery``
and ``trimesh``.

Why not :mod:`multiprocessing`?  ``mp.get_context("spawn").Process`` makes the
child re-import the parent's ``__main__`` module before running the target.  The
model wrappers run as ``python -m cad_trials.models.cadrecode`` (see
``slurm/run_model.sbatch``), so every child re-executed that module's top level --
``import torch``, ``transformers``, the vendored model classes.  On the cluster's
NFS conda envs a cold torch import is 15-45 s, spent *inside* the execution
timeout, which turned almost every call into a false ``"timeout after ...s"``.
A plain ``subprocess`` with ``python -c <driver>`` has no such re-import.

Public surface (unchanged): :class:`ExecResult`, :func:`execute_program`,
:func:`valid_geometry`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Stdlib-only driver: imports cadquery + trimesh and nothing from this package.
# argv[1] = path to the generated program, argv[2] = destination STL.
# Emits a single JSON line on stdout.
_DRIVER = r'''
import json, sys, cadquery as cq, trimesh
code = open(sys.argv[1]).read()
out_stl = sys.argv[2]
try:
    env = {}
    exec(code, env)
    obj = env.get("r")
    if obj is None:
        cands = [v for v in env.values() if isinstance(v, cq.Workplane)]
        if not cands:
            raise RuntimeError("no result solid (no `r`, no Workplane)")
        obj = cands[-1]
    compound = obj.val() if hasattr(obj, "val") else obj
    try:
        diag = float(compound.BoundingBox().DiagonalLength)
    except Exception:
        diag = 0.0
    lin = max(diag * 1e-3, 1e-4) if diag > 0 else 0.01
    verts, faces = compound.tessellate(lin, 0.1)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces, process=True)
    mesh.merge_vertices()
    if len(mesh.faces) == 0:
        raise RuntimeError("empty tessellation")
    import os
    os.makedirs(os.path.dirname(out_stl) or ".", exist_ok=True)
    mesh.export(out_stl)
    print(json.dumps(dict(ok=True, volume=float(mesh.volume),
                          watertight=bool(mesh.is_watertight),
                          n_vertices=int(len(mesh.vertices)))))
except Exception as e:
    print(json.dumps(dict(ok=False, error="%s: %s" % (type(e).__name__, e))))
'''


@dataclass
class ExecResult:
    ok: bool
    error: str | None = None
    stl_path: str | None = None
    volume: float | None = None
    watertight: bool | None = None
    n_vertices: int | None = None


def execute_program(code: str, out_stl: str | Path, timeout: float = 30.0) -> ExecResult:
    """Execute ``code`` in a child interpreter and export its solid to ``out_stl``.

    The program may either bind its result to ``r`` or leave the last
    ``cq.Workplane`` in module scope.  Tessellation tolerance is derived from the
    solid's bounding-box diagonal (``max(diag * 1e-3, 1e-4)``) so small and large
    parts get comparable mesh fidelity.
    """
    out_stl = str(out_stl)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        code_path = fh.name
    try:
        proc = subprocess.run([sys.executable, "-c", _DRIVER, code_path, out_stl],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, error=f"timeout after {timeout}s")
    finally:
        Path(code_path).unlink(missing_ok=True)

    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        stderr = (proc.stderr or "")[-500:]
        return ExecResult(ok=False, error=f"driver produced no output; stderr: {stderr}")
    try:
        d = json.loads(lines[-1])
    except Exception:
        return ExecResult(ok=False, error=f"unparseable driver output: {lines[-1][:300]}")
    if not d.get("ok"):
        return ExecResult(ok=False, error=d.get("error", "unknown"))
    return ExecResult(ok=True, stl_path=out_stl, volume=d["volume"],
                      watertight=d["watertight"], n_vertices=d["n_vertices"])


def valid_geometry(res: ExecResult) -> bool:
    return bool(res.ok and res.volume is not None and res.volume > 1e-9)
