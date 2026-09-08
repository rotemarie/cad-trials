from __future__ import annotations
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    ok: bool
    error: str | None = None
    stl_path: str | None = None
    volume: float | None = None
    watertight: bool | None = None
    n_vertices: int | None = None


def _worker(code: str, out_stl: str, q) -> None:
    try:
        import cadquery as cq
        import trimesh
        env: dict = {}
        exec(code, env)
        obj = env.get("r")
        if obj is None:
            cands = [v for v in env.values() if isinstance(v, cq.Workplane)]
            if not cands:
                raise RuntimeError("no result solid (no `r`, no Workplane)")
            obj = cands[-1]
        compound = obj.val() if hasattr(obj, "val") else obj
        verts, faces = compound.tessellate(0.1, 0.1)
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces, process=False)
        if len(mesh.faces) == 0:
            raise RuntimeError("empty tessellation")
        Path(out_stl).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_stl)
        q.put(dict(ok=True, volume=float(mesh.volume),
                   watertight=bool(mesh.is_watertight), n_vertices=len(mesh.vertices)))
    except Exception as e:  # noqa: BLE001 - report everything
        q.put(dict(ok=False, error=f"{type(e).__name__}: {e}"))


def execute_program(code: str, out_stl: str | Path, timeout: float = 15.0) -> ExecResult:
    out_stl = str(out_stl)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, out_stl, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return ExecResult(ok=False, error=f"timeout after {timeout}s")
    try:
        d = q.get_nowait()
    except Exception:
        return ExecResult(ok=False, error="worker died without result")
    if not d.get("ok"):
        return ExecResult(ok=False, error=d.get("error", "unknown"))
    return ExecResult(ok=True, stl_path=out_stl, volume=d["volume"],
                      watertight=d["watertight"], n_vertices=d["n_vertices"])


def valid_geometry(res: ExecResult) -> bool:
    return bool(res.ok and res.volume is not None and res.volume > 1e-9)
