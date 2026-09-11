"""Reconstruction metrics: voxel IoU, Chamfer distance, CadQuery op count.

Used to score a model-reconstructed CAD program against a ground-truth mesh.
"""
from __future__ import annotations

import re

import numpy as np
import trimesh

from .meshes import normalize_mesh

_OP_RE = re.compile(
    r"\.(extrude|revolve|loft|sweep|box|sphere|cylinder|cut|union|intersect|"
    r"hole|cboreHole|cskHole)\s*\(")


class FillFailed(RuntimeError):
    """Interior voxel fill failed -- the mesh is open / non-watertight.

    Raised instead of silently falling back to a *surface-only* occupancy grid:
    a surface-only grid compared against a solid-filled one yields a plausible
    but meaningless low IoU, indistinguishable from a genuinely wrong shape.  An
    unscorable ``iou=None`` is an honest gap; a fill artifact dressed as a score
    is not.
    """


def _voxel_occupancy(mesh: trimesh.Trimesh, res: int, bounds: np.ndarray) -> np.ndarray:
    """Boolean occupancy grid for ``mesh`` binned into a common ``bounds`` frame.

    ``bounds`` is ``[[lo], [hi]]``; ``pitch`` is derived from the longest axis of
    that shared box so two meshes voxelised against the same ``bounds`` land on the
    same lattice. The mesh is voxelised with ``trimesh`` and interior-filled;
    a failed fill raises :class:`FillFailed`.
    """
    lo = np.asarray(bounds[0], dtype=float)
    hi = np.asarray(bounds[1], dtype=float)
    pitch = float((hi - lo).max()) / res
    dims = np.ceil((hi - lo) / pitch).astype(int) + 2

    try:
        vg = mesh.voxelized(pitch=pitch).fill()
        pts = np.asarray(vg.points)
    except Exception as e:  # noqa: BLE001 - any fill failure is unscorable
        raise FillFailed(f"interior voxel fill failed: {type(e).__name__}: {e}") from e

    grid = np.zeros(tuple(dims), dtype=bool)
    if len(pts) == 0:
        return grid
    ijk = np.floor((pts - lo) / pitch + 1e-9).astype(int)
    ok = ((ijk >= 0) & (ijk < dims)).all(axis=1)
    ijk = ijk[ok]
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return grid


def voxel_iou(a: trimesh.Trimesh, b: trimesh.Trimesh, res: int = 64,
              shared_frame: bool = False) -> float:
    """Volumetric intersection-over-union of two meshes.

    ``shared_frame=False`` (default, used by the harness for model-vs-GT):
    normalise each mesh independently to the unit cube, then voxelise both against
    that cube. This is translation/scale invariant but **not** rotation invariant
    and collapses genuinely disjoint shapes onto each other — a documented
    limitation; there is no rotation search.

    ``shared_frame=True``: skip per-mesh normalisation and voxelise both in a
    common frame (the union of their bounds) for an honest overlap measurement.

    Raises :class:`FillFailed` when either mesh cannot be interior-filled.
    """
    if shared_frame:
        lo = np.minimum(a.bounds[0], b.bounds[0])
        hi = np.maximum(a.bounds[1], b.bounds[1])
        bounds = np.array([lo, hi])
        ga = _voxel_occupancy(a, res, bounds)
        gb = _voxel_occupancy(b, res, bounds)
    else:
        na, nb = normalize_mesh(a), normalize_mesh(b)
        bounds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        ga = _voxel_occupancy(na, res, bounds)
        gb = _voxel_occupancy(nb, res, bounds)

    inter = int(np.logical_and(ga, gb).sum())
    union = int(np.logical_or(ga, gb).sum())
    return float(inter / union) if union else 0.0


def chamfer_distance(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 8192,
                     seed: int = 0) -> float:
    """Symmetric mean-squared nearest-neighbour distance between surface samples.

    Both meshes are normalised to the unit cube first. ``seed`` is applied to both
    surface-sampling calls so the score is deterministic. Scaled x1000 to match the
    cad-recode notebook convention.
    """
    from scipy.spatial import cKDTree

    na, nb = normalize_mesh(a), normalize_mesh(b)
    pa, _ = trimesh.sample.sample_surface(na, n, seed=seed)
    pb, _ = trimesh.sample.sample_surface(nb, n, seed=seed)
    pa, pb = np.asarray(pa), np.asarray(pb)
    da, _ = cKDTree(pb).query(pa)
    db, _ = cKDTree(pa).query(pb)
    return float((np.mean(da ** 2) + np.mean(db ** 2)) * 1000.0)


def count_ops(code: str) -> int:
    """Number of solid-producing CadQuery calls in ``code`` (regex match count)."""
    return len(_OP_RE.findall(code))
