from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    m = trimesh.load(path, force="mesh")
    if m is None or getattr(m, "faces", None) is None or len(m.faces) == 0:
        raise ValueError(f"empty or unreadable mesh: {path}")
    return m


def normalize_mesh(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_translation(-out.bounds.mean(axis=0))
    factor = scale / out.extents.max()
    out.apply_scale(factor)
    return out


def _farthest_point_sample(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(pts)
    if k >= n:
        return np.arange(n)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = rng.integers(n)
    dist = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(pts - pts[sel[i]], axis=1))
    return sel


def sample_point_cloud(mesh: trimesh.Trimesh, n: int, seed: int = 0,
                       n_pre: int = 8192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pre, _ = trimesh.sample.sample_surface(mesh, n_pre, seed=seed)
    idx = _farthest_point_sample(np.asarray(pre), n, rng)
    return np.asarray(pre)[idx].astype(np.float64)
