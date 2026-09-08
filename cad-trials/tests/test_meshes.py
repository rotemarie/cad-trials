import numpy as np
import trimesh
import pytest
from cad_trials.common.meshes import load_mesh, normalize_mesh, sample_point_cloud


def test_normalize_centers_and_scales():
    m = trimesh.creation.box((10, 4, 2))
    m.apply_translation((100, 100, 100))
    out = normalize_mesh(m, scale=1.0)
    assert np.allclose(out.bounds.mean(axis=0), 0, atol=1e-6)
    assert out.extents.max() == pytest.approx(1.0, abs=1e-6)
    # original untouched
    assert not np.allclose(m.bounds.mean(axis=0), 0)


def test_normalize_scale_two():
    out = normalize_mesh(trimesh.creation.box((3, 3, 3)), scale=2.0)
    assert out.extents.max() == pytest.approx(2.0, abs=1e-6)


def test_sample_point_cloud_shape_and_determinism():
    m = trimesh.creation.box((1, 1, 1))
    a = sample_point_cloud(m, 256, seed=0)
    b = sample_point_cloud(m, 256, seed=0)
    c = sample_point_cloud(m, 256, seed=1)
    assert a.shape == (256, 3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # points lie on the cube surface (within tolerance)
    assert np.all(np.abs(a) <= 0.5 + 1e-6)


def test_sample_point_cloud_spread():
    # FPS should cover all 6 faces of a cube -> large bounding box
    pts = sample_point_cloud(trimesh.creation.box((1, 1, 1)), 256, seed=0)
    assert np.ptp(pts, axis=0).min() > 0.9


def test_load_mesh_missing(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        load_mesh(tmp_path / "nope.stl")
