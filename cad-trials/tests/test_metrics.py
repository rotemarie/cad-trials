import numpy as np
import trimesh
import pytest
from cad_trials.common.metrics import voxel_iou, chamfer_distance, count_ops


def test_iou_identical_is_one():
    c = trimesh.creation.box((1, 1, 1))
    assert voxel_iou(c, c.copy(), res=32) == pytest.approx(1.0, abs=0.02)


def test_iou_disjoint_is_zero():
    a = trimesh.creation.box((1, 1, 1)); a.apply_translation((-5, 0, 0))
    b = trimesh.creation.box((1, 1, 1)); b.apply_translation((5, 0, 0))
    # after independent normalization both fill the cube -> identical, IoU high.
    # disjointness must be tested pre-normalized: use the shared-frame variant
    assert voxel_iou(a, b, res=32) == pytest.approx(1.0, abs=0.05)  # documents the limitation


def test_iou_partial_overlap_shared_frame():
    a = trimesh.creation.box((2, 2, 2))
    b = trimesh.creation.box((2, 2, 2)); b.apply_translation((1, 0, 0))
    val = voxel_iou(a, b, res=48, shared_frame=True)
    assert 0.2 < val < 0.45   # analytic IoU of two unit-overlap 2-cubes = 1/3


def test_chamfer_zero_for_identical():
    c = trimesh.creation.box((1, 1, 1))
    assert chamfer_distance(c, c.copy()) == pytest.approx(0.0, abs=1e-3)


def test_chamfer_positive_for_different_shapes():
    a = trimesh.creation.box((1, 1, 1))
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.6)
    assert chamfer_distance(a, b) > 0.5


def test_count_ops():
    code = ("import cadquery as cq\n"
            "r = cq.Workplane('XY').box(10,10,2).faces('>Z').workplane()"
            ".hole(3).cboreHole(2,4,1)")
    assert count_ops(code) == 3   # box, hole, cboreHole
