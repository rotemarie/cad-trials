"""Unit tests for the torch-free shared scoring helpers (``common/scoring.py``)."""
import trimesh
import pytest

from cad_trials.common.execute import ExecResult
from cad_trials.common.scoring import make_record, score


def test_score_identical_box(tmp_path):
    box = trimesh.creation.box((2, 1, 1))
    a = tmp_path / "a.stl"
    b = tmp_path / "b.stl"
    box.export(a)
    box.export(b)
    iou, chamfer = score(str(a), str(b))
    assert iou == pytest.approx(1.0, abs=0.03)
    assert chamfer == pytest.approx(0.0, abs=1e-3)


def test_score_missing_file_returns_none(tmp_path):
    box = trimesh.creation.box((1, 1, 1))
    gt = tmp_path / "gt.stl"
    box.export(gt)
    assert score(str(tmp_path / "nope.stl"), str(gt)) == (None, None)


def test_make_record_success_sets_metrics(tmp_path):
    box = trimesh.creation.box((2, 1, 1))
    pred = tmp_path / "pred.stl"
    gt = tmp_path / "gt.stl"
    box.export(pred)
    box.export(gt)
    res = ExecResult(ok=True, stl_path=str(pred), volume=2.0, watertight=True,
                     n_vertices=8)
    rec = make_record(
        model="cadrecode", weights="filapro/cad-recode-v1.5", problem="p1",
        input_path=tmp_path / "pc_256.ply", kind="pc_256", sample=0,
        out_path=tmp_path / "pc_256+s0.py", res=res,
        code="import cadquery as cq\nr = cq.Workplane('XY').box(2,1,1)",
        gt=str(gt), wall_s=1.0, error=None)
    assert rec.model == "cadrecode"
    assert rec.valid_code is True
    assert rec.valid_geometry is True
    assert rec.iou == pytest.approx(1.0, abs=0.03)
    assert rec.chamfer == pytest.approx(0.0, abs=1e-3)
    assert rec.n_ops == 1


def test_make_record_failure_leaves_metrics_none(tmp_path):
    res = ExecResult(ok=False, error="SyntaxError: bad code")
    rec = make_record(
        model="cadrecode", weights="w", problem="p1",
        input_path=tmp_path / "pc_256.ply", kind="pc_256", sample=2,
        out_path=None, res=res, code=None, gt=None,
        wall_s=0.5, error=None)
    assert rec.valid_code is False
    assert rec.valid_geometry is False
    assert rec.iou is None and rec.chamfer is None
    assert rec.n_ops is None
    assert rec.error == "SyntaxError: bad code"
