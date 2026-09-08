# cad-trials/tests/test_base.py
import json
from cad_trials.models._base import (
    standard_parser, iter_inputs, write_output, gt_for)


def test_parser_defaults():
    p = standard_parser("x")
    ns = p.parse_args(["--inputs", "a.png", "b.png", "--out-dir", "out"])
    assert ns.inputs == ["a.png", "b.png"]
    assert ns.n_samples == 5 and ns.seed_base == 0 and ns.problem == "p1"


def test_iter_inputs_expands_and_sorts(tmp_path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    got = iter_inputs([str(tmp_path / "*.png")])
    assert [p.name for p in got] == ["a.png", "b.png"]


def test_iter_inputs_empty_raises():
    import pytest
    with pytest.raises(SystemExit):
        iter_inputs(["/no/such/*.xyz"])


def test_write_output_creates_files_and_meta(tmp_path):
    write_output(tmp_path, "part", 0, "print(1)", {"wall_s": 2.0})
    write_output(tmp_path, "part", 1, "print(2)", {"wall_s": 3.0})
    assert (tmp_path / "part+s0.py").read_text() == "print(1)"
    meta = json.loads((tmp_path / "part.meta.json").read_text())
    assert len(meta["samples"]) == 2 and meta["samples"][1]["wall_s"] == 3.0


def test_gt_for(tmp_path):
    (tmp_path / "part.stl").write_bytes(b"x")
    assert gt_for("part", str(tmp_path)) == str(tmp_path / "part.stl")
    assert gt_for("missing", str(tmp_path)) is None
    assert gt_for("anything", str(tmp_path / "part.stl")) == str(tmp_path / "part.stl")
