# cad-trials/tests/test_base.py
import json
from cad_trials.models._base import (
    standard_parser, iter_inputs, write_output, gt_for)


def test_parser_defaults():
    p = standard_parser("x")
    ns = p.parse_args(["--inputs", "a.png", "b.png", "--out-dir", "out"])
    assert ns.inputs == ["a.png", "b.png"]
    assert ns.n_samples == 5 and ns.seed_base == 0 and ns.problem == "p1"
    assert ns.part == "part" and ns.retry_errors is False


def test_parser_accepts_part_and_retry_errors():
    p = standard_parser("x")
    ns = p.parse_args(["--inputs", "a.png", "--out-dir", "out",
                       "--part", "widget", "--retry-errors"])
    assert ns.part == "widget" and ns.retry_errors is True


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


def test_write_output_prefix_namespaces_parts(tmp_path):
    # two parts sharing an input stem must not overwrite each other in one out_dir
    a = write_output(tmp_path, "pc_256", 0, "A", {}, prefix="alpha+")
    b = write_output(tmp_path, "pc_256", 0, "B", {}, prefix="beta+")
    assert a != b
    assert a.read_text() == "A" and b.read_text() == "B"
    assert (tmp_path / "alpha+pc_256.meta.json").exists()
    assert (tmp_path / "beta+pc_256.meta.json").exists()


def test_write_output_rerun_replaces_meta_entry(tmp_path):
    write_output(tmp_path, "part", 0, "print(1)", {"wall_s": 2.0})
    write_output(tmp_path, "part", 1, "print(2)", {"wall_s": 3.0})
    write_output(tmp_path, "part", 0, "print(9)", {"wall_s": 9.0})   # resumed re-run
    meta = json.loads((tmp_path / "part.meta.json").read_text())
    assert [s["sample"] for s in meta["samples"]] == [0, 1]           # no duplicate
    assert meta["samples"][0]["wall_s"] == 9.0                        # newest wins


def test_gt_for(tmp_path):
    (tmp_path / "part.stl").write_bytes(b"x")
    assert gt_for("part", str(tmp_path)) == str(tmp_path / "part.stl")
    assert gt_for("missing", str(tmp_path)) is None
    assert gt_for("anything", str(tmp_path / "part.stl")) == str(tmp_path / "part.stl")
