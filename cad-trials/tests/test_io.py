from pathlib import Path
from cad_trials.common.io import RunRecord, append_run, load_runs, run_exists


def _rec(**kw):
    base = dict(model="m", weights_id="w", problem="p1", input_path="a.png",
                input_kind="render", sample=0, output_path="o.py", wall_s=1.0, error=None)
    base.update(kw)
    return RunRecord(**base)


def test_append_then_load_roundtrip(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run(_rec(iou=0.5), p)
    append_run(_rec(sample=1, error="boom", output_path=None), p)
    rows = load_runs(p)
    assert len(rows) == 2
    assert rows[0].iou == 0.5
    assert rows[1].error == "boom"


def test_load_missing_file_is_empty(tmp_path):
    assert load_runs(tmp_path / "nope.jsonl") == []


def test_run_exists_ignores_error_rows(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run(_rec(sample=0), p)
    append_run(_rec(sample=1, error="boom"), p)
    assert run_exists("m", "a.png", 0, p) is True
    assert run_exists("m", "a.png", 1, p) is False   # error row does not count
    assert run_exists("m", "a.png", 2, p) is False


def test_append_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "runs.jsonl"
    append_run(_rec(), p)
    assert p.exists()
