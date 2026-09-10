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


def test_run_exists_matches_any_recorded_attempt(tmp_path):
    # Generation/execution failures are the common outcome here; treating them as
    # "not done" made every array-job resume re-run and double-count them.
    p = tmp_path / "runs.jsonl"
    append_run(_rec(sample=0), p)
    append_run(_rec(sample=1, error="boom"), p)
    assert run_exists("m", "a.png", 0, p) is True
    assert run_exists("m", "a.png", 1, p) is True    # error row counts as attempted
    assert run_exists("m", "a.png", 2, p) is False


def test_run_exists_retry_errors_reopens_error_rows(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run(_rec(sample=0), p)
    append_run(_rec(sample=1, error="boom"), p)
    assert run_exists("m", "a.png", 0, p, retry_errors=True) is True
    assert run_exists("m", "a.png", 1, p, retry_errors=True) is False


def test_load_runs_skips_torn_lines(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run(_rec(sample=0), p)
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"model": "m", "input_pa\n')     # torn concurrent append
    append_run(_rec(sample=1), p)
    rows = load_runs(p)
    assert [r.sample for r in rows] == [0, 1]


def test_load_runs_reads_a_shard_directory(tmp_path):
    d = tmp_path / "runs.d"
    append_run(_rec(sample=0), d / "1.jsonl")
    append_run(_rec(sample=1), d / "2.jsonl")
    rows = load_runs(d)
    assert sorted(r.sample for r in rows) == [0, 1]


def test_new_fields_default_and_roundtrip(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run(_rec(), p)                                   # legacy-shaped row
    append_run(_rec(sample=1, watertight=True, part="widget"), p)
    rows = load_runs(p)
    assert rows[0].watertight is None and rows[0].part is None
    assert rows[1].watertight is True and rows[1].part == "widget"


def test_load_runs_tolerates_rows_without_new_fields(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(
        '{"model":"m","weights_id":"w","problem":"p1","input_path":"a.png",'
        '"input_kind":"render","sample":0,"output_path":null,"wall_s":1.0,'
        '"error":null,"unknown_future_field":42}\n', encoding="utf-8")
    rows = load_runs(p)
    assert len(rows) == 1 and rows[0].part is None


def test_append_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "runs.jsonl"
    append_run(_rec(), p)
    assert p.exists()
