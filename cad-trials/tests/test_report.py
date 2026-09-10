from cad_trials.common.io import RunRecord, append_run
from cad_trials.common.report import build_report, coverage_table


def _rows(tmp_path):
    p = tmp_path / "runs.jsonl"
    for k in range(4):
        append_run(RunRecord(model="cadrille_img", weights_id="w", problem="p1",
                             input_path=f"x{k}.png", input_kind="render_hlr_lines",
                             sample=0, output_path=f"x{k}.py", wall_s=1.0,
                             error=None, valid_code=(k > 0), valid_geometry=(k > 1),
                             iou=(0.3 if k > 1 else None)), p)
    append_run(RunRecord(model="cadrecode", weights_id="w", problem="p1",
                         input_path="pc.ply", input_kind="pc_256", sample=0,
                         output_path="pc.py", wall_s=2.0, error=None,
                         valid_code=True, valid_geometry=True, iou=0.9), p)
    return p


def test_coverage_table(tmp_path):
    from cad_trials.common.io import load_runs
    tbl = coverage_table(load_runs(_rows(tmp_path)))
    cad = next(r for r in tbl if r["model"] == "cadrille_img")
    assert cad["n"] == 4
    assert cad["valid_code_pct"] == 75.0
    assert cad["valid_geom_pct"] == 50.0
    assert abs(cad["mean_iou"] - 0.3) < 1e-6


def test_build_report_writes_html(tmp_path):
    out = tmp_path / "report.html"
    build_report(_rows(tmp_path), out)
    html = out.read_text()
    assert "<title>" in html and "cadrille_img" in html and "coverage" in html.lower()
