import os, shutil, pytest
pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None or not os.environ.get("CADRILLE_WEIGHTS_OK"),
    reason="needs GPU + cadrille weights")


def test_one_render_produces_a_row(tmp_path):
    import trimesh
    from cad_trials.common.render import render_style
    from cad_trials.models.cadrille_img import main
    m = trimesh.creation.box((4, 2, 1))
    img = render_style(m, "hlr_lines", size=256)
    img.save(tmp_path / "part.png")
    gt = tmp_path / "part.stl"; m.export(gt)
    runs = tmp_path / "runs.jsonl"
    main(["--inputs", str(tmp_path / "part.png"), "--out-dir", str(tmp_path / "o"),
          "--n-samples", "1", "--gt", str(gt), "--runs-path", str(runs)])
    from cad_trials.common.io import load_runs
    rows = load_runs(runs)
    assert len(rows) == 1 and rows[0].model == "cadrille_img"
