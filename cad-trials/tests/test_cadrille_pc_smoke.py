import os, shutil, pytest
pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None or not os.environ.get("CADRILLE_WEIGHTS_OK"),
    reason="needs GPU + cadrille weights")


def test_one_point_cloud_produces_a_row(tmp_path):
    import trimesh
    from cad_trials.common.meshes import normalize_mesh, sample_point_cloud
    from cad_trials.models.cadrille_pc import main
    m = normalize_mesh(trimesh.creation.box((4, 2, 1)))
    pts = sample_point_cloud(m, 256, seed=0)
    trimesh.PointCloud(pts).export(tmp_path / "pc_256.ply")
    gt = tmp_path / "part.stl"; m.export(gt)
    runs = tmp_path / "runs.jsonl"
    main(["--inputs", str(tmp_path / "pc_256.ply"), "--out-dir", str(tmp_path / "o"),
          "--n-samples", "1", "--gt", str(gt), "--runs-path", str(runs)])
    from cad_trials.common.io import load_runs
    rows = load_runs(runs)
    assert len(rows) == 1 and rows[0].model == "cadrille_pc"
