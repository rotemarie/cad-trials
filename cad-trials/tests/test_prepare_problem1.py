import json
import trimesh
from cad_trials.data.prepare_problem1 import prepare
from cad_trials.common.render import STYLES


def test_prepare_writes_all_artifacts(tmp_path):
    stl = tmp_path / "in.stl"
    trimesh.creation.box((4, 2, 1)).export(stl)
    out = tmp_path / "prep"
    manifest = prepare(stl, out, seed=0)

    assert (out / "normalized.stl").exists()
    for n in (256, 2048, 8192):
        assert (out / f"pc_{n}.ply").exists()
    for s in STYLES:
        assert (out / f"render_{s}.png").exists()
    assert (out / "tile_4diag.png").exists()
    assert (out / "three_view.png").exists()

    kinds = {i["kind"] for i in manifest["inputs"]}
    assert f"render_{STYLES[0]}" in kinds and "tile_4diag" in kinds
    assert manifest["gt"].endswith("normalized.stl")


def test_prepare_pc_has_right_point_count(tmp_path):
    stl = tmp_path / "in.stl"
    trimesh.creation.box((1, 1, 1)).export(stl)
    prepare(stl, tmp_path / "p", seed=0)
    pc = trimesh.load(tmp_path / "p" / "pc_256.ply")
    assert len(pc.vertices) == 256
