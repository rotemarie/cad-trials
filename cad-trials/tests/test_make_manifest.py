from cad_trials.common.render import STYLES
from cad_trials.slurm.make_manifest import IMAGE_KINDS, build_manifest


def _rows(out):
    return [l.split("\t") for l in out.read_text().strip().splitlines()]


def test_manifest_rows(tmp_path):
    prep = tmp_path / "prepared" / "part"
    prep.mkdir(parents=True)
    for f in ["render_hlr_lines.png", "tile_4diag.png", "pc_256.ply"]:
        (prep / f).write_bytes(b"x")
    out = tmp_path / "manifest.tsv"
    build_manifest([str(prep)], ["cadrille_img", "cadrille_pc", "cadrecode"], str(out))
    lines = out.read_text().strip().splitlines()
    assert any("\tcadrille_img\t" in l and "render_hlr_lines.png" in l for l in lines)
    assert any("\tcadrecode\t" in l and "pc_256.ply" in l for l in lines)
    # cadrille_img must NOT get the point cloud
    assert not any("\tcadrille_img\t" in l and "pc_256" in l for l in lines)


def test_manifest_columns_lead_with_part(tmp_path):
    prep = tmp_path / "prepared" / "widget"
    prep.mkdir(parents=True)
    (prep / "pc_256.ply").write_bytes(b"x")
    out = tmp_path / "manifest.tsv"
    build_manifest([str(prep)], ["cadrecode"], str(out))
    (row,) = _rows(out)
    part, model, input_path, input_kind, n_samples, env = row
    assert part == "widget"          # prepared dir basename, column 1
    assert model == "cadrecode"
    assert input_path.endswith("prepared/widget/pc_256.ply")
    assert input_kind == "pc_256" and n_samples == "5" and env == "cadrecode"


def test_manifest_keeps_parts_distinct(tmp_path):
    for name in ("alpha", "beta"):
        d = tmp_path / "prepared" / name
        d.mkdir(parents=True)
        (d / "pc_256.ply").write_bytes(b"x")
    out = tmp_path / "manifest.tsv"
    build_manifest([str(tmp_path / "prepared" / "alpha"),
                    str(tmp_path / "prepared" / "beta")], ["cadrecode"], str(out))
    rows = _rows(out)
    assert sorted(r[0] for r in rows) == ["alpha", "beta"]


def test_image_kinds_derive_from_styles():
    # a style rename must flow into the manifest without a second edit
    assert IMAGE_KINDS == [f"render_{s}" for s in STYLES] + ["tile_4diag", "three_view"]
    assert "render_hlr_paper" in IMAGE_KINDS
    assert "render_draftsheet" not in IMAGE_KINDS
