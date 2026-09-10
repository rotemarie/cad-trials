from cad_trials.slurm.make_manifest import build_manifest


def test_manifest_rows(tmp_path):
    prep = tmp_path / "prepared" / "part"
    prep.mkdir(parents=True)
    for f in ["render_hlr_lines.png", "tile_4diag.png", "pc_256.ply"]:
        (prep / f).write_bytes(b"x")
    out = tmp_path / "manifest.tsv"
    build_manifest([str(prep)], ["cadrille_img", "cadrille_pc", "cadrecode"], str(out))
    lines = out.read_text().strip().splitlines()
    assert any("cadrille_img\t" in l and "render_hlr_lines.png" in l for l in lines)
    assert any("cadrecode\t" in l and "pc_256.ply" in l for l in lines)
    # cadrille_img must NOT get the point cloud
    assert not any("cadrille_img\t" in l and "pc_256" in l for l in lines)
