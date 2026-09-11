"""Prepare Problem-1 inputs from a single STL file.

Produces:
- Normalized meshes (scale 1 and scale 2)
- Point clouds at 256, 2048, 8192 points
- Renders in all 5 styles
- Tile and three-view renders

Returns a manifest with ground-truth mesh and input files.
"""
from __future__ import annotations

from pathlib import Path
import trimesh

from cad_trials.common.meshes import load_mesh, normalize_mesh, sample_point_cloud
from cad_trials.common.render import STYLES, render_style, four_diagonal_tile, ortho_three_view


def prepare(stl_path: str | Path, out_dir: str | Path, seed: int = 0) -> dict:
    """Convert STL to all Problem-1 model inputs.

    Args:
        stl_path: Path to input STL file
        out_dir: Directory to write outputs
        seed: Random seed for point cloud sampling

    Returns:
        Manifest dict with "gt" (normalized.stl path) and "inputs" (list of input files)
    """
    stl_path = Path(stl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh = load_mesh(stl_path)

    # Normalize and save at scale 1
    norm_1 = normalize_mesh(mesh, scale=1.0)
    norm_1_path = out_dir / "normalized.stl"
    norm_1.export(norm_1_path)

    # Normalize and save at scale 2
    norm_2 = normalize_mesh(mesh, scale=2.0)
    norm_2_path = out_dir / "normalized_2u.stl"
    norm_2.export(norm_2_path)

    # Sample and save point clouds
    inputs = []
    for n in (256, 2048, 8192):
        pc_points = sample_point_cloud(norm_1, n, seed=seed)
        pc_mesh = trimesh.PointCloud(pc_points)
        pc_path = out_dir / f"pc_{n}.ply"
        pc_mesh.export(pc_path)
        inputs.append({"path": str(pc_path), "kind": f"pc_{n}"})

    # Render each style
    for style in STYLES:
        img = render_style(norm_1, style, size=256)
        style_path = out_dir / f"render_{style}.png"
        img.save(style_path)
        inputs.append({"path": str(style_path), "kind": f"render_{style}"})

    # Create and save tile render
    tile_img = four_diagonal_tile(norm_1, size=128)
    tile_path = out_dir / "tile_4diag.png"
    tile_img.save(tile_path)
    inputs.append({"path": str(tile_path), "kind": "tile_4diag"})

    # Create and save three-view render
    three_view_img = ortho_three_view(norm_1, size=256)
    three_view_path = out_dir / "three_view.png"
    three_view_img.save(three_view_path)
    inputs.append({"path": str(three_view_path), "kind": "three_view"})

    # Return manifest
    return {
        "gt": str(norm_1_path),
        "inputs": inputs,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Problem-1 inputs from STL")
    parser.add_argument("--stl", required=True, help="Path to input STL file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point clouds")

    args = parser.parse_args()
    manifest = prepare(args.stl, args.out_dir, seed=args.seed)
    print(f"Manifest GT: {manifest['gt']}")
    print(f"Generated {len(manifest['inputs'])} input files")
