"""Manifest builder for Slurm array job harness."""

import os
import subprocess
from pathlib import Path
from typing import Sequence

# render.STYLES is the single source of truth for the image kinds; importing it
# keeps the manifest in step with a style rename/addition.  render.py is
# torch-free (matplotlib only), so this stays importable in probe-common.
from cad_trials.common.render import STYLES

# every image input_kind prepare_problem1.py emits
IMAGE_KINDS = [f"render_{s}" for s in STYLES] + ["tile_4diag", "three_view"]


# Model → (env, kinds) mapping
MODEL_KINDS = {
    "cadrille_img": {
        "env": "cadrille",
        "kinds": IMAGE_KINDS,
    },
    "cadrille_pc": {
        "env": "cadrille",
        "kinds": ["pc_256"],
    },
    "cadrecode": {
        "env": "cadrecode",
        "kinds": ["pc_256"],
    },
}


def _get_repo_root() -> Path:
    """Find repo root using git, or return current directory if git fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()


def build_manifest(
    prepared_dirs: Sequence[str], models: Sequence[str], out_tsv: str
) -> None:
    """
    Build manifest TSV for Slurm array job.

    Args:
        prepared_dirs: List of paths to prepared directories (e.g., cad-trials/results/prepared/part)
        models: List of model names to include (e.g., ["cadrille_img", "cadrille_pc", "cadrecode"])
        out_tsv: Path to output TSV file

    The TSV has columns:
    part<TAB>model<TAB>input_path<TAB>input_kind<TAB>n_samples<TAB>env

    ``part`` is the prepared dir's basename.  It disambiguates rows once more than
    one part is prepared: output filenames are prefixed with it and the ground-truth
    mesh is resolved next to the input, so two parts sharing an input stem
    (``pc_256.ply``) neither collide on disk nor get scored against each other's GT.

    Each row represents one task for the array job.
    Paths are stored as repo-root-relative (e.g., cad-trials/results/prepared/part/file.png).
    """
    repo_root = _get_repo_root()
    rows = []

    for prep_dir_str in prepared_dirs:
        prep_dir = Path(prep_dir_str)
        if not prep_dir.is_dir():
            continue
        part = prep_dir.name

        for model in models:
            if model not in MODEL_KINDS:
                continue

            config = MODEL_KINDS[model]
            env = config["env"]
            kinds = config["kinds"]
            n_samples = 5  # All models use 5 samples

            # Find files matching the kinds
            for file_path in sorted(prep_dir.glob("*")):
                if not file_path.is_file():
                    continue

                stem = file_path.stem  # filename without extension

                # Check if this file's stem matches any accepted kind
                if stem in kinds:
                    # Convert to repo-root-relative path
                    try:
                        rel_path = os.path.relpath(file_path, repo_root)
                    except ValueError:
                        # Fallback if relpath fails (e.g., different drives on Windows)
                        rel_path = str(file_path)

                    rows.append(
                        f"{part}\t{model}\t{rel_path}\t{stem}\t{n_samples}\t{env}"
                    )

    # Write manifest
    out_path = Path(out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n" if rows else "")


if __name__ == "__main__":
    # Dry-run: build manifest for prepared/part directory
    import sys

    repo_root = _get_repo_root()
    prep_dir = repo_root / "cad-trials" / "results" / "prepared" / "part"
    out_file = repo_root / "cad-trials" / "results" / "manifest.tsv"
    models_list = ["cadrille_img", "cadrille_pc", "cadrecode"]

    build_manifest([str(prep_dir)], models_list, str(out_file))

    if out_file.exists():
        print(f"✓ Manifest written to {out_file}")
        print(f"  ({out_file.read_text().count(chr(10))} tasks)")
        print("\nFirst 10 lines:")
        lines = out_file.read_text().strip().splitlines()
        for line in lines[:10]:
            print(f"  {line}")
    else:
        print(f"✗ Failed to write manifest to {out_file}")
        sys.exit(1)
