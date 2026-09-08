"""Shared CLI contract for model wrappers in the cad-trials harness.

Every model wrapper builds its parser from :func:`standard_parser`, expands its
inputs with :func:`iter_inputs`, writes samples with :func:`write_output`, skips
already-completed work with :func:`already_done`, and resolves ground-truth
meshes with :func:`gt_for`.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from cad_trials.common import io


def standard_parser(description: str) -> argparse.ArgumentParser:
    """Build the argument parser shared by every model wrapper."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="input globs or paths")
    p.add_argument("--out-dir", required=True, help="directory for outputs")
    p.add_argument("--n-samples", type=int, default=5,
                   help="samples to draw per input")
    p.add_argument("--seed-base", type=int, default=0,
                   help="base seed; sample i uses seed_base + i")
    p.add_argument("--weights", default="", help="model weights id or path")
    p.add_argument("--runs-path", default=io.DEFAULT_PATH,
                   help="path to the runs.jsonl ledger")
    p.add_argument("--problem", choices=["p1", "p2"], default="p1",
                   help="which problem variant to run")
    p.add_argument("--gt", default="",
                   help="ground-truth mesh file, or a dir of meshes keyed by stem")
    return p


def iter_inputs(patterns: list[str]) -> list[Path]:
    """Expand globs/paths, dedupe, sort. Raise SystemExit if nothing matched."""
    matched: set[str] = set()
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            matched.update(hits)
        elif Path(pat).exists():
            matched.add(pat)
    if not matched:
        raise SystemExit(f"no inputs matched: {patterns}")
    return sorted((Path(m) for m in matched), key=lambda p: str(p))


def write_output(out_dir, stem: str, sample: int, payload: str, meta: dict,
                 ext: str = "py") -> Path:
    """Write one sample's payload and append its meta to ``<stem>.meta.json``.

    Returns the path of the payload file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{stem}+s{sample}.{ext}"
    out_path.write_text(payload)

    meta_path = out_dir / f"{stem}.meta.json"
    if meta_path.exists():
        doc = json.loads(meta_path.read_text())
    else:
        doc = {"stem": stem, "samples": []}
    doc.setdefault("samples", [])
    entry = {"sample": sample, "output_path": str(out_path), **meta}
    doc["samples"].append(entry)
    meta_path.write_text(json.dumps(doc, indent=2))

    return out_path


def already_done(model: str, input_path: str, sample: int,
                 runs_path: str = io.DEFAULT_PATH) -> bool:
    """True when a successful run for (model, input_path, sample) already exists."""
    return io.run_exists(model, input_path, sample, runs_path)


def gt_for(stem: str, gt_arg: str) -> str | None:
    """Resolve the ground-truth mesh for ``stem``.

    ``gt_arg`` empty -> None; a file -> returned as-is; a dir -> ``<dir>/<stem>.stl``
    when it exists, else None.
    """
    if not gt_arg:
        return None
    p = Path(gt_arg)
    if p.is_file():
        return gt_arg
    if p.is_dir():
        candidate = p / f"{stem}.stl"
        return str(candidate) if candidate.exists() else None
    return None
