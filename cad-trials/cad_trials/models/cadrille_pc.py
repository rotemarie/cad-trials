"""cadrille point-cloud-branch wrapper on the standard cad-trials harness contract.

Runs the pretrained ``cadrille`` model (Qwen2-VL-2B backbone) on a ``.ply`` point
cloud, executes the generated CadQuery, scores it against a ground-truth mesh, and
appends one :class:`~cad_trials.common.io.RunRecord` per (input, sample).

The shared model-load / generate / scoring / RunRecord surface lives in
:mod:`cad_trials.models._cadrille_common`; this module holds only the
point-cloud-specific parts: loading the ``.ply``, the checkpoint-space
normalization, the ``{"point_cloud": ...}`` message shape, and the ``main`` glue.

CLI::

    python -m cad_trials.models.cadrille_pc \
        --inputs "results/prepared/part/pc_256.ply" \
        --out-dir results/cadrille_pc \
        --gt results/prepared/part/normalized.stl \
        --n-samples 5 [--seed-base 0] [--weights maksimko123/cadrille]

``torch`` / ``transformers`` / ``cadrille`` are imported at module load (via
``_cadrille_common``): importing this module requires a working cadrille
environment (see ``envs/cadrille.md``).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh

from cad_trials.common.execute import execute_program  # noqa: E402
from cad_trials.common.io import append_run  # noqa: E402
from cad_trials.models._base import (  # noqa: E402
    already_done,
    gt_for,
    iter_inputs,
    standard_parser,
    write_output,
)
from cad_trials.models._cadrille_common import (  # noqa: E402
    DEFAULT_WEIGHTS,
    DESCRIPTION,
    generate_codes,
    load_model,
    load_processor,
    make_record,
)


def load_points(path) -> np.ndarray:
    """Load a ``.ply`` point cloud as an ``(N, 3)`` float64 array."""
    obj = trimesh.load(path, process=False)
    pts = np.asarray(obj.vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) point cloud, got {pts.shape} from {path}")
    return pts


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Map our prepared point cloud into cadrille's checkpoint input space.

    ``prepare_problem1.py`` samples points from ``normalized.stl`` -- a unit cube
    centred at the origin, so points lie in ``[-0.5, 0.5]`` (see
    ``cad_trials/common/meshes.py:normalize_mesh``).

    cadrille's TEST-split branch expects the point cloud in ``[0, 1]`` and maps it
    to ``[-1, 1]`` via ``point_cloud = (point_cloud - 0.5) * 2``
    (``vendor/cadrille/dataset.py:167``, ``CadRecodeDataset.get_point_cloud`` else
    branch; the DeepCAD test meshes it consumes are centred at ``0.5``).

    So we shift our ``[-0.5, 0.5]`` points to ``[0, 1]`` first, then apply the
    repo transform.  Net effect: ``points * 2`` landing in ``[-1, 1]``.
    """
    points_0_1 = points + 0.5
    return (points_0_1 - 0.5) * 2.0


def run_cadrille_pc(model, processor, points: np.ndarray, n_samples: int,
                    seed_base: int, temperature: float = 0.8,
                    max_new_tokens: int = 768) -> list[str]:
    """Pure inference: return ``n_samples`` CadQuery code strings for one cloud.

    ``points`` is the raw ``[-0.5, 0.5]`` cloud; it is normalized here, then handed
    to :func:`cad_trials.models._cadrille_common.generate_codes` as a
    ``{"point_cloud": ...}`` message.  ``n_points`` (= the actual point count) is
    passed through so ``pc_2048`` / ``pc_8192`` clouds work as well as ``pc_256``.
    """
    pc = normalize_points(points).astype(np.float32)
    item = {"point_cloud": pc, "description": DESCRIPTION, "file_name": "x"}
    return generate_codes(
        model, processor, item, n_samples=n_samples, seed_base=seed_base,
        temperature=temperature, max_new_tokens=max_new_tokens,
        n_points=pc.shape[0])


def main(argv=None) -> None:
    parser = standard_parser(
        "Run pretrained cadrille on a .ply point cloud and score the CadQuery.")
    args = parser.parse_args(argv)

    weights = args.weights or DEFAULT_WEIGHTS
    inputs = iter_inputs(args.inputs)
    out_dir = Path(args.out_dir)

    model = load_model(weights)
    processor = load_processor()

    for input_path in inputs:
        stem = input_path.stem  # pc_256 / pc_2048 / pc_8192
        gt = gt_for(stem, args.gt)
        for k in range(args.n_samples):
            if already_done("cadrille_pc", str(input_path), k, args.runs_path):
                continue

            t0 = time.perf_counter()
            error: str | None = None
            code: str | None = None
            out_path: Path | None = None
            res = None
            n_points: int | None = None

            try:
                points = load_points(input_path)
                n_points = int(points.shape[0])
                code = run_cadrille_pc(
                    model, processor, points, n_samples=1,
                    seed_base=args.seed_base + k)[0]
            except Exception as e:  # noqa: BLE001 - record and continue
                error = f"{type(e).__name__}: {e}"

            if code is not None:
                meta = {"weights": weights, "n_points": n_points,
                        "seed": args.seed_base + k}
                out_path = write_output(out_dir, stem, k, code, meta, ext="py")
                res = execute_program(code, out_dir / f"{stem}+s{k}.stl")

            append_run(
                make_record(
                    model="cadrille_pc", weights=weights, problem=args.problem,
                    input_path=input_path, kind=stem, sample=k, out_path=out_path,
                    res=res, code=code, gt=gt,
                    wall_s=time.perf_counter() - t0, error=error),
                args.runs_path,
            )


if __name__ == "__main__":
    main()
