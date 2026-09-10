"""cadrille image-branch wrapper on the standard cad-trials harness contract.

Runs the pretrained ``cadrille`` model (Qwen2-VL-2B backbone) on rendered PNGs,
executes the generated CadQuery, scores it against a ground-truth mesh, and
appends one :class:`~cad_trials.common.io.RunRecord` per (input, sample).

The shared model-load / generate / scoring / RunRecord surface lives in
:mod:`cad_trials.models._cadrille_common`; this module holds only the
image-specific parts: letterboxing + the training-time black border (with the
``--raw`` / ``tile_4diag`` bypass), the ``{"video": ...}`` message shape, and the
``main`` glue.  Generation logic is ported from the repo-root seed script
``infer_image.py``.

CLI::

    python -m cad_trials.models.cadrille_img \
        --inputs "results/prepared/part/render_*.png" \
        --out-dir results/cadrille_img \
        --gt results/prepared/part/normalized.stl \
        --n-samples 5 [--seed-base 0] [--weights maksimko123/cadrille] [--raw]

``torch`` / ``transformers`` / ``cadrille`` are imported at module load (via
``_cadrille_common``): importing this module requires a working cadrille
environment (see ``envs/cadrille.md``).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

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


def prepare_image(path, raw: bool, img_size: int = 128, border: int = 3,
                  pad_color=(255, 255, 255)) -> Image.Image:
    """Load an image file; letterbox to a square + add the training-time black
    border unless ``raw`` (``raw`` is for the pre-composed ``tile_4diag.png``)."""
    img = Image.open(path).convert("RGB")
    if not raw:
        img = ImageOps.pad(img, (img_size, img_size), color=pad_color)
        img = ImageOps.expand(img, border=border, fill="black")
    return img


def run_cadrille_image(model, processor, image: Image.Image, n_samples: int,
                       seed_base: int, temperature: float = 0.8,
                       max_new_tokens: int = 768) -> list[str]:
    """Pure inference: return ``n_samples`` CadQuery code strings for one image.

    Wraps the image in a ``{"video": [image]}`` message and hands it to
    :func:`cad_trials.models._cadrille_common.generate_codes` (seeded per sample).
    """
    item = {"video": [image], "description": DESCRIPTION, "file_name": "x"}
    return generate_codes(
        model, processor, item, n_samples=n_samples, seed_base=seed_base,
        temperature=temperature, max_new_tokens=max_new_tokens)


def main(argv=None) -> None:
    parser = standard_parser(
        "Run pretrained cadrille on rendered images and score the CadQuery.")
    parser.add_argument(
        "--raw", action="store_true",
        help="force every input through unmodified; by default only the "
             "tile_4diag stem is fed raw and everything else is letterboxed to "
             "128px + given the training-time black border")
    args = parser.parse_args(argv)

    weights = args.weights or DEFAULT_WEIGHTS
    inputs = iter_inputs(args.inputs)
    out_dir = Path(args.out_dir)
    prefix = f"{args.part}+" if args.part else ""

    model = load_model(weights)
    processor = load_processor()

    for input_path in inputs:
        stem = input_path.stem
        # tile_4diag.png is already composed + bordered by prepare_problem1.py, so
        # it must skip the letterbox; render_* / three_view need it. --raw forces all.
        raw = args.raw or stem == "tile_4diag"
        gt = gt_for(stem, args.gt)
        for k in range(args.n_samples):
            if already_done("cadrille_img", str(input_path), k, args.runs_path,
                            retry_errors=args.retry_errors):
                continue

            t0 = time.perf_counter()
            error: str | None = None
            code: str | None = None
            out_path: Path | None = None
            res = None

            # Every per-sample step -- load, generate, write, execute -- is inside
            # this guard: one bad sample must not kill an 8-hour array task.
            try:
                image = prepare_image(input_path, raw=raw)
                code = run_cadrille_image(
                    model, processor, image, n_samples=1,
                    seed_base=args.seed_base + k)[0]
                meta = {"weights": weights, "raw": bool(raw),
                        "seed": args.seed_base + k, "part": args.part}
                out_path = write_output(out_dir, stem, k, code, meta, ext="py",
                                        prefix=prefix)
                res = execute_program(code, out_dir / f"{prefix}{stem}+s{k}.stl")
            except Exception as e:  # noqa: BLE001 - record and continue
                error = f"{type(e).__name__}: {e}"

            try:
                append_run(
                    make_record(
                        model="cadrille_img", weights=weights, problem=args.problem,
                        input_path=input_path, kind=stem, sample=k, out_path=out_path,
                        res=res, code=code, gt=gt, part=args.part,
                        wall_s=time.perf_counter() - t0, error=error),
                    args.runs_path,
                )
            except Exception as e:  # noqa: BLE001 - ledger write must not abort the task
                print(f"warning: could not record {stem}+s{k}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
