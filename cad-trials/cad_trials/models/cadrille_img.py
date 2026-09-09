"""cadrille image-branch wrapper on the standard cad-trials harness contract.

Runs the pretrained ``cadrille`` model (Qwen2-VL-2B backbone) on rendered PNGs,
executes the generated CadQuery, scores it against a ground-truth mesh, and
appends one :class:`~cad_trials.common.io.RunRecord` per (input, sample).

Generation logic is ported from the repo-root seed script ``infer_image.py``:
same ``collate`` + ``model.generate`` + ``batch_decode`` path, restructured onto
:mod:`cad_trials.models._base`.

CLI::

    python -m cad_trials.models.cadrille_img \
        --inputs "results/prepared/part/render_*.png" \
        --out-dir results/cadrille_img \
        --gt results/prepared/part/normalized.stl \
        --n-samples 5 [--seed-base 0] [--weights maksimko123/cadrille] [--raw]

``torch`` / ``transformers`` / ``cadrille`` are imported at module load: importing
this module requires a working cadrille environment (see ``envs/cadrille.md``).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor

# vendored upstream repo (git-ignored: cad-trials/vendor/cadrille) must be importable
sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "cadrille"))
from cadrille import Cadrille, collate  # noqa: E402

from cad_trials.common.execute import execute_program, valid_geometry  # noqa: E402
from cad_trials.common.io import RunRecord, append_run  # noqa: E402
from cad_trials.common.meshes import load_mesh  # noqa: E402
from cad_trials.common.metrics import chamfer_distance, count_ops, voxel_iou  # noqa: E402
from cad_trials.models._base import (  # noqa: E402
    already_done,
    gt_for,
    iter_inputs,
    standard_parser,
    write_output,
)

DEFAULT_WEIGHTS = "maksimko123/cadrille"
PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"
DESCRIPTION = "Generate cadquery code"


def load_model(weights_id: str) -> "Cadrille":
    """Load cadrille weights onto CUDA (or CPU), trying fast attention first."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for attn in ("flash_attention_2", "sdpa"):
        try:
            model = Cadrille.from_pretrained(
                weights_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn,
                device_map=device,
            ).eval()
            print(f"loaded {weights_id} on {device} with attn={attn}")
            return model
        except (ImportError, ValueError) as e:
            print(f"attn={attn} unavailable ({e}); falling back")
    raise RuntimeError("could not load cadrille model")


def load_processor() -> AutoProcessor:
    return AutoProcessor.from_pretrained(
        PROCESSOR_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
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

    A distinct torch seed (``seed_base + k``) is set before each ``generate`` so
    sampled decodes differ. ``temperature > 0`` -> ``do_sample=True``.
    """
    do_sample = temperature is not None and temperature > 0
    item = {"video": [image], "description": DESCRIPTION, "file_name": "x"}
    codes: list[str] = []
    for k in range(n_samples):
        torch.manual_seed(seed_base + k)
        batch = collate([item], processor=processor, n_points=256, eval=True)
        pvv = batch.get("pixel_values_videos")
        vgt = batch.get("video_grid_thw")
        with torch.no_grad():
            generated = model.generate(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                point_clouds=batch["point_clouds"].to(model.device),
                is_pc=batch["is_pc"].to(model.device),
                is_img=batch["is_img"].to(model.device),
                pixel_values_videos=pvv.to(model.device) if pvv is not None else None,
                video_grid_thw=vgt.to(model.device) if vgt is not None else None,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        trimmed = [o[len(i):] for i, o in zip(batch["input_ids"], generated)]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        codes.append(decoded[0])
    return codes


def _score(pred_stl: str, gt_path: str) -> tuple[float | None, float | None]:
    """(voxel_iou, chamfer) for a predicted STL vs GT; None on any load/metric error."""
    try:
        pred_mesh = load_mesh(pred_stl)
        gt_mesh = load_mesh(gt_path)
    except Exception:
        return None, None
    iou = chamfer = None
    try:
        iou = voxel_iou(pred_mesh, gt_mesh)
    except Exception:
        iou = None
    try:
        chamfer = chamfer_distance(pred_mesh, gt_mesh)
    except Exception:
        chamfer = None
    return iou, chamfer


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

    model = load_model(weights)
    processor = load_processor()

    for input_path in inputs:
        stem = input_path.stem
        # tile_4diag.png is already composed + bordered by prepare_problem1.py, so
        # it must skip the letterbox; render_* / three_view need it. --raw forces all.
        raw = args.raw or stem == "tile_4diag"
        gt = gt_for(stem, args.gt)
        for k in range(args.n_samples):
            if already_done("cadrille_img", str(input_path), k, args.runs_path):
                continue

            t0 = time.perf_counter()
            error: str | None = None
            code: str | None = None
            out_path: Path | None = None
            res = None
            iou = chamfer = None
            n_ops = None

            try:
                image = prepare_image(input_path, raw=raw)
                code = run_cadrille_image(
                    model, processor, image, n_samples=1,
                    seed_base=args.seed_base + k)[0]
            except Exception as e:  # noqa: BLE001 - record and continue
                error = f"{type(e).__name__}: {e}"

            if code is not None:
                meta = {"weights": weights, "raw": bool(raw),
                        "seed": args.seed_base + k}
                out_path = write_output(out_dir, stem, k, code, meta, ext="py")
                res = execute_program(code, out_dir / f"{stem}+s{k}.stl")
                if res.error and error is None:
                    error = res.error
                n_ops = count_ops(code)
                if res.ok and gt:
                    iou, chamfer = _score(res.stl_path, gt)

            append_run(
                RunRecord(
                    model="cadrille_img",
                    weights_id=weights,
                    problem=args.problem,
                    input_path=str(input_path),
                    input_kind=stem,
                    sample=k,
                    output_path=str(out_path) if out_path is not None else None,
                    wall_s=time.perf_counter() - t0,
                    error=error,
                    valid_code=bool(res.ok) if res is not None else False,
                    valid_geometry=valid_geometry(res) if res is not None else False,
                    iou=iou,
                    chamfer=chamfer,
                    n_ops=n_ops,
                    gt_path=gt,
                ),
                args.runs_path,
            )


if __name__ == "__main__":
    main()
