"""Shared surface for the cadrille model wrappers (:mod:`cadrille_img`,
:mod:`cadrille_pc`).

Both wrappers drive the same pretrained ``cadrille`` model (Qwen2-VL-2B backbone)
through the same ``collate`` + ``model.generate`` + ``batch_decode`` path and emit
the same :class:`~cad_trials.common.io.RunRecord` shape; only the per-modality
input preparation differs.  That common part lives here:

* the ``vendor/cadrille`` import bootstrap and the ``maksimko123/cadrille`` /
  ``Qwen2-VL`` constants,
* :func:`load_model` / :func:`load_processor`,
* :func:`generate_codes` -- the seeded ``collate``/``generate``/decode loop, given
  a ready per-modality message dict,
* :func:`score` -- predicted-STL-vs-GT metrics,
* :func:`make_record` -- the RunRecord construction (identical bar ``model=``).

``torch`` / ``transformers`` / ``cadrille`` are imported at module load: importing
this module requires a working cadrille environment (see ``envs/cadrille.md``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

# vendored upstream repo (git-ignored: cad-trials/vendor/cadrille) must be importable
sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "cadrille"))
from cadrille import Cadrille, collate  # noqa: E402

from cad_trials.common.execute import valid_geometry  # noqa: E402
from cad_trials.common.io import RunRecord  # noqa: E402
from cad_trials.common.meshes import load_mesh  # noqa: E402
from cad_trials.common.metrics import chamfer_distance, count_ops, voxel_iou  # noqa: E402

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


def generate_codes(model, processor, item: dict, n_samples: int, seed_base: int,
                   temperature: float = 0.8, max_new_tokens: int = 768,
                   n_points: int = 256) -> list[str]:
    """Return ``n_samples`` CadQuery code strings for one prepared message.

    ``item`` is the per-modality message dict already built by the caller:
    ``{"video": [img], "description": ..., "file_name": ...}`` for images or
    ``{"point_cloud": <(N, 3) array>, "description": ..., "file_name": ...}`` for
    point clouds. ``collate`` sets ``is_img`` / ``is_pc`` from its keys; the
    ``generate`` kwargs are identical for both modalities.

    A distinct torch seed (``seed_base + k``) is set before each ``generate`` so
    sampled decodes differ. ``temperature > 0`` -> ``do_sample=True``.
    """
    do_sample = temperature is not None and temperature > 0
    codes: list[str] = []
    for k in range(n_samples):
        torch.manual_seed(seed_base + k)
        batch = collate([item], processor=processor, n_points=n_points, eval=True)
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


def score(pred_stl: str, gt_path: str) -> tuple[float | None, float | None]:
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


def make_record(*, model: str, weights: str, problem: str, input_path, kind: str,
                sample: int, out_path, res, code: str | None, gt: str | None,
                wall_s: float, error: str | None) -> RunRecord:
    """Build the RunRecord shared by both wrappers (identical bar ``model=``).

    Merges ``res.error`` into ``error`` (when no earlier error), computes
    ``n_ops`` from ``code`` and ``iou``/``chamfer`` via :func:`score` when the
    program executed and a GT mesh is available -- matching the pre-refactor
    inline logic in each wrapper's ``main``.
    """
    if res is not None and res.error and error is None:
        error = res.error
    n_ops = count_ops(code) if code is not None else None
    iou = chamfer = None
    if res is not None and res.ok and gt:
        iou, chamfer = score(res.stl_path, gt)
    return RunRecord(
        model=model,
        weights_id=weights,
        problem=problem,
        input_path=str(input_path),
        input_kind=kind,
        sample=sample,
        output_path=str(out_path) if out_path is not None else None,
        wall_s=wall_s,
        error=error,
        valid_code=bool(res.ok) if res is not None else False,
        valid_geometry=valid_geometry(res) if res is not None else False,
        iou=iou,
        chamfer=chamfer,
        n_ops=n_ops,
        gt_path=gt,
    )
