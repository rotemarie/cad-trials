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
  a ready per-modality message dict.

:func:`score` and :func:`make_record` (torch-free, model-agnostic) now live in
:mod:`cad_trials.common.scoring` and are re-exported here for the existing
``from cad_trials.models._cadrille_common import make_record`` call sites.

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

from cad_trials.common.scoring import make_record, score  # noqa: E402,F401

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
