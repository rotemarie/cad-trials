"""cad-recode point-cloud wrapper on the standard cad-trials harness contract.

cad-recode (https://github.com/filaPro/cad-recode) is a *different* model from
``cadrille``: a Qwen2-1.5B decoder with a Fourier point encoder that maps a 256
point cloud straight to CadQuery Python source.  This module runs the pretrained
``filapro/cad-recode-v1.5`` checkpoint on a ``.ply`` point cloud, executes the
generated CadQuery, scores it against a ground-truth mesh, and appends one
:class:`~cad_trials.common.io.RunRecord` per (input, sample).

The model classes (``CADRecode`` + ``FourierPointEncoder``) are hand-copied from
the upstream ``demo.ipynb`` into ``cad-trials/vendor/cad-recode/cadrecode_model.py``
(force-added; ``vendor/`` is git-ignored).  The generation recipe is ported
verbatim from ``demo.ipynb`` cell "Run CAD-Recode on the input point cloud".

CLI::

    python -m cad_trials.models.cadrecode \
        --inputs "results/prepared/part/pc_256.ply" \
        --out-dir results/cadrecode \
        --gt results/prepared/part/normalized.stl \
        --n-samples 5 [--seed-base 0] [--weights filapro/cad-recode-v1.5]

``torch`` / ``transformers`` and the vendored ``cadrecode_model`` are imported at
module load: importing this module requires a working cad-recode environment
(see ``envs/cadrecode.md``).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from transformers import AutoTokenizer

# vendored upstream repo (git-ignored dir; cadrecode_model.py is force-added) must import
sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "cad-recode"))
from cadrecode_model import CADRecode  # noqa: E402,F401

from cad_trials.common.execute import execute_program  # noqa: E402
from cad_trials.common.io import append_run  # noqa: E402
from cad_trials.common.scoring import make_record  # noqa: E402
from cad_trials.models._base import (  # noqa: E402
    already_done,
    gt_for,
    iter_inputs,
    standard_parser,
    write_output,
)

DEFAULT_WEIGHTS = "filapro/cad-recode-v1.5"
TOKENIZER_ID = "Qwen/Qwen2-1.5B"
PAD_TOKEN = "<|im_end|>"
IM_START = "<|im_start|>"
END_TOKEN = "<|endoftext|>"
MAX_NEW_TOKENS = 768


def load_model(weights_id: str) -> "CADRecode":
    """Load cad-recode weights onto CUDA (or CPU), preferring flash-attn on CUDA.

    Mirrors ``demo.ipynb`` cell "Load CAD-Recode checkpoint"
    (``attn_implementation='flash_attention_2' if cuda else None``), but the
    ``flash_attn`` package is an optional, slow-to-build dependency (see
    ``envs/cadrecode.md``) that most nodes won't have installed. Fall back to
    ``sdpa`` — same pattern as ``models/_cadrille_common.load_model`` — instead
    of hard-failing with an ImportError.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_candidates = ("flash_attention_2", "sdpa") if device == "cuda" else (None,)
    last_error: Exception | None = None
    for attn_implementation in attn_candidates:
        try:
            model = CADRecode.from_pretrained(
                weights_id,
                torch_dtype="auto",
                attn_implementation=attn_implementation,
            ).eval().to(device)
            print(f"loaded {weights_id} on {device} with attn={attn_implementation}")
            return model
        except (ImportError, ValueError) as e:
            last_error = e
            print(f"attn={attn_implementation} unavailable ({e}); falling back")
    raise RuntimeError(f"could not load {weights_id}: {last_error}")


def load_tokenizer() -> AutoTokenizer:
    """``demo.ipynb``: ``AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B',
    pad_token='<|im_end|>', padding_side='left')``."""
    return AutoTokenizer.from_pretrained(
        TOKENIZER_ID, pad_token=PAD_TOKEN, padding_side="left")


def load_points(path) -> np.ndarray:
    """Load a ``.ply`` point cloud as an ``(N, 3)`` float64 array."""
    obj = trimesh.load(path, process=False)
    pts = np.asarray(obj.vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) point cloud, got {pts.shape} from {path}")
    return pts


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Map our prepared point cloud into cad-recode's checkpoint input space.

    ``prepare_problem1.py`` samples points from ``normalized.stl`` -- via
    ``common/meshes.py:normalize_mesh`` with ``scale=1.0``: centred at the origin
    and scaled so the longest axis spans ``1.0`` -> points lie in ``[-0.5, 0.5]``.

    cad-recode's ``demo.ipynb`` cell "Load input point cloud" normalizes its input
    mesh "to fit within a cube of size 2, centered at the origin" -- concretely
    ``mesh.apply_translation(-(bounds[0] + bounds[1]) / 2.0)`` then
    ``mesh.apply_scale(2.0 / max(extents))`` -- so the longest axis spans ``2.0``
    and points lie in ``[-1, 1]``.

    Our longest axis is ``1.0`` vs. their ``2.0`` (both origin-centred), so the
    exact map is ``points * 2``.
    """
    return points * 2.0


def run_cadrecode(model, tokenizer, points: np.ndarray, n_samples: int,
                  seed_base: int, temperature: float = 0.7,
                  max_new_tokens: int = MAX_NEW_TOKENS,
                  do_sample: bool | None = None) -> list[str]:
    """Pure inference: return ``n_samples`` CadQuery code strings for one cloud.

    ``points`` is the raw ``[-0.5, 0.5]`` cloud; it is normalized here to
    ``[-1, 1]`` then fed through the ``demo.ipynb`` "Run CAD-Recode" recipe:
    ``input_ids = [pad] * N + [<|im_start|>]``, ``attention_mask = [-1] * N + [1]``,
    ``point_cloud`` a ``(1, N, 3)`` float tensor; then decode and slice the text
    between ``<|im_start|>`` and ``<|endoftext|>``.

    cad-recode decodes greedily/deterministically in the paper.  ``do_sample``
    defaults to ``n_samples > 1`` -- so a single-sample call stays deterministic
    (matching the paper) while a multi-sample request switches on
    ``do_sample=True, temperature=0.7`` (seeded per sample with
    ``torch.manual_seed(seed_base + k)``) to get distinct decodes.  ``main`` passes
    ``do_sample`` explicitly so its per-sample loop still samples when the caller
    asked for ``--n-samples > 1``.  Sampling deviates from the paper's decode.
    """
    pc = normalize_points(points).astype(np.float32)
    n = pc.shape[0]
    im_start_id = tokenizer(IM_START)["input_ids"][0]
    if do_sample is None:
        do_sample = n_samples > 1

    input_ids = [tokenizer.pad_token_id] * n + [im_start_id]
    attention_mask = [-1] * n + [1]

    codes: list[str] = []
    for k in range(n_samples):
        torch.manual_seed(seed_base + k)
        with torch.no_grad():
            batch_ids = model.generate(
                input_ids=torch.tensor(input_ids).unsqueeze(0).to(model.device),
                attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(model.device),
                point_cloud=torch.tensor(pc).unsqueeze(0).to(model.device),
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        text = tokenizer.batch_decode(batch_ids)[0]
        begin = text.find(IM_START) + len(IM_START)
        end = text.find(END_TOKEN)
        codes.append(text[begin:end] if end != -1 else text[begin:])
    return codes


def main(argv=None) -> None:
    parser = standard_parser(
        "Run pretrained cad-recode on a .ply point cloud and score the CadQuery.")
    args = parser.parse_args(argv)

    weights = args.weights or DEFAULT_WEIGHTS
    inputs = iter_inputs(args.inputs)
    out_dir = Path(args.out_dir)
    prefix = f"{args.part}+" if args.part else ""

    model = load_model(weights)
    tokenizer = load_tokenizer()

    for input_path in inputs:
        stem = input_path.stem  # pc_256
        gt = gt_for(stem, args.gt)
        for k in range(args.n_samples):
            if already_done("cadrecode", str(input_path), k, args.runs_path,
                            retry_errors=args.retry_errors):
                continue

            t0 = time.perf_counter()
            error: str | None = None
            code: str | None = None
            out_path: Path | None = None
            res = None
            n_points: int | None = None

            # Every per-sample step -- load, generate, write, execute -- is inside
            # this guard: one bad sample must not kill an 8-hour array task.
            try:
                points = load_points(input_path)
                n_points = int(points.shape[0])
                code = run_cadrecode(
                    model, tokenizer, points, n_samples=1,
                    seed_base=args.seed_base + k,
                    do_sample=args.n_samples > 1)[0]
                meta = {"weights": weights, "n_points": n_points,
                        "seed": args.seed_base + k, "part": args.part}
                out_path = write_output(out_dir, stem, k, code, meta, ext="py",
                                        prefix=prefix)
                res = execute_program(code, out_dir / f"{prefix}{stem}+s{k}.stl")
            except Exception as e:  # noqa: BLE001 - record and continue
                error = f"{type(e).__name__}: {e}"

            try:
                append_run(
                    make_record(
                        model="cadrecode", weights=weights, problem=args.problem,
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
