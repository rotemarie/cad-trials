# `cadrecode` environment (point-cloud branch)

Cluster setup for the pretrained **cad-recode** model
(https://github.com/filaPro/cad-recode) -- a Qwen2-1.5B decoder with a Fourier
point encoder that maps a 256-point cloud straight to CadQuery Python. This is a
*different* model and repo from `cadrille`; it needs its own env. Used by
`cad_trials.models.cadrecode`. This machine has no GPU -- build this on the
cluster.

## Create the env

```bash
conda create -y -n cadrecode python=3.10
conda activate cadrecode

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.47.1 accelerate numpy pillow trimesh cadquery scipy huggingface-hub
```

Optional (faster attention on CUDA): `pip install flash-attn --no-build-isolation`.
`load_model` requests `flash_attention_2` only when CUDA is present, else `None`.

## Vendor the upstream repo

`cad_trials.models.cadrecode` imports `from cadrecode_model import CADRecode`. The
`CADRecode` + `FourierPointEncoder` classes are **not** a package -- they live in
the upstream `demo.ipynb` notebook. They have been hand-copied verbatim (code
cell "Define CAD-Recode model") into
`cad-trials/vendor/cad-recode/cadrecode_model.py`, which **is committed** to our
repo (force-added, since `cad-trials/vendor/` is git-ignored). The wrapper
prepends `cad-trials/vendor/cad-recode` to `sys.path`.

Clone the rest of the upstream repo for reference (licence, changelog, demo):

```bash
git clone https://github.com/filaPro/cad-recode cad-trials/vendor/cad-recode
# cadrecode_model.py is already in git; the clone just refreshes the sibling files
```

Upstream licence: **CC BY-NC 4.0** (`cad-trials/vendor/cad-recode/LICENSE.md`) --
non-commercial research use only.

## Environment variables

```bash
export PYTHONPATH=$PWD/cad-trials:$PWD/cad-trials/vendor/cad-recode:$PYTHONPATH
export HF_HOME=$WORK/hf_cache          # keep model weights off $HOME quota
```

Weights (`filapro/cad-recode-v1.5`, tokenizer `Qwen/Qwen2-1.5B` with
`pad_token='<|im_end|>'`, `padding_side='left'`) download from the Hugging Face
Hub on first run into `$HF_HOME`.

## Smoke test

```bash
conda activate cadrecode
export CADRECODE_WEIGHTS_OK=1     # unblocks tests/test_cadrecode_smoke.py
python -m pytest cad-trials/tests/test_cadrecode_smoke.py -v
```

## Reference-part run

```bash
python -m cad_trials.models.cadrecode \
  --inputs cad-trials/results/prepared/part/pc_256.ply \
  --out-dir cad-trials/results/cadrecode \
  --gt cad-trials/results/prepared/part/normalized.stl \
  --n-samples 5
```

cad-recode decodes greedily/deterministically in the paper. `run_cadrecode` keeps
a single-sample call deterministic; `--n-samples > 1` switches on
`do_sample=True, temperature=0.7` (seeded per sample) so the draws differ -- this
deviates from the paper's deterministic decode.

## Point-cloud transform

Our `pc_256.ply` points come from `normalize_mesh(scale=1.0)` -> origin-centred,
longest axis spans `1.0`, so points lie in `[-0.5, 0.5]`. `demo.ipynb` (cell
"Load input point cloud") normalizes its input mesh "to fit within a cube of size
2, centered at the origin" (`apply_translation(-(bounds[0]+bounds[1])/2)` then
`apply_scale(2.0/max(extents))`), so its points lie in `[-1, 1]`. The wrapper's
`normalize_points` therefore maps ours by `points * 2`.
