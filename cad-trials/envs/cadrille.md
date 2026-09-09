# `cadrille` environment (image + point-cloud branch)

Cluster setup for the pretrained `cadrille` model (Qwen2-VL-2B backbone). Used by
`cad_trials.models.cadrille_img` (image branch) and future point-cloud wrappers.
This machine has no GPU — build this on the cluster.

## Create the env

```bash
conda create -y -n cadrille python=3.10
conda activate cadrille

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.50.3 accelerate==0.34.2 qwen-vl-utils==0.0.10 \
            huggingface-hub==0.27.0 "numpy<2.3" pillow trimesh cadquery scipy
```

Optional (faster attention): `pip install flash-attn --no-build-isolation`. The
wrapper falls back to `sdpa` if `flash_attention_2` is unavailable.

## Vendor the upstream repo

`cad_trials.models.cadrille_img` imports `from cadrille import Cadrille, collate`.
The wrapper prepends `cad-trials/vendor/cadrille` to `sys.path` itself, so only the
clone is required (it is git-ignored via `cad-trials/vendor/`):

```bash
git clone https://github.com/col14m/cadrille cad-trials/vendor/cadrille
```

## Environment variables

```bash
export PYTHONPATH=$PWD/cad-trials:$PWD/cad-trials/vendor/cadrille:$PYTHONPATH
export HF_HOME=$WORK/hf_cache          # keep model weights off $HOME quota
```

Weights (`maksimko123/cadrille`, processor `Qwen/Qwen2-VL-2B-Instruct`) download
from the Hugging Face Hub on first run into `$HF_HOME`.

## Smoke test

```bash
conda activate cadrille
export CADRILLE_WEIGHTS_OK=1     # unblocks tests/test_cadrille_img_smoke.py
python -m pytest cad-trials/tests/test_cadrille_img_smoke.py -v
```

## Reference-part run

```bash
python -m cad_trials.models.cadrille_img \
  --inputs "cad-trials/results/prepared/part/render_*.png" \
           "cad-trials/results/prepared/part/tile_4diag.png" \
           "cad-trials/results/prepared/part/three_view.png" \
  --out-dir cad-trials/results/cadrille_img \
  --gt cad-trials/results/prepared/part/normalized.stl \
  --n-samples 5
```

Add `--raw` when the input set is `tile_4diag.png` (already letterboxed + bordered
by `prepare_problem1.py`); the six Task-5 render styles and `three_view.png` want
the default letterbox path.
