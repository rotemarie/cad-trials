# probe-common Environment

## BGU Cluster (Canonical)

The canonical `probe-common` environment for running the full pipeline on the BGU cluster.

```bash
# on the BGU cluster, in $WORK
module load miniconda 2>/dev/null || true
conda create -y -n probe-common python=3.11
conda activate probe-common
pip install "trimesh[easy]==4.5.3" numpy==2.2.0 pillow==11.0.0 matplotlib==3.10.0 \
            scipy==1.14.1 cadquery==2.5.2 pytest==8.3.4 manifold3d==3.0.0 mapbox-earcut
```

**Important:** Before running any module or test, set:
```bash
export PYTHONPATH=$PWD/cad-trials:$PYTHONPATH
```

And run all `python -m` and `pytest` commands from the **repository root** (`/path/to/CAD`), not from `cad-trials/`.

## Local Development (macOS)

The maintainer's Mac (arm64) cannot install Python 3.11 or cadquery 2.5.2. Instead, use:
- Python 3.12
- cadquery 2.8.0
- trimesh 4.5.3
- numpy 2.2.6

Create and activate a local `.venv`:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "trimesh[easy]==4.5.3" numpy==2.2.6 pillow==11.0.0 matplotlib==3.10.0 \
            scipy==1.14.1 cadquery==2.8.0 pytest==8.3.4 manifold3d==3.0.0 mapbox-earcut
```

Then set the Python path and run tests from the repo root (same as cluster):
```bash
export PYTHONPATH=$PWD/cad-trials:$PYTHONPATH
python -m pytest cad-trials/ -v
```

All tests pass on both Python 3.11/cadquery 2.5.2 (cluster) and Python 3.12/cadquery 2.8.0 (local).
