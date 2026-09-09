# CAD Trials

Harness for reproducing and extending the CAD design task benchmarks.

## Quick Start

1. **Clone and set up environment:**
   ```bash
   cd /path/to/CAD
   # See envs/probe-common.md for cluster and local setup
   # Cluster: conda create -y -n probe-common python=3.11; pip install ...
   # Local:   python3.12 -m venv .venv; source .venv/bin/activate; pip install ...
   ```

2. **Activate environment:**
   ```bash
   source .venv/bin/activate  # or: conda activate probe-common
   export PYTHONPATH=$PWD/cad-trials:$PYTHONPATH
   ```

3. **Run tests (from repo root):**
   ```bash
   python -m pytest cad-trials/ -v
   ```
   Expected: ~34 passing tests (Tasks 1–7).

4. **Prepare Problem 1 inputs:**
   ```bash
   python -m cad_trials.data.prepare_problem1
   ```
   Generates the seed meshes and renders (STL, images) into `cad-trials/data/prepared/`.

5. **Per-model environments & inference:**
   See `envs/` directory for per-model environment specs (added in later plans).
   Each model has its own conda/pip stack and inference entrypoint.

6. **Cluster job submission:**
   See `slurm/` directory for job templates (added in later plans).

## Environment Setup

See [`envs/probe-common.md`](envs/probe-common.md) for:
- **Cluster** (canonical): Python 3.11, cadquery 2.5.2, numpy 2.2.0, trimesh 4.5.3
- **Local dev** (macOS): Python 3.12, cadquery 2.8.0, numpy 2.2.6, trimesh 4.5.3

Both run the full test suite (`pytest cad-trials/ -v`) successfully.

## Test Suite

Run from repo root:
```bash
python -m pytest cad-trials/ -v
```

Tests cover:
- **Render** (`test_render.py`): image rendering (5 styles, diagonal tile, orthographic views)
- **Meshes** (`test_meshes.py`): normalization, scaling, point cloud sampling, loading
- **Problem 1 prep** (`test_prepare_problem1.py`): pipeline, seed generation, point cloud sizing
- **Execution** (`test_execute.py`): CadQuery program execution, timeouts, error handling
- **Metrics** (`test_metrics.py`): IoU, Chamfer distance, operation counting
- **IO/Logging** (`test_io.py`): `runs.jsonl` recording
- **Base** (`test_base.py`): parser, input expansion, output writing, ground-truth reference

No slow tests yet; full suite runs in <1 min on most hardware.

## Run Record Schema

All trial runs are appended to `cad-trials/results/runs.jsonl`, one JSON object per line.

**`RunRecord` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model identifier (e.g., `"code-gpt"`, `"cad-gnn"`) |
| `weights_id` | str | Weights/checkpoint ID (e.g., HuggingFace model hash) |
| `problem` | str | Problem name (e.g., `"problem1"`) |
| `input_path` | str | Path to input file (e.g., `"cad-trials/data/prepared/seed_001.stl"`) |
| `input_kind` | str | Input type (see vocabulary below) |
| `sample` | int | Sample index for this problem |
| `output_path` | str \| None | Path to generated output (STL/STEP/etc); `None` on error |
| `wall_s` | float | Wall-clock seconds (inference time) |
| `error` | str \| None | Error message on failure; `None` if succeeded |
| `valid_code` | bool \| None | Whether model output was valid code/syntax |
| `valid_geometry` | bool \| None | Whether output was a valid mesh |
| `iou` | float \| None | Intersection-over-union vs. ground truth |
| `chamfer` | float \| None | Chamfer distance vs. ground truth |
| `n_ops` | int \| None | Number of CAD operations in output |
| `gt_path` | str \| None | Ground truth reference path |

**Example:**
```json
{
  "model": "code-gpt",
  "weights_id": "huggingface:gpt2-medium",
  "problem": "problem1",
  "input_path": "cad-trials/data/prepared/seed_001/render_shaded_color.png",
  "input_kind": "render_shaded_color",
  "sample": 0,
  "output_path": "cad-trials/results/problem1/code-gpt/seed_001_sample0.step",
  "wall_s": 12.345,
  "error": null,
  "valid_code": true,
  "valid_geometry": true,
  "iou": 0.87,
  "chamfer": 0.042,
  "n_ops": 15,
  "gt_path": "cad-trials/data/problem1_ground_truth/seed_001.step"
}
```

## Input Kind Vocabulary

`input_kind` specifies which rendering or representation is being fed to the model.

### Image Renderings (from `render.py`)

**Style-based 4-viewport renderings** (`2x2` front/left/top/iso layout, 256×256 by default):
- `render_shaded_color` — solid fill, shaded by normal angle (green #4a9d5b)
- `render_shaded_hlr` — grey fill with hidden-line removal edges
- `render_wireframe` — wireframe only (no fill)
- `render_hlr_lines` — white fill with hidden-line removal (thin edges)
- `render_draftsheet` — technical drawing style (white on beige bg, heavy edges)

**Composite renderings:**
- `tile_4diag` — four shaded diagonal views (128×128 ea., 3-px black borders, 2x2 tiled)
- `three_view` — approximate front/top/left orthographic silhouettes with crease lines

### Point Cloud Representations

- `pc_256` — 256-point sampled cloud
- `pc_2048` — 2048-point sampled cloud
- `pc_8192` — 8192-point sampled cloud

## File Layout

```
cad-trials/
├── README.md                    (this file)
├── envs/
│   ├── probe-common.md          (cluster + local Python/cadquery setup)
│   ├── model-a.md               (model A inference env — added later)
│   └── model-b.md               (model B inference env — added later)
├── cad_trials/                  (package)
│   ├── __init__.py
│   ├── common/
│   │   ├── io.py                (RunRecord, append_run)
│   │   ├── render.py            (render_style, four_diagonal_tile, ortho_three_view)
│   │   ├── geometry.py          (mesh I/O and validation)
│   │   └── problem.py           (problem definitions, seed generation)
│   └── data/
│       └── prepare_problem1.py  (generates seeds and ground truth)
├── tests/
│   ├── test_render.py
│   ├── test_geometry.py
│   ├── test_problem.py
│   └── test_io.py
├── data/
│   ├── benchmarks/              (large benchmark meshes, not committed)
│   └── prepared/                (generated seeds/renders, not committed)
├── results/
│   ├── runs.jsonl               (trial log)
│   └── prepared/                (inference outputs, not committed)
└── slurm/                        (cluster job templates — added later)
```

## Development

### Adding a new model

1. Create `envs/model-name.md` with install + inference steps
2. Implement model inference in `cad_trials/models/model_name.py`
3. Add an entrypoint or job template in `slurm/model-name.sbatch`
4. Log results by calling `append_run()` from `cad_trials.common.io`

### Updating the test suite

Tests live in `cad-trials/tests/`. Run with:
```bash
python -m pytest cad-trials/ -v
```

### Editing renders or Problem 1 definition

- Image renders: `cad_trials/common/render.py`
- Mesh I/O and validation: `cad_trials/common/geometry.py`
- Problem definition and seed generation: `cad_trials/common/problem.py`
- Data preparation script: `cad_trials/data/prepare_problem1.py`

Remember to update `pytest.ini` at repo root if test locations change.
