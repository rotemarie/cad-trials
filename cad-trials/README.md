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
   Expected: 55 passed, 3 skipped (the 3 skips are the GPU smoke tests — see below).

4. **Prepare Problem 1 inputs:**
   ```bash
   python -m cad_trials.data.prepare_problem1 --stl output/fine.STL \
       --out-dir cad-trials/results/prepared/part
   ```
   Writes `normalized.stl`, the `render_*.png` styles, `tile_4diag.png`,
   `three_view.png` and the `pc_*.ply` point clouds into that part directory.

5. **Per-model environments & inference:**
   See `envs/` for the per-model environment runbooks; each model has its own
   conda stack and its own wrapper under `cad_trials/models/`.

6. **Cluster job submission:**
   See `slurm/` (`prepare.sbatch`, `run_model.sbatch`) and the end-to-end runbook below.

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

All trial runs are appended to the ledger — `cad-trials/results/runs.d/<task_id>.jsonl`
on the cluster, `cad-trials/results/runs.jsonl` locally — one JSON object per line.

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
| `watertight` | bool \| None | Whether the executed mesh is closed (from `ExecResult`) |
| `part` | str \| None | Part identifier (prepared dir basename); `None` on pre-multi-part rows |

`watertight` and `part` were appended after the first ledgers were written; both
default to `None`, and `load_runs` reads older rows unchanged.

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
- `render_hlr_paper` — `hlr_lines` on a paper-coloured ground (white fill, heavy edges, beige bg).
  Note this is *not* a drafting-convention render: it has no centrelines and no dashed
  hidden edges. A true drafting style is tracked in the Plan 2 backlog.

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
│   ├── cadrille.md              (cadrille_img + cadrille_pc inference env)
│   └── cadrecode.md             (cadrecode inference env)
├── cad_trials/                  (package)
│   ├── __init__.py
│   ├── common/                  (torch-free: runs in probe-common)
│   │   ├── io.py                (RunRecord, append_run, load_runs, run_exists)
│   │   ├── render.py            (render_style, four_diagonal_tile, ortho_three_view)
│   │   ├── meshes.py            (mesh load / normalize / point-cloud sampling)
│   │   ├── execute.py           (subprocess CadQuery → STL, ExecResult)
│   │   ├── metrics.py           (voxel IoU, Chamfer, op count)
│   │   ├── scoring.py           (score + make_record, shared by all wrappers)
│   │   └── report.py            (runs.jsonl → report.html)
│   ├── models/                  (per-model wrappers; import torch)
│   │   ├── _base.py             (standard_parser, iter_inputs, write_output, gt_for)
│   │   ├── _cadrille_common.py  (shared cadrille load/generate surface)
│   │   ├── cadrille_img.py
│   │   ├── cadrille_pc.py
│   │   └── cadrecode.py
│   ├── data/
│   │   └── prepare_problem1.py  (STL → renders + point clouds + normalized.stl)
│   └── slurm/
│       └── make_manifest.py     (prepared dirs × models → manifest.tsv)
├── tests/                       (test_render, test_meshes, test_execute, test_metrics,
│                                 test_io, test_base, test_scoring, test_report,
│                                 test_make_manifest, test_prepare_problem1,
│                                 test_*_smoke — GPU, skipped by default)
├── data/
│   └── benchmarks/              (large benchmark meshes, not committed)
├── results/                     (nothing here is committed)
│   ├── prepared/<part>/         (model INPUTS: renders, point clouds, normalized.stl)
│   ├── <model>/                 (model OUTPUTS: generated .py + executed .stl)
│   ├── manifest.tsv             (array-job task list)
│   ├── runs.d/<task_id>.jsonl   (per-array-task ledger shards — the cluster path)
│   ├── runs.jsonl               (single-file ledger — local / single-process runs)
│   └── report.html              (generated deliverable)
└── slurm/                       (prepare.sbatch, run_model.sbatch)
```

## Run Everything (BGU Cluster)

This is the end-to-end workflow to run all models (cadrille_img, cadrille_pc, cadrecode) on the reference part (Problem 1, seed 001) and generate the report.

### Prerequisites

One-time environment setup. These files are prose runbooks, not scripts — **follow
the steps in** each, don't try to execute the file:

- [`cad-trials/envs/probe-common.md`](envs/probe-common.md) — the `probe-common` env used by
  the prepare step, the report, and the test suite.
- [`cad-trials/envs/cadrille.md`](envs/cadrille.md) — the `cadrille` env, for `cadrille_img`
  and `cadrille_pc`.
- [`cad-trials/envs/cadrecode.md`](envs/cadrecode.md) — the `cadrecode` env, for `cadrecode`.

### Step 1: Prepare reference-part inputs (CPU-only, ~few minutes)

```bash
cd /path/to/CAD
sbatch cad-trials/slurm/prepare.sbatch
# Wait for job to complete
```

### Step 2: Build the manifest (after prepare finishes)

```bash
PYTHONPATH=$PWD/cad-trials python -m cad_trials.slurm.make_manifest
# Output: cad-trials/results/manifest.tsv
```

For one prepared part this is **9 tasks**, not "3 models × 3 kinds" — the models do
not share an input vocabulary:

| model | input kinds | tasks |
|-------|-------------|-------|
| `cadrille_img` | 5 `render_*` styles + `tile_4diag` + `three_view` | 7 |
| `cadrille_pc` | `pc_256` | 1 |
| `cadrecode` | `pc_256` | 1 |

Manifest columns are `part`, `model`, `input_path`, `input_kind`, `n_samples`, `env`.
Preparing a second part adds another 9 rows; `part` keeps their outputs and their
ground truth apart.

### Step 3: Run all model×input tasks as a resumable array job

```bash
sbatch --array=1-$(wc -l < cad-trials/results/manifest.tsv) cad-trials/slurm/run_model.sbatch
# Resumable: re-submitting skips any (model, input, sample) already recorded.
# Add --retry-errors on the wrapper CLI to re-attempt rows that recorded an error.
```

Each array task writes its own ledger shard, `cad-trials/results/runs.d/<task_id>.jsonl` —
concurrent unlocked appends to one `runs.jsonl` on NFS can interleave and tear a line.
A task only ever needs to read its own shard for the resume check, since one task owns
one `(part, model, input)` pair and all its samples.

### Step 4: Build the report

Point `build_report` at the shard **directory** — it reads and concatenates every
`*.jsonl` in it:

```bash
PYTHONPATH=$PWD/cad-trials python -c "from cad_trials.common.report import build_report; build_report('cad-trials/results/runs.d','cad-trials/results/report.html','cad-trials/results/prepared/part')"
# Output: cad-trials/results/report.html
```

A single-file ledger still works (`build_report('cad-trials/results/runs.jsonl', ...)`)
for local single-process runs.

### GPU Smoke Tests (Optional)

To run the GPU smoke tests for real on the cluster (instead of skipping them), enable the model-specific flags:

```bash
CADRILLE_WEIGHTS_OK=1 CADRECODE_WEIGHTS_OK=1 python -m pytest cad-trials/ -v
```

The three smoke tests (`test_cadrille_img_smoke.py`, `test_cadrille_pc_smoke.py`, `test_cadrecode_smoke.py`) are skipped by default because they require GPU and model weights. They check for:
- `CADRILLE_WEIGHTS_OK` — enables cadrille_img and cadrille_pc smoke tests
- `CADRECODE_WEIGHTS_OK` — enables cadrecode smoke test

### Important Notes

- **Clean slate:** ledgers are append-only and accumulate across cluster runs. To start
  a genuinely fresh survey, remove them: `rm -rf cad-trials/results/runs.d cad-trials/results/runs.jsonl`.
  Nothing under `results/` is tracked by git, so a checkout never carries someone
  else's (or a synthetic) ledger into your run.

- **Manifest:** The manifest (`cad-trials/results/manifest.tsv`) lists all part×model×input
  tasks. It is generated fresh each time and determines the array size.

- **Output:** Final results are in `cad-trials/results/report.html`. The raw trial log is
  `cad-trials/results/runs.d/` (or `runs.jsonl` for a local run).

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
- Mesh I/O, normalization, point-cloud sampling: `cad_trials/common/meshes.py`
- Data preparation (problem definition + seed generation): `cad_trials/data/prepare_problem1.py`

Remember to update `pytest.ini` at repo root if test locations change.
