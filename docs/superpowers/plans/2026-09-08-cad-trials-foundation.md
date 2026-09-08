# CAD-Trials Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `cad-trials/` experiment harness — shared infra, Problem-1 data prep from `output/part.STL`, and wrappers for the three already-runnable models (cadrille image, cadrille point-cloud, cad-recode) — producing the first `runs.jsonl` rows and a draft `report.html`.

**Architecture:** A small Python package `cad-trials/` with three layers: `common/` (pure utilities — io schema, metrics, CadQuery execution, rendering), `data/` (input preparation scripts), `models/` (thin per-model inference wrappers sharing one CLI contract). Each model runs in its own environment; wrappers shell out to or import from the upstream repos cloned under `cad-trials/vendor/`. All results append to one JSONL log; a report generator renders it to HTML.

**Tech Stack:** Python 3.11, trimesh, numpy, Pillow, matplotlib, cadquery, pytest. Model envs: cadrille (torch 2.5.1/cu124, transformers 4.50.3), cad-recode (torch 2.5.1, transformers 4.47.1). Cluster: BGU Slurm, RTX 3090 nodes.

**Spec:** `docs/superpowers/specs/2026-09-08-2d-to-cad-model-probe-design.md`

## Global Constraints

- **No training, no fine-tuning** — inference only, pretrained weights only.
- **Python 3.11** for `common/` and `data/` (the `probe-common` env). Model wrappers run under their model's own env and import `common/` via `PYTHONPATH`, so `common/` must stay importable on **Python ≥ 3.9** and depend only on: `trimesh`, `numpy`, `Pillow`, `matplotlib`, `scipy`, `cadquery` (cadquery only inside the execution subprocess).
- **Output contract** (every wrapper, verbatim): for input file `X` and sample `k`, write `<out_dir>/<X.stem>+s<k>.py` (CadQuery program) — or `<X.stem>+s<k>.stl` for mesh-output models — plus one `<out_dir>/<X.stem>.meta.json` per input.
- **`runs.jsonl`** is append-only. One JSON object per line, one line per `(model, input_path, sample)`. Never rewritten in place.
- **Resumability:** before running a `(model, input, sample)`, wrappers/harness skip it if a matching non-error row already exists in `runs.jsonl`.
- **Determinism:** point-cloud sampling and any RNG in `common/` and `data/` take an explicit `seed` argument; default `seed=0`.
- **Units:** `output/part.STL` is in millimetres. All metrics operate on meshes normalized to a unit cube centred at the origin.
- **Repo paths:** work from the project root `/Users/rotemarie/Documents/BGU/Thesis/CAD` (the git repo). Upstream model repos are cloned to `cad-trials/vendor/<repo>` and git-ignored.
- **Commits:** conventional-commit prefixes (`feat:`, `test:`, `chore:`, `docs:`), one commit per completed task.

---

## File Structure

```
cad-trials/
  __init__.py
  conftest.py                  pytest fixtures (tmp meshes)
  common/
    __init__.py
    io.py                      RunRecord dataclass, append_run, load_runs, run_exists
    meshes.py                  load/normalize mesh, sample_point_cloud (FPS)
    metrics.py                 voxel_iou, chamfer_distance, count_ops
    execute.py                 execute_program() — subprocess CadQuery exec + validity
    render.py                  render_style(), four_diagonal_tile(), ortho_three_view()
    report.py                  build_report() — runs.jsonl -> report.html
  data/
    __init__.py
    prepare_problem1.py        part.STL -> normalized.stl, pc_*.ply, render_*.png, tile/sheet
  models/
    __init__.py
    _base.py                   standard_parser(), iter_inputs(), write_output(), already_done()
    cadrille_img.py            wraps vendor/cadrille (image branch)
    cadrille_pc.py             wraps vendor/cadrille (--mode pc)
    cadrecode.py               wraps vendor/cad-recode (notebook logic)
  envs/
    probe-common.md            conda/venv for common+data
    cadrille.md                env for cadrille_img + cadrille_pc
    cadrecode.md               env for cadrecode
  slurm/
    prepare.sbatch
    run_model.sbatch           array job: one task per (model,input)
  vendor/                      (git-ignored) cloned upstream repos
  results/                     (mostly git-ignored) runs.jsonl kept, artifacts ignored
  tests/
    test_io.py  test_meshes.py  test_metrics.py  test_execute.py
    test_render.py  test_base.py  test_prepare_problem1.py  test_report.py
  README.md
pytest.ini                     (project root) — testpaths = cad-trials/tests
```

Existing scripts `infer_image.py`, `render_mesh.py`, `execute_cad.py` at the project root
are the seeds for `models/cadrille_img.py`, `common/render.py`, `common/execute.py`
respectively. They are **superseded** by this plan; delete them in the task that replaces
each (Task 5, Task 4, Task 3).

---

## Task 1: Package skeleton + io schema

**Files:**
- Create: `cad-trials/__init__.py`, `cad-trials/common/__init__.py`, `cad-trials/models/__init__.py`, `cad-trials/data/__init__.py`, `cad-trials/conftest.py`
- Create: `cad-trials/common/io.py`
- Create: `pytest.ini`
- Test: `cad-trials/tests/test_io.py`

**Interfaces:**
- Produces:
  - `RunRecord` dataclass with fields: `model: str`, `weights_id: str`, `problem: str` (`"p1"`/`"p2"`), `input_path: str`, `input_kind: str`, `sample: int`, `output_path: str | None`, `wall_s: float`, `error: str | None`, `valid_code: bool | None = None`, `valid_geometry: bool | None = None`, `iou: float | None = None`, `chamfer: float | None = None`, `n_ops: int | None = None`, `gt_path: str | None = None`
  - `append_run(record: RunRecord, path: str | Path = "cad-trials/results/runs.jsonl") -> None` — creates parent dirs, opens in append mode, writes `json.dumps(asdict(record))` + `"\n"`
  - `load_runs(path=...) -> list[RunRecord]` — returns `[]` if file missing
  - `run_exists(model: str, input_path: str, sample: int, path=...) -> bool` — True iff a row with that triple **and** `error is None` exists

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_io.py
from pathlib import Path
from cad_trials.common.io import RunRecord, append_run, load_runs, run_exists


def _rec(**kw):
    base = dict(model="m", weights_id="w", problem="p1", input_path="a.png",
                input_kind="render", sample=0, output_path="o.py", wall_s=1.0, error=None)
    base.update(kw)
    return RunRecord(**base)


def test_append_then_load_roundtrip(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run(_rec(iou=0.5), p)
    append_run(_rec(sample=1, error="boom", output_path=None), p)
    rows = load_runs(p)
    assert len(rows) == 2
    assert rows[0].iou == 0.5
    assert rows[1].error == "boom"


def test_load_missing_file_is_empty(tmp_path):
    assert load_runs(tmp_path / "nope.jsonl") == []


def test_run_exists_ignores_error_rows(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_run(_rec(sample=0), p)
    append_run(_rec(sample=1, error="boom"), p)
    assert run_exists("m", "a.png", 0, p) is True
    assert run_exists("m", "a.png", 1, p) is False   # error row does not count
    assert run_exists("m", "a.png", 2, p) is False


def test_append_creates_parent_dirs(tmp_path):
    p = tmp_path / "deep" / "nested" / "runs.jsonl"
    append_run(_rec(), p)
    assert p.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/rotemarie/Documents/BGU/Thesis/CAD && python -m pytest cad-trials/tests/test_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cad_trials'`

- [ ] **Step 3: Create skeleton + implement io.py**

`pytest.ini` (project root):
```ini
[pytest]
testpaths = cad-trials/tests
pythonpath = cad-trials
```
Note: `pythonpath = cad-trials` makes `import cad_trials...` resolve to the `cad-trials/` dir. Create `cad-trials/cad_trials` symlink? No — instead set the package name via a top-level `cad-trials/pyproject.toml` is overkill. Simplest: name the importable package `cad_trials` by placing all code under `cad-trials/cad_trials/`. **Revise structure:** the package dir is `cad-trials/cad_trials/` (underscore), tests at `cad-trials/tests/`, `pythonpath = cad-trials`.

Create `cad-trials/cad_trials/__init__.py` (empty), `cad-trials/cad_trials/common/__init__.py` (empty), likewise `models/`, `data/`. Create `cad-trials/conftest.py` (empty for now).

`cad-trials/cad_trials/common/io.py`:
```python
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class RunRecord:
    model: str
    weights_id: str
    problem: str
    input_path: str
    input_kind: str
    sample: int
    output_path: str | None
    wall_s: float
    error: str | None
    valid_code: bool | None = None
    valid_geometry: bool | None = None
    iou: float | None = None
    chamfer: float | None = None
    n_ops: int | None = None
    gt_path: str | None = None


DEFAULT_PATH = "cad-trials/results/runs.jsonl"
_FIELD_NAMES = {f.name for f in fields(RunRecord)}


def append_run(record: RunRecord, path: str | Path = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")


def load_runs(path: str | Path = DEFAULT_PATH) -> list[RunRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out: list[RunRecord] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(RunRecord(**{k: v for k, v in d.items() if k in _FIELD_NAMES}))
    return out


def run_exists(model: str, input_path: str, sample: int,
               path: str | Path = DEFAULT_PATH) -> bool:
    for r in load_runs(path):
        if (r.model, r.input_path, r.sample) == (model, input_path, sample) and r.error is None:
            return True
    return False
```

Update the File Structure note: everywhere the plan says `cad-trials/common/…` the real import path is `cad_trials.common.…` and the file lives at `cad-trials/cad_trials/common/…`. Tests import `from cad_trials.common.io import ...`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest cad-trials/tests/test_io.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add cad-trials/ pytest.ini
git commit -m "feat: cad-trials package skeleton + runs.jsonl io schema"
```

---

## Task 2: Mesh utilities — normalize + point-cloud sampling

**Files:**
- Create: `cad-trials/cad_trials/common/meshes.py`
- Modify: `cad-trials/conftest.py`
- Test: `cad-trials/tests/test_meshes.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `load_mesh(path: str | Path) -> trimesh.Trimesh` — loads, forces a single concatenated mesh (`force="mesh"`), raises `ValueError` if empty
  - `normalize_mesh(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh` — returns a **copy** centred on its bbox midpoint and scaled so `max(extents) == scale`
  - `sample_point_cloud(mesh: trimesh.Trimesh, n: int, seed: int = 0, n_pre: int = 8192) -> np.ndarray` — surface-sample `n_pre` points, farthest-point-subsample to `n`, return `(n, 3) float64`. FPS is a pure-numpy implementation (no pytorch3d).
  - `conftest.py` fixture `unit_cube_mesh` → `trimesh.creation.box((1, 1, 1))`; fixture `shifted_cube_mesh` → same box translated by `(0.5, 0, 0)`

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_meshes.py
import numpy as np
import trimesh
import pytest
from cad_trials.common.meshes import load_mesh, normalize_mesh, sample_point_cloud


def test_normalize_centers_and_scales():
    m = trimesh.creation.box((10, 4, 2))
    m.apply_translation((100, 100, 100))
    out = normalize_mesh(m, scale=1.0)
    assert np.allclose(out.bounds.mean(axis=0), 0, atol=1e-6)
    assert out.extents.max() == pytest.approx(1.0, abs=1e-6)
    # original untouched
    assert not np.allclose(m.bounds.mean(axis=0), 0)


def test_normalize_scale_two():
    out = normalize_mesh(trimesh.creation.box((3, 3, 3)), scale=2.0)
    assert out.extents.max() == pytest.approx(2.0, abs=1e-6)


def test_sample_point_cloud_shape_and_determinism():
    m = trimesh.creation.box((1, 1, 1))
    a = sample_point_cloud(m, 256, seed=0)
    b = sample_point_cloud(m, 256, seed=0)
    c = sample_point_cloud(m, 256, seed=1)
    assert a.shape == (256, 3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # points lie on the cube surface (within tolerance)
    assert np.all(np.abs(a) <= 0.5 + 1e-6)


def test_sample_point_cloud_spread():
    # FPS should cover all 6 faces of a cube -> large bounding box
    pts = sample_point_cloud(trimesh.creation.box((1, 1, 1)), 256, seed=0)
    assert np.ptp(pts, axis=0).min() > 0.9


def test_load_mesh_missing(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        load_mesh(tmp_path / "nope.stl")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest cad-trials/tests/test_meshes.py -v`
Expected: FAIL — `ModuleNotFoundError: cad_trials.common.meshes`

- [ ] **Step 3: Implement meshes.py + conftest fixtures**

```python
# cad-trials/cad_trials/common/meshes.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    m = trimesh.load(path, force="mesh")
    if m is None or getattr(m, "faces", None) is None or len(m.faces) == 0:
        raise ValueError(f"empty or unreadable mesh: {path}")
    return m


def normalize_mesh(mesh: trimesh.Trimesh, scale: float = 1.0) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_translation(-out.bounds.mean(axis=0))
    factor = scale / out.extents.max()
    out.apply_scale(factor)
    return out


def _farthest_point_sample(pts: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(pts)
    if k >= n:
        return np.arange(n)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = rng.integers(n)
    dist = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(pts - pts[sel[i]], axis=1))
    return sel


def sample_point_cloud(mesh: trimesh.Trimesh, n: int, seed: int = 0,
                       n_pre: int = 8192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pre, _ = trimesh.sample.sample_surface(mesh, n_pre, seed=seed)
    idx = _farthest_point_sample(np.asarray(pre), n, rng)
    return np.asarray(pre)[idx].astype(np.float64)
```

`cad-trials/conftest.py`:
```python
import trimesh
import pytest


@pytest.fixture
def unit_cube_mesh():
    return trimesh.creation.box((1, 1, 1))


@pytest.fixture
def shifted_cube_mesh():
    m = trimesh.creation.box((1, 1, 1))
    m.apply_translation((0.5, 0.0, 0.0))
    return m
```

Note on `trimesh.sample.sample_surface` seed kwarg: if the installed trimesh version rejects `seed=`, fall back to `np.random.seed(seed)` before the call and drop the kwarg. Verify with `python -c "import trimesh; help(trimesh.sample.sample_surface)"`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest cad-trials/tests/test_meshes.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add cad-trials/cad_trials/common/meshes.py cad-trials/conftest.py cad-trials/tests/test_meshes.py
git commit -m "feat: mesh normalize + numpy farthest-point point-cloud sampling"
```

---

## Task 3: CadQuery execution + validity

**Files:**
- Create: `cad-trials/cad_trials/common/execute.py`
- Delete: `execute_cad.py` (project root — superseded)
- Test: `cad-trials/tests/test_execute.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass ExecResult`: `ok: bool`, `error: str | None`, `stl_path: str | None`, `volume: float | None`, `watertight: bool | None`, `n_vertices: int | None`
  - `execute_program(code: str, out_stl: str | Path, timeout: float = 15.0) -> ExecResult` — runs `code` in a **spawned subprocess**, expects the result solid bound to `r` (cadrille/cad-recode convention) or, if absent, the last `cq.Workplane` in scope; tessellates, writes `out_stl`, returns metrics. Timeout / exception / empty-result all return `ok=False` with a message.
  - `valid_geometry(res: ExecResult) -> bool` — `res.ok and res.volume is not None and res.volume > 1e-9`

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_execute.py
from cad_trials.common.execute import execute_program, valid_geometry

GOOD = "import cadquery as cq\nr = cq.Workplane('XY').box(2, 2, 2)"
NO_R = "import cadquery as cq\nx = cq.Workplane('XY').box(1, 1, 1)"
BROKEN = "import cadquery as cq\nr = cq.Workplane('XY').box(2, 2)"   # missing arg
HANG = "import cadquery as cq\nwhile True:\n    pass\nr = None"


def test_valid_program_executes(tmp_path):
    res = execute_program(GOOD, tmp_path / "a.stl")
    assert res.ok
    assert (tmp_path / "a.stl").exists()
    assert res.volume ==  __import__("pytest").approx(8.0, rel=0.05)
    assert valid_geometry(res)


def test_result_variable_fallback(tmp_path):
    res = execute_program(NO_R, tmp_path / "b.stl")
    assert res.ok and res.volume == __import__("pytest").approx(1.0, rel=0.05)


def test_broken_program_reports_error(tmp_path):
    res = execute_program(BROKEN, tmp_path / "c.stl")
    assert not res.ok and res.error
    assert not (tmp_path / "c.stl").exists()
    assert not valid_geometry(res)


def test_timeout(tmp_path):
    res = execute_program(HANG, tmp_path / "d.stl", timeout=3)
    assert not res.ok and "timeout" in res.error.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest cad-trials/tests/test_execute.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement execute.py**

Adapt `execute_cad.py` (root). Key design: a module-level `_worker(code, out_stl, q)` function (picklable, top-level) run via `multiprocessing.get_context("spawn").Process`; parent joins with timeout, reads a result dict from the queue. Worker: `exec(code, env)`; pick `env.get("r")` or scan `env.values()` for the last `cadquery.Workplane`; `.val()` if it has one; `compound.tessellate(0.1, 0.1)` → `trimesh.Trimesh` → `.export(out_stl)`; put `dict(ok=True, volume=float(mesh.volume), watertight=bool(mesh.is_watertight), n_vertices=len(mesh.vertices))`. On any exception put `dict(ok=False, error=f"{type(e).__name__}: {e}")`. Parent: if `proc.is_alive()` after join → terminate, `ExecResult(ok=False, error=f"timeout after {timeout}s")`.

Full implementation (write this exactly):
```python
from __future__ import annotations
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    ok: bool
    error: str | None = None
    stl_path: str | None = None
    volume: float | None = None
    watertight: bool | None = None
    n_vertices: int | None = None


def _worker(code: str, out_stl: str, q) -> None:
    try:
        import cadquery as cq
        import trimesh
        env: dict = {}
        exec(code, env)
        obj = env.get("r")
        if obj is None:
            cands = [v for v in env.values() if isinstance(v, cq.Workplane)]
            if not cands:
                raise RuntimeError("no result solid (no `r`, no Workplane)")
            obj = cands[-1]
        compound = obj.val() if hasattr(obj, "val") else obj
        verts, faces = compound.tessellate(0.1, 0.1)
        mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces, process=False)
        if len(mesh.faces) == 0:
            raise RuntimeError("empty tessellation")
        Path(out_stl).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_stl)
        q.put(dict(ok=True, volume=float(mesh.volume),
                   watertight=bool(mesh.is_watertight), n_vertices=len(mesh.vertices)))
    except Exception as e:  # noqa: BLE001 - report everything
        q.put(dict(ok=False, error=f"{type(e).__name__}: {e}"))


def execute_program(code: str, out_stl: str | Path, timeout: float = 15.0) -> ExecResult:
    out_stl = str(out_stl)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, out_stl, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return ExecResult(ok=False, error=f"timeout after {timeout}s")
    try:
        d = q.get_nowait()
    except Exception:
        return ExecResult(ok=False, error="worker died without result")
    if not d.get("ok"):
        return ExecResult(ok=False, error=d.get("error", "unknown"))
    return ExecResult(ok=True, stl_path=out_stl, volume=d["volume"],
                      watertight=d["watertight"], n_vertices=d["n_vertices"])


def valid_geometry(res: ExecResult) -> bool:
    return bool(res.ok and res.volume is not None and res.volume > 1e-9)
```

Then `git rm execute_cad.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest cad-trials/tests/test_execute.py -v`
Expected: 4 passed (needs `cadquery` in the `probe-common` env — see Task 9; if cadquery not yet installed, mark this task blocked on Task 9's env and run `pip install cadquery` first).

- [ ] **Step 5: Commit**

```bash
git add cad-trials/cad_trials/common/execute.py cad-trials/tests/test_execute.py
git rm execute_cad.py
git commit -m "feat: subprocess CadQuery execution with timeout + validity check"
```

---

## Task 4: Metrics — voxel IoU, Chamfer, op count

**Files:**
- Create: `cad-trials/cad_trials/common/metrics.py`
- Test: `cad-trials/tests/test_metrics.py`

**Interfaces:**
- Consumes: `cad_trials.common.meshes.normalize_mesh`
- Produces:
  - `voxel_iou(a: trimesh.Trimesh, b: trimesh.Trimesh, res: int = 64) -> float` — normalize both to the **same** unit cube, voxelize each at `res`, return `|A∩B| / |A∪B|`. No rotation search (documented limitation).
  - `chamfer_distance(a, b, n: int = 8192, seed: int = 0) -> float` — symmetric mean squared nearest-neighbour distance between surface samples of the two normalized meshes, scaled ×1000 (matches cad-recode notebook convention)
  - `count_ops(code: str) -> int` — number of solid-producing CadQuery calls; counts regex matches of `\.(extrude|revolve|loft|sweep|box|sphere|cylinder|cut|union|intersect|hole|cboreHole|cskHole)\s*\(`

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_metrics.py
import numpy as np
import trimesh
import pytest
from cad_trials.common.metrics import voxel_iou, chamfer_distance, count_ops


def test_iou_identical_is_one():
    c = trimesh.creation.box((1, 1, 1))
    assert voxel_iou(c, c.copy(), res=32) == pytest.approx(1.0, abs=0.02)


def test_iou_disjoint_is_zero():
    a = trimesh.creation.box((1, 1, 1)); a.apply_translation((-5, 0, 0))
    b = trimesh.creation.box((1, 1, 1)); b.apply_translation((5, 0, 0))
    # after independent normalization both fill the cube -> identical, IoU high.
    # disjointness must be tested pre-normalized: use the shared-frame variant
    assert voxel_iou(a, b, res=32) == pytest.approx(1.0, abs=0.05)  # documents the limitation


def test_iou_partial_overlap_shared_frame():
    a = trimesh.creation.box((2, 2, 2))
    b = trimesh.creation.box((2, 2, 2)); b.apply_translation((1, 0, 0))
    val = voxel_iou(a, b, res=48, shared_frame=True)
    assert 0.2 < val < 0.45   # analytic IoU of two unit-overlap 2-cubes = 1/3


def test_chamfer_zero_for_identical():
    c = trimesh.creation.box((1, 1, 1))
    assert chamfer_distance(c, c.copy()) == pytest.approx(0.0, abs=1e-3)


def test_chamfer_positive_for_different_shapes():
    a = trimesh.creation.box((1, 1, 1))
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.6)
    assert chamfer_distance(a, b) > 0.5


def test_count_ops():
    code = ("import cadquery as cq\n"
            "r = cq.Workplane('XY').box(10,10,2).faces('>Z').workplane()"
            ".hole(3).cboreHole(2,4,1)")
    assert count_ops(code) == 3   # box, hole, cboreHole
```

Note: `test_iou_partial_overlap_shared_frame` requires a `shared_frame: bool = False`
kwarg on `voxel_iou` — when True, skip per-mesh normalization and voxelize both in a
common frame (union of bounds). This is the honest way to test overlap; the default
`shared_frame=False` path (independent normalization) is what the harness uses for
model-vs-GT and its rotation/scale-invariance limitation is called out in the spec.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest cad-trials/tests/test_metrics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement metrics.py**

```python
from __future__ import annotations
import re
import numpy as np
import trimesh
from .meshes import normalize_mesh

_OP_RE = re.compile(
    r"\.(extrude|revolve|loft|sweep|box|sphere|cylinder|cut|union|intersect|"
    r"hole|cboreHole|cskHole)\s*\(")


def _voxel_occupancy(mesh: trimesh.Trimesh, res: int, bounds: np.ndarray) -> np.ndarray:
    pitch = (bounds[1] - bounds[0]).max() / res
    vg = mesh.voxelized(pitch=pitch)
    vg = vg.fill()
    origin = bounds[0]
    pts = vg.points
    ijk = np.floor((pts - origin) / pitch).astype(int)
    grid = np.zeros((res + 2, res + 2, res + 2), dtype=bool)
    ok = (ijk >= 0).all(axis=1) & (ijk < res + 2).all(axis=1)
    ijk = ijk[ok]
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return grid


def voxel_iou(a: trimesh.Trimesh, b: trimesh.Trimesh, res: int = 64,
              shared_frame: bool = False) -> float:
    if shared_frame:
        lo = np.minimum(a.bounds[0], b.bounds[0])
        hi = np.maximum(a.bounds[1], b.bounds[1])
        bounds = np.array([lo, hi])
        ga, gb = _voxel_occupancy(a, res, bounds), _voxel_occupancy(b, res, bounds)
    else:
        na, nb = normalize_mesh(a), normalize_mesh(b)
        bounds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        ga, gb = _voxel_occupancy(na, res, bounds), _voxel_occupancy(nb, res, bounds)
    inter = np.logical_and(ga, gb).sum()
    union = np.logical_or(ga, gb).sum()
    return float(inter / union) if union else 0.0


def chamfer_distance(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 8192,
                     seed: int = 0) -> float:
    from scipy.spatial import cKDTree
    na, nb = normalize_mesh(a), normalize_mesh(b)
    pa, _ = trimesh.sample.sample_surface(na, n)
    pb, _ = trimesh.sample.sample_surface(nb, n)
    da, _ = cKDTree(pb).query(pa)
    db, _ = cKDTree(pa).query(pb)
    return float((np.mean(da ** 2) + np.mean(db ** 2)) * 1000.0)


def count_ops(code: str) -> int:
    return len(_OP_RE.findall(code))
```

If `mesh.voxelized(...).fill()` is unreliable for open/non-watertight predicted meshes,
catch the exception and fall back to `trimesh.voxel.creation.voxelize` on the surface
without fill; record that the value is surface-only. Keep the fallback simple; note it in
a docstring.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest cad-trials/tests/test_metrics.py -v`
Expected: 6 passed. If `test_iou_partial_overlap_shared_frame` lands outside `(0.2, 0.45)`, adjust the voxel occupancy fill logic (the analytic answer is 1/3) — do not widen the test to hide a real bug.

- [ ] **Step 5: Commit**

```bash
git add cad-trials/cad_trials/common/metrics.py cad-trials/tests/test_metrics.py
git commit -m "feat: voxel IoU, chamfer distance, CadQuery op count"
```

---

## Task 5: Rendering — the six Problem-1 styles + cadrille tile

**Files:**
- Create: `cad-trials/cad_trials/common/render.py`
- Delete: `render_mesh.py` (project root — superseded)
- Test: `cad-trials/tests/test_render.py`

**Interfaces:**
- Consumes: `cad_trials.common.meshes.normalize_mesh`
- Produces:
  - `STYLES = ["shaded_color", "shaded_hlr", "wireframe", "hlr_lines", "draftsheet"]`
  - `FRONTS = [(1,1,1), (-1,-1,-1), (-1,1,-1), (1,-1,1)]` (cadrille's diagonal set)
  - `ORTHO = {"front": (0,-1,0), "top": (0,0,1), "left": (-1,0,0), "iso": (1,-1,1)}`
  - `render_style(mesh: trimesh.Trimesh, style: str, size: int = 256) -> PIL.Image.Image` — 2×2 viewport (Front/Left/Top/Iso, matching the SolidWorks layout in `extracts/`), styled per `style`. Matplotlib `Agg`, `Poly3DCollection`.
  - `four_diagonal_tile(mesh, size: int = 128) -> PIL.Image.Image` — cadrille's native input: 4 shaded views from `FRONTS`, pale-yellow on white, 3-px black border, tiled 2×2. (Port from the deleted `render_mesh.py`.)
  - `ortho_three_view(mesh, size: int = 256) -> PIL.Image.Image` — Front/Top/Left silhouette+crease line art side by side on white (approximate; labelled as such by callers)

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_render.py
import numpy as np
import trimesh
import pytest
from PIL import Image
from cad_trials.common.render import (
    render_style, four_diagonal_tile, ortho_three_view, STYLES)


@pytest.fixture
def part():
    m = trimesh.creation.box((4, 2, 1))
    m.apply_translation((0, 0, 0.5))
    return m


@pytest.mark.parametrize("style", STYLES)
def test_render_style_returns_image(part, style):
    im = render_style(part, style, size=128)
    assert isinstance(im, Image.Image)
    assert im.size == (128, 128)
    # not blank: some non-background pixels
    arr = np.asarray(im.convert("L"))
    assert arr.std() > 3


def test_four_diagonal_tile_shape(part):
    im = four_diagonal_tile(part, size=64)
    # 2x2 of (64 + 6 border) -> 140
    assert im.size == (140, 140)


def test_ortho_three_view(part):
    im = ortho_three_view(part, size=100)
    assert im.size[0] == 300 and im.size[1] == 100
    assert np.asarray(im.convert("L")).min() < 128   # has dark line pixels


def test_render_style_rejects_unknown(part):
    with pytest.raises(ValueError):
        render_style(part, "nope")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest cad-trials/tests/test_render.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement render.py**

Use `matplotlib` with `matplotlib.use("Agg")` at import. One helper `_view(ax, mesh, front, style)` that adds a `Poly3DCollection`; per-style parameters:

| style | facecolor | edgecolor | linewidth | background |
|---|---|---|---|---|
| `shaded_color` | lambert-shaded green `#4a9d5b` | none | 0 | white |
| `shaded_hlr` | lambert-shaded grey `#b8b8b8` | `#222` | 0.3 | white |
| `wireframe` | none (`alpha=0`) | `#222` all edges | 0.4 | white |
| `hlr_lines` | white fill (occludes back edges) | `#111` | 0.6 | white |
| `draftsheet` | white fill | `#111` | 0.7 | `#f0efe6` |

`render_style` builds a 2×2 matplotlib figure, one subplot per `ORTHO` direction
(`front,left,top,iso`), calls `_view`, returns the figure as a PIL image resized to
`(size, size)`. Port `four_diagonal_tile` verbatim from the git history of `render_mesh.py`
(`git show HEAD~2:render_mesh.py` — it's the `render_view` + tiling logic). `ortho_three_view`:
for each of front/top/left, project vertices to 2D, draw mesh edges whose adjacent faces
either both face away/toward (silhouette) or meet at >20° (crease), thick black on white,
hstack the three.

Then `git rm render_mesh.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest cad-trials/tests/test_render.py -v`
Expected: 8 passed (5 parametrized + 3).

- [ ] **Step 5: Commit**

```bash
git add cad-trials/cad_trials/common/render.py cad-trials/tests/test_render.py
git rm render_mesh.py
git commit -m "feat: six-style Problem-1 renders + cadrille diagonal tile + ortho three-view"
```

---

## Task 6: `models/_base.py` — wrapper CLI contract

**Files:**
- Create: `cad-trials/cad_trials/models/_base.py`
- Test: `cad-trials/tests/test_base.py`

**Interfaces:**
- Consumes: `cad_trials.common.io.run_exists`
- Produces:
  - `standard_parser(description: str) -> argparse.ArgumentParser` with: `--inputs` (nargs=+, globs or paths), `--out-dir` (required), `--n-samples` (int, default 5), `--seed-base` (int, default 0), `--weights` (str, default ""), `--runs-path` (default `cad_trials.common.io.DEFAULT_PATH`), `--problem` (choices `p1`/`p2`, default `p1`), `--gt` (str, default "": path or dir of GT meshes keyed by stem)
  - `iter_inputs(patterns: list[str]) -> list[pathlib.Path]` — expand globs, dedupe, sort, error if empty
  - `write_output(out_dir, stem: str, sample: int, payload: str, meta: dict, ext: str = "py") -> Path` — writes `<out_dir>/<stem>+s<sample>.<ext>`, merges `meta` into `<out_dir>/<stem>.meta.json` (list under key `samples`)
  - `already_done(model: str, input_path: str, sample: int, runs_path: str) -> bool` — thin wrapper over `run_exists`
  - `gt_for(stem: str, gt_arg: str) -> str | None` — if `gt_arg` is a file return it; if a dir, return `<dir>/<stem>.stl` when it exists; else None

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_base.py
import json
from cad_trials.models._base import (
    standard_parser, iter_inputs, write_output, gt_for)


def test_parser_defaults():
    p = standard_parser("x")
    ns = p.parse_args(["--inputs", "a.png", "b.png", "--out-dir", "out"])
    assert ns.inputs == ["a.png", "b.png"]
    assert ns.n_samples == 5 and ns.seed_base == 0 and ns.problem == "p1"


def test_iter_inputs_expands_and_sorts(tmp_path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    got = iter_inputs([str(tmp_path / "*.png")])
    assert [p.name for p in got] == ["a.png", "b.png"]


def test_iter_inputs_empty_raises():
    import pytest
    with pytest.raises(SystemExit):
        iter_inputs(["/no/such/*.xyz"])


def test_write_output_creates_files_and_meta(tmp_path):
    write_output(tmp_path, "part", 0, "print(1)", {"wall_s": 2.0})
    write_output(tmp_path, "part", 1, "print(2)", {"wall_s": 3.0})
    assert (tmp_path / "part+s0.py").read_text() == "print(1)"
    meta = json.loads((tmp_path / "part.meta.json").read_text())
    assert len(meta["samples"]) == 2 and meta["samples"][1]["wall_s"] == 3.0


def test_gt_for(tmp_path):
    (tmp_path / "part.stl").write_bytes(b"x")
    assert gt_for("part", str(tmp_path)) == str(tmp_path / "part.stl")
    assert gt_for("missing", str(tmp_path)) is None
    assert gt_for("anything", str(tmp_path / "part.stl")) == str(tmp_path / "part.stl")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest cad-trials/tests/test_base.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `_base.py`** per the Interfaces block. `iter_inputs` on empty calls `parser.error`-style `raise SystemExit("no inputs matched: ...")`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest cad-trials/tests/test_base.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add cad-trials/cad_trials/models/_base.py cad-trials/tests/test_base.py
git commit -m "feat: shared model-wrapper CLI contract (_base)"
```

---

## Task 7: `data/prepare_problem1.py` — STL → all Problem-1 inputs

**Files:**
- Create: `cad-trials/cad_trials/data/prepare_problem1.py`
- Test: `cad-trials/tests/test_prepare_problem1.py`

**Interfaces:**
- Consumes: `common.meshes` (`load_mesh`, `normalize_mesh`, `sample_point_cloud`), `common.render` (`render_style`, `four_diagonal_tile`, `ortho_three_view`, `STYLES`)
- Produces:
  - `prepare(stl_path: str | Path, out_dir: str | Path, seed: int = 0) -> dict` — writes and returns a manifest:
    - `<out_dir>/normalized.stl` (scale 1) and `normalized_2u.stl` (scale 2)
    - `<out_dir>/pc_256.ply`, `pc_2048.ply`, `pc_8192.ply` (via `trimesh.PointCloud(...).export`)
    - `<out_dir>/render_<style>.png` for each style in `STYLES`
    - `<out_dir>/tile_4diag.png`
    - `<out_dir>/three_view.png`
    - returns `{"gt": ".../normalized.stl", "inputs": [{"path": ..., "kind": "render_shaded_color"}, ...]}`
  - `__main__`: `python -m cad_trials.data.prepare_problem1 --stl output/part.STL --out-dir cad-trials/results/prepared/part`

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_prepare_problem1.py
import json
import trimesh
from cad_trials.data.prepare_problem1 import prepare
from cad_trials.common.render import STYLES


def test_prepare_writes_all_artifacts(tmp_path):
    stl = tmp_path / "in.stl"
    trimesh.creation.box((4, 2, 1)).export(stl)
    out = tmp_path / "prep"
    manifest = prepare(stl, out, seed=0)

    assert (out / "normalized.stl").exists()
    for n in (256, 2048, 8192):
        assert (out / f"pc_{n}.ply").exists()
    for s in STYLES:
        assert (out / f"render_{s}.png").exists()
    assert (out / "tile_4diag.png").exists()
    assert (out / "three_view.png").exists()

    kinds = {i["kind"] for i in manifest["inputs"]}
    assert f"render_{STYLES[0]}" in kinds and "tile_4diag" in kinds
    assert manifest["gt"].endswith("normalized.stl")


def test_prepare_pc_has_right_point_count(tmp_path):
    stl = tmp_path / "in.stl"
    trimesh.creation.box((1, 1, 1)).export(stl)
    prepare(stl, tmp_path / "p", seed=0)
    pc = trimesh.load(tmp_path / "p" / "pc_256.ply")
    assert len(pc.vertices) == 256
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest cad-trials/tests/test_prepare_problem1.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `prepare_problem1.py`.** Straight orchestration of Task 2 + Task 5 functions. `input_kind` strings: `render_<style>`, `tile_4diag`, `three_view`, and `pc_256` (PC kinds are added by the model wrappers, but include `pc_256/2048/8192` entries in the manifest `inputs` too so the harness can enumerate them).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest cad-trials/tests/test_prepare_problem1.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run it for real + commit**

```bash
python -m cad_trials.data.prepare_problem1 --stl output/part.STL --out-dir cad-trials/results/prepared/part
# eyeball: open cad-trials/results/prepared/part/render_*.png and three_view.png
git add cad-trials/cad_trials/data/prepare_problem1.py cad-trials/tests/test_prepare_problem1.py
git commit -m "feat: prepare_problem1 — STL to renders, tiles, point clouds"
```

Also add to `.gitignore` (already covers `cad-trials/data/prepared/`; update to `cad-trials/results/prepared/`).

---

## Task 8: `probe-common` environment + README

**Files:**
- Create: `cad-trials/envs/probe-common.md`
- Create: `cad-trials/README.md`
- Modify: `.gitignore` (fix `results/prepared/` path)

**Interfaces:** none (docs). This task has no test; its deliverable is a reproducible env and the earlier tasks' tests passing inside it.

- [ ] **Step 1: Write `envs/probe-common.md`**

Exact commands:
```bash
# on the BGU cluster, in $WORK
module load miniconda 2>/dev/null || true
conda create -y -n probe-common python=3.11
conda activate probe-common
pip install "trimesh[easy]==4.5.3" numpy==2.2.0 pillow==11.0.0 matplotlib==3.10.0 \
            scipy==1.14.1 cadquery==2.5.2 pytest==8.3.4 manifold3d==3.0.0 mapbox-earcut
```
Document: `export PYTHONPATH=$PWD/cad-trials:$PYTHONPATH` before running modules;
`cd` to the project root for all `python -m` and `pytest` calls.

- [ ] **Step 2: Write `cad-trials/README.md`** — the reproduction path: clone, create `probe-common`, `pytest`, `prepare_problem1`, then per-model envs (link `envs/*.md`), then `slurm/`. Include the `runs.jsonl` schema and the `input_kind` vocabulary.

- [ ] **Step 3: Fix `.gitignore`**

Change `cad-trials/data/prepared/` → `cad-trials/results/prepared/`. Add `cad-trials/vendor/`.

- [ ] **Step 4: Full test run**

Run: `python -m pytest cad-trials/ -v`
Expected: all tasks 1–7 tests pass (green).

- [ ] **Step 5: Commit**

```bash
git add cad-trials/envs/probe-common.md cad-trials/README.md .gitignore
git commit -m "docs: probe-common env + cad-trials README"
```

---

## Task 9: cadrille image wrapper

**Files:**
- Create: `cad-trials/cad_trials/models/cadrille_img.py`
- Create: `cad-trials/envs/cadrille.md`
- Delete: `infer_image.py` (project root — superseded)
- Test: `cad-trials/tests/test_cadrille_img_smoke.py` (marked `@pytest.mark.slow`, skipped without weights)

**Interfaces:**
- Consumes: `models._base` (parser, `iter_inputs`, `write_output`, `already_done`, `gt_for`), `common.io.append_run`, `common.execute.execute_program` + `valid_geometry`, `common.metrics` (`voxel_iou`, `chamfer_distance`, `count_ops`)
- Produces:
  - `models/cadrille_img.py` `__main__` implementing the standard CLI. Weights id `maksimko123/cadrille`. For each input × sample: build the cadrille "video" message (reuse `vendor/cadrille` `collate` + `Cadrille`), generate, decode → CadQuery string, `write_output(..., ext="py")`, then execute + score vs GT, `append_run(RunRecord(...))`.
  - `--raw` flag: pass image through unmodified (for `tile_4diag.png`); default letterboxes to 128 + black border (the Task-5 `hlr_lines`/`draftsheet`/etc. renders and `three_view.png` go in un-raw).
  - `run_cadrille_image(model, processor, image: PIL.Image, n_samples, seed_base, temperature=0.8, max_new_tokens=768) -> list[str]` — pure inference, returns N code strings.

- [ ] **Step 1: Write `envs/cadrille.md`**

```bash
conda create -y -n cadrille python=3.10
conda activate cadrille
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.50.3 accelerate==0.34.2 qwen-vl-utils==0.0.10 \
            huggingface-hub==0.27.0 "numpy<2.3" pillow trimesh cadquery scipy
git clone https://github.com/col14m/cadrille cad-trials/vendor/cadrille
export PYTHONPATH=$PWD/cad-trials:$PWD/cad-trials/vendor/cadrille:$PYTHONPATH
export HF_HOME=$WORK/hf_cache
```

- [ ] **Step 2: Write the smoke test**

```python
# cad-trials/tests/test_cadrille_img_smoke.py
import os, shutil, pytest
pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None or not os.environ.get("CADRILLE_WEIGHTS_OK"),
    reason="needs GPU + cadrille weights")


def test_one_render_produces_a_row(tmp_path):
    import trimesh
    from cad_trials.common.render import render_style
    from cad_trials.models.cadrille_img import main
    m = trimesh.creation.box((4, 2, 1))
    img = render_style(m, "hlr_lines", size=256)
    img.save(tmp_path / "part.png")
    gt = tmp_path / "part.stl"; m.export(gt)
    runs = tmp_path / "runs.jsonl"
    main(["--inputs", str(tmp_path / "part.png"), "--out-dir", str(tmp_path / "o"),
          "--n-samples", "1", "--gt", str(gt), "--runs-path", str(runs)])
    from cad_trials.common.io import load_runs
    rows = load_runs(runs)
    assert len(rows) == 1 and rows[0].model == "cadrille_img"
```

- [ ] **Step 3: Run to verify it fails / skips**

Run: `python -m pytest cad-trials/tests/test_cadrille_img_smoke.py -v`
Expected: SKIPPED (no `CADRILLE_WEIGHTS_OK`) — acceptable; real check is Step 5.

- [ ] **Step 4: Implement `cadrille_img.py`.** Port generation logic from `infer_image.py` (git: `git show HEAD~N:infer_image.py`), restructured: `main(argv=None)` uses `standard_parser`; model/processor loaded once; loop inputs → `already_done` guard → `run_cadrille_image` → per sample `write_output` + `execute_program` to `<out>/<stem>+s<k>.stl` + metrics vs `gt_for(stem, args.gt)` → `append_run`. `git rm infer_image.py`.

- [ ] **Step 5: Real smoke on the cluster**

```bash
conda activate cadrille
export CADRILLE_WEIGHTS_OK=1
python -m pytest cad-trials/tests/test_cadrille_img_smoke.py -v      # PASS
# then the real reference-part run:
python -m cad_trials.models.cadrille_img \
  --inputs "cad-trials/results/prepared/part/render_*.png" \
           "cad-trials/results/prepared/part/tile_4diag.png" \
           "cad-trials/results/prepared/part/three_view.png" \
  --out-dir cad-trials/results/cadrille_img \
  --gt cad-trials/results/prepared/part/normalized.stl \
  --n-samples 5
```

- [ ] **Step 6: Commit**

```bash
git add cad-trials/cad_trials/models/cadrille_img.py cad-trials/envs/cadrille.md cad-trials/tests/test_cadrille_img_smoke.py
git rm infer_image.py
git commit -m "feat: cadrille image-branch wrapper on the standard harness contract"
```

---

## Task 10: cadrille point-cloud wrapper

**Files:**
- Create: `cad-trials/cad_trials/models/cadrille_pc.py`
- Test: `cad-trials/tests/test_cadrille_pc_smoke.py` (slow/skipped like Task 9)

**Interfaces:**
- Consumes: same as Task 9, plus reads `.ply` point clouds via `trimesh.load(...).vertices`
- Produces:
  - `cadrille_pc.py` `__main__`, weights id `maksimko123/cadrille`, `--mode pc` path of the cadrille repo. Input kind `pc_256` (also accept `pc_2048`, `pc_8192` — pass through, but default runs feed `pc_256` to match training).
  - `run_cadrille_pc(model, processor, points: np.ndarray, n_samples, seed_base, ...) -> list[str]` — normalize points to `(pts-0.5)*2` per the repo's test convention **after** the `prepare` step already unit-cube-normalized them; verify against `vendor/cadrille/dataset.py` `get_point_cloud` test branch and match exactly.

- [ ] **Step 1: Write the smoke test** (mirror Task 9's, feeding `pc_256.ply`, asserting `rows[0].model == "cadrille_pc"`).

- [ ] **Step 2: Run — SKIPPED without weights.**

- [ ] **Step 3: Implement.** Reuse `vendor/cadrille` `Cadrille` + `collate` with the point-cloud message path. Confirm the exact input-normalization expected by the checkpoint by reading `vendor/cadrille/test.py` and `dataset.py`; the `prepare` output is unit-cube [-0.5,0.5], the repo test path expects `(x-0.5)*2` on [0,1] data — reconcile (likely feed points shifted to [0,1] first). Document the transform in a comment with the file:line reference.

- [ ] **Step 4: Real smoke + reference-part run**

```bash
python -m cad_trials.models.cadrille_pc \
  --inputs cad-trials/results/prepared/part/pc_256.ply \
  --out-dir cad-trials/results/cadrille_pc \
  --gt cad-trials/results/prepared/part/normalized.stl --n-samples 5
```

- [ ] **Step 5: Commit**

```bash
git add cad-trials/cad_trials/models/cadrille_pc.py cad-trials/tests/test_cadrille_pc_smoke.py
git commit -m "feat: cadrille point-cloud wrapper"
```

---

## Task 11: cad-recode wrapper

**Files:**
- Create: `cad-trials/cad_trials/models/cadrecode.py`
- Create: `cad-trials/envs/cadrecode.md`
- Test: `cad-trials/tests/test_cadrecode_smoke.py` (slow/skipped)

**Interfaces:**
- Consumes: same harness pieces as Task 10.
- Produces:
  - `cadrecode.py` `__main__`, weights id `filapro/cad-recode-v1.5`, input kind `pc_256`.
  - `run_cadrecode(model, tokenizer, points: np.ndarray, n_samples, seed_base) -> list[str]` — the notebook's generation: `input_ids = [pad]*256 + [im_start]`, `attention_mask = [-1]*256 + [1]`, `point_cloud` tensor; decode between `<|im_start|>` and `<|endoftext|>`. cad-recode is greedy/deterministic — for `n_samples > 1`, enable sampling (`do_sample=True, temperature=0.7`) and note that this deviates from the paper's deterministic decode.

- [ ] **Step 1: Write `envs/cadrecode.md`**

```bash
conda create -y -n cadrecode python=3.10
conda activate cadrecode
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.47.1 accelerate numpy "pillow" trimesh cadquery scipy huggingface-hub
git clone https://github.com/filaPro/cad-recode cad-trials/vendor/cad-recode
# the model class lives in the demo notebook; extract it:
jupyter nbconvert --to script cad-trials/vendor/cad-recode/demo.ipynb --stdout \
  | sed -n '/class FourierPointEncoder/,/return model_inputs/p' > cad-trials/vendor/cad-recode/cadrecode_model.py
export PYTHONPATH=$PWD/cad-trials:$PWD/cad-trials/vendor/cad-recode:$PYTHONPATH
```

- [ ] **Step 2: Write smoke test** (mirror Task 10; `rows[0].model == "cadrecode"`).

- [ ] **Step 3: Run — SKIPPED without weights.**

- [ ] **Step 4: Implement `cadrecode.py`.** Import `CADRecode` + `FourierPointEncoder` from the extracted `cadrecode_model.py` (verify the nbconvert sed range actually captured the class; if fragile, hand-copy the two classes into `cad-trials/vendor/cad-recode/cadrecode_model.py` and check it in — it's ~90 lines, MIT). Generation exactly as `demo.ipynb` cell "Run CAD-Recode".

- [ ] **Step 5: Real smoke + reference-part run**

```bash
python -m cad_trials.models.cadrecode \
  --inputs cad-trials/results/prepared/part/pc_256.ply \
  --out-dir cad-trials/results/cadrecode \
  --gt cad-trials/results/prepared/part/normalized.stl --n-samples 5
```

- [ ] **Step 6: Commit**

```bash
git add cad-trials/cad_trials/models/cadrecode.py cad-trials/envs/cadrecode.md cad-trials/tests/test_cadrecode_smoke.py cad-trials/vendor/cad-recode/cadrecode_model.py
git commit -m "feat: cad-recode point-cloud wrapper"
```

---

## Task 12: `common/report.py` — runs.jsonl → report.html (draft)

**Files:**
- Create: `cad-trials/cad_trials/common/report.py`
- Test: `cad-trials/tests/test_report.py`

**Interfaces:**
- Consumes: `common.io.load_runs`
- Produces:
  - `build_report(runs_path: str | Path, out_html: str | Path, prepared_dir: str | Path | None = None) -> None`
  - `coverage_table(rows: list[RunRecord]) -> list[dict]` — one dict per `(model, input_kind)`: `n`, `valid_code_pct`, `valid_geom_pct`, `mean_iou` (over non-null), `mean_chamfer`
  - HTML: reuses the visual language of the lit-map artifact (IBM Plex, drafting palette). Sections: **Coverage matrix** (models × input_kind, cells colour-scaled by valid_geom_pct), **Per-run table** (sortable, all columns), **Thumbnails** (if `prepared_dir` given and `<stem>+s<k>.stl` renders exist, embed as `<img>` base64 — else skip). No external JS/CSS; inline only.

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_report.py
from cad_trials.common.io import RunRecord, append_run
from cad_trials.common.report import build_report, coverage_table


def _rows(tmp_path):
    p = tmp_path / "runs.jsonl"
    for k in range(4):
        append_run(RunRecord(model="cadrille_img", weights_id="w", problem="p1",
                             input_path=f"x{k}.png", input_kind="render_hlr_lines",
                             sample=0, output_path=f"x{k}.py", wall_s=1.0,
                             error=None, valid_code=(k > 0), valid_geometry=(k > 1),
                             iou=(0.3 if k > 1 else None)), p)
    append_run(RunRecord(model="cadrecode", weights_id="w", problem="p1",
                         input_path="pc.ply", input_kind="pc_256", sample=0,
                         output_path="pc.py", wall_s=2.0, error=None,
                         valid_code=True, valid_geometry=True, iou=0.9), p)
    return p


def test_coverage_table(tmp_path):
    from cad_trials.common.io import load_runs
    tbl = coverage_table(load_runs(_rows(tmp_path)))
    cad = next(r for r in tbl if r["model"] == "cadrille_img")
    assert cad["n"] == 4
    assert cad["valid_code_pct"] == 75.0
    assert cad["valid_geom_pct"] == 50.0
    assert abs(cad["mean_iou"] - 0.3) < 1e-6


def test_build_report_writes_html(tmp_path):
    out = tmp_path / "report.html"
    build_report(_rows(tmp_path), out)
    html = out.read_text()
    assert "<title>" in html and "cadrille_img" in html and "coverage" in html.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest cad-trials/tests/test_report.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `report.py`.** Plain string templating (no jinja). `coverage_table` aggregates; `build_report` writes a full standalone HTML doc.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest cad-trials/tests/test_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Generate the real draft + commit**

```bash
python -c "from cad_trials.common.report import build_report; build_report('cad-trials/results/runs.jsonl', 'cad-trials/results/report.html', 'cad-trials/results/prepared/part')"
git add cad-trials/cad_trials/common/report.py cad-trials/tests/test_report.py cad-trials/results/runs.jsonl
git commit -m "feat: draft report.html generator from runs.jsonl"
```

---

## Task 13: Slurm harness

**Files:**
- Create: `cad-trials/slurm/prepare.sbatch`, `cad-trials/slurm/run_model.sbatch`
- Create: `cad-trials/slurm/make_manifest.py`
- Test: `cad-trials/tests/test_make_manifest.py`

**Interfaces:**
- Produces:
  - `make_manifest.py` — `build_manifest(prepared_dirs: list[str], models: list[str], out_tsv: str) -> None`, columns `model<TAB>input_path<TAB>input_kind<TAB>n_samples<TAB>env`. Model→(env, kinds) map: `cadrille_img`→(`cadrille`, render/tile/sheet kinds), `cadrille_pc`→(`cadrille`, `pc_256`), `cadrecode`→(`cadrecode`, `pc_256`).
  - `run_model.sbatch` — `#SBATCH --array=1-N`; reads line `$SLURM_ARRAY_TASK_ID` of the manifest; `conda activate $env`; `python -m cad_trials.models.$model --inputs $input_path --out-dir cad-trials/results/$model --gt cad-trials/results/prepared/<...>/normalized.stl --n-samples $n --runs-path cad-trials/results/runs.jsonl`. Resumability is handled inside the wrapper (`already_done`).
  - `prepare.sbatch` — single task: `conda activate probe-common`; runs `prepare_problem1` for `output/part.STL`.

- [ ] **Step 1: Write the failing test**

```python
# cad-trials/tests/test_make_manifest.py
from cad_trials.slurm.make_manifest import build_manifest


def test_manifest_rows(tmp_path):
    prep = tmp_path / "prepared" / "part"
    prep.mkdir(parents=True)
    for f in ["render_hlr_lines.png", "tile_4diag.png", "pc_256.ply"]:
        (prep / f).write_bytes(b"x")
    out = tmp_path / "manifest.tsv"
    build_manifest([str(prep)], ["cadrille_img", "cadrille_pc", "cadrecode"], str(out))
    lines = out.read_text().strip().splitlines()
    assert any("cadrille_img\t" in l and "render_hlr_lines.png" in l for l in lines)
    assert any("cadrecode\t" in l and "pc_256.ply" in l for l in lines)
    # cadrille_img must NOT get the point cloud
    assert not any("cadrille_img\t" in l and "pc_256" in l for l in lines)
```

- [ ] **Step 2: Run to verify it fails.** `python -m pytest cad-trials/tests/test_make_manifest.py -v` → module missing.

- [ ] **Step 3: Implement `make_manifest.py`** + the two `.sbatch` files (model the sbatch on `pycharm_rtx_3090_32h.sh`'s SBATCH header: `--partition main --qos normal --gpus rtx_3090:1 --cpus-per-task 4 --mem 24G --time 8:00:00`).

- [ ] **Step 4: Run to verify it passes.** `python -m pytest cad-trials/tests/test_make_manifest.py -v` → 1 passed.

- [ ] **Step 5: Dry-run the manifest + commit**

```bash
python -m cad_trials.slurm.make_manifest   # writes cad-trials/results/manifest.tsv for part
git add cad-trials/slurm/ cad-trials/tests/test_make_manifest.py
git commit -m "feat: slurm array-job harness + manifest builder"
```

---

## Task 14: End-to-end dry run + plan-1 report

**Files:**
- Modify: `cad-trials/README.md` (add the "run everything" section)
- Create: `cad-trials/results/report.html` (regenerated, committed)

**Interfaces:** none new.

- [ ] **Step 1: Full local test suite**

Run: `python -m pytest cad-trials/ -v -m "not slow"`
Expected: every non-slow test green.

- [ ] **Step 2: On the cluster — prepare + all three models on the reference part**

```bash
sbatch cad-trials/slurm/prepare.sbatch
# after it finishes:
python -m cad_trials.slurm.make_manifest
sbatch cad-trials/slurm/run_model.sbatch    # array over the manifest
```

- [ ] **Step 3: Build the report**

```bash
python -c "from cad_trials.common.report import build_report; build_report('cad-trials/results/runs.jsonl','cad-trials/results/report.html','cad-trials/results/prepared/part')"
```

- [ ] **Step 4: Sanity-check the numbers.** cad-recode on the reference part should produce valid geometry (it is a clean CAD-ish solid); if IoU is implausibly low (<0.2) inspect the predicted STL vs GT orientation — note it, don't hide it. cadrille image results are expected to be poor on the non-tile renders; that's a finding.

- [ ] **Step 5: Commit + report handoff**

```bash
git add cad-trials/README.md cad-trials/results/report.html cad-trials/results/runs.jsonl
git commit -m "chore: plan-1 end-to-end dry run — 3 models on reference part"
```

Then publish `report.html` as an Artifact and share the link.

---

## Self-Review

**1. Spec coverage (Plan 1 slice = spec M0–M2):**
- Spec §2.1 render styles + point clouds → Tasks 5, 7 ✓
- Spec §2.1.1 (HLR resolved by dataset choice) → `three_view.png` is approximate/best-effort per Task 5; full drawing datasets are Plan 2 ✓ (out of Plan 1 scope, noted)
- Spec §3 metrics (valid_code, valid_geometry, IoU, chamfer, n_ops) → Tasks 3, 4; wired in Tasks 9–11 ✓
- Spec §3 calibration run → Plan 2 (needs benchmark fetch) — **out of Plan 1 scope**, flagged here
- Spec §4 cadrille_img, cadrille_pc, cadrecode → Tasks 9, 10, 11 ✓
- Spec §5 harness tree, per-model envs, Slurm array, output contract, runs.jsonl → Tasks 1, 6, 8, 13 ✓
- Spec §6 report → Task 12 (draft; full version Plan 3) ✓
- Spec §7 M0–M2 → this plan; M3–M7 → Plans 2 and 3 ✓

**2. Placeholder scan:** No "TBD"/"handle errors appropriately". Tasks 10 §Step 3 and 11 §Step 4 defer exact input-normalization / class-extraction detail to "read the vendored repo and match" — this is unavoidable (the transform is defined by an external checkpoint, not by us) but is bounded: the task says which file to read and to document the transform with a file:line reference. Acceptable.

**3. Type consistency:** `RunRecord` fields (Task 1) are used consistently in Tasks 9–12. `ExecResult` (Task 3) consumed in Tasks 9–11 and `valid_geometry` in Task 12's inputs. `render_style`/`STYLES`/`four_diagonal_tile`/`ortho_three_view` (Task 5) consumed in Tasks 7, 9. `standard_parser`/`iter_inputs`/`write_output`/`gt_for` (Task 6) consumed in Tasks 9–11. `load_runs` (Task 1) consumed in Tasks 12, 13. Consistent.

**4. Scope:** Plan 1 is M0–M2 only — foundation + 3 models + draft report. Plans 2 (breadth: Point2CAD, VLM, benchmarks, calibration, PlankAssembly/Drawing2CAD/CReFT-CAD) and 3 (Problem 2 + final report) follow. Each plan leaves a shippable report.
