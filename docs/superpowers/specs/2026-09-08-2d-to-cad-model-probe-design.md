# 2D-to-CAD Model Probe — Design

**Date:** 2026-09-08
**Status:** approved shortlist, spec under review
**Goal:** Run as many existing 2D→CAD reconstruction models as feasible on a controlled
set of inputs, produce a model×input results matrix + visual grid + writeup, so the
mentor can pick a thesis direction between two problems:

- **Problem 1** — orthographic views (Front/Top/Left/iso) → editable CAD
- **Problem 2** — freehand engineering sketch → editable CAD

This is a **breadth-first survey** (Approach A). Each model gets "does it run + what
does it output", with light quantitative metrics where ground truth exists. It is *not*
a depth study of any single model.

---

## 1. Scope

### In scope
- Fixed model shortlist (Section 4), inference only, pretrained weights only. **No training, no fine-tuning.**
- One reference part (`output/part.STL`, 1916 tris) rendered/derived into all input modalities.
- ~15–25 extra shapes sampled from public benchmarks for shape breadth.
- One calibration run reproducing a published metric.
- A published HTML report artifact for the mentor.

### Out of scope
- Training or fine-tuning any model (SECAD-Net's per-shape optimization is therefore excluded).
- Building a new model.
- Dimensional-accuracy evaluation (we check shape similarity, not whether Ø18 came out as 18).
- Problem 2 quantitative metrics (no GT 3D model for the Afeka part; validity + eyeball only).
- Fixing upstream repos beyond small compatibility shims.

---

## 2. Inputs

### 2.1 Reference part (Problem 1)
Source: `output/part.STL` (binary, mm units, a pillow-block: rectangular base +
central bored boss + 2 counterbored side holes + fillets). `output/id209015569.SLDPRT`
is kept for reference but not parsed.

**Optional SolidWorks exports (used only for the user-part OOD drawing probe, §4):**
- `output/part.STEP` — clean B-rep; lets PlankAssembly's own render script make a
  proper drawing of the reference part.
- `output/drawing.pdf` or `output/drawing_{front,top,right}.dxf` — the three standard
  views from a SolidWorks drawing; best-case OOD input for the drawing models.
Neither blocks anything (see §2.1.1).

#### 2.1.1 The HLR question — resolved by dataset choice
PlankAssembly and Drawing2CAD need *real engineering drawings* (clean per-view line art,
visible vs hidden edges), which needs a B-rep, not our triangle mesh. **We do not write
HLR code.** Instead:
- Those two models are evaluated **primarily on their own drawing-native datasets**
  (§2.3) — PlankAssembly's HF dataset rendered by its bundled `dataset/render_*.py`
  pythonocc scripts, Drawing2CAD's `svg_raw`, SPARE3D, TriView2CAD.
- The **reference part is a secondary OOD probe** for them: run PlankAssembly's own
  `render_visible_svg.py` on `part.STEP` if provided, else use the SolidWorks DXF/PDF,
  else skip and mark `drawing: OOD-unavailable`.
- For all *non-drawing* models, the STL renders (§2.1 table) are sufficient and carry no
  HLR risk.
The `views/*.svg` row below is therefore best-effort and only for the reference part.

`data/prepare_problem1.py` produces, from the STL:

| Artifact | How | Consumers |
|---|---|---|
| `normalized.stl` | center + scale to unit cube (and a 2-unit variant) | metrics, PC models |
| `pc_{256,2048,8192}.ply` | `trimesh.sample.sample_surface` → FPS | cad-recode, cadrille-pc, Point2CAD |
| `render_shaded_color.png` | matplotlib/trimesh, 4-viewport (Front/Left/Top/Iso) | cadrille, VLM |
| `render_shaded_hlr.png` | shaded + hidden-line overlay | cadrille, VLM |
| `render_wireframe.png` | all edges | cadrille, VLM |
| `render_hlr_lines.png` | visible edges only, black on white | cadrille, VLM, PlankAssembly-raster |
| `render_draftsheet.png` | HLR + centrelines + dashed hidden, paper bg | cadrille, VLM |
| `tile_4diag.png` | cadrille's native 4-diagonal shaded tile (via `render_mesh.py`) | cadrille (control) |
| `views/{front,top,right}.svg` + `.png` | best-effort, reference part only: PlankAssembly's `render_visible_svg.py` on `part.STEP`, or the SolidWorks DXF | PlankAssembly, Drawing2CAD (OOD probe) |
| `views/three_view.png` | the 3 ortho views composited on one sheet (from the render above, or from `render_hlr_lines`) | cadrille, VLM |

The 6 render styles mirror `extracts/image*.png` (what the user pulled from SolidWorks)
so results are comparable to the user's manual exploration. If `views/*` can't be built
cleanly for a given shape, that shape is still run on every non-drawing model and marked
`drawing: unavailable` in the matrix.

### 2.2 Afeka sketch (Problem 2)
Source: `extracts/sketch.png` (cropped hand drawing, 2 views, hand-lettered dims,
highlighter mark). `data/prepare_sketch.py` produces:
- `sketch_raw.png` — as-is, RGB→square pad
- `sketch_clean.png` — grayscale, adaptive threshold, despeckle, deskew
- `sketch_top.png`, `sketch_front.png` — the two views cropped separately
- `sketch_notext.png` — best-effort dimension-text removal (morphological / connected-component filter) — for models that only want geometry

### 2.3 Benchmark samples (`data/fetch_benchmarks.py`)
The drawing-input models are evaluated **primarily on drawing-native datasets** (their
own or benchmarks that ship line drawings) — this is the in-distribution, fair test and
carries no HLR risk. The reference part is the OOD probe on top.

| Dataset | Source | Take | Primary use |
|---|---|---|---|
| **PlankAssembly data** | `manycore-research/PlankAssembly` (HF) + bundled `dataset/render_*.py` | its test split, ~30 shapes | PlankAssembly in-distribution eval |
| **Drawing2CAD data** | Google Drive (`svg_raw` + `svg_vec`) | its test split, ~30 shapes | Drawing2CAD in-distribution eval |
| DeepCAD test meshes | `maksimko123/deepcad_test_mesh` (HF) | 25 (+ 50 for calibration) | PC models, calibration |
| Fusion360 test meshes | `maksimko123/fusion360_test_mesh` (HF) | 15 | PC models |
| SPARE3D | `ai4ce/SPARE3D` (Google Drive) | 15 three-view line-drawing sets | cross-model drawing input (Plank/Drawing2CAD/cadrille-img/VLM) + meshes as GT |
| TriView2CAD | `zhuofanChen/TriView2CAD` (ModelScope) | 15 real ortho+dimension sheets | cadrille-img, VLM, CReFT-CAD |
| cad-recode release STL | GitHub release | 1 | smoke test |

Each benchmark shape that has a mesh also goes through the Problem-1 `prepare` pipeline
so every non-drawing model sees a consistent input format.

---

## 3. Ground truth & metrics

- **Problem 1 GT:** `output/part.STL` (and each benchmark shape's own mesh).
- **Metrics per generated model:**
  - `valid_code` — CadQuery executes without error/timeout (subprocess, 15 s)
  - `valid_geometry` — result is a non-empty watertight-ish solid
  - `IoU` — voxelized (64³) intersection-over-union vs GT, after ICP-free canonical align (both normalized to unit cube)
  - `chamfer` — symmetric CD on 8192 surface samples
  - `n_primitives` / `n_ops` — parsed from the output program (proxy for parsimony)
- **Point2CAD** outputs surfaces/edges, not a program → report CD + #patches only.
- **Calibration:** cad-recode on 50 DeepCAD test meshes, `--mode pc`, 1 sample; expect
  mean IoU ≈ 0.90–0.95 (paper reports ~0.94). Pass = within ~0.05.
- **Problem 2:** `valid_code`, `valid_geometry`, and a rendered thumbnail only.

All numbers land in `results/runs.jsonl` (one row per `model × input × sample`).

---

## 4. Models

Each model runs in its **own conda/venv or container** (`envs/<model>.md` documents
setup). All wrappers expose the same CLI:

```
python -m models.<name> --inputs <glob> --out-dir results/<name> --n-samples N [--seed-base S]
```
and write, per input: `<stem>+s<k>.py` (or `.stl` for Point2CAD) + a `meta.json`.

### Problem 1 — editable CAD
| Wrapper | Repo / weights | Input | Notes / risk |
|---|---|---|---|
| `cadrille_img` | col14m/cadrille, `maksimko123/cadrille` | render PNGs, tiles, 3-view sheet | **exists** (`infer_image.py`), refactor into `models/`. `--raw` for prepared tiles. |
| `cadrille_pc` | same | `.ply` point cloud | new wrapper; reuse repo `collate`, `--mode pc` path. |
| `plankassembly` | manycore-research/PlankAssembly | **primary:** its own HF test split (SVGs via bundled `dataset/render_*.py`). **OOD:** reference part via `render_visible_svg.py` on `part.STEP` | furniture-domain — expected to do poorly on mech parts; that is a finding. Check checkpoint release; if none → "documented, not run". torch 1.10/cu113 env. |
| `drawing2cad` | lllssc/Drawing2CAD | **primary:** its own `svg_raw` test split. **OOD:** reference-part SVG if available | ACM-MM'25. Verify weights on their Google Drive; if absent → documented only. |
| `creft_cad` | KeNiu042/CReFT-CAD + ModelScope ckpt `zhuofanChen/CReFT-CAD` | ortho sheet (+ dims) | training code gated on acceptance, but **checkpoint + TriView2CAD are on ModelScope**. Attempt inference with the ckpt (it's a Qwen2-VL SFT). If ckpt won't load → documented only. |
| `vlm_baseline` | Qwen2.5-VL-7B-Instruct (HF), local on 3090 | any render + 3-view sheet | prompt → CadQuery code; fixed prompt template in `models/vlm_baseline.py`. Also an optional API path (Claude/GPT-4o) behind an env var, off by default. |

### Problem 1 — mesh / point cloud
| Wrapper | Repo / weights | Input | Notes |
|---|---|---|---|
| `cadrecode` | filapro/cad-recode, `filapro/cad-recode-v1.5` | `pc_256.ply` | **exists** (notebook logic), wrap into `models/`. |
| `cadrille_pc` | (above) | `pc_256.ply` | shared with Problem 1 list. |
| `point2cad` | prs-eth/point2cad, Docker `toshas/point2cad:v1` | `pc_8192.ply` (+ normals) | outputs surfaces/edges/corners; ~5 min/shape; run via Docker or Singularity on the node. |

### Problem 2 — sketch
| Wrapper | Input | Notes |
|---|---|---|
| `cadrille_img` | `sketch_*.png` | reuse Problem-1 wrapper. |
| `vlm_baseline` | `sketch_*.png` | reuse; prompt asks for CadQuery from the drawing + dims. |
| `free2cad` | — | data + weights hosting broken per README; attempt, expect blocked, document. |

---

## 5. Harness architecture

```
cad-trials/                      (new top-level dir in the project)
  data/
    prepare_problem1.py         STL -> renders / SVGs / point clouds
    prepare_sketch.py           sketch.png -> cleaned variants
    fetch_benchmarks.py         download + subsample benchmark inputs
  models/
    cadrille_img.py  cadrille_pc.py  cadrecode.py
    plankassembly.py  drawing2cad.py  creft_cad.py
    point2cad.py  vlm_baseline.py  free2cad.py
    _base.py                    shared CLI arg parsing + output contract
  common/
    execute_cad.py              (exists) CadQuery exec + validity   [moved in]
    render.py                   (exists render_mesh.py) + the 6 style renders + multiview  [moved in]
    metrics.py                  IoU (voxel), chamfer, program parse
    grid.py                     contact-sheet / matrix figure builder
    io.py                       runs.jsonl append, meta.json schema
  envs/                         one markdown per model: exact install steps
  slurm/
    prepare.sbatch
    run_model.sbatch            array job: one task per (model,input); resumable
    calibrate.sbatch
  results/
    runs.jsonl                  append-only log
    <model>/…                   raw outputs (.py/.stl/.json + thumbnails)
    report.html                 mentor deliverable (published as Artifact)
  README.md                     how to reproduce end to end
```

**Env isolation:** `envs/` holds per-model setup. Conflicts are real (PlankAssembly
torch 1.10 vs cadrille 2.5.1 vs Point2CAD docker vs Free2CAD TensorFlow). No shared
mega-env. A small `probe-common` env runs `data/` and `common/` (trimesh, pythonocc,
numpy, matplotlib, cadquery).

**Slurm:** `run_model.sbatch` is a `--array` job reading a manifest
(`results/manifest.tsv`, columns: model, input_path, n_samples). Each task activates the
right env, runs one wrapper on one input, appends to `runs.jsonl`. Re-running skips rows
already present. One RTX 3090 per task (the existing `pycharm_rtx_3090_32h.sh` pattern,
adapted to a batch job).

**Output contract (every wrapper):** for input `X` writes
`results/<model>/<X-stem>+s<k>.py` (program) or `.stl` (Point2CAD), plus one
`results/<model>/<X-stem>.meta.json` with `{model, weights_id, input, params, wall_s,
error}`. `common/` then executes programs, computes metrics, appends `runs.jsonl`.

---

## 6. Deliverable — `results/report.html`

Published as an Artifact (like the earlier lit-map). Sections:

1. **Setup** — the reference part, the 6 render styles side by side, the sketch variants, the benchmark sample.
2. **Coverage matrix** — models (rows) × input types (cols); each cell = ran / valid-code% / valid-geom% / mean IoU, color-coded.
3. **Visual grid** — for the reference part: each model's best-of-N reconstruction rendered next to the GT, per input style. Same for 4–6 benchmark shapes.
4. **Problem 2 panel** — sketch in, each model's output rendered, plus the failure-mode notes.
5. **Calibration** — cad-recode vs DeepCAD paper number.
6. **What we learned** — per problem: which model families produced anything usable, how sensitive to render style, where the wall is. Ends with a recommended thesis direction and the 2–3 concrete open gaps it would target.
7. **Repro appendix** — env list, dataset versions/licenses, commit.

---

## 7. Order of work (milestones)

1. **M0 – common infra:** move `execute_cad.py` + `render_mesh.py` into `common/`,
   add `metrics.py`, `io.py`, `grid.py`, `probe-common` env. Verify on existing
   `work_dirs/dc0` outputs.
2. **M1 – prepare Problem 1:** `prepare_problem1.py` on `output/part.STL` → all
   artifacts; eyeball the renders. Reference-part SVG is best-effort (needs `part.STEP`).
3. **M2 – already-running models:** wrap `cadrille_img`, `cadrille_pc`, `cadrecode`;
   run on the reference part; first rows in `runs.jsonl`; first draft `report.html`.
4. **M3 – easy adds:** `point2cad` (docker), `vlm_baseline` (Qwen2.5-VL-7B).
5. **M4 – benchmarks:** `fetch_benchmarks.py`; rerun M2–M3 models over the sample;
   calibration run.
6. **M5 – harder adds:** `plankassembly`, `drawing2cad`, `creft_cad` — each is
   time-boxed; if weights/inference don't come up in ~half a day it becomes
   "documented, not run" in the report.
7. **M6 – Problem 2:** `prepare_sketch.py`; run cadrille + VLM + attempt Free2CAD.
8. **M7 – report:** finalize `report.html`, publish Artifact, write "what we learned".

Milestones 3–6 each leave the report in a shippable state, so the mentor meeting can
happen against whatever depth is reached.

---

## 8. Open risks

| Risk | Mitigation |
|---|---|
| Clean engineering drawings of the *reference part* | resolved by design: drawing models eval on drawing-native datasets; reference part is best-effort OOD only (§2.1.1) |
| PlankAssembly / Drawing2CAD / CReFT-CAD weights not actually downloadable | each is time-boxed; degrade to "documented, not run" |
| Env conflicts eat time | strict per-model isolation; Point2CAD stays containerized |
| Voxel IoU misleading for thin features | report Chamfer alongside; 64³ minimum, note limitation |
| Cluster queue slow | array jobs + resumable; models are cheap (<1 GPU-hr each) once set up |
| No GT for Afeka part | Problem 2 is explicitly qualitative; offer user to model it once for a bonus GT |

---

## 9. Non-goals restated
No training. No new model. No dimensional metrology. No repo forking beyond shims.
If a model needs fine-tuning to produce output (SECAD-Net), it is documented, not run.
