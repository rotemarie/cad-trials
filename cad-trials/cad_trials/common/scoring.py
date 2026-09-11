"""Model-agnostic scoring + RunRecord construction for the cad-trials harness.

These two helpers used to live in :mod:`cad_trials.models._cadrille_common`, but
that module imports ``torch`` / ``transformers`` / ``cadrille`` at load time, so a
wrapper running in a *different* conda env (e.g. :mod:`cad_trials.models.cadrecode`)
could not reuse them.  They are torch-free -- only :mod:`cad_trials.common`
(``meshes``, ``metrics``, ``io``, ``execute``) is touched -- so they belong here.

* :func:`score` -- predicted-STL-vs-GT ``(voxel_iou, chamfer)``.
* :func:`make_record` -- the :class:`~cad_trials.common.io.RunRecord` shared by
  every model wrapper (identical bar ``model=``).
"""
from __future__ import annotations

from cad_trials.common.execute import valid_geometry
from cad_trials.common.io import RunRecord
from cad_trials.common.meshes import load_mesh
from cad_trials.common.metrics import FillFailed, chamfer_distance, count_ops, voxel_iou

IOU_FILL_NOTE = "iou: fill fallback (non-watertight predicted mesh)"


def score(pred_stl: str, gt_path: str, with_note: bool = False):
    """(voxel_iou, chamfer) for a predicted STL vs GT; None on any load/metric error.

    ``with_note=True`` returns a 3-tuple ``(iou, chamfer, note)`` instead, where
    ``note`` is :data:`IOU_FILL_NOTE` when the IoU is ``None`` specifically because
    the predicted mesh could not be interior-filled (see
    :class:`cad_trials.common.metrics.FillFailed`) -- an *unscorable* IoU rather
    than a bad one.  ``chamfer`` is unaffected: it only samples surfaces.
    """
    note: str | None = None
    try:
        pred_mesh = load_mesh(pred_stl)
        gt_mesh = load_mesh(gt_path)
    except Exception:
        return (None, None, None) if with_note else (None, None)
    iou = chamfer = None
    try:
        iou = voxel_iou(pred_mesh, gt_mesh)
    except FillFailed:
        iou = None
        note = IOU_FILL_NOTE
    except Exception:
        iou = None
    try:
        chamfer = chamfer_distance(pred_mesh, gt_mesh)
    except Exception:
        chamfer = None
    return (iou, chamfer, note) if with_note else (iou, chamfer)


def make_record(*, model: str, weights: str, problem: str, input_path, kind: str,
                sample: int, out_path, res, code: str | None, gt: str | None,
                wall_s: float, error: str | None, part: str | None = None) -> RunRecord:
    """Build the RunRecord shared by every model wrapper (identical bar ``model=``).

    Merges ``res.error`` into ``error`` (when no earlier error), computes ``n_ops``
    from ``code`` and ``iou``/``chamfer`` via :func:`score` when the program
    executed and a GT mesh is available.  An IoU that is ``None`` because the
    predicted mesh could not be interior-filled is flagged in ``error`` with
    :data:`IOU_FILL_NOTE` so the gap is visible in the report.
    """
    if res is not None and res.error and error is None:
        error = res.error
    n_ops = count_ops(code) if code is not None else None
    iou = chamfer = None
    if res is not None and res.ok and gt:
        iou, chamfer, note = score(res.stl_path, gt, with_note=True)
        if note:
            error = note if error is None else f"{error}; {note}"
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
        watertight=res.watertight if res is not None else None,
        part=part,
    )
