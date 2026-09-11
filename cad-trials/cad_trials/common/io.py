"""The run ledger: one JSON object per (model, input, sample) attempt.

``runs.jsonl`` is append-only and is read back by the resume check
(:func:`run_exists`) and the report builder.  On the cluster each array task
writes its **own** shard under ``results/runs.d/<task_id>.jsonl`` -- concurrent
unlocked appends to a single NFS file can tear a line -- and
:func:`load_runs` accepts either a single file or such a directory.
"""
from __future__ import annotations
import json
import sys
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
    # appended (keep at the end: existing positional/keyword construction must work)
    watertight: bool | None = None
    part: str | None = None


DEFAULT_PATH = "cad-trials/results/runs.jsonl"
_FIELD_NAMES = {f.name for f in fields(RunRecord)}


def append_run(record: RunRecord, path: str | Path = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")


def _ledger_files(path: Path) -> list[Path]:
    """The ``*.jsonl`` files behind ``path`` (a single file, or a shard dir)."""
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return [path] if path.exists() else []


def load_runs(path: str | Path = DEFAULT_PATH) -> list[RunRecord]:
    """Read a ledger file -- or every ``*.jsonl`` in a ledger *directory*.

    Unparseable lines (a torn append from a concurrent NFS writer) are reported
    on stderr and skipped: one bad line must not halt a resume or a report.
    """
    out: list[RunRecord] = []
    for f in _ledger_files(Path(path)):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                print(f"warning: skipping unparseable line {n} in {f}", file=sys.stderr)
                continue
            out.append(RunRecord(**{k: v for k, v in d.items() if k in _FIELD_NAMES}))
    return out


def run_exists(model: str, input_path: str, sample: int,
               path: str | Path = DEFAULT_PATH,
               retry_errors: bool = False) -> bool:
    """True when ``(model, input_path, sample)`` has already been *attempted*.

    Any recorded row counts as done, error or not.  Generation/execution failures
    are the common outcome in this survey, so treating them as "not done" made
    every array-job resume re-run them and append a *second* row -- silently
    inflating the denominators in :func:`~cad_trials.common.report.coverage_table`.

    ``retry_errors=True`` (the ``--retry-errors`` CLI flag) restores the old
    behaviour: only error-free rows count as done.
    """
    for r in load_runs(path):
        if (r.model, r.input_path, r.sample) != (model, input_path, sample):
            continue
        if retry_errors and r.error is not None:
            continue
        return True
    return False
