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
