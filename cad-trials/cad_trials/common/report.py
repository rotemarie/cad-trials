"""Turn ``results/runs.jsonl`` into a standalone ``results/report.html``.

This is the *draft* mentor deliverable for the cad-trials harness: a single
self-contained HTML file (inline CSS, no JS, no external assets bar the IBM Plex
web-font ``<link>``) with

* a **coverage matrix** -- models (rows) x input_kind (cols), each cell shaded
  by valid-geometry percentage,
* a **per-run table** -- every column of every :class:`~cad_trials.common.io.RunRecord`,
* an optional **thumbnails** strip -- rendered model outputs (``<stem>+s<k>.png``)
  found in ``prepared_dir``, embedded as base64 ``<img>``; skipped entirely when
  none are present.

Torch-free by construction -- only :mod:`cad_trials.common.io` is imported, so it
runs in the lightweight ``probe-common`` env.
"""
from __future__ import annotations

import base64
import html
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from cad_trials.common.io import RunRecord, load_runs

_FIELDS = [
    "model", "weights_id", "problem", "input_path", "input_kind", "sample",
    "output_path", "wall_s", "error", "valid_code", "valid_geometry",
    "iou", "chamfer", "n_ops", "gt_path",
]

# drafting palette (shared with the lit-map artifact)
_PAPER = "#f6f5f1"
_INK = "#1b2431"
_RED = "#b23b30"
_HAIRLINE = "#dcd9cf"


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _mean(vals: list[float | None]) -> float | None:
    present = [v for v in vals if v is not None]
    return sum(present) / len(present) if present else None


def coverage_table(rows: list[RunRecord]) -> list[dict]:
    """One dict per ``(model, input_kind)`` group.

    Keys: ``model``, ``input_kind``, ``n``, ``valid_code_pct``,
    ``valid_geom_pct``, ``mean_iou``, ``mean_chamfer``.  Percentages are over the
    ``n`` rows in the group (``True`` count / n * 100).  ``mean_iou`` /
    ``mean_chamfer`` average only the non-``None`` values, and are ``None`` when
    the group has none.
    """
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for r in rows:
        groups.setdefault((r.model, r.input_kind), []).append(r)

    out: list[dict] = []
    for (model, kind), grp in sorted(groups.items()):
        n = len(grp)
        out.append({
            "model": model,
            "input_kind": kind,
            "n": n,
            "valid_code_pct": sum(1 for r in grp if r.valid_code is True) / n * 100,
            "valid_geom_pct": sum(1 for r in grp if r.valid_geometry is True) / n * 100,
            "mean_iou": _mean([r.iou for r in grp]),
            "mean_chamfer": _mean([r.chamfer for r in grp]),
        })
    return out


# --------------------------------------------------------------------------- #
# html rendering
# --------------------------------------------------------------------------- #
def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "&mdash;"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return _e(x)


def _shade(pct: float | None) -> str:
    """Interpolate cell background paper -> drafting green by ``pct`` (0..100)."""
    if pct is None:
        return _PAPER
    t = max(0.0, min(1.0, pct / 100.0))
    p = (0xf6, 0xf5, 0xf1)
    g = (0x4a, 0x9d, 0x5b)
    r, gg, b = (round(p[i] + (g[i] - p[i]) * t) for i in range(3))
    return f"#{r:02x}{gg:02x}{b:02x}"


def _coverage_section(rows: list[RunRecord]) -> str:
    tbl = coverage_table(rows)
    if not tbl:
        return "<section><h2>Coverage matrix</h2><p class='empty'>No runs recorded yet.</p></section>"

    models = sorted({r["model"] for r in tbl})
    kinds = sorted({r["input_kind"] for r in tbl})
    cell = {(r["model"], r["input_kind"]): r for r in tbl}

    head = "".join(f"<th>{_e(k)}</th>" for k in kinds)
    body = []
    for m in models:
        tds = []
        for k in kinds:
            c = cell.get((m, k))
            if c is None:
                tds.append("<td class='cov-cell empty'>&mdash;</td>")
                continue
            tds.append(
                f"<td class='cov-cell' style='background:{_shade(c['valid_geom_pct'])}'>"
                f"<span class='big'>{c['valid_geom_pct']:.0f}%</span> geom"
                f"<span class='sub'>n={c['n']} &middot; code {c['valid_code_pct']:.0f}%"
                f" &middot; IoU {_fmt(c['mean_iou'], 2)}</span></td>"
            )
        body.append(f"<tr><th class='rowhead'>{_e(m)}</th>{''.join(tds)}</tr>")

    return (
        "<section><h2>Coverage matrix</h2>"
        "<p class='note'>Cell shading tracks valid-geometry rate. "
        "Each cell: geom% (large), then run count, valid-code%, mean IoU.</p>"
        "<div class='scroll'><table class='cov'>"
        f"<thead><tr><th>model \\ input_kind</th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div></section>"
    )


def _runs_section(rows: list[RunRecord]) -> str:
    if not rows:
        return "<section><h2>Per-run table</h2><p class='empty'>No runs recorded yet.</p></section>"
    head = "".join(f"<th>{_e(f)}</th>" for f in _FIELDS)
    body = []
    for r in rows:
        d = r.__dict__
        tds = []
        for f in _FIELDS:
            v = d.get(f)
            if f in ("iou", "chamfer", "wall_s"):
                tds.append(f"<td class='num'>{_fmt(v)}</td>")
            elif f in ("valid_code", "valid_geometry"):
                mark = "&#10003;" if v is True else ("&times;" if v is False else "&mdash;")
                cls = "ok" if v is True else ("bad" if v is False else "")
                tds.append(f"<td class='mark {cls}'>{mark}</td>")
            elif f == "error":
                tds.append(f"<td class='err'>{_e(v) if v else ''}</td>")
            else:
                tds.append(f"<td>{_fmt(v)}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    return (
        f"<section><h2>Per-run table</h2><p class='note'>{len(rows)} run(s).</p>"
        "<div class='scroll'><table class='runs'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>"
    )


def _thumbs_section(prepared_dir: Path | None) -> str:
    if prepared_dir is None:
        return ""
    prepared_dir = Path(prepared_dir)
    if not prepared_dir.is_dir():
        return ""
    # rendered model outputs: "<stem>+s<k>.png"
    imgs = sorted(p for p in prepared_dir.glob("*+s*.png") if p.is_file())
    if not imgs:
        return ""
    cards = []
    for p in imgs:
        mime = mimetypes.guess_type(p.name)[0] or "image/png"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        cards.append(
            f"<figure><img alt='{_e(p.stem)}' src='data:{mime};base64,{b64}'>"
            f"<figcaption>{_e(p.stem)}</figcaption></figure>"
        )
    return (
        "<section><h2>Thumbnails</h2>"
        "<p class='note'>Rendered model reconstructions found in the prepared dir.</p>"
        f"<div class='thumbs'>{''.join(cards)}</div></section>"
    )


_CSS = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: {_PAPER}; color: {_INK};
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5; -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 3rem 1.5rem 5rem; }}
header {{ border-bottom: 2px solid {_INK}; padding-bottom: 1.25rem; margin-bottom: 2.5rem; }}
h1 {{ font-size: 1.6rem; font-weight: 600; margin: 0 0 .35rem; letter-spacing: -0.01em; }}
h1 .accent {{ color: {_RED}; }}
h2 {{
  font-size: 1.05rem; font-weight: 600; margin: 0 0 .6rem;
  text-transform: uppercase; letter-spacing: 0.08em;
}}
section {{ margin-bottom: 3rem; }}
.meta {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .8rem; color: #5c6672; }}
.note {{ font-size: .85rem; color: #5c6672; margin: 0 0 1rem; }}
.empty {{ font-style: italic; color: #8a8f97; }}
.scroll {{ overflow-x: auto; border: 1px solid {_HAIRLINE}; border-radius: 3px; }}
table {{
  border-collapse: collapse; width: 100%;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .78rem; font-variant-numeric: tabular-nums;
}}
thead th {{
  background: {_INK}; color: {_PAPER}; font-weight: 500; text-align: left;
  padding: .5rem .7rem; white-space: nowrap; position: sticky; top: 0;
}}
tbody td, tbody th {{ padding: .45rem .7rem; border-top: 1px solid {_HAIRLINE}; vertical-align: top; }}
tbody th.rowhead {{ text-align: left; font-weight: 600; background: #efeee8; white-space: nowrap; }}
table.cov td.cov-cell {{ min-width: 8.5rem; }}
.cov-cell .big {{ display: block; font-size: 1rem; font-weight: 600; }}
.cov-cell .sub {{ display: block; font-size: .68rem; color: #3f4854; margin-top: .2rem; }}
td.num, td.mark {{ text-align: right; white-space: nowrap; }}
td.mark {{ text-align: center; }}
td.mark.ok {{ color: #2f7d43; }}
td.mark.bad {{ color: {_RED}; }}
td.err {{ color: {_RED}; max-width: 22rem; }}
tbody tr:nth-child(even) td {{ background: rgba(0,0,0,0.015); }}
.thumbs {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
.thumbs figure {{ margin: 0; width: 200px; border: 1px solid {_HAIRLINE}; background: #fff; padding: .5rem; }}
.thumbs img {{ width: 100%; display: block; }}
.thumbs figcaption {{
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .68rem; color: #5c6672; margin-top: .4rem; word-break: break-all;
}}
footer {{ margin-top: 4rem; padding-top: 1.25rem; border-top: 1px solid {_HAIRLINE};
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .72rem; color: #8a8f97; }}
"""


def build_report(runs_path, out_html, prepared_dir=None) -> None:
    """Write a standalone ``report.html`` summarising ``runs_path``."""
    rows = load_runs(runs_path)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_models = len({r.model for r in rows})

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cad-trials &mdash; draft report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>cad-trials <span class="accent">&mdash;</span> draft report</h1>
  <p class="meta">generated {generated} &middot; {len(rows)} run(s) &middot; {n_models} model(s)</p>
</header>
{_coverage_section(rows)}
{_runs_section(rows)}
{_thumbs_section(prepared_dir)}
<footer>Draft mentor deliverable. Regenerate with
<code>cad_trials.common.report.build_report</code>.</footer>
</div>
</body>
</html>
"""
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
