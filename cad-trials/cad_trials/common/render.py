"""Render a single mesh into the input images the image-based models consume.

Three products:

* ``render_style`` -- the six Problem-1 line/shaded styles, each as a 2x2
  viewport (Front / Left / Top / Iso), matching the SolidWorks 4-viewport
  layout used by the ``extracts/`` screenshots.
* ``four_diagonal_tile`` -- cadrille's native input: 4 shaded views from the
  diagonal ``FRONTS`` set, pale-yellow solid on white, 3-px black border,
  tiled 2x2.  Ported from the (now removed) root ``render_mesh.py``.
* ``ortho_three_view`` -- an *approximate* Front/Top/Left silhouette + crease
  line-art strip.  Callers are expected to label it as approximate.

Matplotlib runs on the headless ``Agg`` backend.
"""
from __future__ import annotations

import numpy as np
import trimesh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

# NOTE: `hlr_paper` was called `draftsheet` until the foundation final review.
# It is hlr_lines on a paper-coloured ground -- it has no centrelines and no dashed
# hidden edges, so it is *not* the drafting-convention style the spec describes.
# A real drafting render is tracked in the Plan 2 backlog.
STYLES = ["shaded_color", "shaded_hlr", "wireframe", "hlr_lines", "hlr_paper"]

FRONTS = [(1, 1, 1), (-1, -1, -1), (-1, 1, -1), (1, -1, 1)]

ORTHO = {"front": (0, -1, 0), "top": (0, 0, 1), "left": (-1, 0, 0), "iso": (1, -1, 1)}

# cadrille's pale-yellow base colour ([255, 255, 136])
_BASE_COLOR = np.array([1.0, 1.0, 136 / 255])

# per-style rendering parameters (see task brief table)
_STYLE_PARAMS = {
    "shaded_color": dict(fill="#4a9d5b", shade=True, edge=None, lw=0.0, bg="white"),
    "shaded_hlr": dict(fill="#b8b8b8", shade=True, edge="#222222", lw=0.3, bg="white"),
    "wireframe": dict(fill=None, shade=False, edge="#222222", lw=0.4, bg="white"),
    "hlr_lines": dict(fill="#ffffff", shade=False, edge="#111111", lw=0.6, bg="white"),
    "hlr_paper": dict(fill="#ffffff", shade=False, edge="#111111", lw=0.7, bg="#f0efe6"),
}


def _hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def _fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(rgb)


def _view_angles(front) -> tuple[float, float]:
    front = np.asarray(front, float)
    azim = np.degrees(np.arctan2(front[1], front[0]))
    elev = np.degrees(np.arctan2(front[2], np.hypot(front[0], front[1])))
    return elev, azim


def _view(ax, mesh: trimesh.Trimesh, front, style: str) -> None:
    """Add one styled ``Poly3DCollection`` of ``mesh`` to a 3D axis."""
    p = _STYLE_PARAMS[style]
    tris = mesh.vertices[mesh.faces]

    if p["fill"] is None:
        facecolors = (0.0, 0.0, 0.0, 0.0)  # transparent -> wireframe
    elif p["shade"]:
        light = np.asarray(front, float)
        light = light / np.linalg.norm(light)
        shade = 0.35 + 0.65 * np.clip(mesh.face_normals @ light, 0, 1)
        rgb = np.clip(shade[:, None] * _hex_rgb(p["fill"])[None, :], 0, 1)
        facecolors = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
    else:
        facecolors = tuple(_hex_rgb(p["fill"])) + (1.0,)

    edgecolors = "none" if p["edge"] is None else p["edge"]
    poly = Poly3DCollection(
        tris, facecolors=facecolors, edgecolors=edgecolors, linewidths=p["lw"]
    )
    ax.add_collection3d(poly)

    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    c = mesh.vertices.mean(axis=0)
    r = np.ptp(mesh.vertices, axis=0).max() / 2 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)

    elev, azim = _view_angles(front)
    ax.view_init(elev=elev, azim=azim)


def render_style(mesh: trimesh.Trimesh, style: str, size: int = 256) -> Image.Image:
    """Render ``mesh`` as a 2x2 Front/Left/Top/Iso viewport in the given style."""
    if style not in _STYLE_PARAMS:
        raise ValueError(f"unknown style: {style!r}; choose from {STYLES}")
    p = _STYLE_PARAMS[style]

    fig = plt.figure(figsize=(4, 4), dpi=max(64, size // 2))
    fig.patch.set_facecolor(p["bg"])
    for i, key in enumerate(["front", "left", "top", "iso"]):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.set_facecolor(p["bg"])
        _view(ax, mesh, ORTHO[key], style)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return _fig_to_image(fig).resize((size, size))


# --------------------------------------------------------------------------- #
# cadrille's diagonal tile  (ported from render_mesh.py)
# --------------------------------------------------------------------------- #
def _render_diagonal_view(mesh: trimesh.Trimesh, front, size: int = 128) -> Image.Image:
    fig = plt.figure(figsize=(1, 1), dpi=size)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    tris = mesh.vertices[mesh.faces]
    light = np.asarray(front, float)
    light = light / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(mesh.face_normals @ light, 0, 1)
    colors = np.clip(shade[:, None] * _BASE_COLOR[None, :], 0, 1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolors="none"))

    c = mesh.vertices.mean(axis=0)
    r = np.ptp(mesh.vertices, axis=0).max() / 2 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)

    elev, azim = _view_angles(front)
    ax.view_init(elev=elev, azim=azim)
    return _fig_to_image(fig).resize((size, size))


def _add_border(im: Image.Image, border: int = 3) -> Image.Image:
    return Image.fromarray(
        np.pad(
            np.array(im),
            ((border, border), (border, border), (0, 0)),
            constant_values=0,
        )
    )


def four_diagonal_tile(mesh: trimesh.Trimesh, size: int = 128) -> Image.Image:
    """4 shaded diagonal views, 3-px black border each, tiled 2x2."""
    views = [_add_border(_render_diagonal_view(mesh, f, size)) for f in FRONTS]
    top = np.hstack([np.array(views[0]), np.array(views[1])])
    bot = np.hstack([np.array(views[2]), np.array(views[3])])
    return Image.fromarray(np.vstack([top, bot]))


# --------------------------------------------------------------------------- #
# approximate ortho three-view
# --------------------------------------------------------------------------- #
_PROJ_AXES = {"front": (0, 2), "top": (0, 1), "left": (1, 2)}
_PROJ_VIEWDIR = {"front": (0, -1, 0), "top": (0, 0, 1), "left": (-1, 0, 0)}


def _line_edges(mesh: trimesh.Trimesh, view_dir, crease_deg: float = 20.0) -> np.ndarray:
    """Vertex-index pairs for silhouette + crease + boundary edges."""
    v = np.asarray(view_dir, float)
    v = v / np.linalg.norm(v)

    edge_lists = []
    fa = mesh.face_adjacency
    if len(fa):
        n = mesh.face_normals
        d = np.column_stack([n[fa[:, 0]] @ v, n[fa[:, 1]] @ v])
        silhouette = (d[:, 0] * d[:, 1]) < 0
        crease = mesh.face_adjacency_angles > np.radians(crease_deg)
        keep = silhouette | crease
        edge_lists.append(mesh.face_adjacency_edges[keep])

    sorted_edges = np.sort(mesh.edges, axis=1)
    uniq, counts = np.unique(sorted_edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary):
        edge_lists.append(boundary)

    if not edge_lists:
        return np.zeros((0, 2), dtype=np.int64)
    return np.vstack(edge_lists)


def _ortho_panel(mesh: trimesh.Trimesh, name: str, size: int) -> np.ndarray:
    cols = _PROJ_AXES[name]
    edges = _line_edges(mesh, _PROJ_VIEWDIR[name])
    verts2d = mesh.vertices[:, cols]

    fig = plt.figure(figsize=(1, 1), dpi=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    if len(edges):
        ax.add_collection(
            LineCollection(verts2d[edges], colors="black", linewidths=1.2)
        )
    mn, mx = verts2d.min(axis=0), verts2d.max(axis=0)
    c = (mn + mx) / 2
    r = (mx - mn).max() / 2 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_aspect("equal")
    img = _fig_to_image(fig).resize((size, size))
    return np.asarray(img)


def ortho_three_view(mesh: trimesh.Trimesh, size: int = 256) -> Image.Image:
    """Front / Top / Left line-art panels side by side (approximate)."""
    panels = [_ortho_panel(mesh, name, size) for name in ("front", "top", "left")]
    return Image.fromarray(np.hstack(panels))
