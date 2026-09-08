"""Render an STL/OBJ mesh into the 4-view tiled image cadrille's image branch expects.

Mirrors dataset.py's `get_img`: 4 viewpoints, pale-yellow shaded solid on white,
128 px each, 3 px black border, tiled 2x2. Uses matplotlib so it runs headless
(no Open3D). Feed the output to `infer_image.py --raw`.

Usage:
    python render_mesh.py model.stl --out model_input.png
"""

import os
import argparse

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

FRONTS = [(1, 1, 1), (-1, -1, -1), (-1, 1, -1), (1, -1, 1)]
BASE_COLOR = np.array([1.0, 1.0, 136 / 255])  # the [255,255,136] cadrille uses


def render_view(mesh, front, size=128):
    fig = plt.figure(figsize=(1, 1), dpi=size)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))

    tris = mesh.vertices[mesh.faces]
    light = np.asarray(front, float)
    light = light / np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(mesh.face_normals @ light, 0, 1)
    colors = np.clip(shade[:, None] * BASE_COLOR[None, :], 0, 1)

    ax.add_collection3d(Poly3DCollection(tris, facecolors=colors, edgecolors='none'))

    c = mesh.vertices.mean(axis=0)
    r = np.ptp(mesh.vertices, axis=0).max() / 2 * 1.15
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)

    azim = np.degrees(np.arctan2(front[1], front[0]))
    elev = np.degrees(np.arctan2(front[2], np.hypot(front[0], front[1])))
    ax.view_init(elev=elev, azim=azim)

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(rgb).resize((size, size))


def add_border(im, border=3):
    return Image.fromarray(np.pad(
        np.array(im), ((border, border), (border, border), (0, 0)), constant_values=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh')
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--num-imgs', type=int, default=4, choices=[1, 2, 4])
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force='mesh')
    # normalize into a unit cube centred at 0.5 (cadrille test meshes are unit-cube)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.0 / max(mesh.extents))
    mesh.apply_translation([0.5, 0.5, 0.5])

    views = [add_border(render_view(mesh, f, args.size)) for f in FRONTS]

    if args.num_imgs == 1:
        out = views[0]
    elif args.num_imgs == 2:
        out = Image.fromarray(np.hstack([np.array(views[0]), np.array(views[1])]))
    else:
        top = np.hstack([np.array(views[0]), np.array(views[1])])
        bot = np.hstack([np.array(views[2]), np.array(views[3])])
        out = Image.fromarray(np.vstack([top, bot]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.save(args.out)
    print(f'saved {args.out}  {out.size}')


if __name__ == '__main__':
    main()
