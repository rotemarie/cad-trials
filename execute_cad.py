"""Execute cadrille-generated CadQuery code safely and export meshes.

CadQuery code from a model can hang or leak memory (see
https://github.com/CadQuery/cadquery/issues/1665), so each file is executed in a
separate process with a timeout. Produces an STL (+ optional STEP) and a
validity log.

Usage:
    python execute_cad.py work_dirs/test/2d+s0.py --out-dir work_dirs/test
    python execute_cad.py 'out/**/*.py' --out-dir out --timeout 5 --log out/validity.csv
"""

import os
import csv
import glob
import argparse
import multiprocessing as mp


def _worker(code, stl_path, step_path, q):
    try:
        import cadquery as cq  # noqa: F401
        import trimesh

        env = {}
        exec(code, env)
        # cadrille/cad-recode convention: the result solid is bound to `r`
        obj = env.get('r')
        if obj is None:
            raise RuntimeError("no variable `r` produced by the code")
        compound = obj.val() if hasattr(obj, 'val') else obj

        vertices, faces = compound.tessellate(0.001, 0.1)
        mesh = trimesh.Trimesh(
            [(v.x, v.y, v.z) for v in vertices], faces, process=False)
        mesh.export(stl_path)
        if step_path:
            cq.exporters.export(compound, step_path)

        q.put(dict(
            ok=True,
            volume=float(mesh.volume),
            area=float(mesh.area),
            bbox=mesh.bounds.tolist(),
            n_vertices=len(mesh.vertices),
            n_faces=len(mesh.faces),
            watertight=bool(mesh.is_watertight),
        ))
    except Exception as e:
        q.put(dict(ok=False, error=f'{type(e).__name__}: {e}'))


def run_one(py_path, out_dir, timeout, write_step):
    stem = os.path.splitext(os.path.basename(py_path))[0]
    stl_path = os.path.join(out_dir, f'{stem}.stl')
    step_path = os.path.join(out_dir, f'{stem}.step') if write_step else None
    with open(py_path) as f:
        code = f.read()

    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, stl_path, step_path, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(file=py_path, ok=False, error=f'timeout after {timeout}s')

    try:
        result = q.get_nowait()
    except Exception:
        result = dict(ok=False, error='process died without result')
    result['file'] = py_path
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='+',
                    help='.py files or globs (quote globs so the shell does not expand them)')
    ap.add_argument('--out-dir', default='work_dirs/executed')
    ap.add_argument('--timeout', type=float, default=5.0)
    ap.add_argument('--step', action='store_true', help='also export STEP')
    ap.add_argument('--log', default=None, help='CSV path for the validity log')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = []
    for pat in args.patterns:
        files.extend(sorted(glob.glob(pat, recursive=True)) or ([pat] if os.path.exists(pat) else []))
    if not files:
        raise SystemExit('no matching .py files')

    rows = []
    n_ok = 0
    for py_path in files:
        r = run_one(py_path, args.out_dir, args.timeout, args.step)
        rows.append(r)
        if r['ok']:
            n_ok += 1
            print(f'OK    {py_path}  volume={r["volume"]:.1f}  faces={r["n_faces"]}  '
                  f'watertight={r["watertight"]}')
        else:
            print(f'FAIL  {py_path}  {r["error"]}')

    print(f'\n{n_ok}/{len(files)} executed successfully')

    log_path = args.log or os.path.join(args.out_dir, 'validity.csv')
    keys = ['file', 'ok', 'error', 'volume', 'area', 'n_vertices', 'n_faces', 'watertight', 'bbox']
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'log: {log_path}')


if __name__ == '__main__':
    main()
