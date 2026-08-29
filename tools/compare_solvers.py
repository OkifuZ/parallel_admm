#!/usr/bin/env python3
"""Compare two solver runs of the same scene frame-by-frame.

Runs `main.exe` once per solver (headless batch export), then compares the
per-frame vertex positions and step metrics (contacts, per-step time).

Usage:
  python tools/compare_solvers.py \
      --scene resource/scene/test_XPBD_ADMM.toml \
      --resource resource \
      --solvers admm,xpbd \
      --frames 30 \
      [--exe build/Release/main.exe] [--out build/compare_out] [--no-plot]

Outputs (under --out):
  <solver>/          OBJ frame dumps + step_metrics.csv per solver run
  frames.csv         per-frame comparison (mean/max vertex diff, metrics)
  compare.png        plots (requires matplotlib; skip with --no-plot)
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="scene TOML path")
    p.add_argument("--resource", required=True, help="resource directory (meshes)")
    p.add_argument("--solvers", default="admm,xpbd", help="comma-separated solver list")
    p.add_argument("--frames", type=int, default=None, help="frames to run (default: scene end_frame)")
    p.add_argument("--exe", default="build/Release/main.exe", help="path to main.exe")
    p.add_argument("--out", default="build/compare_out", help="output directory")
    p.add_argument("--no-plot", action="store_true", help="skip matplotlib plots")
    p.add_argument("--keep", action="store_true", help="keep per-solver run dirs")
    return p.parse_args()


SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]")


def patch_scene(src, solver, frames, out_dir):
    """Copy the scene TOML, forcing one solver + headless batch export."""
    lines = open(src, encoding="utf-8", errors="replace").read().splitlines()
    out = []
    section = ""
    solver_type_set = False
    for line in lines:
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1)
        stripped = line.strip()
        if section == "solver" and stripped.startswith("type"):
            out.append(f"type = '{solver}'")
            solver_type_set = True
            continue
        if section == "xpbd" and stripped.startswith("enable"):
            # The legacy [xpbd] enable flag would force XPBD even for admm runs.
            out.append(f"enable = {'true' if solver == 'xpbd' else 'false'}")
            continue
        out.append(line)
    # top-level / [solver] type insertion
    if not solver_type_set:
        for i, line in enumerate(out):
            if SECTION_RE.match(line) and SECTION_RE.match(line).group(1) == "solver":
                out.insert(i + 1, f"type = '{solver}'")
                break
    # force headless + batch export + frames
    def set_kv(out, key, value):
        for i, line in enumerate(out):
            if not SECTION_RE.match(line) and line.strip().startswith(key + " "):
                out[i] = f"{key} = {value}"
                return
        out.insert(0, f"{key} = {value}")  # top-level scalar, insert at top
    set_kv(out, "show_windows", "false")
    set_kv(out, "export_obj", "true")
    set_kv(out, "separate_out", "false")
    if frames:
        set_kv(out, "end_frame", str(frames))
    set_kv(out, "out_file", f"'{out_dir}'")
    return "\n".join(out) + "\n"


def run_solver(args, solver, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # export_frame writes into <out_path>/<scene_name>/, so create it.
    scene_name = None
    for line in open(args.scene, encoding="utf-8", errors="replace"):
        if line.strip().startswith("scene_name"):
            scene_name = line.split("=", 1)[1].strip().strip("'\"")
            break
    if scene_name:
        os.makedirs(os.path.join(out_dir, scene_name), exist_ok=True)
    scene = os.path.join(out_dir, "scene.toml")
    patched = patch_scene(args.scene, solver, args.frames, out_dir)
    with open(scene, "w", encoding="utf-8") as f:
        f.write(patched)
    cmd = [args.exe, scene, args.resource]
    print(f"[{solver}] running: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-2000:])
        sys.exit(f"[{solver}] main.exe failed with exit code {r.returncode}")
    print(f"[{solver}] done")


def load_objs(run_dir, scene_name):
    """frame -> (n_verts,3) numpy-free list-of-lists of vertex positions."""
    frames = {}
    # export_frame writes <scene_name>\_NNNNN.obj under the run dir.
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "*.obj"), recursive=True)):
        m = re.search(r"_(\d{5})\.obj$", path)
        if not m:
            continue
        frame = int(m.group(1))
        verts = []
        for line in open(path, encoding="utf-8", errors="replace"):
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        frames[frame] = verts
    return frames


def load_metrics(run_dir):
    path = os.path.join(run_dir, "step_metrics.csv")
    if not os.path.exists(path):
        return {}
    metrics = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            metrics[int(row["frame"])] = (float(row["step_ms"]), int(row["n_contacts"]))
    return metrics


def compare(args, solvers):
    scene_name = None
    for line in open(args.scene, encoding="utf-8", errors="replace"):
        if line.strip().startswith("scene_name"):
            scene_name = line.split("=", 1)[1].strip().strip("'\"")
            break
    if not scene_name:
        sys.exit("scene_name not found in scene TOML")

    objs = {}
    metrics = {}
    for s in solvers:
        run_dir = os.path.join(args.out, s)
        objs[s] = load_objs(run_dir, scene_name)
        metrics[s] = load_metrics(run_dir)
        print(f"[{s}] {len(objs[s])} frames loaded, metrics {len(metrics[s])} entries")

    common = sorted(set(objs[solvers[0]]) & set(objs[solvers[1]]))
    if not common:
        sys.exit("no common frames to compare")
    print(f"comparing {len(common)} frames ({common[0]}..{common[-1]})")

    rows = []
    for f in common:
        a, b = objs[solvers[0]][f], objs[solvers[1]][f]
        n = min(len(a), len(b))
        mean = maxd = 0.0
        for i in range(n):
            d = sum((a[i][k] - b[i][k]) ** 2 for k in range(3)) ** 0.5
            mean += d
            maxd = max(maxd, d)
        mean /= max(n, 1)
        ma = metrics[solvers[0]].get(f, ("", ""))
        mb = metrics[solvers[1]].get(f, ("", ""))
        rows.append((f, mean, maxd, ma[1], mb[1], ma[0], mb[0]))

    out_csv = os.path.join(args.out, "frames.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "mean_vertex_diff", "max_vertex_diff",
                    f"contacts_{solvers[0]}", f"contacts_{solvers[1]}",
                    f"step_ms_{solvers[0]}", f"step_ms_{solvers[1]}"])
        w.writerows(rows)
    print(f"comparison CSV: {out_csv}")

    # summary
    means = [r[1] for r in rows]
    maxs = [r[2] for r in rows]
    print("\n== summary ==")
    print(f"mean vertex diff : {sum(means)/len(means):.6g} (max over frames {max(means):.6g})")
    print(f"max vertex diff  : {max(maxs):.6g} @ frame {max(rows, key=lambda r: r[2])[0]}")
    for s in solvers:
        ms = [r[5 if solvers.index(s) == 0 else 6] for r in rows if r[5 if solvers.index(s) == 0 else 6] != ""]
        if ms:
            print(f"{s} avg step ms : {sum(ms)/len(ms):.1f}")
    return rows


def plot(args, solvers, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return
    frames = [r[0] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(frames, [r[1] for r in rows], label="mean")
    axes[0].plot(frames, [r[2] for r in rows], label="max")
    axes[0].set_ylabel("vertex diff")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for i, s in enumerate(solvers):
        col = 4 + i  # contacts_*, step_ms_*
        axes[1].plot(frames, [r[col] if r[col] != "" else 0 for r in rows], label=s)
        axes[2].plot(frames, [r[col + 2] if r[col + 2] != "" else 0 for r in rows], label=s)
    axes[1].set_ylabel("n_contacts"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("step ms"); axes[2].set_xlabel("frame"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"compare: {' vs '.join(solvers)} ({args.scene})")
    out_png = os.path.join(args.out, "compare.png")
    fig.savefig(out_png, dpi=130)
    print(f"plot: {out_png}")


def main():
    args = parse_args()
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    if len(solvers) < 2:
        sys.exit("need at least two solvers")
    os.makedirs(args.out, exist_ok=True)
    for s in solvers:
        run_solver(args, s, os.path.join(args.out, s))
    rows = compare(args, solvers)
    if not args.no_plot:
        plot(args, solvers, rows)
    if not args.keep:
        for s in solvers:
            shutil.rmtree(os.path.join(args.out, s), ignore_errors=True)
        print("per-solver run dirs removed (--keep to retain)")


if __name__ == "__main__":
    main()
