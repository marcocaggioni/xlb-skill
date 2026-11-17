#!/usr/bin/env python3
"""Render videos from the Couette field snapshots (runs/gamma_*/field_*.csv).

Outputs (in examples/couette/):
  couette-gamma10.mp4  — velocity field vs time at gamma = 10 1/s (the
                         run is initialised with the analytic Couette
                         profile, so the field is steady from the
                         start — this video shows it holding steady;
                         the bob itself renders dark — solid-body nodes
                         carry no fluid velocity — and the bright
                         annulus is the sheared fluid).
  couette-sweep.mp4    — steady-state field at each sweep shear rate,
                         gamma = 0.1 -> 100 1/s.
Frames are rendered with matplotlib (Agg) and encoded with ffmpeg
(libx264). Requires the sweep to have been run first.
"""
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
RI, RO = 0.020, 0.025


def read_field(path):
    xs, ys, ux, uy = [], [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            xs.append(int(row["iX"])); ys.append(int(row["iY"]))
            ux.append(float(row["ux_m_s"])); uy.append(float(row["uy_m_s"]))
    nx = max(xs) // 2 + 1
    ny = max(ys) // 2 + 1
    # The CSV rows are ordered i (x) outer, j (y) inner, so the flat
    # array reshapes to [x, y]; imshow/quiver need [y, x] — transpose.
    UX = np.array(ux).reshape(nx, ny).T
    UY = np.array(uy).reshape(nx, ny).T
    # Node coordinates in metres (grid step 2 * dx, dx = 0.5 mm;
    # grid centre is at index 50.5 for the 102 x 102 XLB grid).
    X = (np.arange(nx) * 2 - 50.5) * 0.0005
    Y = (np.arange(ny) * 2 - 50.5) * 0.0005
    return X, Y, UX, UY


def render_frame(path, out_png, title, vmax):
    X, Y, UX, UY = read_field(path)
    speed = np.hypot(UX, UY)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(speed, extent=[X[0], X[-1], Y[0], Y[-1]], origin="lower",
                   cmap="viridis", vmin=0, vmax=vmax)
    ax.quiver(X[::4], Y[::4], UX[::4, ::4], UY[::4, ::4], color="white",
              scale=2.5 * vmax, width=0.003)
    for r in (RI, RO):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="white", lw=1.5))
    ax.set_xlim(-0.026, 0.026); ax.set_ylim(-0.026, 0.026)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="speed (m/s)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def encode(frames_dir, out_mp4, fps):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(frames_dir / "frame_%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)],
                   check=True)
    print("wrote", out_mp4)


def main():
    tmp = Path("/tmp/couette-frames")
    # Clean the frame dirs first: a previous render (e.g. the Palabos
    # video, which uses the same /tmp path) leaves stale frames behind,
    # and ffmpeg would silently include them.
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    # --- time evolution at gamma = 10 ---
    run10 = HERE / "runs" / "gamma_10"
    fields = sorted(run10.glob("field_*.csv"),
                    key=lambda p: int(p.stem.split("_")[1]))
    d = tmp / "g10"; d.mkdir(parents=True, exist_ok=True)
    dt = 4.16667e-05
    for k, f in enumerate(fields):
        t = int(f.stem.split("_")[1]) * dt
        render_frame(f, d / f"frame_{k:03d}.png",
                     f"Couette, gamma = 10 1/s, t = {t:.2f} s", vmax=0.04)
    encode(d, HERE / "couette-gamma10.mp4", fps=10)
    # --- steady state across the sweep (each gamma held 1 s = 10 frames) ---
    import shutil
    gammas = [0.1, 0.3, 1, 3, 10, 30, 100]
    d2 = tmp / "sweep"; d2.mkdir(parents=True, exist_ok=True)
    for k, g in enumerate(gammas):
        run = HERE / "runs" / f"gamma_{g}"
        last = sorted(run.glob("field_*.csv"),
                      key=lambda p: int(p.stem.split("_")[1]))[-1]
        src = tmp / f"sweep-src-{k}.png"
        render_frame(last, src, f"Couette, steady state, gamma = {g} 1/s",
                     vmax=0.4)
        for j in range(10):
            shutil.copy(src, d2 / f"frame_{k*10+j:03d}.png")
    encode(d2, HERE / "couette-sweep.mp4", fps=10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
