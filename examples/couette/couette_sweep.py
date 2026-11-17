#!/usr/bin/env python3
"""Sweep the XLB Couette case over the shear rates in case.yaml.

For each gamma: build the lattice parameters, run the XLB case
(couette.py, via `uv run python` from the repo root), and collect
runs/gamma_<value>/torque.csv. Default is a dry run: print the plan
and the lattice parameters, run nothing.

Usage (from the repo root):
    uv run python examples/couette/couette_sweep.py            # dry run
    uv run python examples/couette/couette_sweep.py --run      # actually run
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lattice import lattice_params, load_case  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually run the solver")
    args = ap.parse_args()

    case = load_case()
    lat = lattice_params(case)
    if not lat["mach_ok"]:
        print("FAIL: lattice check failed (u_lb too high) — fix case.yaml first")
        return 1
    n_steps = case["xlb"].get("steps_per_point", lat["n_steps"])
    print(f"lattice: dx={lat['dx_m']*1e3:.3f} mm dt={lat['dt_s']:.3e} s "
          f"steps/point={n_steps:,} (XLB override, see case.yaml) "
          f"u_lb,max={lat['u_lb_max']:.4f}")

    gammas = case["sweep"]["shear_rate_1_s"]
    if not args.run:
        print(f"DRY RUN: would run {len(gammas)} points: {gammas} 1/s")
        print("Pass --run to execute. Case script:", HERE / "couette.py")
        return 0

    for gamma in gammas:
        out = HERE / "runs" / f"gamma_{gamma}"
        cmd = ["uv", "run", "python", str(HERE / "couette.py"),
               "--gamma", str(gamma), "--out", str(out)]
        print("running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"run failed for gamma={gamma}")
            return r.returncode
    print("sweep done — post-process with: uv run python examples/couette/couette_post.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
