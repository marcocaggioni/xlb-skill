#!/usr/bin/env python3
"""Sweep the XLB TC Couette case over the shear rates in case.yaml.

For each gamma: build the per-point lattice parameters, run the XLB
case (couette.py, via `uv run python` from the repo root), and collect
runs/gamma_<value>/torque.csv + summary.json. Default is a dry run:
print the per-point lattice table and the plan, run nothing.

Usage (from the repo root):
    uv run python examples/couette-tc/couette_sweep.py            # dry run
    uv run python examples/couette-tc/couette_sweep.py --run      # actually run
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lattice import lattice_params_for_gamma, load_case  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually run the solver")
    args = ap.parse_args()

    case = load_case()
    gammas = case["sweep"]["shear_rate_1_s"]
    n_steps = case["xlb"].get("steps_per_point", 3000)
    for gamma in gammas:
        lat = lattice_params_for_gamma(case, gamma)
        if not lat["mach_ok"]:
            print(f"FAIL: lattice check failed for gamma={gamma} (u_lb too high)")
            return 1
        print(f"gamma={gamma:>5}  dt={lat['dt_s']:.3e} s  u_lb={lat['u_lb']:.2e}  "
              f"eta_label={lat['eta_label_Pa_s']:.3f}  omega_inner={lat['omega_inner']:.3f}  "
              f"omega_plug={lat['omega_plug']:.5f}  steps/point={n_steps:,}")
    if not args.run:
        print(f"DRY RUN: would run {len(gammas)} points: {gammas} 1/s")
        print("Pass --run to execute. Case script:", HERE / "couette.py")
        return 0

    for gamma in gammas:
        out = HERE / "runs" / f"gamma_{gamma}"
        # sys.executable is the venv python (uv run resolves to it too);
        # "uv" itself is not on PATH for subprocesses in this VM.
        cmd = [sys.executable, str(HERE / "couette.py"),
               "--gamma", str(gamma), "--out", str(out)]
        print("running:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"run failed for gamma={gamma}")
            return r.returncode
    print("sweep done — post-process with: uv run python examples/couette-tc/couette_post.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
