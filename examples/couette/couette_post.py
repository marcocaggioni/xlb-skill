#!/usr/bin/env python3
"""Couette post-processing: torque.csv -> flow curve + Newtonian validation.

Input per sweep point: runs/gamma_<value>/torque.csv with columns
    time_s, torque_N_m_per_m
(torque per unit length; the 2D case. Multiply by L for 3D torque.)

Conversions (same as the M-Star Couette example):
    gamma_i = 2*Omega*Ro^2 / (Ro^2 - Ri^2)      Omega from gamma via case.yaml
    tau_i   = T / (2*pi*Ri^2*L),  T = torque_per_m * L
    eta_app = tau_i / gamma_i
    T_analytic = 4*pi*L*eta*Omega*Ri^2*Ro^2 / (Ro^2 - Ri^2)

Validation gate: for the Newtonian case the simulated torque must match
the analytic torque within 10% (default). Exits non-zero on failure —
do not proceed to a non-Newtonian curve until it passes.

Usage: python3 couette_post.py [--runs runs] [--validation-tol 0.10]
Writes flow_curve.csv and flow_curve.png in examples/couette/.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from lattice import load_case  # noqa: E402

HERE = Path(__file__).parent


def read_torque(path):
    times, torques = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            times.append(float(row["time_s"]))
            torques.append(float(row["torque_N_m_per_m"]))
    return times, torques


def steady_mean(times, torques, frac):
    n = len(times)
    k = max(1, int(n * frac))
    window = torques[-k:]
    mean = sum(window) / len(window)
    sd = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5
    return mean, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--validation-tol", type=float, default=0.10)
    args = ap.parse_args()

    case = load_case()
    g, f, s = case["geometry"], case["fluid"], case["sweep"]
    Ri, Ro, L = g["R_i"], g["R_o"], g["L"]
    eta = f["eta_Pa_s"]
    frac = s["steady_window_fraction"]
    rows = []
    for gamma in s["shear_rate_1_s"]:
        path = Path(args.runs) / f"gamma_{gamma}" / "torque.csv"
        if not path.exists():
            print(f"missing {path} — run the sweep first")
            return 1
        times, torques = read_torque(path)
        T_per_m, sd = steady_mean(times, torques, frac)
        T = T_per_m * L
        Omega = gamma * (Ro * Ro - Ri * Ri) / (2.0 * Ro * Ro)
        tau = T / (2.0 * math.pi * Ri * Ri * L)
        T_an = 4 * math.pi * L * eta * Omega * Ri * Ri * Ro * Ro / (Ro * Ro - Ri * Ri)
        err = abs(T - T_an) / T_an if T_an else float("nan")
        rows.append({"gamma": gamma, "omega": Omega, "tau": tau,
                     "eta_app": tau / gamma, "T": T, "T_analytic": T_an,
                     "rel_err": err, "torque_sd": sd * L})
        print(f"gamma={gamma:>6} 1/s  tau={tau:9.4f} Pa  eta_app={tau/gamma:.4f} "
              f"Pa.s  T={T:.6e} N.m  analytic err={err*100:.2f}%")

    max_err = max(r["rel_err"] for r in rows)
    passed = max_err <= args.validation_tol
    print(f"Newtonian validation: max analytic error {max_err*100:.2f}% "
          f"(tol {args.validation_tol*100:.0f}%) — {'PASSED' if passed else 'FAILED'}")

    with open(HERE / "flow_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gamma_1_s", "omega_rad_s", "tau_Pa",
                                           "eta_app_Pa_s", "torque_N_m",
                                           "torque_analytic_N_m", "rel_err"])
        w.writeheader()
        for r in rows:
            w.writerow({"gamma_1_s": r["gamma"], "omega_rad_s": r["omega"],
                        "tau_Pa": r["tau"], "eta_app_Pa_s": r["eta_app"],
                        "torque_N_m": r["T"], "torque_analytic_N_m": r["T_analytic"],
                        "rel_err": r["rel_err"]})

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog([r["gamma"] for r in rows], [r["tau"] for r in rows], "o-")
    ax[0].set_xlabel("shear rate (1/s)"); ax[0].set_ylabel("shear stress (Pa)")
    ax[0].set_title("Couette flow curve (XLB)")
    ax[1].loglog([r["gamma"] for r in rows], [r["eta_app"] for r in rows], "o-")
    ax[1].set_xlabel("shear rate (1/s)"); ax[1].set_ylabel("apparent viscosity (Pa.s)")
    ax[1].set_title("Apparent viscosity")
    fig.tight_layout()
    fig.savefig(HERE / "flow_curve.png", dpi=150)
    print("wrote flow_curve.csv and flow_curve.png")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
