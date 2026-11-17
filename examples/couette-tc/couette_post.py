#!/usr/bin/env python3
"""Couette post-processing for the TC case: torque.csv -> flow curve + validation.

Input per sweep point: runs/gamma_<value>/torque.csv with columns
    time_s, torque_N_m_per_m
(torque per unit length; the 2D case. Multiply by L for 3D torque.)
plus runs/gamma_<value>/summary.json (density drift, profile check).

Validation is on TORQUE against the analytic TC torque
(tc_analytic.torque_of_omega), NOT on eta_app: the label gamma is the
Newtonian inner-wall-rate convention, and the TC fluid's true inner-wall
rate exceeds the label rate (gamma_i = 1.10 at label 0.1, 162 at label
100), so eta_app = tau_i/gamma_label is a reporting quantity, not the
model viscosity at the label rate. At gamma <= 10 the analytic solution
has a genuine unyielded plug attached to the outer wall (yield radius
R_y < R_o); the flow curve must still validate on torque.

Validation gate: max torque error <= 10% (default). Exits non-zero on
failure — do not proceed to a flow-curve claim until it passes.

Usage: python3 couette_post.py [--runs runs] [--validation-tol 0.10]
Writes flow_curve.csv and flow_curve.png in examples/couette-tc/.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from lattice import load_case  # noqa: E402
from tc_analytic import torque_of_omega, tau_inner  # noqa: E402

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
    g, s = case["geometry"], case["sweep"]
    Ri, Ro, L = g["R_i"], g["R_o"], g["L"]
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
        T_an = torque_of_omega(Omega)
        err = abs(T - T_an) / T_an if T_an else float("nan")
        r_y = math.sqrt(T_an / (2.0 * math.pi * L * case["fluid"]["sigma_y_Pa"]))
        drift = float("nan")
        summ = Path(args.runs) / f"gamma_{gamma}" / "summary.json"
        if summ.exists():
            drift = json.loads(summ.read_text())["density_drift_pct"]
        rows.append({"gamma": gamma, "omega": Omega, "tau": tau,
                     "eta_app": tau / gamma, "T": T, "T_analytic": T_an,
                     "rel_err": err, "torque_sd": sd * L,
                     "drift": drift, "r_y_mm": min(r_y, Ro) * 1000.0})
        print(f"gamma={gamma:>6} 1/s  tau={tau:9.4f} Pa  eta_app={tau/gamma:.4f} "
              f"Pa.s  T={T:.6e} N.m  analytic err={err*100:.2f}%  "
              f"R_y={min(r_y, Ro)*1000:.2f} mm  drift={drift:+.2f}%")

    max_err = max(r["rel_err"] for r in rows)
    passed = max_err <= args.validation_tol
    print(f"TC validation: max analytic-torque error {max_err*100:.2f}% "
          f"(tol {args.validation_tol*100:.0f}%) — {'PASSED' if passed else 'FAILED'}")

    with open(HERE / "flow_curve.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["gamma_1_s", "omega_rad_s", "tau_Pa",
                                           "eta_app_Pa_s", "torque_N_m",
                                           "torque_analytic_N_m", "rel_err",
                                           "density_drift_pct", "yield_radius_mm"])
        w.writeheader()
        for r in rows:
            w.writerow({"gamma_1_s": r["gamma"], "omega_rad_s": r["omega"],
                        "tau_Pa": r["tau"], "eta_app_Pa_s": r["eta_app"],
                        "torque_N_m": r["T"], "torque_analytic_N_m": r["T_analytic"],
                        "rel_err": r["rel_err"],
                        "density_drift_pct": r["drift"],
                        "yield_radius_mm": r["r_y_mm"]})

    gam = [r["gamma"] for r in rows]
    tau_an = [tau_inner(r["T_analytic"]) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(gam, [r["tau"] for r in rows], "o-", label="XLB")
    ax[0].loglog(gam, tau_an, "k--", label="analytic TC")
    ax[0].set_xlabel("shear rate (label, 1/s)")
    ax[0].set_ylabel("shear stress at inner wall (Pa)")
    ax[0].set_title("Couette flow curve, TC fluid (XLB)")
    ax[0].legend()
    ax[1].loglog(gam, [r["eta_app"] for r in rows], "o-")
    ax[1].set_xlabel("shear rate (label, 1/s)")
    ax[1].set_ylabel("apparent viscosity (Pa.s)")
    ax[1].set_title("Apparent viscosity (label rate)")
    fig.tight_layout()
    fig.savefig(HERE / "flow_curve.png", dpi=150)
    print("wrote flow_curve.csv and flow_curve.png")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
