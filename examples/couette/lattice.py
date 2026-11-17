#!/usr/bin/env python3
"""Physical -> lattice parameters for the XLB Couette case.

Reads examples/couette/case.yaml, solves dt from the chosen relaxation
frequency omega, and checks the low-Mach condition at the highest shear
rate. Writes lattice.json next to case.yaml.

Formulas (D2Q9/D3Q19, cs^2 = 1/3):
    dx     = gap / N_gap
    nu_lb  = nu * dt / dx^2
    omega  = 1 / (3*nu_lb + 0.5)      ->  nu_lb = (1/omega - 0.5)/3
    u_lb   = u_phys * dt / dx         ->  must stay < max_lattice_velocity
    torque conversion: lattice force -> N uses rho * dx^4 / dt^2
"""
import json
import math
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # keep the file usable without PyYAML for a quick check
    yaml = None

CASE = Path(__file__).with_name("case.yaml")


def load_case():
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    return yaml.safe_load(CASE.read_text())


def lattice_params(case):
    g = case["geometry"]
    f = case["fluid"]
    s = case["sweep"]
    lat = case["lattice"]
    Ri, Ro = g["R_i"], g["R_o"]
    gap = Ro - Ri
    nu = f["eta_Pa_s"] / f["rho_kg_m3"]
    N = lat["cells_across_gap"]
    dx = gap / N
    omega = lat["omega"]
    if not (0.5 < omega < 2.0):
        raise ValueError(f"omega={omega} outside (0.5, 2.0); unstable or overdamped")
    nu_lb = (1.0 / omega - 0.5) / 3.0
    dt = nu_lb * dx * dx / nu
    # Highest shear rate -> highest wall speed.
    gamma_max = max(s["shear_rate_1_s"])
    Omega_max = gamma_max * (Ro * Ro - Ri * Ri) / (2.0 * Ro * Ro)
    u_wall_max = Omega_max * Ri
    u_lb_max = u_wall_max * dt / dx
    n_steps = int(math.ceil(s["run_time_s"] / dt))
    return {
        "dx_m": dx,
        "dt_s": dt,
        "nu_lb": nu_lb,
        "omega": omega,
        "u_wall_max_m_s": u_wall_max,
        "u_lb_max": u_lb_max,
        "max_lattice_velocity": lat["max_lattice_velocity"],
        "mach_ok": u_lb_max < lat["max_lattice_velocity"],
        "n_steps": n_steps,
        "cells_across_gap": N,
        "gap_m": gap,
        "force_conversion_N_per_lb_force": f["rho_kg_m3"] * dx**4 / dt**2,
    }


def main():
    case = load_case()
    p = lattice_params(case)
    out = CASE.with_name("lattice.json")
    out.write_text(json.dumps(p, indent=2) + "\n")
    print(f"dx = {p['dx_m']*1e3:.3f} mm   dt = {p['dt_s']:.3e} s   "
          f"omega = {p['omega']:.3f}   nu_lb = {p['nu_lb']:.4f}")
    print(f"u_wall,max = {p['u_wall_max_m_s']:.3f} m/s   "
          f"u_lb,max = {p['u_lb_max']:.4f}   steps = {p['n_steps']:,}")
    if not p["mach_ok"]:
        print(f"FAIL: u_lb,max {p['u_lb_max']:.4f} >= "
              f"{p['max_lattice_velocity']} — reduce dt or coarsen dx")
        return 1
    print("lattice check PASSED — wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
