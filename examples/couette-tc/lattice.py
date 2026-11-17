#!/usr/bin/env python3
"""Physical -> lattice parameters for the XLB TC Couette case.

Reads examples/couette-tc/case.yaml. Unlike the Newtonian case, dt is
solved PER SWEEP POINT from the viscosity at that point's label rate:

    eta_label = sigma_y/gamma + sigma_y/sqrt(gamma*gamma_c) + eta_bg
    nu_label  = eta_label / rho
    dt        = nu_lb_ref * dx^2 / nu_label,  nu_lb_ref = (1/omega_ref-0.5)/3

with omega_ref = 1.0 (nu_lb_ref = 1/6). The local omega then follows the
local viscosity: omega = 1/(3*nu_lb + 0.5), nu_lb = eta_local*dt/(rho*dx^2).
At the inner wall omega is 1.17-1.82 across the sweep (inside
[0.8, 1.9]); in the unyielded plug eta is the cap eta_max = 7.02214e4
Pa s, so omega there falls to 0.02 (gamma = 0.1) down to 0.00003
(gamma = 100) — the plug being near-solid, documented in case.yaml
and RUN_REPORT.md, not a stability problem (plug near-stationary,
3,000-step runs). The Mach check is per point: u_lb = Omega*R_i*dt/dx.

Per-point table (dx = 0.5 mm, omega_ref = 1.0; computed by this file,
`python3 lattice.py`):

    gamma  dt (s)     u_lb      eta_label  eta_inner  omega_inner  omega_plug
    0.1    5.770e-08  4.15e-08  722.19     70.42      1.822        0.0204
    0.3    1.693e-07  3.66e-07  246.16     34.70      1.753        0.0070
    1      5.408e-07  3.89e-06  77.05      16.16      1.653        0.0022
    3      1.519e-06  3.28e-05  27.43      8.17       1.541        0.0008
    10     4.498e-06  3.24e-04  9.26       3.95       1.402        0.0003
    30     1.138e-05  2.46e-03  3.66       2.08       1.276        0.0001
    100    2.874e-05  2.07e-02  1.45       1.03       1.169        0.00003

(eta_inner / omega_inner use the analytic TC torque: tau_i from
tc_analytic.torque_of_omega, gamma_i = gamma_of_tau(tau_i),
eta_inner = tau_i/gamma_i. Note the task brief's "70.75 at 100" for
eta at the label rate is a slip: eta(100) = 0.7 + 0.7 + 0.05 = 1.45
Pa s from the TC law as written.)

Writes lattice.json next to case.yaml (params at the highest gamma,
for the record) and lattice_<gamma>.json per point when run with
--all. couette.py writes the per-point lattice.json into each run
directory.
"""
import json
import math
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # keep the file usable without PyYAML for a quick check
    yaml = None

sys.path.insert(0, str(Path(__file__).parent))
from tc_analytic import gamma_of_tau, torque_of_omega, tau_inner  # noqa: E402

CASE = Path(__file__).with_name("case.yaml")


def load_case():
    if yaml is None:
        raise SystemExit("PyYAML required: pip install pyyaml")
    return yaml.safe_load(CASE.read_text())


def tc_eta(gamma, f):
    """Uncapped TC viscosity (Pa s) at shear rate gamma (1/s)."""
    return (f["sigma_y_Pa"] / gamma
            + f["sigma_y_Pa"] / math.sqrt(gamma * f["gamma_dot_c_1_s"])
            + f["eta_bg_Pa_s"])


def eta_max(f):
    """Viscosity cap from gamma_reg (computed, not hard-coded)."""
    g_reg = f["gamma_reg_1_s"]
    return tc_eta(g_reg, f)


def lattice_params_for_gamma(case, gamma):
    g = case["geometry"]
    f = case["fluid"]
    lat = case["lattice"]
    Ri, Ro = g["R_i"], g["R_o"]
    gap = Ro - Ri
    dx = gap / lat["cells_across_gap"]
    omega_ref = lat["omega_ref"]
    if not (0.5 < omega_ref < 2.0):
        raise ValueError(f"omega_ref={omega_ref} outside (0.5, 2.0)")
    nu_lb_ref = (1.0 / omega_ref - 0.5) / 3.0
    eta_label = tc_eta(gamma, f)
    nu_label = eta_label / f["rho_kg_m3"]
    dt = nu_lb_ref * dx * dx / nu_label
    Omega = gamma * (Ro * Ro - Ri * Ri) / (2.0 * Ro * Ro)
    u_wall = Omega * Ri
    u_lb = u_wall * dt / dx
    # Inner-wall omega from the analytic TC torque.
    T_an = torque_of_omega(Omega)
    tau_i = tau_inner(T_an)
    gamma_i = gamma_of_tau(tau_i)
    eta_inner = tau_i / gamma_i if gamma_i > 0 else float("inf")

    def omega_of_eta(eta):
        nu_lb = eta / f["rho_kg_m3"] * dt / dx ** 2
        return 1.0 / (3.0 * nu_lb + 0.5)

    n_steps = int(math.ceil(case["sweep"]["run_time_s"] / dt))
    return {
        "gamma_1_s": gamma,
        "dx_m": dx,
        "dt_s": dt,
        "nu_lb_ref": nu_lb_ref,
        "omega_ref": omega_ref,
        "eta_label_Pa_s": eta_label,
        "eta_inner_Pa_s": eta_inner,
        "gamma_inner_1_s": gamma_i,
        "tau_inner_Pa": tau_i,
        "torque_analytic_N_m": T_an,
        "omega_inner": omega_of_eta(eta_inner),
        "omega_plug": omega_of_eta(eta_max(f)),
        "eta_max_Pa_s": eta_max(f),
        "u_wall_m_s": u_wall,
        "u_lb": u_lb,
        "max_lattice_velocity": lat["max_lattice_velocity"],
        "mach_ok": u_lb < lat["max_lattice_velocity"],
        "n_steps_1s": n_steps,
        "cells_across_gap": lat["cells_across_gap"],
        "gap_m": gap,
        "force_conversion_N_per_lb_force": f["rho_kg_m3"] * dx ** 4 / dt ** 2,
    }


def lattice_params(case):
    """Params at the highest sweep gamma (record / Mach check)."""
    return lattice_params_for_gamma(case, max(case["sweep"]["shear_rate_1_s"]))


def main():
    case = load_case()
    rows = []
    for gamma in case["sweep"]["shear_rate_1_s"]:
        p = lattice_params_for_gamma(case, gamma)
        rows.append(p)
        print(f"gamma={gamma:>5}  dt={p['dt_s']:.3e} s  u_lb={p['u_lb']:.2e}  "
              f"eta_label={p['eta_label_Pa_s']:.3f}  eta_inner={p['eta_inner_Pa_s']:.3f}  "
              f"omega_inner={p['omega_inner']:.3f}  omega_plug={p['omega_plug']:.5f}  "
              f"{'OK' if p['mach_ok'] else 'FAIL mach'}")
    out = CASE.with_name("lattice.json")
    out.write_text(json.dumps(lattice_params(case), indent=2) + "\n")
    for p in rows:
        CASE.with_name(f"lattice_{p['gamma_1_s']}.json").write_text(
            json.dumps(p, indent=2) + "\n")
    if not all(p["mach_ok"] for p in rows):
        print("FAIL: u_lb too high at some point — reduce dt or coarsen dx")
        return 1
    print("lattice check PASSED — wrote", out, "and per-point lattice_<gamma>.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
