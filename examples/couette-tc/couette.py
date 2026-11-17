#!/usr/bin/env python3
"""Couette case for XLB, three-component (TC) fluid (2D annulus, JAX backend, CPU).

Rotating inner cylinder (bob), fixed outer cylinder (cup), TC fluid
    tau = sigma_y + sigma_y*sqrt(gamma_dot/gamma_dot_c) + eta_bg*gamma_dot
with sigma_y = 70 Pa, gamma_dot_c = 100 1/s, eta_bg = 0.05 Pa s.
Torque on the bob -> flow curve, validated against the analytic TC
Couette torque (tc_analytic.py, shared reference implementation).
Same geometry/sweep as the Newtonian example in examples/couette/,
so the solvers compare point for point.

XLB API used (xlb 0.3.2, read from the package source, not invented):
same as examples/couette/couette.py (see its docstring), plus the TC
collision operator in tc_collision.py: XLB ships no rheology dropdown,
so the local relaxation frequency is computed inside a custom BGK
collision operator from the local strain rate, following the pattern
of XLB's own SmagorinskyLESBGK operator. The stepper is built with
collision_type="BGK" and its .collision is then replaced by the TC
operator before the first step (the stepper resolves self.collision
when its jitted implementation is first traced).

Rotating wall: angular sectors of HalfwayBounceBackBC, as in the
Newtonian case (32 sectors, one constant wall velocity per BC object
in the JAX backend). No per-step ramp: the run is initialised with
the ANALYTIC TC Couette velocity profile (yielded region shears,
unyielded plug attached to the fixed outer wall is stationary) —
computed from tc_analytic.py by integrating gamma_dot(r) inward
from R_o. The profile is also the cross-check: the final field is
binned against it in profile.csv.

Precision: FP64FP64 (case.yaml). At gamma = 0.1 the lattice wall
speed is u_lb ~ 4e-8 (dt is set by eta = 722 Pa s); FP32 cannot
carry the momentum-exchange torque — a small difference of two
large terms — at that signal level. XLB enables JAX x64 for FP64
policies automatically.

Torque sign: as in the Newtonian case, XLB's momentum-exchange
force is the force on the solid, so its z-torque on the bob is
negative; the reported torque is the negation (driving torque).

Run length: steps_per_point from case.yaml (3,000 steps), an XLB-only
override of the shared 1 s run time — XLB 0.3.2's halfway bounce-back
leaks mass at a curved moving wall (see examples/couette/RUN_REPORT.md).
Density drift is measured per point and written to summary.json.

Usage (from the repo root):
    uv run python examples/couette-tc/couette.py --gamma 10 --out examples/couette-tc/runs/gamma_10
Writes torque.csv, profile.csv, lattice.json and summary.json in --out,
plus field_<step>.csv snapshots.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.helper.initializers import initialize_eq

sys.path.insert(0, str(Path(__file__).parent))
from lattice import lattice_params_for_gamma, load_case, eta_max  # noqa: E402
from tc_analytic import gamma_of_tau, torque_of_omega  # noqa: E402
from tc_collision import TCCollision  # noqa: E402


def tc_profile(r_eval, Omega, case):
    """Analytic TC azimuthal velocity u_theta(r) for angular velocity Omega.

    gamma_dot(r) from inverting the TC law at tau(r) = T/(2*pi*L*r^2)
    (0 in the plug, where tau < sigma_y); u_theta(r) =
    r * integral_r^Ro gamma_dot(r')/r' dr'. Computed on a fine radial
    mesh and interpolated to r_eval.
    """
    g = case["geometry"]
    Ri, Ro, L = g["R_i"], g["R_o"], g["L"]
    T = torque_of_omega(Omega)
    n = 20000
    rr = np.linspace(Ri, Ro, n)
    tau = T / (2.0 * math.pi * L * rr ** 2)
    gam = np.array([gamma_of_tau(t) for t in tau])
    # integral_r^Ro gam/r' dr' by cumulative trapezoid from Ro inward
    integ = np.zeros(n)
    integ[-1] = 0.0
    # trapezoid segments, accumulated from the outer wall inward
    seg = 0.5 * (gam[1:] / rr[1:] + gam[:-1] / rr[:-1]) * np.diff(rr)
    integ[:-1] = np.cumsum(seg[::-1])[::-1]
    uth = rr * integ
    return np.interp(r_eval, rr, uth)


def build_case(case, lat, gamma):
    """Build grid, BCs, stepper and the analytic-TC-profile initial condition."""
    g, f = case["geometry"], case["fluid"]
    Ri, Ro = g["R_i"], g["R_o"]
    nx = ny = int(round(2 * Ro / lat["dx_m"])) + 2
    cx = cy = (nx - 1) / 2.0
    Ri_lb = Ri / lat["dx_m"]
    Ro_lb = Ro / lat["dx_m"]
    dx, dt = lat["dx_m"], lat["dt_s"]
    omega = lat["omega_ref"]

    Omega_phys = gamma * (Ro * Ro - Ri * Ri) / (2.0 * Ro * Ro)
    Omega_lb = Omega_phys * dt

    backend = ComputeBackend.JAX
    pp = PrecisionPolicy.FP64FP64
    vs = xlb.velocity_set.D2Q9(precision_policy=pp, compute_backend=backend)
    xlb.init(velocity_set=vs, default_backend=backend, default_precision_policy=pp)
    grid = grid_factory((nx, ny), compute_backend=backend)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    r = np.hypot(ii - cx, jj - cy)
    theta = np.arctan2(jj - cy, ii - cx)
    inner_solid = r <= Ri_lb
    outer_solid = r >= Ro_lb
    fluid = (~inner_solid) & (~outer_solid)

    bcs = []
    ox, oy = np.where(outer_solid)
    bcs.append(HalfwayBounceBackBC(indices=[ox.tolist(), oy.tolist()],
                                   prescribed_value=(0.0, 0.0)))
    n_sectors = case["xlb"]["wall_sectors"]
    sector_bcs = []
    for s in range(n_sectors):
        lo = -math.pi + s * 2 * math.pi / n_sectors
        hi = -math.pi + (s + 1) * 2 * math.pi / n_sectors
        m = inner_solid & (theta >= lo) & (theta < hi)
        if m.sum() == 0:
            continue
        th_c = 0.5 * (lo + hi)
        uw = Omega_lb * Ri_lb * np.array([-math.sin(th_c), math.cos(th_c)])
        sx, sy = np.where(m)
        bc = HalfwayBounceBackBC(indices=[sx.tolist(), sy.tolist()],
                                 prescribed_value=(float(uw[0]), float(uw[1])))
        sector_bcs.append(bc)
        bcs.append(bc)

    stepper = IncompressibleNavierStokesStepper(
        grid=grid, boundary_conditions=bcs, collision_type="BGK")
    # TC rheology: replace the BGK collision with the local-omega TC
    # operator (resolved when the jitted stepper is first traced).
    stepper.collision = TCCollision(
        velocity_set=vs, precision_policy=pp, compute_backend=backend,
        dt=dt, dx=dx, rho_phys=f["rho_kg_m3"],
        sigma_y=f["sigma_y_Pa"], gamma_c=f["gamma_dot_c_1_s"],
        eta_bg=f["eta_bg_Pa_s"], eta_max=eta_max(f))

    # --- analytic TC Couette profile as initial condition ---
    r_phys = r * dx
    vth = np.where(fluid, tc_profile(r_phys, Omega_phys, case), 0.0)
    ux = np.where(fluid, -vth * (jj - cy) / np.where(r == 0, 1, r), 0.0)
    uy = np.where(fluid, vth * (ii - cx) / np.where(r == 0, 1, r), 0.0)
    u_lb = np.stack([ux * dt / dx, uy * dt / dx]).astype(np.float64)

    def initializer(bc_mask, f_0):
        rho = grid.create_field(cardinality=1, fill_value=1.0,
                                dtype=pp.compute_precision)
        u = jnp.array(u_lb)
        return initialize_eq(f_0, grid, vs, pp, backend, rho=rho, u=u)

    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields(initializer=initializer)

    mt = [MomentumTransfer(bc, compute_backend=backend) for bc in sector_bcs]
    return dict(grid=grid, vs=vs, pp=pp, backend=backend, stepper=stepper,
                f_0=f_0, f_1=f_1, bc_mask=bc_mask, missing_mask=missing_mask,
                sector_bcs=sector_bcs, mt=mt, omega=omega, cx=cx, cy=cy,
                fluid=fluid, r=r, theta=theta, nx=nx, ny=ny,
                Omega_phys=Omega_phys)


def sample_torque(state):
    """Per-node momentum-exchange torque (lattice units, z-component).

    Same replay of MomentumTransfer's JAX formula as the Newtonian
    case (see examples/couette/couette.py): phi = f_post_collision[opp]
    + f_post_stream at the missing directions of each sector BC's edge
    nodes, force = c[:, opp] . phi, torque = sum r_node x F_node.
    """
    vs = state["vs"]
    C = np.array(vs.c)                       # (2, 9)
    OPP = np.array(vs.opp_indices)           # (9,)
    bc_mask = np.array(state["bc_mask"])
    mm = np.array(state["missing_mask"])
    nx, ny = state["nx"], state["ny"]
    cx, cy = state["cx"], state["cy"]
    f_0, f_1 = state["f_0"], state["f_1"]
    total = 0.0
    for bc, mt in zip(state["sector_bcs"], state["mt"]):
        f_pc, f_ps = mt.fetcher(f_0, f_1, state["bc_mask"], state["missing_mask"])
        f_pc, f_ps = np.array(f_pc), np.array(f_ps)
        is_edge = (bc_mask[0] == bc.id) & (~mm[0])
        if not is_edge.any():
            continue
        phi = f_pc[OPP] + f_ps
        mask = mm & np.broadcast_to(is_edge, (9, nx, ny))
        phi = np.where(mask, phi, 0.0)
        force = np.tensordot(C[:, OPP], phi, axes=(-1, 0))  # (2, nx, ny)
        xs, ys = np.where(is_edge)
        total += np.sum((xs - cx) * force[1][is_edge]
                        - (ys - cy) * force[0][is_edge])
    return total


def rho_mean(state, macro):
    rho, _ = macro(state["f_0"])
    rho = np.array(rho)
    if rho.ndim == 3:
        rho = rho[0]
    return float(rho[state["fluid"]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override n_steps from case.yaml")
    args = ap.parse_args()
    gamma = args.gamma

    case = load_case()
    lat = lattice_params_for_gamma(case, gamma)
    if not lat["mach_ok"]:
        print("FAIL: lattice check failed (u_lb too high)")
        return 1
    n_steps = args.steps or case["xlb"].get("steps_per_point", lat["n_steps_1s"])
    dt = lat["dt_s"]
    sample_every = case["xlb"]["sample_every_steps"]
    snapshot_every = case["xlb"].get("snapshot_every_steps", 1000)
    L = case["geometry"]["L"]
    force_conv = lat["force_conversion_N_per_lb_force"]

    t0 = time.time()
    state = build_case(case, lat, gamma)
    print(f"gamma={gamma} 1/s  Omega={state['Omega_phys']:.4f} rad/s  "
          f"dt={dt:.3e} s  steps={n_steps:,}  build {time.time()-t0:.1f}s")

    macro = Macroscopic(compute_backend=state["backend"],
                        precision_policy=state["pp"],
                        velocity_set=xlb.velocity_set.D2Q9(
                            precision_policy=state["pp"],
                            compute_backend=state["backend"]))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "lattice.json").write_text(json.dumps(lat, indent=2) + "\n")
    rho_start = rho_mean(state, macro)
    print(f"rho_start (fluid mean) = {rho_start:.6f}")

    def snapshot(step):
        rho, u = macro(state["f_0"])
        u = np.array(u) * lat["dx_m"] / lat["dt_s"]  # (2, nx, ny) m/s
        with open(out / f"field_{step}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["iX", "iY", "ux_m_s", "uy_m_s"])
            for i in range(0, state["nx"], 2):
                for j in range(0, state["ny"], 2):
                    w.writerow([i, j, f"{u[0, i, j]:.6e}", f"{u[1, i, j]:.6e}"])

    torque_rows = []
    snapshot(0)
    for step in range(n_steps):
        state["f_0"], state["f_1"] = state["stepper"](
            state["f_0"], state["f_1"], state["bc_mask"],
            state["missing_mask"], state["omega"], step)
        state["f_0"], state["f_1"] = state["f_1"], state["f_0"]
        if (step + 1) % sample_every == 0:
            tq_lb = sample_torque(state)
            tq_per_m = -tq_lb * force_conv
            torque_rows.append((step + 1, (step + 1) * dt, tq_lb, tq_per_m))
        if (step + 1) % snapshot_every == 0:
            snapshot(step + 1)

    with open(out / "torque.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "time_s", "torque_lb", "torque_N_m_per_m"])
        for s, t, tq_lb, tq in torque_rows:
            w.writerow([s, f"{t:.6e}", f"{tq_lb:.6e}", f"{tq:.6e}"])

    # --- steady torque (last 20%) vs analytic TC torque ---
    n = len(torque_rows)
    k = max(1, int(n * case["sweep"]["steady_window_fraction"]))
    steady = [r[3] for r in torque_rows[-k:]]
    T_per_m = sum(steady) / len(steady)
    sd = (sum((x - T_per_m) ** 2 for x in steady) / len(steady)) ** 0.5
    T = T_per_m * L
    Ri, Ro = case["geometry"]["R_i"], case["geometry"]["R_o"]
    T_an = torque_of_omega(state["Omega_phys"])
    err = abs(T - T_an) / T_an
    # first-half vs second-half of the steady window, as a drift check
    half = max(1, len(steady) // 2)
    first_mean = sum(steady[:half]) / half
    second_mean = sum(steady[half:]) / (len(steady) - half)
    print(f"steady torque: {T:.6e} N.m  (sd {sd*L:.2e})  analytic {T_an:.6e}  "
          f"err {err*100:.2f}%  window drift {(second_mean-first_mean)/T_per_m*100:+.2f}%  "
          f"runtime {time.time()-t0:.1f}s")

    # --- density drift (mass leak) ---
    rho_end = rho_mean(state, macro)
    drift = (rho_end - rho_start) / rho_start * 100.0
    print(f"density drift: rho {rho_start:.6f} -> {rho_end:.6f}  ({drift:+.3f}% over {n_steps} steps)")

    # --- profile cross-check: binned u_theta vs analytic TC profile ---
    rho, u = macro(state["f_0"])
    u = np.array(u) * lat["dx_m"] / lat["dt_s"]
    ii, jj = np.meshgrid(np.arange(state["nx"]), np.arange(state["ny"]), indexing="ij")
    r_phys = state["r"] * lat["dx_m"]
    uth = np.where(state["fluid"],
                   (-u[0] * (jj - state["cy"]) + u[1] * (ii - state["cx"]))
                   / np.where(state["r"] == 0, 1, state["r"]), 0.0)
    fr, fu = r_phys[state["fluid"]], uth[state["fluid"]]
    with open(out / "profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r_m", "u_theta_m_s", "u_theta_analytic_m_s", "count"])
        edges = np.linspace(Ri, Ro, 21)
        prof_errs = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (fr >= a) & (fr < b)
            if m.sum() == 0:
                continue
            rc = 0.5 * (a + b)
            ua = float(tc_profile(np.array([rc]), state["Omega_phys"], case)[0])
            w.writerow([f"{rc:.6e}", f"{fu[m].mean():.6e}", f"{ua:.6e}", int(m.sum())])
            if abs(ua) > 0.1 * float(np.max(np.abs(
                    tc_profile(np.linspace(Ri, Ro, 50), state["Omega_phys"], case)))):
                prof_errs.append(abs(fu[m].mean() - ua) / abs(ua))
    prof_err = max(prof_errs) if prof_errs else float("nan")
    print(f"profile cross-check: max rel err in yielded region {prof_err*100:.2f}%")

    summary = {
        "gamma_1_s": gamma, "omega_rad_s": state["Omega_phys"],
        "torque_N_m": T, "torque_sd_N_m": sd * L,
        "torque_analytic_N_m": T_an, "rel_err": err,
        "steady_window_drift_pct": (second_mean - first_mean) / T_per_m * 100.0,
        "rho_start": rho_start, "rho_end": rho_end,
        "density_drift_pct": drift,
        "profile_max_rel_err_yielded": prof_err,
        "n_steps": n_steps, "dt_s": dt,
        "runtime_s": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", out / "torque.csv,", out / "profile.csv,", out / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
