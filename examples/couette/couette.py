#!/usr/bin/env python3
"""Couette case for XLB (2D annulus, JAX backend, CPU).

Rotating inner cylinder (bob), fixed outer cylinder (cup), Newtonian
fluid. Torque on the bob -> flow curve. Same geometry/fluid/sweep as
the M-Star and Palabos Couette examples, so the solvers compare
point for point.

XLB API used (xlb 0.3.2, read from the package source, not invented):
  - xlb.init(velocity_set, default_backend, default_precision_policy)
  - grid_factory(grid_shape, compute_backend)  -> Grid
  - IncompressibleNavierStokesStepper(grid, boundary_conditions,
      collision_type="BGK"); stepper.prepare_fields(initializer=...)
      returns (f_0, f_1, bc_mask, missing_mask); each step is
      f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, step)
      followed by a buffer swap (examples/cfd/lid_driven_cavity_2d.py).
  - HalfwayBounceBackBC(indices=[x_array, y_array], prescribed_value=(ux, uy))
      — moving-wall bounce-back with a constant wall velocity per BC
      object (xlb/operator/boundary_condition/bc_halfway_bounce_back.py:
      the JAX implementation calls profile() with no arguments, so one
      BC object carries exactly one constant velocity).
  - Macroscopic(compute_backend, precision_policy, velocity_set) -> (rho, u).
  - MomentumTransfer(bc, compute_backend) -> net force on that BC's
      geometry; its .fetcher (FetchPopulations) reconstructs
      (f_post_collision, f_post_stream). The per-node torque here
      replays MomentumTransfer's own JAX formula
      (phi = f_post_collision[opp] + f_post_stream at missing
      directions; force = c[:, opp] . phi) node by node, because the
      per-sector net force cannot give a torque: the pressure (radial)
      part of the force is ~100x the shear part and only cancels when
      the lever arm sits at the force's own angle.
  - initialize_eq(f, grid, velocity_set, precision_policy,
      compute_backend, rho=..., u=...) for the analytic-profile IC
      (xlb/helper/initializers.py).

Rotating wall: the inner cylinder is split into `wall_sectors`
angular sectors (case.yaml, 32), each its own HalfwayBounceBackBC
with the wall velocity at the sector centre. Directional error per
sector <= half the sector angle (~0.5% at 32 sectors). BC ids are
uint8 (1..253), so a per-node BC (~312 wall nodes) is impossible
(xlb/cell_type.py, boundary_condition_registry.py).

No per-step ramp: wall velocities are baked into the jitted stepper,
so changing them mid-run would recompile every step. Instead the
run is initialised with the analytic Couette profile (same purpose:
no start-up shock). The profile is also the cross-check: the final
field is fitted to u_theta = A*r + B/r and the wall shear stress
gives a second, independent torque estimate.

Torque sign: XLB's momentum-exchange force is the force exerted on
the solid, so its z-torque on the bob is negative (it opposes the
rotation). The reported torque is the negation — the driving torque
the rheometer applies — the same convention as the Palabos example.

Run length: steps_per_point from case.yaml (3,000 steps = 0.125 s),
an XLB-only override of the shared 1 s run time. XLB 0.3.2's
halfway bounce-back leaks mass at a curved moving wall (~0.5% of
the fluid density per 1,000 steps at gamma = 10 1/s, ~5% at
gamma = 100, linear in wall speed; identical at 128 sectors, so it
is the staircase wall, not the sectors). The velocity inflates as
1/rho while the momentum-exchange torque stays analytic, so a long
run corrupts the velocity field but not the torque. 3,000 steps is
long past steady state and bounds the leak; see RUN_REPORT.md.

Usage (from the repo root):
    uv run python examples/couette/couette.py --gamma 10 --out examples/couette/runs/gamma_10
Writes torque.csv, profile.csv and field_<step>.csv snapshots in --out.
"""
import argparse
import csv
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
from lattice import lattice_params, load_case  # noqa: E402


def build_case(case, lat, gamma):
    """Build grid, BCs, stepper and the analytic-profile initial condition."""
    g, f = case["geometry"], case["fluid"]
    Ri, Ro = g["R_i"], g["R_o"]
    # Square grid just big enough to contain the outer cylinder (2*Ro/dx
    # cells across) plus one cell of margin on each side.
    nx = ny = int(round(2 * Ro / lat["dx_m"])) + 2
    cx = cy = (nx - 1) / 2.0
    Ri_lb = Ri / lat["dx_m"]
    Ro_lb = Ro / lat["dx_m"]
    dx, dt = lat["dx_m"], lat["dt_s"]
    omega = lat["omega"]

    # --- lattice angular velocity and wall speed ---
    Omega_phys = gamma * (Ro * Ro - Ri * Ri) / (2.0 * Ro * Ro)
    Omega_lb = Omega_phys * dt

    backend = ComputeBackend.JAX
    pp = PrecisionPolicy.FP32FP32
    vs = xlb.velocity_set.D2Q9(precision_policy=pp, compute_backend=backend)
    xlb.init(velocity_set=vs, default_backend=backend, default_precision_policy=pp)
    grid = grid_factory((nx, ny), compute_backend=backend)

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    r = np.hypot(ii - cx, jj - cy)
    theta = np.arctan2(jj - cy, ii - cx)
    inner_solid = r <= Ri_lb
    outer_solid = r >= Ro_lb
    fluid = (~inner_solid) & (~outer_solid)

    # --- boundary conditions ---
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

    # --- analytic Couette profile as initial condition ---
    # v_theta(r) = A*r + B/r, A = -Om*Ri^2/(Ro^2-Ri^2),
    # B = Om*Ri^2*Ro^2/(Ro^2-Ri^2)  (physical units)
    A = -Omega_phys * Ri**2 / (Ro**2 - Ri**2)
    B = Omega_phys * Ri**2 * Ro**2 / (Ro**2 - Ri**2)
    r_phys = r * dx
    vth = np.where(fluid, A * r_phys + B / np.where(r_phys == 0, 1, r_phys), 0.0)
    ux = np.where(fluid, -vth * (jj - cy) / np.where(r == 0, 1, r), 0.0)
    uy = np.where(fluid, vth * (ii - cx) / np.where(r == 0, 1, r), 0.0)
    u_lb = np.stack([ux * dt / dx, uy * dt / dx]).astype(np.float32)

    def initializer(bc_mask, f_0):
        rho = grid.create_field(cardinality=1, fill_value=1.0,
                                dtype=pp.compute_precision)
        u = jnp.array(u_lb)
        return initialize_eq(f_0, grid, vs, pp, backend, rho=rho, u=u)

    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields(initializer=initializer)

    # Momentum-transfer operators, built ONCE: their jitted fetchers are
    # keyed on the operator instance, so rebuilding them per sample
    # would recompile every time (measured ~6 s per sample).
    mt = [MomentumTransfer(bc, compute_backend=backend) for bc in sector_bcs]
    return dict(grid=grid, vs=vs, pp=pp, backend=backend, stepper=stepper,
                f_0=f_0, f_1=f_1, bc_mask=bc_mask, missing_mask=missing_mask,
                sector_bcs=sector_bcs, mt=mt, omega=omega, cx=cx, cy=cy,
                fluid=fluid, r=r, theta=theta, nx=nx, ny=ny,
                Omega_phys=Omega_phys, A=A, B=B)


def sample_torque(state):
    """Per-node momentum-exchange torque (lattice units, z-component).

    Replays MomentumTransfer's JAX formula node by node (see module
    docstring): the fetcher of each sector BC reconstructs the
    post-streaming populations with that BC's moving-wall correction,
    phi = f_post_collision[opp] + f_post_stream at the missing
    directions of that BC's edge nodes, force = c[:, opp] . phi, and
    the torque sums r_node x F_node over the edge nodes.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override n_steps from lattice.py")
    args = ap.parse_args()
    gamma = args.gamma

    case = load_case()
    lat = lattice_params(case)
    if not lat["mach_ok"]:
        print("FAIL: lattice check failed (u_lb too high)")
        return 1
    n_steps = args.steps or case["xlb"].get("steps_per_point", lat["n_steps"])
    dt = lat["dt_s"]
    sample_every = case["xlb"]["sample_every_steps"]
    snapshot_every = case["xlb"].get("snapshot_every_steps", 1000)
    L = case["geometry"]["L"]
    eta = case["fluid"]["eta_Pa_s"]
    force_conv = lat["force_conversion_N_per_lb_force"]  # lattice force -> N (2D, per unit length)

    t0 = time.time()
    state = build_case(case, lat, gamma)
    print(f"gamma={gamma} 1/s  Omega={state['Omega_phys']:.4f} rad/s  "
          f"steps={n_steps:,}  build {time.time()-t0:.1f}s")

    macro = Macroscopic(compute_backend=state["backend"],
                        precision_policy=state["pp"],
                        velocity_set=xlb.velocity_set.D2Q9(
                            precision_policy=state["pp"],
                            compute_backend=state["backend"]))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

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
            # driving torque: negate the (opposing) force on the solid
            tq_per_m = -tq_lb * force_conv
            torque_rows.append((step + 1, (step + 1) * dt, tq_lb, tq_per_m))
        if (step + 1) % snapshot_every == 0:
            snapshot(step + 1)

    with open(out / "torque.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "time_s", "torque_lb", "torque_N_m_per_m"])
        for s, t, tq_lb, tq in torque_rows:
            w.writerow([s, f"{t:.6e}", f"{tq_lb:.6e}", f"{tq:.6e}"])

    # --- steady torque (last 20%) vs analytic ---
    n = len(torque_rows)
    k = max(1, int(n * case["sweep"]["steady_window_fraction"]))
    steady = [r[3] for r in torque_rows[-k:]]
    T_per_m = sum(steady) / len(steady)
    sd = (sum((x - T_per_m) ** 2 for x in steady) / len(steady)) ** 0.5
    T = T_per_m * L
    Ri, Ro = case["geometry"]["R_i"], case["geometry"]["R_o"]
    T_an = (4 * math.pi * L * eta * state["Omega_phys"] * Ri**2 * Ro**2
            / (Ro**2 - Ri**2))
    err = abs(T - T_an) / T_an
    print(f"steady torque: {T:.6e} N.m  (sd {sd*L:.2e})  analytic {T_an:.6e}  "
          f"err {err*100:.2f}%  runtime {time.time()-t0:.1f}s")

    # --- profile cross-check: fit u_theta = A*r + B/r over fluid nodes ---
    rho, u = macro(state["f_0"])
    u = np.array(u) * lat["dx_m"] / lat["dt_s"]
    ii, jj = np.meshgrid(np.arange(state["nx"]), np.arange(state["ny"]), indexing="ij")
    r_phys = state["r"] * lat["dx_m"]
    uth = np.where(state["fluid"],
                   (-u[0] * (jj - state["cy"]) + u[1] * (ii - state["cx"]))
                   / np.where(state["r"] == 0, 1, state["r"]), 0.0)
    fr, fu = r_phys[state["fluid"]], uth[state["fluid"]]
    # least squares for A, B in u = A*r + B/r
    M = np.stack([fr, 1.0 / fr], axis=1)
    (A_fit, B_fit), *_ = np.linalg.lstsq(M, fu, rcond=None)
    # Shear rate magnitude at the inner wall: |gamma| = 2*B/Ri^2
    # (the tensorial shear rate r*d/dr(u_theta/r) = -2*B/r^2 is
    # negative by convention; the Couette gamma_i is its magnitude).
    gamma_i_fit = 2 * B_fit / Ri**2
    tau_fit = eta * gamma_i_fit
    T_fit = tau_fit * 2 * math.pi * Ri**2 * L
    print(f"profile fit: A={A_fit:.4f} B={B_fit:.6e}  gamma_i={gamma_i_fit:.3f} 1/s  "
          f"T_fit={T_fit:.6e} N.m  err vs analytic {abs(T_fit-T_an)/T_an*100:.2f}%")
    with open(out / "profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r_m", "u_theta_m_s", "u_theta_analytic_m_s", "count"])
        edges = np.linspace(Ri, Ro, 21)
        for a, b in zip(edges[:-1], edges[1:]):
            m = (fr >= a) & (fr < b)
            if m.sum() == 0:
                continue
            rc = 0.5 * (a + b)
            w.writerow([f"{rc:.6e}", f"{fu[m].mean():.6e}",
                        f"{state['A']*rc + state['B']/rc:.6e}", int(m.sum())])
    print("wrote", out / "torque.csv", "and", out / "profile.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
