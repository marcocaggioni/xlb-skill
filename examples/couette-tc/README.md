# Couette example, three-component (TC) fluid — XLB

Fluid in a Couette rheometer: inner cylinder rotates, outer cylinder
fixed, torque on the bob gives the flow curve. Same geometry and sweep
as the Newtonian example in `examples/couette/`, so the solvers can be
compared point for point — but the fluid is the three-component (TC)
model fluid (Marco, Nov 17, voice):

    tau = sigma_y + sigma_y*sqrt(gamma_dot/gamma_dot_c) + eta_bg*gamma_dot

with sigma_y = 70 Pa, gamma_dot_c = 100 1/s, eta_bg = 0.05 Pa·s (50 cP).
The sweep labels are the Newtonian inner-wall-rate convention
(Omega = gamma*(Ro^2-Ri^2)/(2*Ro^2)); the TC fluid's true inner-wall
rate exceeds the label rate (1.10 1/s at label 0.1, 162 1/s at label
100), so the flow curve is validated on **torque** against the analytic
TC torque — not by comparing eta_app to the TC law at the label rate.

## What is here

| File | What it does |
|---|---|
| `case.yaml` | The contract: geometry, TC fluid + regularization, sweep, per-point lattice, XLB settings. Confirm before running. |
| `tc_analytic.py` | The shared analytic TC Couette reference (copied from `~/workspace/tc-analytic.py`, self-contained). |
| `tc_collision.py` | Custom BGK collision operator: local omega from the local strain rate via the TC law. **Run for real.** |
| `lattice.py` | Physical → lattice parameters, solved **per sweep point** from eta at the label rate. **Tested.** |
| `couette.py` | The XLB case (2D annulus, JAX backend, FP64). **Run for real — see RUN_REPORT.md.** |
| `couette_sweep.py` | Runs the sweep, one run per shear rate. Dry-run by default. |
| `couette_post.py` | Torque → flow curve + TC analytic-torque validation gate. **Tested on the real sweep.** |
| `make_video.py` | Renders `couette-tc-gamma10.mp4` + `couette-tc-sweep.mp4` from the snapshots. |

## Defaults

R_i = 0.020 m, R_o = 0.025 m, L = 0.060 m, TC fluid as above,
ρ = 1000 kg/m³, γ̇ = 0.1 → 100 s⁻¹ log-spaced (labels), 10 cells across
the 5 mm gap, ω_ref = 1.0 (dt solved per point from eta at the label
rate), 3,000 steps per point, precision FP64FP64. 2D cross-section
(torque per unit length); 3D is a follow-up. Wall sectors: 32.

**Regularization (required):** eta(gamma) diverges as gamma → 0, so
eta = min(eta(gamma), eta_max) with eta_max = eta(1e-3 1/s) = 7.02214e4
Pa·s. The cap only bites inside the unyielded plug. See `case.yaml`
and `RUN_REPORT.md`.

## Run it

```bash
uv sync
bash scripts/check-setup.sh
.venv/bin/python examples/couette-tc/couette_sweep.py --run   # full sweep
.venv/bin/python examples/couette-tc/couette_post.py          # flow curve + gate
.venv/bin/python examples/couette-tc/make_video.py            # videos
```

## Honest status

- `lattice.py`: verified — per-point dt from 5.77e-8 s (γ̇ = 0.1) to
  2.87e-5 s (γ̇ = 100), ω_inner = 1.17–1.82, u_lb,max = 0.021.
- `tc_collision.py`: the strain-rate-from-Pi_neq factor was validated
  numerically on the Newtonian steady field (median γ̇_Pi/γ̇_analytic
  = 1.003 at ω = 1.0, 1.001 at ω = 0.6, 1.012 at ω = 1.5), and the
  implicit local-ω solve was verified to recover the input γ̇ to
  <1e-5 relative over γ̇ = 1e-4 → 1000 1/s.
- `couette.py`: run for real on this VM (XLB 0.3.2, JAX backend, CPU,
  FP64). Full sweep done Nov 17, 2026.
- `couette_post.py`: TC validation **PASSED — max analytic-torque
  error 8.02%** (at γ̇ = 100; tolerance 10%). See `RUN_REPORT.md` for
  the numbers and the known limitations.
- **Known limitations:** the halfway bounce-back mass leak (same as
  Newtonian) hits the TC torque harder at γ̇ = 100 (−10.7% density
  over 3,000 steps → −8.0% torque, because shear-thinning breaks the
  Newtonian self-compensation); and the velocity profile deviates
  from the analytic TC profile near the yield surface — a ~1% stress
  discrepancy amplified by γ̇ ∝ (τ−σ_y)² near yield — while the
  torque stays accurate. Both are documented in `RUN_REPORT.md`.
  Do not trust the velocity field (or a profile-derived torque) near
  the plug; the torque is the validated observable.

## Analytic check

γ̇_i = 2·Ω·R_o²/(R_o²−R_i²) is the *Newtonian* label; the TC torque
for a given Ω comes from `tc_analytic.torque_of_omega` (bisection on
Ω(T) = ∫_{Ri}^{Ry} γ̇(τ(r)) dr/r, Ry = min(Ro, √(T/(2πLσ_y)))).
At γ̇ = 10 s⁻¹: Ω = 1.8 rad/s (17.2 rpm), T = 1.628e-2 N·m,
τ_i = 108.0 Pa, R_y = 24.84 mm (thin plug at the outer wall).
