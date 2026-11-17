# Couette example — XLB

Fluid in a Couette rheometer: inner cylinder rotates, outer cylinder
fixed, torque on the bob gives the flow curve. Same geometry and
fluid as the M-Star and Palabos examples, so the solvers can be
compared point for point.

## What is here

| File | What it does |
|---|---|
| `case.yaml` | The contract: geometry, fluid, sweep, lattice, XLB settings. Confirm before running. |
| `lattice.py` | Physical → lattice parameters (dx, dt, omega, Mach check). **Tested.** |
| `couette.py` | The XLB case (2D annulus, JAX backend). **Run for real — see RUN_REPORT.md.** |
| `couette_sweep.py` | Runs the sweep, one run per shear rate. Dry-run by default. |
| `couette_post.py` | Torque → flow curve + Newtonian validation gate. **Tested on synthetic torque.** |
| `make_video.py` | Renders `couette-gamma10.mp4` + `couette-sweep.mp4` from the snapshots. |

## Defaults

R_i = 0.020 m, R_o = 0.025 m, L = 0.060 m, Newtonian η = 1.0 Pa·s,
ρ = 1000 kg/m³, γ̇ = 0.1 → 100 s⁻¹ log-spaced, 10 cells across the
5 mm gap, ω = 1.0, 1 s per point, steady window = last 20%.
2D cross-section first (torque per unit length); 3D is a follow-up.
Wall sectors: 32.

## Run it

```bash
uv sync
bash ../../scripts/check-setup.sh
./../../run-couette.sh --run     # gamma = 10 check, then the full sweep
uv run python make_video.py     # videos from the snapshots
```

## Honest status (v0.1)

- `lattice.py`: verified — dx 0.5 mm, dt 4.17e-5 s, ω = 1.0,
  u_lb,max = 0.030 at γ̇ = 100 s⁻¹, 24,000 steps per 1 s point
  (XLB runs use 3,000 steps per point — see below).
- `couette_post.py`: verified on synthetic torque data — recovers
  η_app ≈ 1.0 Pa·s, analytic error ~0%, validation PASSED; and on
  the real sweep — max error 1.16%, validation PASSED.
- `couette.py`: run for real on this VM (XLB 0.3.2, JAX backend,
  CPU). Full sweep done Nov 17, 2026: γ̇ = 0.1 → 100 s⁻¹,
  η_app = 0.988–1.000 Pa·s, max analytic error 1.16% (at
  γ̇ = 100). See `RUN_REPORT.md` for the numbers and the known
  limitations.
- **Known limitation — mass leak:** XLB 0.3.2's halfway
  bounce-back leaks mass at the curved moving wall (~0.5% of the
  fluid density per 1,000 steps at γ̇ = 10, ~5% at γ̇ = 100).
  Runs are therefore capped at 3,000 steps per point
  (`xlb.steps_per_point` in `case.yaml`); the torque stays
  analytic but the velocity field inflates as 1/ρ on long runs.
  Do not run 1 s per point.

## Analytic check

γ̇_i = 2·Ω·R_o²/(R_o²−R_i²), τ_i = T/(2π·R_i²·L),
T_analytic = 4π·L·η·Ω·R_i²·R_o²/(R_o²−R_i²).
At γ̇ = 10 s⁻¹: Ω = 1.8 rad/s (17.2 rpm), T = 1.508e-3 N·m, τ = 10 Pa.
