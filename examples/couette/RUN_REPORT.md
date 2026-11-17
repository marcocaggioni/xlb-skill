# RUN_REPORT — XLB Couette example

Newtonian validation of the XLB Couette case, run for real on this
VM. Same geometry, fluid and sweep as the M-Star and Palabos
Couette examples, so the three solvers compare point for point.

## What ran

- **XLB version:** 0.3.2 (PyPI, commit `272dc8e` of
  https://github.com/Autodesk/XLB.git — "Prepare 0.3.2 PyPI release"),
  Apache-2.0. Installed with `uv sync` (`xlb[warp]`, `jax`, numpy,
  matplotlib, pyyaml); `uv.lock` in the repo.
- **Backend:** JAX, CPU only (`CpuDevice`). This VM has no GPU; the
  Warp/Neon backends need NVIDIA hardware and were not tried.
  (`warp-lang` 1.19.0 is installed as an XLB import-time dependency
  and falls back to CPU with a "No GPU support" warning.)
- **Machine:** 4 CPUs, ~7.7 GB RAM, Python 3.12.3.
- **Date:** 2026-11-17.
- **Case:** `examples/couette/case.yaml` — 2D annulus, R_i = 0.020 m,
  R_o = 0.025 m, L = 0.060 m; Newtonian η = 1.0 Pa·s, ρ = 1000 kg/m³;
  γ̇ = 0.1, 0.3, 1, 3, 10, 30, 100 s⁻¹. Lattice: 10 cells across the
  5 mm gap (dx = 0.5 mm), ω = 1.0, dt = 4.167e-5 s, u_lb,max = 0.030.
- **Run length:** 3,000 steps per point (0.125 s) — an XLB-only
  override of the shared 1 s run time, forced by the mass leak
  (see below). Steady state is reached by ~step 500 (verified from
  a zero-velocity start: profile within 1% of analytic by step 500,
  torque steady from step ~100). Torque averaged over the last 20%
  of each run (steps 2,400–3,000). Total sweep time 161 s
  (~12–15 s per point including the ~4–5 s JIT build).

## How the moving wall and torque were implemented

(XLB API read from the 0.3.2 sources and examples, not invented.)

- **Rotating wall — angular sectors.** `HalfwayBounceBackBC` with
  `prescribed_value=(ux, uy)`. In the JAX backend the velocity
  profile is called with no per-node argument
  (`xlb/operator/boundary_condition/bc_halfway_bounce_back.py`,
  `jax_implementation`: `u_wall = self.profile()`), so one BC object
  carries exactly one constant velocity. The inner cylinder is
  therefore split into 32 angular sectors (`xlb.wall_sectors`),
  each its own `HalfwayBounceBackBC` with the wall velocity at the
  sector centre. Directional error per sector ≤ half the sector
  angle (~0.5% at 32). BC ids are uint8 (1–253), so a per-node BC
  for the ~312 wall nodes is impossible
  (`xlb/cell_type.py`, `boundary_condition_registry.py`).
  The outer wall is a single `HalfwayBounceBackBC` with
  velocity (0, 0).
- **No per-step ramp.** Wall velocities are baked into the jitted
  stepper, so the run is initialised with the analytic Couette
  velocity profile via `initialize_eq`
  (`xlb/helper/initializers.py`) instead of a quarter-sine ramp.
- **Torque — per-node momentum exchange.** XLB's `MomentumTransfer`
  returns the *net* force on a BC's geometry, which cannot give a
  torque: the pressure (radial) part of the wall force is ~100×
  the shear part and only cancels when the lever arm sits at the
  force's own angle. (First attempt — one lever arm per sector —
  read −2.3 N/m against an analytic 0.0251 N/m.) The run script
  therefore replays `MomentumTransfer`'s own JAX formula node by
  node, using each sector BC's `.fetcher` (`FetchPopulations`) to
  reconstruct the post-streaming populations with that BC's
  moving-wall correction: φ = f_post-collision[opp] + f_post-stream
  at the missing directions of that BC's edge nodes,
  F = c[:, opp] · φ, torque = Σ_nodes r_node × F_node.
  XLB's momentum-exchange force is the force on the solid, so its
  z-torque on the bob is negative (it opposes the rotation); the
  reported torque is the negation — the driving torque — the same
  convention as the Palabos example.
- **Torque conversion (2D, per unit length):** lattice force → N
  with ρ·dx⁴/dt² = 0.036 N per lattice-force unit; 3D torque = (N/m) × L.
- **Performance note:** the `MomentumTransfer` operators must be
  built once and reused — their jitted fetchers are keyed on the
  operator instance, so rebuilding them per torque sample costs
  ~6 s per sample (the first gamma=10 run took 2 min for 2,000
  steps before it was killed and fixed).

## Results — the gate

| γ̇ (1/s) | Ω (rad/s) | T sim (N·m) | T analytic (N·m) | err | η_app (Pa·s) | profile-fit err |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.018 | 1.5071e-05 | 1.5080e-05 | 0.06% | 0.9994 | 0.29% |
| 0.3 | 0.054 | 4.5247e-05 | 4.5239e-05 | 0.02% | 1.0002 | 0.23% |
| 1 | 0.18 | 1.5086e-04 | 1.50796e-04 | 0.04% | 1.0004 | 0.13% |
| 3 | 0.54 | 4.5257e-04 | 4.5239e-04 | 0.04% | 1.0004 | 0.01% |
| 10 | 1.8 | 1.5072e-03 | 1.5080e-03 | 0.05% | 0.9995 | 1.10% |
| 30 | 5.4 | 4.5112e-03 | 4.5239e-03 | 0.28% | 0.9972 | 4.35% |
| 100 | 18.0 | 1.4904e-02 | 1.5080e-02 | 1.16% | 0.9884 | 17.56% |

- **Newtonian validation: PASSED — max analytic error 1.16%
  (tolerance 10%).** η_app = 0.988–1.000 Pa·s against the true 1.0.
- γ̇ = 10 check (the gate point): **T = 1.50724e-03 N·m vs analytic
  1.50796e-03 N·m, error 0.05%** (steady-window sd 2.6e-08 N·m).
- The "profile-fit err" column is the independent cross-check: the
  final velocity field fitted to u_θ = A·r + B/r, wall shear rate
  from the fit, torque from that shear rate. It agrees with the
  momentum-exchange torque at low γ̇ and diverges at high γ̇ —
  that divergence is the mass leak (below), not a torque error.
- Full data: `flow_curve.csv`, `flow_curve.png`;
  per-point `runs/gamma_<value>/torque.csv` (30 samples each),
  `profile.csv` (20 radial bins), `field_<step>.csv` snapshots.
- Videos (matplotlib + ffmpeg, from the real snapshots):
  `couette-gamma10.mp4` (13 frames, 1.3 s, γ̇ = 10 over 0.125 s)
  and `couette-sweep.mp4` (70 frames, 7 s, steady state per γ̇).

## Known limitation — mass leak at the curved moving wall

Measured 2026-11-17, and it is the reason the runs are 3,000 steps
instead of 1 s:

- Mean fluid density falls steadily during a run: **~0.5% per
  1,000 steps at γ̇ = 10 1/s, ~5.2% per 1,000 steps at γ̇ = 100**
  (linear in wall speed; measured from a zero-velocity start with
  the macroscopic density). The velocity then inflates as 1/ρ
  (momentum is maintained by the wall), so over a 1 s (24,000-step)
  run at γ̇ = 10 the profile ends up +13% fast while the torque
  stays analytic (0.04%) — the torque self-compensates because
  ρ·(velocity gradient) stays constant.
- The leak is the staircase curved wall, not the sectors: **128
  sectors leak exactly the same as 32** (ρ = 0.98507 vs 0.98455
  after 3,000 steps at γ̇ = 10). XLB 0.3.2 has no
  interpolated/curved-wall BC to switch to (the BC list is
  DoNothing, Equilibrium, ExtrapolationOutflow, FullwayBounceBack,
  HalfwayBounceBack, Hybrid, Regularized, ZouHe — and Fullway has
  no moving-wall support).
- The torque is physical, not vacuous: from a zero-velocity start
  it reads 0.519 N/m at step 0 (start-up transient) and decays to
  the analytic 0.0251 N/m as the profile develops. Decomposed, the
  wall-correction term alone is a constant 0.519 N/m and the fluid
  part (−0.494 N/m) cancels most of it — the reported torque is
  the small difference of two large terms, and it converges to the
  analytic value.
- Effect on this sweep (3,000 steps/point): density loss ≈ 0.02%
  (γ̇ = 0.1) to ≈ 15% (γ̇ = 100; intermediate values scale linearly
  with wall speed). The γ̇ = 100 torque is 1.16% low and its
  profile fit 17.6% high — both consistent with the leak — but
  still inside the 10% gate. **Do not run 1 s per point, and do
  not trust the velocity field (or a profile-derived torque) at
  high γ̇ without quoting the leak.**

## Still unverified

- Non-Newtonian fluids (Carreau-Yasuda / Herschel–Bulkley as a
  local-ω collision operator — the skill describes the route;
  the Newtonian gate must pass first, and it now does).
- 3D (end effects, ~100× cost).
- Finer resolution (20 cells across the gap) — note the leak per
  unit physical time scales ~1/dx, so finer grids need the leak
  re-measured, not assumed better.
- MPI / multi-device runs; Warp/Neon GPU backends (need NVIDIA
  hardware — the Warp backend's per-voxel wall profile would
  remove the sector approximation, but whether it also removes the
  mass leak is untested).
- A mass-conserving curved moving wall in XLB (upstream fix, or a
  custom BC) — the item that would lift the run-length cap.
