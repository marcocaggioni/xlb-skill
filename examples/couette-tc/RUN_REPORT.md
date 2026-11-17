# RUN_REPORT — XLB Couette example, three-component (TC) fluid

TC validation of the XLB Couette case, run for real on this VM. Same
geometry and sweep as the Newtonian example in `examples/couette/`
and the other solvers' TC cases, so the solvers compare point for
point — with the TC model fluid instead of Newtonian.

## What ran

- **XLB version:** 0.3.2 (PyPI), Apache-2.0. Installed with `uv sync`
  (`xlb[warp]`, `jax`, numpy, matplotlib, pyyaml); `uv.lock` in the repo.
- **Backend:** JAX, CPU only (`CpuDevice`). This VM has no GPU; the
  Warp/Neon backends need NVIDIA hardware and were not tried.
- **Precision:** FP64FP64 (case.yaml). At γ̇ = 0.1 the lattice wall
  speed is u_lb ~ 4e-8 (dt is set by eta = 722 Pa·s); FP32 cannot
  carry the momentum-exchange torque — a small difference of two
  large terms — at that signal level. XLB enables JAX x64 for FP64
  policies automatically.
- **Machine:** 4 CPUs, ~7.7 GB RAM, Python 3.12.3.
- **Date:** 2026-11-17.
- **Case:** `examples/couette-tc/case.yaml` — 2D annulus, R_i = 0.020 m,
  R_o = 0.025 m, L = 0.060 m; TC fluid σ_y = 70 Pa, γ̇_c = 100 1/s,
  η_bg = 0.05 Pa·s, ρ = 1000 kg/m³; γ̇ (label) = 0.1, 0.3, 1, 3, 10,
  30, 100 s⁻¹. Lattice: 10 cells across the 5 mm gap (dx = 0.5 mm).
- **Run length:** 3,000 steps per point (same XLB-only cap as the
  Newtonian case — the mass leak, below). Torque averaged over the
  last 20% of each run (steps 2,400–3,000). Total sweep time 316 s
  of per-point runtime (26–68 s per point including the ~14–18 s JIT
  build). One runtime restart interrupted the first sweep attempt
  after 6 of 7 points; γ̇ = 100 was run separately afterwards with
  the same script and settings — no numbers were tuned.

## How the TC rheology was implemented

(XLB API read from the 0.3.2 sources and examples, not invented.
XLB ships no rheology dropdown: BGK/KBC/Smagorinsky only.)

- **Custom collision operator** (`tc_collision.py`, subclass of
  `xlb.operator.collision.Collision`, JAX backend only), following
  the pattern of XLB's own `SmagorinskyLESBGK`: the local strain
  rate is read off the non-equilibrium stress tensor, no velocity
  gradients are needed. The stepper is built with
  `collision_type="BGK"` and its `.collision` is replaced by the TC
  operator before the first step.
- **Strain rate from the populations.** Π_neq[a,b] = Σ_i c_ia c_ib
  (f_i − f_i^eq), S = −Π_neq/(2ρc_s²τ), γ̇_lb = √(2S:S) =
  3|Π_neq|_F/(√2·ρ·τ). The τ (not τ−½) denominator was validated
  numerically on the Newtonian Couette steady field: median
  γ̇_Pi/γ̇_analytic = 1.003 at ω = 1.0, 1.001 at ω = 0.6, 1.012 at
  ω = 1.5 (τ−½ gives 2.02 at ω = 1.0 — decisively rejected).
- **Implicit local-ω solve.** τ itself depends on γ̇ through the
  viscosity, so per node the pair (γ̇, τ) is solved by bisection on
  h(g) = g·(0.5 + C₀·η(g)) = K/dt, K = 3|Π_neq|_F/(√2·ρ),
  C₀ = 3·dt/(ρ_phys·dx²), η(g) the capped TC viscosity (30
  iterations, vectorised over the grid, inside the jitted
  collision). The scalar ω the stepper passes in is ignored — it is
  only the reference ω the lattice was built at. The solve was
  verified in isolation to recover the input γ̇ to <1e-5 relative
  over γ̇ = 1e-4 → 1000 1/s. ω is clipped above at 1.9 (η → η_bg at
  very high local γ̇) for stability; it is not clipped below.
- **Regularization.** η(γ̇) = σ_y/γ̇ + σ_y/√(γ̇·γ̇_c) + η_bg
  diverges as γ̇ → 0, so η = min(η(γ̇), η_max) with η_max =
  η(γ̇_reg), γ̇_reg = 1e-3 1/s, η_max = 7.02214e4 Pa·s (computed from
  γ̇_reg in the code, not hard-coded). The cap only bites where
  γ̇ < 1e-3 1/s, i.e. inside the unyielded plug (γ̇ = 0 there
  analytically). The plug therefore creeps at γ̇ ~ τ/η_max ~ 1e-3
  1/s instead of exactly 0 — a <1% effect on Ω at the lowest sweep
  point.
- **Per-point lattice** (`lattice.py`): dt is solved per sweep point
  from the viscosity at that point's label rate, ω_ref = 1.0:
  dt = ν_lb,ref·dx²/ν_label, ν_lb,ref = 1/6. The local ω then
  follows the local viscosity. ω at the inner wall is 1.17–1.82
  across the sweep (inside [0.8, 1.9]); ω in the plug falls to
  0.02 (γ̇ = 0.1) down to 0.00003 (γ̇ = 100) because η there is the
  cap — the plug being near-solid, not an instability: the plug is
  near-stationary (populations near rest equilibrium) and the runs
  are 3,000 steps. u_lb,max = 0.021 (at γ̇ = 100).

  | γ̇ label | dt (s) | u_lb | η_label | η_inner | γ̇_inner | ω_inner | ω_plug |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 0.1 | 5.770e-08 | 4.15e-08 | 722.19 | 70.42 | 1.10 | 1.822 | 0.0204 |
  | 0.3 | 1.693e-07 | 3.66e-07 | 246.16 | 34.70 | 2.33 | 1.753 | 0.0070 |
  | 1 | 5.408e-07 | 3.89e-06 | 77.05 | 16.16 | 5.35 | 1.653 | 0.0022 |
  | 3 | 1.519e-06 | 3.28e-05 | 27.43 | 8.17 | 11.56 | 1.541 | 0.0008 |
  | 10 | 4.498e-06 | 3.24e-04 | 9.26 | 3.95 | 27.33 | 1.402 | 0.0003 |
  | 30 | 1.138e-05 | 2.46e-03 | 3.66 | 2.08 | 61.72 | 1.276 | 0.0001 |
  | 100 | 2.874e-05 | 2.07e-02 | 1.45 | 1.03 | 162.44 | 1.169 | 0.00003 |

  (η_inner/γ̇_inner from the analytic TC torque. Note: η at the
  label rate is 1.45 Pa·s at γ̇ = 100, not 70.75 — the "70.75"
  figure in the task brief is a slip: η(100) = 0.7 + 0.7 + 0.05.)
- **Initial condition:** the analytic TC Couette profile (yielded
  region shears, plug attached to the fixed outer wall is
  stationary), computed from `tc_analytic.py` by integrating
  γ̇(r) inward from R_o. Verified: the linear-mesh integral
  reproduces Ω to 1e-7, and the step-0 field matches the analytic
  profile to ~1–4% per radial bin.
- **Torque:** per-node momentum exchange, exactly as in the
  Newtonian case (replay of `MomentumTransfer`'s JAX formula node
  by node; reported torque is the negation of the force on the
  solid = the driving torque). Same conversion, ρ·dx⁴/dt² per
  point (dt varies per point).
- **Moving wall:** 32 angular sectors of `HalfwayBounceBackBC`,
  as in the Newtonian case.

## Results — the gate

Validation is on **torque** against the analytic TC torque
(`tc_analytic.torque_of_omega`). η_app = τ_i/γ̇_label is reported
only — the label rate is a Newtonian convention and the TC fluid's
true inner-wall rate exceeds it (see γ̇_inner above).

| γ̇ label | Ω (rad/s) | T sim (N·m) | T analytic (N·m) | err | η_app (Pa·s) | R_y (mm) | density drift | profile err* |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.018 | 1.194744e-02 | 1.167063e-02 | +2.37% | 792.29 | 21.03 | −0.00% | 198% |
| 0.3 | 0.054 | 1.236848e-02 | 1.218414e-02 | +1.51% | 273.40 | 21.49 | −0.00% | 199% |
| 1 | 0.18 | 1.314250e-02 | 1.303766e-02 | +0.80% | 87.15 | 22.23 | −0.00% | 126% |
| 3 | 0.54 | 1.428214e-02 | 1.423130e-02 | +0.36% | 31.57 | 23.22 | −0.02% | 66% |
| 10 | 1.8 | 1.627107e-02 | 1.627965e-02 | −0.05% | 10.79 | 24.84 | −0.17% | 32% |
| 30 | 5.4 | 1.912002e-02 | 1.931375e-02 | −1.00% | 4.23 | 25.00 | −1.27% | 8.5% |
| 100 | 18.0 | 2.320978e-02 | 2.523402e-02 | −8.02% | 1.54 | 25.00 | −10.67% | 15% |

\* "profile err" is the max relative error of the binned final
velocity field vs the analytic TC profile over bins where the
analytic velocity exceeds 10% of its maximum. It is a cross-check
only, not the gate — see "Known limitation: velocity profile" below.

- **TC validation: PASSED — max analytic-torque error 8.02%
  (tolerance 10%).** γ̇ = 10 check (the gate point):
  **T = 1.627107e-02 N·m vs analytic 1.627965e-02 N·m, error
  −0.05%** (steady-window sd 1.2e-06 N·m, window drift −0.01%).
- The error is systematic in γ̇: +2.4% at the bottom of the sweep
  falling monotonically to −8.0% at the top. The low-γ̇ end reads
  high (effective stress ~1% high in the yielded region — see the
  profile analysis below); the high-γ̇ end reads low because of the
  mass leak (below).
- Steady-window drift is ≤0.01% at γ̇ ≤ 10, −0.10% at γ̇ = 30 and
  −0.87% at γ̇ = 100 (torque still falling slowly at the end of the
  γ̇ = 100 run as the leak grows).
- Full data: `flow_curve.csv`, `flow_curve.png`; per-point
  `runs/gamma_<value>/torque.csv` (30 samples each), `profile.csv`
  (20 radial bins), `summary.json`, `lattice.json`,
  `field_<step>.csv` snapshots.
- Videos (matplotlib + ffmpeg, from the real snapshots):
  `couette-tc-gamma10.mp4` (13 frames, 1.3 s, γ̇ = 10 over
  0.0135 s) and `couette-tc-sweep.mp4` (70 frames, 7 s, steady
  state per γ̇, colour scale per frame since the wall speed spans
  0.00036–0.36 m/s).

## Known limitation — mass leak, and why it bites TC harder

Measured in the Newtonian case (Nov 17) and present here: XLB
0.3.2's halfway bounce-back leaks mass at the curved moving wall,
mean fluid density falling ~0.5% per 1,000 steps at γ̇ = 10 and
~5% per 1,000 steps at γ̇ = 100 (linear in wall speed). In this
sweep the drift is −0.00% (γ̇ = 0.1) to −10.67% (γ̇ = 100 over
3,000 steps).

In the Newtonian case the torque self-compensates (ρ·(velocity
gradient) stays constant, so the γ̇ = 100 torque was only 1.16%
low despite a ~15% density loss). For a shear-thinning fluid the
compensation breaks: the leak inflates the velocity as 1/ρ, the
inflated γ̇ lowers η, and the torque drops — hence −8.02% at
γ̇ = 100, still inside the 10% gate but marginal. A shorter run
would leak less (the torque is steady from ~step 500), but the
3,000-step length is the case-file contract and was not changed
to improve the number.

## Known limitation — velocity profile near the yield surface

The final velocity field deviates from the analytic TC profile,
most at low γ̇ (max binned error 198% at γ̇ = 0.1, 32% at γ̇ = 10,
8.5% at γ̇ = 30) — while the torque stays within 2.4% at those
points. Diagnosed (Nov 17, from the γ̇ = 10 run, step 1,000):

- The stress measured from Π_neq in the fluid matches the
  momentum-balance τ(r) = T/(2πLr²) to ~0.5% at every radius, and
  the Π-based γ̇ matches the finite-difference γ̇ of the velocity
  field — so the local rheology and the stress are both right at
  the ~1% level.
- The implied effective yield stress from the simulated (τ, γ̇)
  pairs is 69.3–70.0 Pa through most of the gap (true 70 Pa),
  drifting to ~67–68 Pa right at the yield surface.
- Near yield, γ̇ ∝ (τ − σ_y)², so a ~1% stress-level discrepancy
  (≈1 Pa on τ − σ_y ≈ 3–10 Pa) amplifies into a factor-2–5 error
  in γ̇ — and therefore in u_θ — near the plug. The profile error
  is concentrated in the bins just inside R_y, where the analytic
  velocity is small.
- The plug itself (γ̇ = 0 analytically) creeps at up to ~6e-5 m/s
  at γ̇ = 10 (regularization + staircase wall; the plug is only
  0.16 mm thick there, below one cell).

Consequence: **do not trust the velocity field (or a
profile-derived torque) near the plug; the torque is the validated
observable**, which is why this benchmark validates on torque.
The profile cross-check is reported, not hidden, for exactly this
reason — it is the TC analogue of the Newtonian case's 17.6%
profile error at γ̇ = 100.

## Still unverified

- Non-Newtonian behaviour at finer resolution (20 cells across the
  gap) — note the leak per unit physical time scales ~1/dx, so a
  finer grid needs the leak re-measured, not assumed better; the
  near-yield profile error may or may not improve.
- 3D (end effects, ~100× cost).
- MPI / multi-device runs; Warp/Neon GPU backends (need NVIDIA
  hardware — the Warp backend's per-voxel wall profile would
  remove the sector approximation, but whether it also removes the
  mass leak is untested).
- A mass-conserving curved moving wall in XLB (upstream fix, or a
  custom BC) — the item that would lift the run-length cap and
  clean up the γ̇ = 100 torque.
- Whether the ~1% effective-stress discrepancy near yield is a
  variable-τ Chapman–Enskog correction, a staircase-wall effect,
  or the regularization — it is small in stress but visible in the
  profile only because of the (τ − σ_y)² sensitivity.
