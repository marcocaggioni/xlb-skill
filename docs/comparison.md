# Couette flow curve — solver comparison

Same test for every solver, so the numbers are comparable: a 2D
Couette rheometer, inner cylinder rotating, outer fixed, Newtonian
fluid η = 1.0 Pa·s, ρ = 1000 kg/m³, R_i = 0.020 m, R_o = 0.025 m,
L = 0.060 m, shear rates 0.1 → 100 s⁻¹. The check is the analytic
Couette torque: the simulated torque must match it, and the recovered
apparent viscosity must come back ≈ 1.0 Pa·s.

Written 2026-11-17; updated the same day with the OpenLB Newtonian run,
and again with the three-component (TC) benchmark below. A copy
of this file lives in every solver repo (`docs/comparison.md`); the
numbers are the same everywhere.

## Scoreboard

| Solver | Type | Validation error (max) | η_app recovered | Status |
|---|---|---|---|---|
| M-Star | LBM, GPU-native, licensed | not yet run (no licensed install here) | — | skill + case ready, v0.1 |
| OpenFOAM | Finite-volume, GPL | **0.03%** | 1.0003 | ran, published |
| XLB | LBM, Python/JAX | 1.16% | 0.988–1.000 | ran, published, with a caveat |
| Palabos | LBM, C++/MPI, AGPLv3 | 1.43% | 1.0143 | ran, published |
| OpenLB | LBM, C++/CUDA, GPLv2 | 0.94% | 1.0093 | ran, published |

## Three-component (TC) model fluid — Couette benchmark (Nov 17, 2026)

Same Couette case, Newtonian fluid replaced by Marco's TC model:
σ = σ_y + σ_y√(γ̇/γ̇_c) + η_bg·γ̇, with σ_y = 70 Pa,
γ̇_c = 100 1/s, η_bg = 0.05 Pa·s (50 cP), ρ = 1000 kg/m³. Same
geometry (R_i = 0.020 m, R_o = 0.025 m, L = 0.060 m) and the same
seven sweep labels γ̇ = 0.1 → 100 s⁻¹ (Ω set by the Newtonian
inner-wall-rate convention, Ω = γ̇(R_o²−R_i²)/(2R_o²)).

Validation is torque vs the analytic TC response: τ(r) = T/(2πLr²),
the TC law inverted for γ̇(τ) (γ̇ = 0 where τ < σ_y), Ω = ∫γ̇ dr/r
out to the yield radius R_y = min(R_o, √(T/2πLσ_y)), solved for T
by bisection. The shared reference implementation (`tc-analytic.py`,
one copy per repo) reproduces the known Newtonian analytic torque to
0.0000%. Reference torques: 0.01167, 0.01218, 0.01304, 0.01423,
0.01628, 0.01931, 0.02523 N·m. At γ̇ ≤ 10 there is a genuine
unyielded plug attached to the outer wall (R_y = 21.0–24.8 mm); at
30 and 100 the gap is fully yielded.

Regularization (same everywhere): η(γ̇) = σ_y/γ̇ + σ_y/√(γ̇γ̇_c) +
η_bg diverges at γ̇ → 0, so every solver caps it:
η = min(η(γ̇), η_max), η_max = η(10⁻³ s⁻¹) = 70,221 Pa·s.

### TC scoreboard

| Solver | TC max error | Worst point | Status |
|---|---|---|---|
| M-Star | not run | — | case ready (Custom Fluid UDF), Marco to run at work |
| OpenFOAM | **0.82%** | γ̇ = 0.1 (+0.82%) | ran, published |
| OpenLB | **1.14%** | γ̇ = 0.1 (+1.14%) | ran, published |
| Palabos | 3.80% | γ̇ = 0.1 (+3.80%) | ran, published |
| XLB | 8.02% | γ̇ = 100 (−8.02%) | ran, published, marginal at γ̇ = 100 |

### Per-solver TC notes

- **OpenFOAM (0.82%)** — custom `threeComponent` viscosity model
  (no built-in TC model exists; nothing substituted). All seven
  errors +0.2–0.8%, largest at the lowest rate where the yielded
  layer is thinnest (~8 cells). Sweep 75 s. γ̇ = 30/100 flagged
  `converged: false` on the residual criterion, but the torque is
  flat and 6000-iteration checks move those points toward analytic.
- **OpenLB (1.14%)** — local-ω BGK via Π^neq, Bouzidi interpolated
  curved wall, momentum-exchange torque. Errors +0.1–1.1%, sweep
  ~75 s. No mass leak (density drift ≤ +0.34%). Best open-LBM TC
  number — and the only open LBM here that also did the cavity and
  the RBC test below.
- **Palabos (3.80%)** — `TCBGKdynamics`, local ω via Π^neq,
  fixed-point iterated like Palabos' own Carreau dynamics. Errors
  all positive, falling with γ̇ — systematic staircase-wall bias,
  worst where the yielded zone is thinnest (~2 cells at γ̇ = 0.1).
  Sweep ~8 min.
- **XLB (8.02%)** — custom BGK collision operator, local ω from
  Π^neq. Passes the 10% gate but marginal at γ̇ = 100 (−8.02%),
  where the halfway bounce-back mass leak (−10.7% density) breaks
  the Newtonian self-compensation for a shear-thinning fluid.
  Velocity profile also deviates near the yield surface while
  torque stays accurate. Sweep 316 s.
- **M-Star** — case prepared (`examples/couette-tc/`), not run. TC
  is not a built-in rheology, but the public docs document a
  Custom Fluid UDF (Kinematic Viscosity UDF, a local expression
  evaluated at each lattice position each step) — the case is
  prepared to use it. One gap: the exact shear-rate variable name
  inside the UDF is not in the public docs and must be read off
  Pre's Input Variables Panel on the licensed install.

### Lid-driven cavity with the TC fluid (OpenLB, for the RTX 3090 video)

2D square cavity, L = 0.100 m, U_lid = 1.0 m/s (characteristic rate
10 1/s → τ ≈ 93 Pa at the lid, above the 70 Pa yield; quiet corners
unyielded), N = 128 intervals per side, 455,325 steps, 1315 s on
this VM's serial CPU (5.76 MLUPS). Newtonian Re = 100 check vs Ghia
et al.: 2.84% PASSED. TC run converged (KE drift 0.00009%, plug
38.7% of cells, density drift −0.25%): yielded lid layer, rotating
unyielded plug at the vortex core, stagnant bottom plug. Video:
`examples/cavity-tc/cavity-tc.mp4` in the OpenLB repo. RTX 3090
estimate for the same run on `matteo` (WSL): ~10–60 s of solver
time — an estimate, not a measurement (basis: 7.58e9 lattice
updates; A5000 711/1077 MLUPS published; 3090 ballpark 700–1100
MLUPS pure BGK, derated 300–700 for the TC path, plus a
kernel-launch floor on 16.6k cells); one-time CUDA build 10–30 min
not included. 3D not attempted (would cost ~100× at this resolution).

### Immersed-boundary red blood cell (OpenLB, first test)

OpenLB 1.9's shipped benchmark `fsi/deformableParticleInShearFlow3d`
(Krüger, Varnik & Raabe 2011), copied unchanged: single spherical
capsule in shear, 50³, Re = 0.02, G = 0.01, 9000 steps to κt = 120.
Steady D/G = 7.019 vs analytic 6.25 (+12.3%), PASSED at the 15%
gate (the shipped Δx = 1 run is itself ~+9% off analytic). Run
359.6 s, 3.13 MLUPS. Scope, stated honestly: spherical capsule, not
a biconcave RBC; Skalak membrane with no bending and no
volume/area conservation. Suspension not attempted — the
single-cell pass makes it a legitimate next step.

## M-Star (Simcenter, Siemens) — the licensed reference

The thing everything else is measured against. GPU-native lattice
Boltzmann, built for transient mixing and bioprocess: agitated tanks,
bioreactors, fermentation, particles, heat transfer, multiphase.
Pre GUI builds the case, Solve runs it on GPU, Post renders it.
Siemens positions it as a validated digital twin for scale-up
(AbbVie / Bristol Myers Squibb bioreactor cases).

**Pros**
- Turnkey: GUI, built-in rheologies (Newtonian, power-law,
  Herschel–Bulkley, Carreau, Cross, Bingham), torque output
  built in, no code to write for a standard case.
- GPU-native and fast; vendor support and validated workflows.
- Marco holds a work license.

**Cons**
- ~USD 60,000/year reported, quote-based, no free trial; a license
  seat is burned on every run.
- Closed: no public input.xml schema, Post has no batch mode,
  nothing can be published openly.
- The Couette skill is ready (private repo `mstar-skill`) but has
  not yet run on a licensed install.

**Use it for:** production mixing / scale-up work at P&G, where
speed, validation, and support matter more than cost.

## OpenFOAM — cleanest validation so far

Finite-volume, not lattice Boltzmann. Mature, GPL, huge community.
Non-Newtonian models are built in (Herschel–Bulkley, Bird–Carreau,
Casson, power-law), rotating walls and force/torque output are
standard. Our run used a 4° wedge sector, 40 cells across the gap.

**Pros**
- Best number in this test: 0.03% error, η_app = 1.0003 at all
  seven shear rates.
- Rheology is a menu choice, not a coding project — the shortest
  path to a real non-Newtonian flow curve.
- Free, runs anywhere, enormous example and tutorial base.

**Cons**
- Not LBM: different numerics, different failure modes; a
  steeper, dictionary-based case setup with no GUI by default.
- Our run is a thin wedge sector, not a full 3D annulus — no end
  effects yet.

**Use it for:** the first real non-Newtonian flow curve, and as
the accuracy reference the LBM codes are judged against.

## Palabos — the open LBM workhorse

C++ library with MPI, AGPLv3, from the University of Geneva.
Practically no dependencies (Posix + MPI). GPU version in
pre-release. You write a small C++ program against it and compile
it; there is no GUI and no shipped Taylor–Couette example.

**Pros**
- Free, open, deploys anywhere with a compiler; public repo and
  CI are possible.
- Validation passed at 1.43% (η_app = 1.0143, constant across the
  sweep — a systematic staircase-wall bias, so finer resolution
  should shrink it). Sweep took 43 s on CPU.
- MPI parallelism is built in.

**Cons**
- You write the case, the moving curved wall, and the torque
  extraction (momentum exchange) yourself — that was the hard
  part of this build.
- No rheology menu: non-Newtonian means a local relaxation
  frequency you implement.
- GPU support is pre-release, not the solid path.

**Use it for:** open LBM work you want to publish or run anywhere
without a license, once the case code exists.

## XLB — the Python / GPU bet

Pure-Python LBM on JAX, pip-installable, Apache-2.0. The easiest
to install and the easiest to extend; GPU comes along with JAX.

**Pros**
- `pip install`, no C++ build, runs in a notebook; the fastest
  route from an idea to a running simulation.
- Validation passed at 1.16% max error; torque is right across
  the sweep.
- GPU in WSL is realistic here (JAX + CUDA), unlike the C++
  codes' pre-release GPU paths.

**Cons**
- Young (v0.3.2): no interpolated curved-wall boundary condition,
  and the halfway bounce-back **leaks mass at the curved moving
  wall** — ~0.5% of fluid density per 1,000 steps at γ̇ = 10,
  ~5% at γ̇ = 100. Runs had to be capped short, and the γ̇ = 100
  velocity profile comes out 17.6% too fast.
- Least mature of the three that ran; treat results with the
  leak in mind until the boundary condition improves.

**Use it for:** fast prototyping and GPU experiments, not yet for
numbers you would defend without the caveat.

## OpenLB — second-best validation, best open LBM number

C++ with CUDA, GPLv2, the other big open LBM code. On paper the
closest Palabos alternative, with more ready-made examples and
more mature CUDA support. Ran 2026-11-17 in its own repo
(`openlb-skill`): 1.9r0, D2Q9/BGK, serial CPU, 10 cells across the
gap, Bouzidi interpolated bounce-back moving wall (the aorta3d
pattern) with the analytical circle as the wall position — so the
inner wall is not a staircase, unlike Palabos.

**Pros**
- Validation passed at **0.94%** (η_app = 1.0093, constant across
  the sweep — a systematic bias, so finer resolution should shrink
  it). Sweep took 32.5 s on CPU.
- Best open-LBM number in this test, and the only open LBM here
  with an interpolated curved moving wall that holds up: no mass
  leak was seen (unlike XLB), torque steady to 7 digits.
- Ships a real Taylor–Couette example (`fsi/taylorCouette3d`) and
  non-Newtonian patterns (`powerLaw2d`,
  `nonNewtonianPoiseuille3d`).

**Cons**
- You still write the case, the moving wall, and the torque
  extraction yourself — and the obvious first attempt
  (`boundary::InterpolatedVelocity` on the solid disc) throws
  "Could not set Boundary"; the Bouzidi route is the one that works.
- No rheology menu: non-Newtonian means a local relaxation
  frequency you implement.
- 2D cross-section here; the shipped 3D Taylor–Couette example was
  not used for v0.1.

**Use it for:** open LBM work you want to publish or run anywhere
without a license — the strongest open LBM candidate in this test
so far.

## Bottom line

- **Accuracy so far:** OpenFOAM (0.03%) > OpenLB (0.94%) > XLB (1.16%) ≈ Palabos
  (1.43%), all four passing the 10% gate.
- **TC accuracy:** OpenFOAM (0.82%) > OpenLB (1.14%) > Palabos
  (3.80%) > XLB (8.02%), all four passing the 10% gate — and the
  TC ranking matches the Newtonian one.
- **Best open LBM for the TC fluid:** OpenLB (1.14%, interpolated
  curved wall, no leak) — and it is also the only one with a TC
  cavity video and a first RBC test.
- **Fastest to a real non-Newtonian curve:** OpenFOAM, because the
  models are built in.
- **Torque is the robust observable** for a yield-stress fluid in
  every LBM code here; velocity profiles near the yield surface are
  not — a ~1% stress error maps to a 10–30% shear-rate error where
  the TC law is flat near yield.
- **Production mixing at work:** still M-Star — nothing open
  matches its validated tank/impeller/multiphase package.
- **Next steps:** finer resolution (the low-rate TC points are
  resolution-limited everywhere); the M-Star TC run at work; the
  RBC suspension; a 3D cavity if the 2D video earns it.

## Repos

- M-Star (private): github.com/marcocaggioni/mstar-skill
- Palabos: github.com/marcocaggioni/palabos-skill
- XLB: github.com/marcocaggioni/xlb-skill
- OpenFOAM: github.com/marcocaggioni/openfoam-skill
- OpenLB: github.com/marcocaggioni/openlb-skill (ran 2026-11-17)
