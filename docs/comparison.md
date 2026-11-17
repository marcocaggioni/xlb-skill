# Couette flow curve — solver comparison

Same test for every solver, so the numbers are comparable: a 2D
Couette rheometer, inner cylinder rotating, outer fixed, Newtonian
fluid η = 1.0 Pa·s, ρ = 1000 kg/m³, R_i = 0.020 m, R_o = 0.025 m,
L = 0.060 m, shear rates 0.1 → 100 s⁻¹. The check is the analytic
Couette torque: the simulated torque must match it, and the recovered
apparent viscosity must come back ≈ 1.0 Pa·s.

Written 2026-11-17, updated the same day with the OpenLB run. A copy
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
- **Fastest to a real non-Newtonian curve:** OpenFOAM, because the
  models are built in.
- **Best open LBM for publishing:** OpenLB today (0.94%, interpolated
  curved wall, no leak); Palabos close behind; XLB once its
  curved-wall boundary is fixed.
- **Production mixing at work:** still M-Star — nothing open
  matches its validated tank/impeller/multiphase package.
- **Next step:** run a Herschel–Bulkley or Carreau fluid through
  the same Couette case in whichever solver you pick; that is
  where the solvers will actually separate.

## Repos

- M-Star (private): github.com/marcocaggioni/mstar-skill
- Palabos: github.com/marcocaggioni/palabos-skill
- XLB: github.com/marcocaggioni/xlb-skill
- OpenFOAM: github.com/marcocaggioni/openfoam-skill
- OpenLB: github.com/marcocaggioni/openlb-skill (ran 2026-11-17)
