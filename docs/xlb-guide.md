# XLB — short guide for this repo

Sources: https://github.com/Autodesk/XLB, the XLB docs in the repo
(`docs/` in the checkout), and XLB's own examples
(`examples/cfd/lid_driven_cavity_2d.py`,
`examples/cfd/rotating_sphere_3d.py`). Anything not in those
sources is marked as unverified in this repo — do not treat it as fact.

## What XLB is

Open-source lattice Boltzmann library in Python, built on JAX,
from Autodesk Research, Apache-2.0. `pip install xlb` (or
`uv sync` in this repo). Backends: JAX (CPU, single/multi-GPU,
distributed), Warp (single-GPU, NVIDIA), Neon (multi-GPU). It is
differentiable — the same code can be differentiated through with
JAX. There is no GUI and no case file format — you write a Python
script against the library.

## Install

```bash
uv sync   # in this repo: XLB + JAX (CPU) + warp-lang + tooling
```

`warp-lang` is required at XLB import time even when using the JAX
backend (XLB raises if `warp` cannot be imported); installing
`xlb[warp]` satisfies that. The Warp/Neon *compute* backends need
NVIDIA hardware and are not used in v0.1.

## How a simulation is put together

Following `examples/cfd/lid_driven_cavity_2d.py`:

```python
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic

xlb.init(velocity_set=xlb.velocity_set.D2Q9(...),
         default_backend=ComputeBackend.JAX,
         default_precision_policy=PrecisionPolicy.FP32FP32)
grid = grid_factory(grid_shape, compute_backend=ComputeBackend.JAX)
stepper = IncompressibleNavierStokesStepper(
    grid=grid, boundary_conditions=[...], collision_type="BGK")
f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()
for i in range(num_steps):
    f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, i)
    f_0, f_1 = f_1, f_0
```

Boundary conditions are index-based: you pass explicit lattice
indices. `HalfwayBounceBackBC(indices=...)` is no-slip;
`EquilibriumBC(rho=..., u=..., indices=...)` prescribes velocity;
`HalfwayBounceBackBC(indices=..., prescribed_value=(ux, uy))`
is a moving wall with one constant velocity. For a rotating wall
the JAX backend needs one BC per angular sector (see below).
`MomentumTransfer(bc, ...)` computes the force on a BC's geometry
by momentum exchange (used in `rotating_sphere_3d.py` for
drag/lift). `Macroscopic` returns density and velocity fields.

## Lattice units — the one thing to get right

XLB works in lattice units internally. Convert physical → lattice
before setting anything, and convert forces/torques back afterwards:

```
dx = characteristic length / resolution
dt from omega:  omega = 1 / (3*nu_lb + 0.5),  nu_lb = nu * dt / dx^2
u_lb = u_phys * dt / dx          keep < 0.1
rho_lb = 1.0 (by convention)
force conversion (2D, per unit length) = rho * dx^4 / dt^2
```

`examples/couette/lattice.py` does this for the Couette case.
Choosing `omega` near 1 is a good default; near 2 the scheme goes
unstable, near 0.5 it is over-damped.

## Boundaries — the rotating-wall problem

- No-slip walls: `HalfwayBounceBackBC` on solid nodes (the
  halfway bounce-back scheme; the wall sits half a cell between a
  solid node and a fluid node).
- Moving walls: `HalfwayBounceBackBC` with `prescribed_value`
  (constant velocity) or `profile` (a callable). **In the JAX
  backend the profile is called with no per-node argument, so it
  can only return one constant velocity per BC object**
  (`xlb/operator/boundary_condition/bc_halfway_bounce_back.py`,
  `jax_implementation`). A rotating wall therefore has to be
  split into angular sectors, each its own BC. In the Warp
  backend the profile is a `@wp.func(index, timestep)` returning
  the velocity at that voxel — the pattern used in
  `rotating_sphere_3d.py` — but Warp needs NVIDIA hardware.
- Interior geometry with index BCs: pass the *solid* indices;
  the masker pads them to find the boundary nodes
  (`IndicesBoundaryMasker`, `pad_indices()`). BC ids are uint8
  (1–253), so a per-node BC for every wall node is impossible —
  another reason for sectors.

## Non-Newtonian fluids

There is no rheology menu. The documented pattern (same as in the
Palabos literature) is to make the relaxation frequency a local
function of the shear rate, as a custom collision operator — the
local strain rate is available from the distribution functions.
Herschel–Bulkley needs the same plus a yield stress and a viscosity
cap at low shear rate. Implement Newtonian first, validate it, then
add the model — never the other way round.

## Limits (v0.1)

- No shipped Couette example in XLB — the case here is written
  from XLB's own examples (`lid_driven_cavity_2d.py`,
  `rotating_sphere_3d.py`), not copied.
- The JAX-backend rotating wall is piecewise-constant (sectors);
  the error is bounded by half the sector angle and is measured
  by the Newtonian validation gate, not assumed away.
- **Mass leak at a curved moving wall (measured Nov 17, 2026).**
  Halfway bounce-back leaks mass at the staircase curved wall:
  mean fluid density falls ~0.5% per 1,000 steps at γ̇ = 10 1/s
  and ~5% per 1,000 steps at γ̇ = 100 (linear in wall speed).
  128 sectors leak exactly the same as 32, so it is the staircase
  wall, not the sectors — and XLB 0.3.2 has no interpolated /
  curved-wall BC to switch to (the BC list is DoNothing,
  Equilibrium, ExtrapolationOutflow, FullwayBounceBack,
  HalfwayBounceBack, Hybrid, Regularized, ZouHe; Fullway has no
  moving-wall support). The velocity inflates as 1/ρ while the
  momentum-exchange torque stays analytic, so a long run corrupts
  the velocity field but not the torque. Runs are therefore capped
  at `xlb.steps_per_point` = 3,000 steps (0.125 s) — long past
  steady state, reached by ~step 500 — and the density drift is
  reported per point in `RUN_REPORT.md`.
- GPU backends (Warp/Neon) are not exercised in v0.1.
- XLB is Apache-2.0: permissive — fine for a public repo and for
  internal use alike.
