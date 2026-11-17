---
name: xlb-simulation
description: "Run an open-source XLB (JAX lattice Boltzmann) simulation from an English discussion: elicit the geometry, fluid and operating conditions in plain language, turn them into a checked case file, run the XLB case from the command line, and turn the wall torque back into a physical result (e.g. a Couette flow curve). Use when the user wants a free, pip-installable, GPU-ready LBM simulation — a Couette rheometer, a channel, a flow past a body — or asks for the XLB counterpart to an M-Star or Palabos run."
---

# XLB Simulation

XLB (Autodesk Research) is an open-source lattice Boltzmann library
in Python, built on JAX, Apache-2.0, installable with `pip install
xlb`. It runs on CPU, single GPU (Warp backend), multi-GPU (Neon
backend) and distributed systems, and it is differentiable — the
same code that runs a flow can be differentiated through. This skill
is the XLB counterpart of the `mstar-simulation` and
`palabos-simulation` skills: same discussion → case → run → report
workflow, same Couette example, different solver — no license, no
C++ compiler, no separate library checkout.

## The workflow, in order

```
1. Discuss (English)     -> what fluid, what geometry, what do you want to measure?
2. Write case.yaml       -> every number in one file, with units. Show it. Get a yes.
3. Lattice parameters    -> lattice.py: dx, dt, omega, lattice velocity. Check Mach < 0.1.
4. Run the case          -> python couette.py (JAX backend, CPU or GPU). No build step.
5. Sweep / run           -> one run per shear rate, torque CSV per run.
6. Post-process          -> torque -> stress, omega -> shear rate, flow curve + validation.
7. Report                -> plot + CSV + video + what was assumed, checked, and unverified.
```

**Never skip step 2.** Same rule as in the M-Star and Palabos
skills: a simulation with an unconfirmed shear-rate range, fluid
model, or geometry is a wasted run. If the user has not named a
number, ask — do not invent one and bury it in code. Defaults in
the examples are labelled `default:` and must be confirmed.

**Running is free here** (no license seat), but it is still not free:
a 3D run at fine resolution costs real compute time. Default to a
cheap first run (coarse lattice, short time, one shear rate) and
only then sweep.

## 1. Discuss — what to elicit

Same checklist as the M-Star and Palabos skills, plus one XLB-specific item:

### Geometry
- Inner (rotating) radius `R_i`, outer (fixed) radius `R_o`, immersed
  length `L`. Metres.
- **First version is a 2D cross-section** (annulus in the plane,
  torque per unit length) unless the user asks for 3D. The flow in an
  ideal Couette cell has no axial variation except end effects, so
  2D validates the physics at a fraction of the cost; 3D is the
  follow-up, with end effects included. Say which you are running.

### Fluid — pick one rheology, with numbers
XLB ships BGK, KBC and Smagorinsky collision operators, but **no
rheology dropdown**. Newtonian is built in (constant relaxation
frequency). Non-Newtonian means making the relaxation frequency a
function of the local shear rate — the Carreau-Yasuda pattern used
in the Palabos literature applies here too, as a custom collision
operator (the local strain rate is available from the distribution
functions). Herschel–Bulkley needs the same treatment plus a yield
stress, with a viscosity cap at low shear rate (the yield stress
makes the apparent viscosity diverge as γ̇ → 0 — cap it, and say
what cap you used). If the user names a real material, do not invent
its parameters: ask for a measured flow curve, a rheofit fit, or a
literature value, or start Newtonian as a validation case.

### Operating conditions — the sweep
- Shear-rate range and spacing, e.g. 0.1 → 100 s⁻¹, log-spaced.
- Lattice resolution: cells across the gap (default 10, i.e.
  dx = gap / 10). More cells = more accurate torque, much more cost.
- Run time per point and the steady window for averaging torque.
  Viscous diffusion time is gap²/ν — for the default case that is
  ~0.025 s, so 1 s per point is generous.

### What is being measured
- Couette: torque on the rotating bob → shear stress, rotation rate
  → shear rate. The flow curve. Name the file it comes from
  (`torque.csv`, written by the run script) before running.

## 2. Write case.yaml

One file per simulation, in `examples/<name>/case.yaml`. It is the
contract: the scripts read it, they do not re-derive numbers from
chat. Template: `examples/couette/case.yaml`. Show it to the user
and get a yes before building.

## 3. Lattice parameters — the step M-Star hides

XLB works in **lattice units**. Converting physical → lattice
is the agent's job, and getting it wrong is the most common failure.
`examples/couette/lattice.py` does it; the rules are:

```
dx = gap / N_gap                     lattice spacing (m)
nu_lb = nu * dt / dx^2               lattice kinematic viscosity
omega = 1 / (3*nu_lb + 0.5)         relaxation frequency (D2Q9/D3Q19, cs^2 = 1/3)
u_lb  = u_phys * dt / dx             lattice velocity — keep max u_lb < 0.1 (low Mach)
rho_lb = 1.0                         by convention
```

Choose `dt` by fixing `omega` (default 1.0, so nu_lb = 1/6) and
solving `dt = nu_lb * dx^2 / nu`. Then check `u_lb` at the highest
shear rate. If `u_lb >= 0.1`, reduce `dt` (more steps) or coarsen
`dx` — and say which you did. Never run with `omega` outside
roughly [0.8, 1.9] without saying so: near 2 the scheme goes
unstable, near 0.5 it is over-damped.

Force/torque conversion: a lattice force converts with
`rho * dx^4 / dt^2` (per unit length in 2D).
The run script writes torque already converted to N·m per unit
length; `couette_post.py` checks the units by comparing with the
analytic torque (step 6).

## 4. Run the case

XLB is a Python library, not a compiled program: there is no build
step. Install once with `uv sync` (it installs XLB, JAX, warp-lang
and the tooling), then:

```bash
bash scripts/check-setup.sh     # python, uv, xlb import, backend
./run-couette.sh                # dry-run: lattice params only
./run-couette.sh --run          # gamma = 10 check, then the sweep
```

The Couette case (`examples/couette/couette.py`) follows XLB's own
example pattern (`lid_driven_cavity_2d.py`, `rotating_sphere_3d.py`):
`xlb.init(...)`, `grid_factory(...)`, `HalfwayBounceBackBC` for the
walls, `IncompressibleNavierStokesStepper` with BGK collision,
`MomentumTransfer` for the wall force, `Macroscopic` for the
velocity field. **Two XLB-specific facts shape the implementation:**

- **Moving wall, JAX backend.** A `HalfwayBounceBackBC` takes one
  constant wall velocity per BC object (its JAX implementation
  calls the velocity profile with no per-node argument), so a
  rotating wall — whose velocity varies with position — is split
  into angular sectors (default 32), each sector its own BC with
  the wall velocity at the sector centre. The directional error per
  sector is at most half the sector angle (~0.5% at 32 sectors).
  The Warp backend does support a per-voxel velocity profile, but
  it needs NVIDIA hardware; this repo's runs use the JAX backend
  on CPU.
- **No per-step ramp.** The wall velocities are baked into the
  compiled stepper, so changing them mid-run would recompile every
  step. Instead the run is initialised with the analytic Couette
  velocity profile (same purpose as the ramp: no start-up shock).
  This is documented in `case.yaml` and `RUN_REPORT.md`.
- **Mass leak at a curved moving wall — the known limitation.**
  XLB 0.3.2's halfway bounce-back leaks mass at the staircase
  curved wall: mean fluid density falls ~0.5% per 1,000 steps at
  γ̇ = 10 1/s and ~5% per 1,000 steps at γ̇ = 100 (linear in wall
  speed; identical at 128 sectors, so it is the staircase wall,
  not the sectors — and there is no interpolated/curved-wall BC
  in XLB 0.3.2 to switch to). The velocity then inflates as 1/ρ
  while the momentum-exchange torque stays analytic, so a long run
  corrupts the velocity field but not the torque. Keep runs short
  (`xlb.steps_per_point` in `case.yaml`, 3,000 steps = 0.125 s —
  long past steady state, which is reached by ~step 500) and
  report the density drift per point. Do not run 1 s per point.

**Status: the XLB case has been run on this VM (JAX backend, CPU)
and the Newtonian validation in §6 passes — see
`examples/couette/RUN_REPORT.md` for the numbers.** The API names
were checked against the cloned XLB sources (v0.3.2) and its
examples, not invented.

Non-Newtonian: implement it as a custom collision operator with a
local relaxation frequency, `omega(x, t)` from the local shear
rate via the chosen model (Carreau-Yasuda first;
Herschel–Bulkley second, with a viscosity cap). Keep the Newtonian
path as the default and the validation gate — never debug a
non-Newtonian model on an unvalidated geometry.

## 5. Run

```bash
./run-couette.sh --run           # one run per shear rate, in examples/couette/runs/
# or, per point:
uv run python examples/couette/couette.py --gamma 10 --out examples/couette/runs/gamma_10
```

Each run writes `torque.csv` (time, torque per unit length),
`profile.csv` (radius, time-averaged azimuthal velocity) and
`field_*.csv` snapshots (velocity field, for the video). If a run
is slow, reduce the resolution or the run time first — do not
silently change the geometry to make it fast.

## 6. Post-process — torque to flow curve (Couette)

Same conversion as the M-Star and Palabos skills, per unit length
in 2D (multiply by `L` for the 3D torque):

```
shear rate at the inner wall:  gamma_i = 2*Omega*Ro^2 / (Ro^2 - Ri^2)
shear stress at the inner wall: tau_i  = T / (2*pi*Ri^2*L)   (T = torque * L in 2D)
apparent viscosity:             eta_app = tau_i / gamma_i
```

Average torque over the steady window named in `case.yaml`, report
its standard deviation as the steadiness check.

**Validation gate (mandatory for the first run of a new geometry).**
Run a Newtonian fluid of known eta first and compare with the
analytic Couette torque:

```
T_analytic = 4*pi*L*eta*Omega*Ri^2*Ro^2 / (Ro^2 - Ri^2)
```

If the simulated torque differs by more than ~10%, do **not**
proceed to a non-Newtonian flow curve. Stop and diagnose: lattice
resolution across the gap, `omega` / `dt`, run time, moving-wall
boundary, torque extraction. Record the validation number in the
report. `couette_post.py` enforces this gate and exits non-zero on
failure.

Then each sweep point gives one (gamma_i, tau_i) point. Plot tau vs
gamma (log-log) and eta_app vs gamma, write `flow_curve.csv`, and —
if the user wants parameters — fit the curve with rheofit
(Herschel–Bulkley, Carreau, …) in a separate step.

## 7. Report

Deliver, per run: `case.yaml` (as run), `lattice.json` (dx, dt,
omega, max u_lb), `flow_curve.csv`, `flow_curve.png`, the video,
and a short note stating the geometry (2D or 3D), fluid, sweep,
validation result, steady-state criterion, and anything
unverified. If a run failed, report the solver log tail — do not
silently drop the point from the curve.

## Starting a session with GitHub Copilot CLI

Same setup as `mstar-skill` and `palabos-skill`: Copilot CLI
discovers project skills in `.github/skills/`, and this skill is
mirrored there. From the repo root:

```bash
git clone https://github.com/marcocaggioni/xlb-skill.git
cd xlb-skill
uv sync
bash scripts/check-setup.sh
copilot
```

Then `/skills list` (you should see `xlb-simulation`), and:

```
/xlb-simulation run the Couette example in examples/couette/case.yaml
```

This skill deliberately has **no `allowed-tools` frontmatter**:
Copilot must ask before running shell commands, and in particular
before a long solver run. If the skill was added during a session,
run `/skills reload` first.

## What this skill does NOT do

- It does not replace M-Star for tanks, impellers, multiphase, or
  particle-laden flows — those are out of scope for v0.1. This repo
  is the Couette flow curve, done openly, as the first proof that
  the open route works.
- It does not claim GPU performance: the runs here use the JAX
  backend on CPU. The Warp (single-GPU) and Neon (multi-GPU)
  backends exist and are the reason to pick XLB over Palabos when
  a GPU is available, but they need NVIDIA hardware and are not
  exercised in v0.1.

## Files in this repo

- `skills/xlb-simulation/SKILL.md` — this file. Mirrored
  byte-for-byte at `.github/skills/xlb-simulation/SKILL.md`
  (the copy GitHub Copilot CLI discovers). **Keep the two copies
  in sync** — edit both in the same commit.
- `docs/xlb-guide.md` — install, backends, lattice units,
  non-Newtonian notes, limits.
- `examples/couette/` — the worked Couette example: `case.yaml`,
  `lattice.py`, `couette.py`, `couette_sweep.py`,
  `couette_post.py`, `make_video.py`, `README.md`.
- XLB itself: https://github.com/Autodesk/XLB (Apache-2.0)
