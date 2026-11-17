# AGENTS.md — xlb-skill

XLB counterpart of the private `mstar-skill` repo and the public
`palabos-skill` repo: same discussion → case → run → report
workflow, same Couette example, XLB (JAX lattice Boltzmann) solver
instead of licensed M-Star or C++ Palabos. No license, no C++
compiler — XLB installs with pip/uv.

- Skill: `skills/xlb-simulation/SKILL.md`, mirrored byte-for-byte
  at `.github/skills/xlb-simulation/SKILL.md` (the copy Copilot
  CLI discovers). Keep the two copies in sync; edit both per commit.
- No `allowed-tools` in the skill frontmatter, deliberately — Copilot
  must ask before shell/solver runs.
- `case.yaml` is the contract. Defaults are labelled; confirm before
  running. Never invent a number.
- Lattice units are the agent's job: `lattice.py` converts physical
  → lattice and enforces u_lb < 0.1. Never run with omega outside
  roughly [0.8, 1.9] without saying so.
- Validation gate: Newtonian torque vs analytic Couette torque within
  ~10% before any non-Newtonian curve. `couette_post.py` enforces it.
- XLB specifics: JAX backend on CPU for v0.1 runs (Warp/Neon need
  NVIDIA hardware). Rotating wall = angular sectors of
  `HalfwayBounceBackBC` (one constant wall velocity per BC object
  in the JAX backend). No per-step ramp — initialise with the
  analytic Couette profile instead.
- **Mass leak (measured Nov 17):** halfway bounce-back leaks mass
  at the curved moving wall (~0.5% of fluid density per 1,000
  steps at γ̇ = 10 1/s, ~5% at γ̇ = 100; staircase wall, not the
  sectors — 128 sectors leak the same). Runs are capped at
  `xlb.steps_per_point` = 3,000 steps in case.yaml; do not run
  1 s per point. The torque stays analytic; the velocity field
  inflates as 1/ρ, so report the density drift per point.
- XLB itself is Apache-2.0; this repo's own code is MIT.
