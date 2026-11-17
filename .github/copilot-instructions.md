# Copilot instructions — xlb-skill

This repo teaches an agent to run open-source XLB (JAX lattice
Boltzmann) simulations from an English discussion. The skill is
`skills/xlb-simulation/SKILL.md`, mirrored at
`.github/skills/xlb-simulation/SKILL.md`.

Rules for every simulation in this repo:

1. Discuss in English first, then write `case.yaml`, show it, and
   get a yes before running. Never invent a number.
2. Convert physical → lattice units with `lattice.py` and check
   the low-Mach condition before running.
3. Default to a dry run (lattice check only). Never run the solver
   without saying so first.
4. Validate a Newtonian case against the analytic Couette torque
   (within ~10%) before any non-Newtonian flow curve.
5. XLB is a Python library: no build step, `uv sync` installs it.
   The JAX backend runs on CPU; Warp/Neon backends need NVIDIA
   hardware and are out of scope for v0.1 runs.
6. The rotating wall is split into angular sectors of
   `HalfwayBounceBackBC` (one constant wall velocity per BC object
   in the JAX backend). Initialise with the analytic Couette
   profile instead of a per-step ramp.
