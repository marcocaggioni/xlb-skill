# Quickstart — xlb-skill

```bash
git clone https://github.com/marcocaggioni/xlb-skill.git
cd xlb-skill
uv sync            # XLB + JAX (CPU) + warp-lang + numpy/matplotlib/pyyaml
bash scripts/check-setup.sh
```

Then start Copilot CLI in the repo root and invoke the skill:

```bash
copilot
```

```
/xlb-simulation run the Couette example in examples/couette/case.yaml
```

Or in one go:

```bash
copilot -i "Use the /xlb-simulation skill to run the Couette example in examples/couette/case.yaml"
```

Copilot must be started from the repo root for skill discovery.
If the skill was added during a session, run `/skills reload` first.
The skill has no `allowed-tools` frontmatter on purpose — Copilot
must ask before running shell commands or a solver run.

To run the Couette example without Copilot:

```bash
./run-couette.sh            # dry run: lattice check only
./run-couette.sh --run      # gamma = 10 check, then the full sweep
./run-couette.sh --post     # post-process runs/ into flow_curve.csv/png
uv run python examples/couette/make_video.py   # videos from the snapshots
```

GPU note: the runs in this repo use the JAX backend on CPU.
XLB's Warp (single-GPU) and Neon (multi-GPU) backends need NVIDIA
hardware; switching backend is a `case.yaml` change plus the
matching install (`xlb[warp]` is already installed — warp-lang is
required at XLB import time — but the compute still runs on CPU
without a GPU).
