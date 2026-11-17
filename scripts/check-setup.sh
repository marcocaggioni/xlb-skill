#!/usr/bin/env bash
# Check the machine is ready to run the XLB Couette example.
# Exits 0 when ready, 1 otherwise.
set -u
fail=0
say() { printf '%-28s %s\n' "$1" "$2"; }

check() {  # name, command...
  local name="$1"; shift
  if command -v "$1" >/dev/null 2>&1; then say "$name" "OK ($1)"; else say "$name" "MISS"; fail=1; fi
}

check "copilot CLI" copilot
check "uv" uv
check "python3" python3
check "ffmpeg (video)" ffmpeg

if [ -f pyproject.toml ] && [ -d .venv ]; then
  if uv run python -c "import xlb; print(xlb.__version__)" 2>/dev/null | grep -q "0\."; then
    say "xlb (uv venv)" "OK ($(uv run python -c 'import xlb; print(xlb.__version__)' 2>/dev/null))"
  else
    say "xlb (uv venv)" "MISS (run: uv sync)"; fail=1
  fi
else
  say "xlb (uv venv)" "MISS (run: uv sync)"; fail=1
fi

uv run python -c "import jax; print(jax.devices())" 2>/dev/null | grep -q "CpuDevice" && say "JAX CPU backend" "OK" || { say "JAX CPU backend" "MISS"; fail=1; }

[ -f skills/xlb-simulation/SKILL.md ] && say "skill (skills/)" "OK" || { say "skill (skills/)" "MISS"; fail=1; }
[ -f .github/skills/xlb-simulation/SKILL.md ] && say "skill (.github/skills/)" "OK" || { say "skill (.github/skills/)" "MISS"; fail=1; }
[ -f examples/couette/case.yaml ] && say "Couette case.yaml" "OK" || { say "Couette case.yaml" "MISS"; fail=1; }

if [ "$fail" -eq 0 ]; then echo "READY"; else echo "NOT READY — fix the MISS lines above"; fi
exit "$fail"
