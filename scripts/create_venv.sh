#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

echo "[manual-loop-closure] Repo root: $REPO_ROOT"
echo "[manual-loop-closure] Python: $PYTHON_BIN"
echo "[manual-loop-closure] Venv: $VENV_DIR"

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required.")

try:
    import ensurepip  # noqa: F401
except ModuleNotFoundError:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    raise SystemExit(
        "Python venv bootstrap is unavailable. On Ubuntu/Debian, install "
        f"python{version}-venv first, then rerun make venv."
    )
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

if [[ -n "${PYTHONPATH:-}" ]]; then
  echo "[manual-loop-closure] Clearing PYTHONPATH while installing venv packages."
fi

env -u PYTHONPATH PYTHONNOUSERSITE=1 python -m pip install --upgrade pip wheel setuptools
env -u PYTHONPATH PYTHONNOUSERSITE=1 python -m pip install -r "$REPO_ROOT/requirements.txt"

echo
echo "[manual-loop-closure] Virtual environment is ready."
echo "[manual-loop-closure] Activate it with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "[manual-loop-closure] Install the GTSAM 4.3 Python wrapper with:"
echo "  make gtsam-python"
echo "[manual-loop-closure] Then launch:"
echo "  env -u PYTHONPATH python \"$REPO_ROOT/launch_gui.py\" --session-root /path/to/session"
