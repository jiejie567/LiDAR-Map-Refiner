#!/usr/bin/env python3
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"MISSING ({exc})"
    output = (result.stdout or result.stderr).strip()
    return output or f"exit={result.returncode}"


def import_version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"MISSING ({exc})"
    if module_name == "PyQt5":
        from PyQt5 import QtCore  # type: ignore

        return f"{QtCore.PYQT_VERSION_STR} / Qt {QtCore.QT_VERSION_STR}"
    return str(getattr(module, "__version__", "unknown"))


def first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return "not found"


def main() -> int:
    print("== LiDAR Map Refiner: environment check ==")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print()

    print("[Python packages]")
    for name in ["open3d", "PyQt5", "numpy", "scipy", "matplotlib", "gtsam"]:
        print(f"  {name}: {import_version(name)}")
    print()

    print("[Python build toolchain]")
    print(f"  cmake: {run(['cmake', '--version']).splitlines()[0]}")
    print(f"  g++: {run(['g++', '--version']).splitlines()[0]}")
    print()

    print("[Common CMake package paths]")
    gtsam_candidates = [
        Path("/usr/local/lib/cmake/GTSAM/GTSAMConfig.cmake"),
        Path("/usr/lib/x86_64-linux-gnu/cmake/GTSAM/GTSAMConfig.cmake"),
    ]
    print(f"  GTSAMConfig.cmake: {first_existing(gtsam_candidates)}")
    print()

    print("[Python optimizer]")
    optimizer_cli = REPO_ROOT / "gui" / "manual_loop_closure" / "python_optimizer" / "cli.py"
    print(f"  optimizer cli: {first_existing([optimizer_cli])}")
    print("  gtsam wrapper helper:   bash scripts/install_gtsam_python.sh")
    print("  docker image (optional): docker build -t manual-loop-closure-tools:latest .")
    print("  GUI launch:    python launch_gui.py --session-root /path/to/session")
    print("  note:          The GUI and optimizer are Python-only; ROS is not required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
