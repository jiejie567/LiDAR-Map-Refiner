from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BACKEND_PREFERENCE_PYTHON = "python"
OPTIMIZE_MODE_LM = "lm"
OPTIMIZE_MODE_ISAM2 = "isam2"
OPTIMIZE_MODE_GNC_TLS = "gnc_tls"
OPTIMIZE_MODE_LM_HUBER = "lm_huber"
OPTIMIZE_MODE_LM_CHORDAL = "lm_chordal"


# ===== BEGIN CHANGE: optimizer backend adapter =====
@dataclass(frozen=True)
class OptimizerRunOptions:
    session_root: Path
    g2o_path: Path
    tum_path: Path
    keyframe_dir: Path
    constraints_csv: Path
    output_dir: Path
    map_voxel_leaf: float
    optimize_mode: str = OPTIMIZE_MODE_LM
    skip_map_build: bool = False
    manual_information_scale: float = 1.0
    loop_correlation_window_keyframes: int = 0
    loop_cluster_information_budget: float = 0.0
    loop_correlation_policy: str = "pair"
    loop_cluster_allocation: str = "equal"
    skip_graph_plot: bool = False

    def to_cli_args(self) -> list[str]:
        args = [
            "--session-root",
            str(self.session_root),
            "--g2o",
            str(self.g2o_path),
            "--tum",
            str(self.tum_path),
            "--keyframe-dir",
            str(self.keyframe_dir),
            "--constraints-csv",
            str(self.constraints_csv),
            "--output-dir",
            str(self.output_dir),
            "--map-voxel-leaf",
            f"{self.map_voxel_leaf:.6f}",
            "--optimize-mode",
            self.optimize_mode,
            "--manual-information-scale",
            f"{self.manual_information_scale:.12g}",
            "--loop-correlation-window-keyframes",
            str(self.loop_correlation_window_keyframes),
            "--loop-cluster-information-budget",
            f"{self.loop_cluster_information_budget:.12g}",
            "--loop-correlation-policy",
            self.loop_correlation_policy,
            "--loop-cluster-allocation",
            self.loop_cluster_allocation,
        ]
        if self.skip_map_build:
            args.append("--skip-map-build")
        if self.skip_graph_plot:
            args.append("--skip-graph-plot")
        return args


@dataclass(frozen=True)
class OptimizerRunResult:
    output_dir: Path
    output_g2o: Path
    output_tum: Path
    output_map_pcd: Path
    output_trajectory_pcd: Path
    output_report_json: Path
    factor_count: int
    pose_count: int
    enabled_constraints: int
    map_built: bool = True


@dataclass(frozen=True)
class ResolvedOptimizerBackend:
    key: str
    display_name: str
    program: str
    arguments_prefix: tuple[str, ...]

    def build_process_args(self, options: OptimizerRunOptions) -> list[str]:
        return [*self.arguments_prefix, *options.to_cli_args()]


def _supports_imports(python_executable: Path, import_statement: str) -> bool:
    try:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                import_statement,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _supports_python_optimizer(python_executable: Path) -> bool:
    return _supports_imports(python_executable, "import gtsam, numpy, scipy")


def _candidate_python_executables(project_root: Optional[Path]) -> list[Path]:
    candidates: list[Path] = []
    current = Path(sys.executable).expanduser()

    def append_candidate(path: Optional[Path]) -> None:
        if path is None:
            return
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = candidate.resolve()
        if not candidate.is_file():
            return
        if candidate not in candidates:
            candidates.append(candidate)

    append_candidate(Path(os.environ["MANUAL_LOOP_OPTIMIZER_PYTHON"])) if os.environ.get(
        "MANUAL_LOOP_OPTIMIZER_PYTHON"
    ) else None
    append_candidate(current)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        append_candidate(Path(conda_prefix) / "bin" / "python3")
        append_candidate(Path(conda_prefix) / "bin" / "python")

    if project_root is not None:
        append_candidate(project_root / ".venv" / "bin" / "python3")
        append_candidate(project_root / ".venv" / "bin" / "python")

    repo_root = os.environ.get("MANUAL_LOOP_CLOSURE_REPO")
    if repo_root:
        repo_path = Path(repo_root)
        append_candidate(repo_path / ".venv" / "bin" / "python3")
        append_candidate(repo_path / ".venv" / "bin" / "python")

    default_standalone_repo = Path.home() / "my_git" / "Mannual-Loop-Closure-Tools"
    append_candidate(default_standalone_repo / ".venv" / "bin" / "python3")
    append_candidate(default_standalone_repo / ".venv" / "bin" / "python")

    validated_workspace_repo = (
        Path.home() / "slam_repo" / "Manual-Loop-Closure-Tools")
    append_candidate(validated_workspace_repo / ".venv" / "bin" / "python3")
    append_candidate(validated_workspace_repo / ".venv" / "bin" / "python")

    home = Path.home()
    for base_dir in (home / "anaconda3", home / "miniconda3"):
        append_candidate(base_dir / "bin" / "python3")
        append_candidate(base_dir / "bin" / "python")

    return candidates


def resolve_python_optimizer_backend(
    *,
    script_dir: Path,
    project_root: Optional[Path],
) -> Optional[ResolvedOptimizerBackend]:
    cli_script = script_dir / "manual_loop_closure" / "python_optimizer" / "cli.py"
    if not cli_script.is_file():
        return None

    for candidate in _candidate_python_executables(project_root):
        if not _supports_python_optimizer(candidate):
            continue
        return ResolvedOptimizerBackend(
            key=BACKEND_PREFERENCE_PYTHON,
            display_name=f"python:{candidate}",
            program=str(candidate),
            arguments_prefix=("-u", str(cli_script)),
        )
    return None
# ===== END CHANGE: optimizer backend adapter =====


# ===== BEGIN CHANGE: balm backend adapter =====
@dataclass(frozen=True)
class BalmRunOptions:
    tum_path: Path
    keyframe_dir: Path
    output_dir: Path
    source_run_dir: Optional[Path] = None
    root_voxel_size: float = 1.0
    max_iterations: int = 20
    coarse_plane_thickness: float = 0.12
    downsample_leaf: float = 0.2
    max_range: float = 80.0
    max_observation_range: float = 20.0
    double_sided_enable: bool = True
    diagnose_map_inconsistency: bool = False

    def to_cli_args(self) -> list[str]:
        args = [
            "--tum",
            str(self.tum_path),
            "--keyframe-dir",
            str(self.keyframe_dir),
            "--output-dir",
            str(self.output_dir),
            "--root-voxel",
            f"{self.root_voxel_size:.6f}",
            "--max-iterations",
            str(self.max_iterations),
            "--coarse-plane-thickness",
            f"{self.coarse_plane_thickness:.6f}",
            "--downsample-leaf",
            f"{self.downsample_leaf:.6f}",
            "--max-range",
            f"{self.max_range:.6f}",
            "--max-observation-range",
            f"{self.max_observation_range:.6f}",
            "--double-sided" if self.double_sided_enable else "--no-double-sided",
        ]
        if self.source_run_dir is not None:
            args.extend(["--source-run-dir", str(self.source_run_dir)])
        if self.diagnose_map_inconsistency:
            args.append("--diagnose-map-inconsistency")
        return args


def resolve_python_balm_backend(
    *,
    script_dir: Path,
    project_root: Optional[Path],
) -> Optional[ResolvedOptimizerBackend]:
    cli_script = script_dir / "manual_loop_closure" / "python_optimizer" / "balm_cli.py"
    if not cli_script.is_file():
        return None

    for candidate in _candidate_python_executables(project_root):
        if not _supports_imports(candidate, "import numpy, scipy"):
            continue
        return ResolvedOptimizerBackend(
            key="balm-python",
            display_name=f"balm:{candidate}",
            program=str(candidate),
            arguments_prefix=("-u", str(cli_script)),
        )
    return None
# ===== END CHANGE: balm backend adapter =====
