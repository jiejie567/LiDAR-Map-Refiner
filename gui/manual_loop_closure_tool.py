#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

INITIAL_LOOP_SEARCH_LABEL = "Initial Loop Search"


def _gui_compute_marker_path() -> Path:
    """PID-owned marker used by formal replay's compute-quiet gate."""
    override = os.environ.get("GHOSTLOOP_GUI_COMPUTE_DIR")
    if override:
        root = Path(override)
    else:
        runtime = os.environ.get("XDG_RUNTIME_DIR")
        root = Path(runtime) if runtime else Path("/tmp") / f"ghostloop-{os.getuid()}"
        root = root / "ghostloop-gui-compute"
    return root / f"{os.getpid()}.active"


def _python_supports_manual_loop_dependencies(python_executable: Path) -> bool:
    try:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                "import open3d, PyQt5, numpy, scipy",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _python_supports_canonical_repair(python_executable: Path) -> bool:
    """Require the union of GUI geometry and headless PGO dependencies."""
    try:
        result = subprocess.run(
            [
                str(python_executable),
                "-c",
                "import open3d, numpy, scipy, gtsam",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _candidate_python_executables() -> list[Path]:
    # A venv's Python is commonly a symlink to the system binary. Preserve the
    # invocation path: resolving it erases pyvenv.cfg discovery and turns a
    # valid venv candidate back into the dependency-missing system Python.
    current = Path(sys.executable).absolute()
    candidates: list[Path] = []

    def append_candidate(path: Path) -> None:
        candidate = path.expanduser().absolute()
        if candidate == current or not candidate.is_file():
            return
        if candidate not in candidates:
            candidates.append(candidate)

    override = os.environ.get("MANUAL_LOOP_GUI_PYTHON")
    if override:
        append_candidate(Path(override))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        append_candidate(Path(conda_prefix) / "bin" / "python3")
        append_candidate(Path(conda_prefix) / "bin" / "python")

    home = Path.home()
    for base_dir in (home / "anaconda3", home / "miniconda3"):
        append_candidate(base_dir / "bin" / "python3")
        append_candidate(base_dir / "bin" / "python")

    repo_venv = Path(__file__).resolve().parent.parent / ".venv"
    append_candidate(repo_venv / "bin" / "python3")
    append_candidate(repo_venv / "bin" / "python")

    # Compatibility with the validated standalone installation documented by
    # this workspace's root README. This lets the simple `python3 launch_gui.py`
    # entry point recover automatically on machines where Open3D lives there.
    standalone_venv = (
        home / "slam_repo" / "Manual-Loop-Closure-Tools" / ".venv")
    append_candidate(standalone_venv / "bin" / "python3")
    append_candidate(standalone_venv / "bin" / "python")

    return candidates


def _bootstrap_python_environment() -> None:
    if importlib.util.find_spec("open3d") is not None:
        return
    if os.environ.get("MS_MANUAL_LOOP_REEXEC") == "1":
        return

    for candidate in _candidate_python_executables():
        if not _python_supports_manual_loop_dependencies(candidate):
            continue
        os.environ["MS_MANUAL_LOOP_REEXEC"] = "1"
        print(
            f"[manual_loop_closure_tool] Re-launching with {candidate} because "
            f"{sys.executable} cannot import open3d.",
            file=sys.stderr,
        )
        os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])


_bootstrap_python_environment()

import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets


class BackgroundTask(QtCore.QObject):
    """Run one CPU-heavy callable without blocking the Qt event loop."""

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int, int)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def report_progress(self, text: str, current: int = 0, total: int = 0) -> None:
        """Thread-safe progress callback passed to the worker callable."""
        self.progress.emit(str(text), int(current), int(total))

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(self._fn(self.report_progress))
        except Exception:
            self.failed.emit(traceback.format_exc())

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from manual_loop_closure import (  # noqa: E402
        EdgeRecord,
        OFFICE_DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
        OFFICE_DEFAULT_MAX_ITERATIONS,
        OFFICE_DEFAULT_TARGET_CLOUD_MODE,
        OFFICE_DEFAULT_TARGET_MAP_VOXEL_SIZE,
        OFFICE_DEFAULT_TARGET_MIN_TIME_GAP_SEC,
        OFFICE_DEFAULT_TARGET_NEIGHBORS,
        OFFICE_DEFAULT_VARIANCE_R_RAD2,
        OFFICE_DEFAULT_VARIANCE_T,
        OFFICE_DEFAULT_VOXEL_SIZE,
        PoseGraphData,
        PoseGraphValidationError,
        RegistrationConfig,
        RegistrationPreview,
        RegistrationResult,
        RegistrationWorkspace,
        SessionPaths,
        SessionResolutionError,
        TARGET_CLOUD_MODE_RS_SPATIAL_SUBMAP,
        TARGET_CLOUD_MODE_TEMPORAL_WINDOW,
        TrajectoryData,
        TrajectoryValidationError,
        align_pose_graph_to_frame_count,
        load_pose_graph,
        load_tum_trajectory,
        list_numbered_pcds,
        matrix_to_quat_xyzw,
        matrix_to_xyz_rpy_deg,
        resolve_session_paths,
        write_filtered_pose_graph,
    )
    from manual_loop_closure.scan_context_io import (  # noqa: E402
        _gravity_canonical_rotation,
        load_scan_context_config,
        load_scan_context_gravity,
    )
    from manual_loop_closure.seed_loops import (  # noqa: E402
        build_descriptor_stack,
        descriptor_env_setup,
        find_seed_candidates,
        ghost_badness,
        pair_yaw_peaks,
    )
    from manual_loop_closure.python_optimizer.balm import (  # noqa: E402
        BalmParams,
    )
    from manual_loop_closure.concurrency import hypothesis_workers  # noqa: E402
    import manual_loop_closure.registration as _registration  # noqa: E402
    from manual_loop_closure.open3d_viewer import (  # noqa: E402
        INTERACTION_MODE_CAMERA,
        INTERACTION_MODE_EDIT_SOURCE,
        EDIT_OPERATION_ROTATE,
        EDIT_OPERATION_TRANSLATE,
        LOCK_MODE_XY_YAW,
        LOCK_MODE_Z_ONLY,
        LOCK_MODE_XZ_PITCH,
        LOCK_MODE_YZ_ROLL,
        ManualAlignUpdate,
        EmbeddedOpen3DWidget,
        PreviewScene,
        VIEW_PRESET_SIDE_X,
        VIEW_PRESET_SIDE_Y,
        VIEW_PRESET_TOP,
    )
    # ===== BEGIN CHANGE: optimizer backend imports =====
    from manual_loop_closure.optimizer_backend import (  # noqa: E402
        OPTIMIZE_MODE_GNC_TLS,
        OPTIMIZE_MODE_ISAM2,
        OPTIMIZE_MODE_LM,
        BalmRunOptions,
        OptimizerRunOptions,
        resolve_python_balm_backend,
        resolve_python_optimizer_backend,
    )
    # ===== END CHANGE: optimizer backend imports =====
    from manual_loop_closure.pcd_io import (  # noqa: E402
        PcdValidationError,
        load_xyz_points,
        validate_keyframe_numbering,
    )
    from manual_loop_closure.registration import (  # noqa: E402
        build_delta_transform,
        transform_points,
    )
    from manual_loop_closure.repair_presets import (  # noqa: E402
        NEAR_DUPLICATE_KEYFRAMES,
        PRODUCTION_PGO_PROFILE,
        REPAIR_ENV_PRESETS,
        REPAIR_GATE,
        REPAIR_POLICY,
        production_loop_variances,
    )
    from manual_loop_closure.python_optimizer.exporters import (  # noqa: E402
        build_scan_context_from_tum,
        build_map_and_trajectory_from_tum,
        colorize_binary_xyzi_pcd,
        update_report_map_fields,
    )
except ModuleNotFoundError as exc:
    if exc.name != "open3d":
        raise
    raise SystemExit(
        "Failed to import Python package 'open3d'. The tool already tried to re-launch "
        f"with a compatible Python, but none worked. Current interpreter: {sys.executable}\n"
        "Install the tested dependencies first, then try again. On Ubuntu 24.04, "
        "install python3-pip and python3.12-venv if this Python has no pip. Example:\n"
        "  conda env create -f environment.yml\n"
        "  conda activate manual-loop-closure\n"
        "  python launch_gui.py --session-root /path/to/session\n"
        "Or set MANUAL_LOOP_GUI_PYTHON=/path/to/venv/bin/python."
    ) from exc


@dataclass(frozen=True)
class SelectedEdgeRef:
    edge_kind: str
    edge_uid: int


@dataclass
class ManualConstraint:
    manual_uid: int
    enabled: bool
    source_id: int
    target_id: int
    target_cloud_mode: str
    target_neighbors: int
    min_time_gap_sec: float
    target_map_voxel_size: float
    transform_world_source_final: np.ndarray
    transform_target_source_final: np.ndarray
    source_points_world_final: np.ndarray
    fitness: float
    inlier_rmse: float
    variance_t_m2: tuple[float, float, float]
    variance_r_rad2: tuple[float, float, float]
    replaces_edge_uid: Optional[int] = None
    accepted_rev: int = 0
    applied_rev: Optional[int] = None
    note: str = ""

    def csv_row(self) -> list[str]:
        translation = self.transform_target_source_final[:3, 3]
        quat_xyzw = matrix_to_quat_xyzw(self.transform_target_source_final)
        sigma_t = [math.sqrt(max(value, 0.0)) for value in self.variance_t_m2]
        # The UI and project file store rotation variance in rad^2, while the
        # optimizer's legacy CSV columns are explicitly degrees.  The old code
        # omitted this conversion, turning 0.316 rad into 0.316 degrees and
        # making every loop about 57x more certain in rotation than requested.
        sigma_r = [
            math.degrees(math.sqrt(max(value, 0.0)))
            for value in self.variance_r_rad2
        ]
        return [
            "1" if self.enabled else "0",
            str(self.source_id),
            str(self.target_id),
            f"{translation[0]:.12f}",
            f"{translation[1]:.12f}",
            f"{translation[2]:.12f}",
            f"{quat_xyzw[0]:.12f}",
            f"{quat_xyzw[1]:.12f}",
            f"{quat_xyzw[2]:.12f}",
            f"{quat_xyzw[3]:.12f}",
            f"{sigma_t[0]:.6f}",
            f"{sigma_t[1]:.6f}",
            f"{sigma_t[2]:.6f}",
            f"{sigma_r[0]:.6f}",
            f"{sigma_r[1]:.6f}",
            f"{sigma_r[2]:.6f}",
        ]

    def as_edge_record(self) -> EdgeRecord:
        return EdgeRecord(
            edge_uid=self.manual_uid,
            source_id=self.source_id,
            target_id=self.target_id,
            edge_type="manual_added",
            tag="MANUAL",
            line_index=None,
            raw_line=None,
            enabled=self.enabled,
            deletable=True,
        )


@dataclass
class ExistingLoopChange:
    edge_uid: int
    enabled: bool = True
    accepted_rev: int = 0
    applied_rev: Optional[int] = None
    note: str = ""


@dataclass
class UndoSnapshot:
    pose_graph: Optional[PoseGraphData]
    trajectory: Optional[TrajectoryData]
    constraints: list[ManualConstraint]
    disabled_loop_changes: dict[int, ExistingLoopChange]
    source_id: Optional[int]
    target_id: Optional[int]
    selected_edge_ref: Optional[SelectedEdgeRef]
    candidate_replace_edge_uid: Optional[int]
    working_revision: int
    session_dirty: bool
    last_output_dir: Optional[Path]
    pick_mode: str


class TrajectoryCanvas(FigureCanvas):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self._figure = Figure(figsize=(8.0, 6.0))
        self._axes = self._figure.add_subplot(111)
        self._figure.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.95)
        super().__init__(self._figure)
        self.setParent(parent)

        self._positions_xy: Optional[np.ndarray] = None
        self._ghost_positions_xy: Optional[np.ndarray] = None
        self._selected_target_positions_xy: Optional[np.ndarray] = None
        self._pose_graph: Optional[PoseGraphData] = None
        self._constraints: list[ManualConstraint] = []
        self._source_id: Optional[int] = None
        self._target_id: Optional[int] = None
        self._selected_edge_ref: Optional[SelectedEdgeRef] = None
        self._interaction_mode = "nodes"
        self._select_callback = None
        self._hover_callback = None
        self._toolbar: Optional[NavigationToolbar] = None

        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_toolbar(self, toolbar: NavigationToolbar) -> None:
        self._toolbar = toolbar

    def set_callbacks(self, select_callback, hover_callback) -> None:
        self._select_callback = select_callback
        self._hover_callback = hover_callback

    def set_interaction_mode(self, mode: str) -> None:
        self._interaction_mode = mode

    def set_plot_data(
        self,
        *,
        positions_xy: Optional[np.ndarray],
        ghost_positions_xy: Optional[np.ndarray],
        selected_target_positions_xy: Optional[np.ndarray],
        pose_graph: Optional[PoseGraphData],
        constraints: list[ManualConstraint],
        source_id: Optional[int],
        target_id: Optional[int],
        selected_edge_ref: Optional[SelectedEdgeRef],
        preserve_view: bool,
    ) -> None:
        self._positions_xy = positions_xy
        self._ghost_positions_xy = ghost_positions_xy
        self._selected_target_positions_xy = selected_target_positions_xy
        self._pose_graph = pose_graph
        self._constraints = constraints
        self._source_id = source_id
        self._target_id = target_id
        self._selected_edge_ref = selected_edge_ref
        self.redraw(preserve_view=preserve_view)

    def fit_view(self) -> None:
        self.redraw(preserve_view=False)

    def redraw(self, *, preserve_view: bool) -> None:
        previous_limits = None
        if preserve_view and self._axes.has_data():
            previous_limits = (self._axes.get_xlim(), self._axes.get_ylim())

        self._axes.clear()
        self._axes.set_title("Pose Graph")
        self._axes.set_xlabel("X [m]")
        self._axes.set_ylabel("Y [m]")
        self._axes.grid(True, linestyle="--", alpha=0.3)
        self._axes.set_aspect("equal", adjustable="box")

        if self._positions_xy is None or self._pose_graph is None:
            self.draw_idle()
            return

        positions = self._positions_xy
        if self._ghost_positions_xy is not None and len(self._ghost_positions_xy) > 1:
            self._axes.plot(
                self._ghost_positions_xy[:, 0],
                self._ghost_positions_xy[:, 1],
                color="#94a3b8",
                linewidth=1.2,
                alpha=0.55,
                linestyle="--",
                label="Reference trajectory",
            )
        self._axes.scatter(
            positions[:, 0],
            positions[:, 1],
            s=8,
            c="#666666",
            alpha=0.4,
        )

        if len(positions) > 1:
            self._axes.plot(
                positions[:, 0],
                positions[:, 1],
                color="#4c78a8",
                linewidth=1.0,
                alpha=0.75,
                label="Trajectory",
            )

        if (
            self._selected_target_positions_xy is not None
            and len(self._selected_target_positions_xy) > 0
        ):
            selected = self._selected_target_positions_xy
            self._axes.scatter(
                selected[:, 0],
                selected[:, 1],
                s=26,
                c="#f6d32d",
                alpha=0.95,
                edgecolors="black",
                linewidths=0.3,
                label="Selected target frames",
                zorder=4,
            )

        enabled_loops = [edge for edge in self._pose_graph.loop_edges if edge.enabled]
        disabled_loops = [edge for edge in self._pose_graph.loop_edges if not edge.enabled]
        for edge in enabled_loops:
            self._draw_edge(edge, "#d62728", linewidth=1.3, alpha=0.85)
        if enabled_loops:
            self._axes.plot([], [], color="#d62728", linewidth=1.3, label="Existing loop")

        for edge in disabled_loops:
            self._draw_edge(edge, "#8a8a8a", linewidth=1.2, alpha=0.7, linestyle="--")
        if disabled_loops:
            self._axes.plot([], [], color="#8a8a8a", linewidth=1.2, linestyle="--", label="Disabled loop")

        enabled_manual = [constraint for constraint in self._constraints if constraint.enabled]
        disabled_manual = [constraint for constraint in self._constraints if not constraint.enabled]
        for constraint in enabled_manual:
            self._draw_edge(constraint.as_edge_record(), "#2ca02c", linewidth=1.8, alpha=0.95)
        if enabled_manual:
            self._axes.plot([], [], color="#2ca02c", linewidth=1.8, label="Manual loop")

        for constraint in disabled_manual:
            self._draw_edge(constraint.as_edge_record(), "#7f7f7f", linewidth=1.4, alpha=0.7, linestyle="--")
        if disabled_manual:
            self._axes.plot([], [], color="#7f7f7f", linewidth=1.4, linestyle="--", label="Disabled manual")

        if self._source_id is not None and self._target_id is not None:
            self._axes.plot(
                [positions[self._source_id, 0], positions[self._target_id, 0]],
                [positions[self._source_id, 1], positions[self._target_id, 1]],
                color="#bc5090",
                linewidth=2.0,
                alpha=0.95,
                label="Active pair",
            )

        selected_edge = self._edge_from_ref(self._selected_edge_ref)
        if selected_edge is not None:
            highlight_color = "#ffd166"
            if selected_edge.edge_type == "odom":
                highlight_color = "#ff9f1c"
            elif selected_edge.edge_type == "manual_added":
                highlight_color = "#00f5d4"
            self._draw_edge(
                selected_edge,
                highlight_color,
                linewidth=3.2,
                alpha=1.0,
                linestyle="--" if not selected_edge.enabled else "-",
            )

        if self._source_id is not None:
            source = positions[self._source_id]
            self._axes.scatter(
                [source[0]],
                [source[1]],
                c="#ff7f0e",
                s=90,
                marker="*",
                edgecolors="black",
                label="Source",
                zorder=5,
            )
        if self._target_id is not None:
            target = positions[self._target_id]
            self._axes.scatter(
                [target[0]],
                [target[1]],
                c="#17becf",
                s=64,
                marker="o",
                edgecolors="black",
                label="Target",
                zorder=5,
            )

        if previous_limits is not None:
            self._axes.set_xlim(previous_limits[0])
            self._axes.set_ylim(previous_limits[1])
        self.draw_idle()

    def _draw_edge(
        self,
        edge: EdgeRecord,
        color: str,
        *,
        linewidth: float,
        alpha: float,
        linestyle: str = "-",
    ) -> None:
        if self._positions_xy is None:
            return
        positions = self._positions_xy
        self._axes.plot(
            [positions[edge.source_id, 0], positions[edge.target_id, 0]],
            [positions[edge.source_id, 1], positions[edge.target_id, 1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
        )

    def _manual_edges(self) -> list[EdgeRecord]:
        return [constraint.as_edge_record() for constraint in self._constraints]

    def _edge_from_ref(self, edge_ref: Optional[SelectedEdgeRef]) -> Optional[EdgeRecord]:
        if edge_ref is None or self._pose_graph is None:
            return None
        if edge_ref.edge_kind == "existing":
            return self._pose_graph.edge_index_by_uid.get(edge_ref.edge_uid)
        for edge in self._manual_edges():
            if edge.edge_uid == edge_ref.edge_uid:
                return edge
        return None

    def _nearest_node(self, event) -> Optional[int]:
        if self._positions_xy is None or event.xdata is None or event.ydata is None:
            return None

        positions = self._positions_xy
        click = np.asarray([event.xdata, event.ydata], dtype=np.float64)
        deltas = positions - click
        distances = np.linalg.norm(deltas, axis=1)
        index = int(np.argmin(distances))

        global_span = max(float(np.ptp(positions[:, 0])), float(np.ptp(positions[:, 1])), 1.0)
        x_limits = self._axes.get_xlim()
        y_limits = self._axes.get_ylim()
        visible_span = max(
            abs(float(x_limits[1] - x_limits[0])),
            abs(float(y_limits[1] - y_limits[0])),
            1.0,
        )
        max_distance = min(0.02 * global_span, max(0.08, 0.02 * visible_span))
        if float(distances[index]) > max_distance:
            return None
        return index

    def _nearest_edge(self, event) -> Optional[SelectedEdgeRef]:
        if self._positions_xy is None or self._pose_graph is None or event.xdata is None or event.ydata is None:
            return None

        point = np.asarray([event.xdata, event.ydata], dtype=np.float64)
        best_ref: Optional[SelectedEdgeRef] = None
        best_distance = float("inf")

        all_edges: list[tuple[SelectedEdgeRef, EdgeRecord]] = [
            (SelectedEdgeRef("existing", edge.edge_uid), edge)
            for edge in self._pose_graph.edge_records
        ]
        all_edges.extend(
            (SelectedEdgeRef("manual_added", edge.edge_uid), edge)
            for edge in self._manual_edges()
        )

        for edge_ref, edge in all_edges:
            segment_start = self._positions_xy[edge.source_id]
            segment_end = self._positions_xy[edge.target_id]
            distance = _point_to_segment_distance(point, segment_start, segment_end)
            if distance < best_distance:
                best_distance = distance
                best_ref = edge_ref

        x_limits = self._axes.get_xlim()
        y_limits = self._axes.get_ylim()
        visible_span = max(
            abs(float(x_limits[1] - x_limits[0])),
            abs(float(y_limits[1] - y_limits[0])),
            1.0,
        )
        if best_distance > max(0.12, 0.015 * visible_span):
            return None
        return best_ref

    def _on_click(self, event) -> None:
        if self._toolbar is not None and self._toolbar.mode:
            return
        if event.inaxes != self._axes or self._select_callback is None:
            return

        if self._interaction_mode == "edges":
            edge_ref = self._nearest_edge(event)
            if edge_ref is not None:
                self._select_callback("edge", edge_ref)
            return

        node_id = self._nearest_node(event)
        if node_id is not None:
            self._select_callback("node", node_id)

    def _on_motion(self, event) -> None:
        if self._hover_callback is None or event.inaxes != self._axes:
            return

        if self._interaction_mode == "edges":
            self._hover_callback("edge", self._nearest_edge(event))
        else:
            self._hover_callback("node", self._nearest_node(event))


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / denom)
    projection = max(0.0, min(1.0, projection))
    nearest = start + projection * segment
    return float(np.linalg.norm(point - nearest))


def _gravity_error_deg(
    measured_target_source: np.ndarray,
    source_up_local: np.ndarray | None,
    target_up_local: np.ndarray | None,
) -> float | None:
    """Yaw-invariant roll/pitch disagreement for a registered loop."""
    if source_up_local is None or target_up_local is None:
        return None
    source = np.asarray(source_up_local, dtype=np.float64).reshape(3)
    target = np.asarray(target_up_local, dtype=np.float64).reshape(3)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if (
        not np.isfinite(source).all() or not np.isfinite(target).all()
        or source_norm < 1e-9 or target_norm < 1e-9
    ):
        return None
    predicted = (
        np.asarray(measured_target_source, dtype=np.float64)[:3, :3]
        @ (source / source_norm)
    )
    cosine = float(np.clip(
        np.dot(predicted, target / target_norm), -1.0, 1.0
    ))
    return float(np.degrees(np.arccos(cosine)))


def _compact_registration_result(result: RegistrationResult | None):
    """Drop render-only point arrays before queueing an automatic result."""
    if result is None:
        return None
    empty = np.empty((0, 3), dtype=np.float64)
    preview = replace(
        result.preview,
        target_points_world=empty,
        source_points_local=empty,
        source_points_world_initial=empty,
        source_points_world_adjusted=empty,
    )
    return replace(
        result,
        preview=preview,
        source_points_world_final=empty,
    )


# Rendering is the one preview step that must run on Qt's main thread.  The
# repeated large-session A/B benchmark found that reducing a 744k-point indoor
# target to ~149k shortens its Open3D upload, while striding an already compact
# 267k outdoor target is slightly slower because it loses the contiguous-array
# fast path.  Keep compact targets intact and cap only genuinely large maps;
# RegistrationPreview still retains every point for GICP/metrics.
# The optimizer CLI's constraint-file columns. Written from two places now --
# the working solve and the consistency leave-one-out trials -- so it lives in
# one place rather than as two literals that can drift apart.
CONSTRAINT_CSV_HEADER = [
    "enabled", "source_id", "target_id", "tx", "ty", "tz",
    "qx", "qy", "qz", "qw", "sigma_tx", "sigma_ty", "sigma_tz",
    "sigma_roll_deg", "sigma_pitch_deg", "sigma_yaw_deg",
]

PREVIEW_TARGET_DISPLAY_LIMIT = 180_000
PREVIEW_TARGET_DISPLAY_CAP_THRESHOLD = 280_000
PREVIEW_SOURCE_DISPLAY_LIMIT = 150_000


def _bounded_display_view(points: Optional[np.ndarray], limit: int) -> Optional[np.ndarray]:
    """Return a deterministic strided view for rendering, never for GICP.

    Open3D geometry upload runs on Qt's GUI thread. Multi-million-point target
    maps can therefore freeze an otherwise asynchronous preview. A slice keeps
    spatial coverage without copying, while RegistrationPreview retains every
    point for registration and quality metrics.
    """
    if points is None or points.shape[0] <= limit:
        return points
    stride = max(1, math.ceil(points.shape[0] / limit))
    return points[::stride]


def _bounded_target_display_view(
    points: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if points is None or points.shape[0] <= PREVIEW_TARGET_DISPLAY_CAP_THRESHOLD:
        return points
    return _bounded_display_view(points, PREVIEW_TARGET_DISPLAY_LIMIT)


class ManualLoopClosureWindow(QtWidgets.QMainWindow):
    MAX_UNDO_SNAPSHOTS = 20

    def __init__(
        self,
        *,
        initial_session_root: Optional[Path] = None,
        initial_g2o_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("LiDAR Map Refiner")
        self.setMinimumSize(760, 560)
        self._settings = QtCore.QSettings("JokerJohn", "ManualLoopClosureTools")

        self.session_paths: Optional[SessionPaths] = None
        self.original_pose_graph: Optional[PoseGraphData] = None
        self.original_trajectory: Optional[TrajectoryData] = None
        self.pose_graph: Optional[PoseGraphData] = None
        self.trajectory: Optional[TrajectoryData] = None
        self.workspace: Optional[RegistrationWorkspace] = None
        self.keyframe_paths: list[Path] = []

        self.source_id: Optional[int] = None
        self.target_id: Optional[int] = None
        self.selected_edge_ref: Optional[SelectedEdgeRef] = None
        self.constraints: list[ManualConstraint] = []
        self.disabled_loop_changes: dict[int, ExistingLoopChange] = {}
        self.current_preview: Optional[RegistrationPreview] = None
        self.last_result: Optional[RegistrationResult] = None
        self.pick_mode = "nodes"
        self._table_updating = False
        self._optimizer_process: Optional[QtCore.QProcess] = None
        self._last_output_dir: Optional[Path] = None
        self._preview_scene_key: Optional[tuple] = None
        self._next_manual_uid = 1
        self._pending_preview_reset_camera = False
        self._working_revision = 0
        self._session_dirty = False
        self._pending_export_after_optimize = False
        self._undo_stack: list[UndoSnapshot] = []
        self._graph_change_rows: list[tuple[str, int]] = []
        self._pre_optimize_snapshot: Optional[UndoSnapshot] = None
        self._graph_change_status_filter = "All Status"
        self._graph_change_type_filter = "All Types"
        self._candidate_replace_edge_uid: Optional[int] = None
        self._active_optimizer_backend_name = ""
        self._optimizer_started_at: Optional[float] = None
        self._project_dir: Optional[Path] = None
        self._project_state_path: Optional[Path] = None
        self._project_log_path: Optional[Path] = None
        self._project_ops_path: Optional[Path] = None
        self._project_id: Optional[str] = None
        self._latest_export_dir: Optional[Path] = None
        self._requested_project_dir: Optional[Path] = None
        self._balm_ghost_suggestions: list[dict] = []
        # Map inconsistency as the last BALM pass left it, and what that pass
        # was given to refine -- the two inputs to the sanity gate in
        # _balm_output_is_sane. Both are per-session state; see the reset in
        # _load_session_preloaded.
        self._prev_ghost_badness: Optional[float] = None
        self._balm_source_run_dir: Optional[Path] = None
        self._repair_map_stage: Optional[str] = None
        self._repair_map_stop_requested = False
        self._canonical_repair_contract: Optional[Path] = None
        self._canonical_repair_environment: Optional[str] = None
        self._canonical_repair_process_grouped = False
        self._canonical_repair_cancelled = False
        self._canonical_repair_snapshot: Optional[UndoSnapshot] = None
        self._balm_cancel_requested = False
        self._seed_probation: Optional[dict] = None
        self._background_thread: Optional[QtCore.QThread] = None
        self._background_worker: Optional[BackgroundTask] = None
        self._background_label = ""
        self._background_result = None
        self._background_error: Optional[str] = None
        self._background_on_finished = None
        self._preview_refresh_queued = False
        self._preloaded_session_payload = None

        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self.refresh_preview)

        self._optimizer_heartbeat_timer = QtCore.QTimer(self)
        self._optimizer_heartbeat_timer.setInterval(5000)
        self._optimizer_heartbeat_timer.timeout.connect(self._on_optimizer_heartbeat)

        self._build_ui()
        # The combo box starts on Indoor before its signal is connected, so no
        # currentIndexChanged event is emitted on startup. Apply the visible
        # preset explicitly; otherwise the legacy registration default (voxel
        # 0.0) leaks into Repair Map until the user toggles the environment.
        #
        # _apply_environment_preset is also where _balm_downsample_leaf and
        # _balm_max_range come into existence, so a window that skips it cannot
        # reach Repair Map at all -- which settles the question of whether any
        # past repair ran without the preset: none did.
        self._apply_environment_preset()
        # Keep safe widget values available for manual preview, but do not let
        # a fresh session silently inherit "Indoor" merely because it was the
        # first combo-box item.  Loading a manifest/project restores a recorded
        # choice; otherwise Repair Map requires the operator to choose once.
        self.env_combo.setCurrentIndex(-1)
        self._apply_initial_window_geometry()
        self._apply_styles()
        self._configure_plot_toolbar()
        self._configure_button_cursors()

        self._restore_input_history(
            initial_session_root=initial_session_root,
            initial_g2o_path=initial_g2o_path,
        )

        self._update_plot_help_state()
        self._update_trajectory_legend_label()
        self._update_cloud_display_controls()

    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        root_layout = QtWidgets.QVBoxLayout(central_widget)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        root_layout.addWidget(self._build_loader_group())

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(10)
        self.plot_panel = self._build_plot_panel()
        self.cloud_panel = self._build_cloud_panel()
        self.control_scroll_area = self._build_control_scroll_area()
        self.plot_panel.setMinimumWidth(500)
        self.cloud_panel.setMinimumWidth(360)
        self.content_splitter.addWidget(self.plot_panel)
        self.content_splitter.addWidget(self.cloud_panel)
        self.content_splitter.addWidget(self.control_scroll_area)
        self.content_splitter.setStretchFactor(0, 6)
        self.content_splitter.setStretchFactor(1, 5)
        self.content_splitter.setStretchFactor(2, 0)
        self.content_splitter.setSizes([560, 460, 320])

        bottom_tabs = QtWidgets.QTabWidget()
        bottom_tabs.addTab(self._build_constraint_tab(), "Graph Changes")
        bottom_tabs.addTab(self._build_log_tab(), "Execution Log")

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(self.content_splitter)
        main_splitter.addWidget(bottom_tabs)
        main_splitter.setStretchFactor(0, 7)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setSizes([820, 240])
        root_layout.addWidget(main_splitter, stretch=1)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f4f7fb;
                color: #1f2937;
                font-size: 12px;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d9e1ea;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                background: #ffffff;
                border: 1px solid #ccd6e2;
                border-radius: 6px;
                padding: 2px 6px;
                min-height: 24px;
            }
            QPushButton, QToolButton {
                background: #ffffff;
                border: 1px solid #ccd6e2;
                border-radius: 6px;
                padding: 4px 8px;
                min-height: 24px;
            }
            QPushButton:hover, QToolButton:hover {
                background: #eef4ff;
                border-color: #9db6d8;
            }
            QPushButton:pressed, QToolButton:pressed {
                background: #dbeafe;
                border-color: #60a5fa;
                color: #0f172a;
            }
            QPushButton:checked, QToolButton:checked {
                background: #dbeafe;
                border-color: #60a5fa;
                color: #1d4ed8;
                font-weight: 600;
            }
            QPushButton:disabled, QToolButton:disabled {
                color: #98a4b3;
                background: #f8fafc;
            }
            QPushButton[buttonRole="primary"], QToolButton[buttonRole="primary"] {
                background: #eff6ff;
                border-color: #93c5fd;
                color: #1d4ed8;
                font-weight: 600;
            }
            QPushButton[buttonRole="primary"]:hover, QToolButton[buttonRole="primary"]:hover {
                background: #dbeafe;
                border-color: #60a5fa;
            }
            QPushButton[buttonRole="primary"]:pressed, QToolButton[buttonRole="primary"]:pressed {
                background: #bfdbfe;
                border-color: #3b82f6;
            }
            QPushButton[buttonRole="secondary"], QToolButton[buttonRole="secondary"] {
                background: #f8fbff;
                border-color: #bfd5f7;
                color: #31527c;
                font-weight: 600;
            }
            QPushButton[buttonRole="secondary"]:hover, QToolButton[buttonRole="secondary"]:hover {
                background: #e8f1ff;
                border-color: #93c5fd;
                color: #1d4ed8;
            }
            QPushButton[buttonRole="secondary"]:pressed, QToolButton[buttonRole="secondary"]:pressed {
                background: #dbeafe;
                border-color: #60a5fa;
            }
            QPushButton[buttonRole="quiet"], QToolButton[buttonRole="quiet"] {
                background: #ffffff;
                border-color: #dde6f0;
                color: #64748b;
            }
            QPushButton[buttonRole="quiet"]:hover, QToolButton[buttonRole="quiet"]:hover {
                background: #f8fafc;
                border-color: #cbd5e1;
                color: #334155;
            }
            QPushButton[buttonRole="quiet"]:pressed, QToolButton[buttonRole="quiet"]:pressed {
                background: #eef2f7;
                border-color: #b8c6d8;
            }
            QToolBar {
                background: #ffffff;
                border: 1px solid #d9e1ea;
                border-radius: 12px;
                spacing: 3px;
                padding: 4px;
            }
            QToolBar QToolButton {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                color: #334155;
                qproperty-iconSize: 17px;
                border-radius: 9px;
                padding: 5px;
                min-width: 30px;
                min-height: 30px;
            }
            QToolBar QToolButton:hover {
                background: #eef4ff;
                border-color: #bfdbfe;
            }
            QToolBar QToolButton:pressed, QToolBar QToolButton:checked {
                background: #dbeafe;
                border-color: #60a5fa;
            }
            QToolBar::separator {
                width: 8px;
                background: transparent;
            }
            QToolBar QToolButton[toolbarRole="quiet"] {
                background: transparent;
                border: 1px solid transparent;
                color: #64748b;
            }
            QFrame#SegmentedControl {
                background: #eef2f7;
                border: 1px solid #d9e1ea;
                border-radius: 10px;
            }
            QToolButton[segmentRole="segmented"] {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 4px 10px;
                min-height: 24px;
                font-weight: 600;
                color: #475569;
            }
            QToolButton[segmentRole="segmented"]:hover {
                background: #e2e8f0;
                border-color: #cbd5e1;
            }
            QToolButton[segmentRole="segmented"]:checked {
                background: #ffffff;
                border-color: #93c5fd;
                color: #1d4ed8;
            }
            QFrame#PrimarySegmentedControl {
                background: #eef6ff;
                border: 1px solid #bfd5f7;
                border-radius: 10px;
            }
            QToolButton[segmentRole="segmentedPrimary"] {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 5px 12px;
                min-height: 26px;
                font-weight: 700;
                color: #31527c;
            }
            QToolButton[segmentRole="segmentedPrimary"]:hover {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            QToolButton[segmentRole="segmentedPrimary"]:checked {
                background: #ffffff;
                border-color: #60a5fa;
                color: #1d4ed8;
            }
            QHeaderView::section {
                background: #edf3f9;
                border: none;
                border-bottom: 1px solid #d9e1ea;
                padding: 4px;
            }
            QLabel#StatusBadge {
                background: #eef2f7;
                border: 1px solid #d6dde6;
                border-radius: 7px;
                padding: 1px 5px;
                font-weight: 600;
                font-size: 10px;
            }
            QFrame#TrajectoryInfoStrip {
                background: #f8fafc;
                border: 1px solid #dde6f0;
                border-radius: 10px;
            }
            QLabel#TrajectoryInfoText {
                color: #334155;
                font-size: 11px;
                font-weight: 600;
                padding: 0px 2px;
            }
            QLabel#PanelLegend {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 3px 6px;
                color: #334155;
                font-size: 11px;
            }
            QLabel#SubtleText {
                color: #64748b;
                font-size: 10px;
            }
            QLabel#CompactValue {
                color: #0f172a;
                padding: 1px 0;
            }
            QFrame#SummaryCard {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 7px;
            }
            QLabel#SummaryCardTitle {
                color: #475569;
                background: #eef2f7;
                border: 1px solid #d6dde6;
                border-radius: 6px;
                padding: 0px 5px;
                font-size: 9px;
                font-weight: 600;
                letter-spacing: 0.1px;
            }
            QLabel#SummaryCardBody {
                color: #0f172a;
                font-size: 10px;
            }
            QFrame#ActionSection {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            QLabel#ActionSectionTitle {
                color: #475569;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.3px;
            }
            QSplitter::handle {
                background: #d9e1ea;
            }
            QTabWidget::pane {
                border: 1px solid #d9e1ea;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #f8fafc;
                border: 1px solid #d9e1ea;
                border-bottom-color: #cfd8e3;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 4px 10px;
                min-width: 72px;
                color: #475569;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #1d4ed8;
                border-color: #93c5fd;
                border-bottom-color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background: #f1f5f9;
                border-color: #cbd5e1;
            }
            """
        )

    def _configure_button_cursors(self) -> None:
        pointing_hand = QtGui.QCursor(QtCore.Qt.PointingHandCursor)
        for button in self.findChildren(QtWidgets.QPushButton):
            button.setCursor(pointing_hand)
        for button in self.findChildren(QtWidgets.QToolButton):
            button.setCursor(pointing_hand)

    def _configure_plot_toolbar(self) -> None:
        if not hasattr(self, "toolbar") or self.toolbar is None:
            return

        self.toolbar.setIconSize(QtCore.QSize(17, 17))
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        removable_actions = {"Subplots", "Customize"}
        for action in list(self.toolbar.actions()):
            text = action.text().strip()
            if text in removable_actions:
                self.toolbar.removeAction(action)
                continue
            action.triggered.connect(self._on_plot_toolbar_action_triggered)

        for button in self.toolbar.findChildren(QtWidgets.QToolButton):
            button.setAutoRaise(False)
            text = button.defaultAction().text().strip() if button.defaultAction() is not None else ""
            if text in {"Clear selection", "Fit trajectory view"}:
                button.setProperty("toolbarRole", "quiet")
                button.style().unpolish(button)
                button.style().polish(button)

    def _on_plot_toolbar_action_triggered(self, *_args) -> None:
        QtCore.QTimer.singleShot(0, self._update_plot_help_state)

    def _restore_input_history(
        self,
        *,
        initial_session_root: Optional[Path],
        initial_g2o_path: Optional[Path],
    ) -> None:
        session_root = initial_session_root or self._settings_existing_path("browser/last_session_root")
        if session_root is not None:
            self.session_root_edit.setText(str(session_root))

        g2o_path = initial_g2o_path or self._settings_existing_path("browser/last_g2o_path")
        if g2o_path is not None:
            self.g2o_edit.setText(str(g2o_path))

    def _settings_existing_path(self, key: str) -> Optional[Path]:
        raw_value = self._settings.value(key, "", type=str)
        if not raw_value:
            return None
        path = Path(raw_value).expanduser()
        if path.exists():
            return path
        return None

    def _existing_dialog_directory(self, text: str) -> Optional[Path]:
        text = text.strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if path.is_dir():
            return path
        if path.is_file():
            return path.parent
        return None

    def _session_root_dialog_directory(self) -> str:
        for candidate in (
            self._existing_dialog_directory(self.session_root_edit.text()),
            self._settings_existing_path("browser/last_session_root"),
            self._settings_existing_path("browser/last_g2o_path"),
        ):
            if candidate is None:
                continue
            return str(candidate if candidate.is_dir() else candidate.parent)
        return str(Path.home())

    def _g2o_dialog_directory(self) -> str:
        for candidate in (
            self._existing_dialog_directory(self.g2o_edit.text()),
            self._existing_dialog_directory(self.session_root_edit.text()),
            self._settings_existing_path("browser/last_g2o_path"),
            self._settings_existing_path("browser/last_session_root"),
        ):
            if candidate is None:
                continue
            return str(candidate if candidate.is_dir() else candidate.parent)
        return str(Path.home())

    def _project_state_dialog_directory(self) -> str:
        session_root_dir = self._existing_dialog_directory(self.session_root_edit.text())
        if session_root_dir is not None:
            projects_dir = session_root_dir / "manual_loop_projects"
            if projects_dir.is_dir():
                return str(projects_dir)
            return str(session_root_dir)
        for candidate in (
            self._settings_existing_path("browser/last_project_state"),
            self._settings_existing_path("browser/last_session_root"),
            Path.home(),
        ):
            if candidate is None:
                continue
            return str(candidate if candidate.is_dir() else candidate.parent)
        return str(Path.home())

    def _is_g2o_under_session_root(self, session_root: Path, g2o_path: Path) -> bool:
        try:
            g2o_path.resolve().relative_to(session_root.resolve())
            return True
        except ValueError:
            return False

    def _remember_loaded_paths(self, paths: SessionPaths) -> None:
        self._settings.setValue("browser/last_session_root", str(paths.session_root))
        self._settings.setValue("browser/last_g2o_path", str(paths.g2o_path))
        self._settings.sync()

    def _remember_project_state_path(self, project_state_path: Path) -> None:
        self._settings.setValue("browser/last_project_state", str(project_state_path))
        self._settings.sync()

    def _release_plot_toolbar_navigation(self) -> bool:
        if not hasattr(self, "toolbar") or self.toolbar is None or not self.toolbar.mode:
            return False

        mode = str(self.toolbar.mode).lower()
        if "pan" in mode:
            self.toolbar.pan()
        elif "zoom" in mode:
            self.toolbar.zoom()
        else:
            return False
        self._update_plot_help_state()
        return True

    def _update_plot_help_state(self) -> None:
        if not self._is_working_view():
            self.plot_help_label.setText(
                "Original view is read-only. Enable Overlay to compare against working."
            )
            return

        toolbar_mode = str(self.toolbar.mode).lower() if hasattr(self, "toolbar") and self.toolbar is not None else ""
        if "zoom" in toolbar_mode:
            self.plot_help_label.setText(
                "Zoom active · click Pick Nodes or Pick Edges to resume selection."
            )
            return
        if "pan" in toolbar_mode:
            self.plot_help_label.setText(
                "Pan active · click Pick Nodes or Pick Edges to resume selection."
            )
            return

        if self.pick_mode == "edges":
            self.plot_help_label.setText(
                "Pick edges to inspect, disable, or replace."
            )
            return

        self.plot_help_label.setText(
            "Pick nodes to add closures, or pick edges to inspect/replace."
        )

    def _apply_initial_window_geometry(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 900)
            return
        available = screen.availableGeometry()
        self.setMinimumSize(
            min(self.minimumWidth(), max(640, available.width() - 24)),
            min(self.minimumHeight(), max(520, available.height() - 24)),
        )
        target_width = min(available.width(), max(self.minimumWidth(), int(available.width() * 0.96)))
        target_height = min(available.height(), max(self.minimumHeight(), int(available.height() * 0.94)))
        self.resize(min(target_width, available.width()), min(target_height, available.height()))
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _build_loader_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Session Input")
        layout = QtWidgets.QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)

        self.session_root_edit = QtWidgets.QLineEdit()
        self.g2o_edit = QtWidgets.QLineEdit()
        browse_root_button = QtWidgets.QPushButton("Browse Root")
        browse_root_button.setToolTip("Open the session-root browser from the most recent folder.")
        browse_root_button.setProperty("buttonRole", "quiet")
        browse_root_button.clicked.connect(self._browse_session_root)
        browse_g2o_button = QtWidgets.QPushButton("Browse G2O")
        browse_g2o_button.setToolTip("Open the g2o file browser from the most recent folder.")
        browse_g2o_button.setProperty("buttonRole", "quiet")
        browse_g2o_button.clicked.connect(self._browse_g2o)
        open_project_button = QtWidgets.QPushButton("Resume Project")
        open_project_button.setToolTip(
            "Restore a saved edit project directly by choosing its project_state.json."
        )
        open_project_button.setProperty("buttonRole", "secondary")
        open_project_button.clicked.connect(self._browse_project_state)
        self.load_button = QtWidgets.QPushButton("Load Session")
        self.load_button.setToolTip("Load the current Session Root and G2O File fields as a new or resumed session.")
        self.load_button.setProperty("buttonRole", "primary")
        self.load_button.clicked.connect(self.load_session)

        button_panel = QtWidgets.QFrame()
        button_layout = QtWidgets.QGridLayout(button_panel)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setHorizontalSpacing(6)
        button_layout.setVerticalSpacing(4)
        button_layout.addWidget(browse_root_button, 0, 0)
        button_layout.addWidget(browse_g2o_button, 0, 1)
        button_layout.addWidget(open_project_button, 1, 0)
        button_layout.addWidget(self.load_button, 1, 1)

        layout.addWidget(QtWidgets.QLabel("Session Root"), 0, 0)
        layout.addWidget(self.session_root_edit, 0, 1)
        layout.addWidget(QtWidgets.QLabel("G2O File"), 1, 0)
        layout.addWidget(self.g2o_edit, 1, 1)
        layout.addWidget(button_panel, 0, 2, 2, 1)
        layout.setColumnStretch(1, 1)

        self.session_hint_label = QtWidgets.QLabel(
            "Load Session uses the current Root + G2O. Resume Project restores a saved project_state.json."
        )
        self.session_hint_label.setObjectName("SubtleText")
        self.session_hint_label.setWordWrap(True)
        layout.addWidget(self.session_hint_label, 2, 1, 1, 2)

        self.session_info_label = QtWidgets.QLabel("No session loaded.")
        self.session_info_label.setWordWrap(True)
        layout.addWidget(self.session_info_label, 3, 0, 1, 3)
        return group

    def _build_plot_panel(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Trajectory")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(7, 8, 7, 7)
        layout.setSpacing(5)
        self.trajectory_canvas = TrajectoryCanvas(group)
        self.toolbar = NavigationToolbar(self.trajectory_canvas, group)
        self.trajectory_canvas.set_toolbar(self.toolbar)
        self.trajectory_canvas.set_callbacks(self._handle_plot_selection, self._handle_plot_hover)
        style = self.style()
        self.toolbar.addSeparator()
        clear_icon = style.standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
        fit_icon = style.standardIcon(QtWidgets.QStyle.SP_TitleBarMaxButton)
        self.clear_selection_action = self.toolbar.addAction(clear_icon, "Clear selection", self._clear_selection)
        self.clear_selection_action.setToolTip("Clear the current node or edge selection.")
        self.fit_view_action = self.toolbar.addAction(
            fit_icon,
            "Fit trajectory view",
            lambda: self._refresh_plot(preserve_view=False),
        )
        self.fit_view_action.setToolTip("Reset the trajectory view to the full pose graph.")
        layout.addWidget(self.toolbar)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(5)
        header_row.addWidget(QtWidgets.QLabel("View"))
        self.trajectory_view_combo = QtWidgets.QComboBox()
        self.trajectory_view_combo.addItems(["Working", "Original"])
        self.trajectory_view_combo.currentIndexChanged.connect(self._on_trajectory_view_changed)
        header_row.addWidget(self.trajectory_view_combo)
        self.show_ghost_check = QtWidgets.QCheckBox("Overlay")
        self.show_ghost_check.setToolTip("Overlay the other trajectory as a reference.")
        self.show_ghost_check.setChecked(True)
        self.show_ghost_check.stateChanged.connect(self._on_show_ghost_changed)
        header_row.addWidget(self.show_ghost_check)
        self.pick_nodes_button = QtWidgets.QToolButton()
        self.pick_nodes_button.setText("Nodes")
        self.pick_nodes_button.setCheckable(True)
        self.pick_nodes_button.setChecked(True)
        self.pick_nodes_button.setProperty("segmentRole", "segmentedPrimary")
        self.pick_nodes_button.setToolTip("Exit pan/zoom mode and pick a source-target node pair.")
        self.pick_nodes_button.clicked.connect(lambda: self._set_pick_mode("nodes"))

        self.pick_edges_button = QtWidgets.QToolButton()
        self.pick_edges_button.setText("Edges")
        self.pick_edges_button.setCheckable(True)
        self.pick_edges_button.setProperty("segmentRole", "segmentedPrimary")
        self.pick_edges_button.setToolTip("Exit pan/zoom mode and inspect an existing or manual edge.")
        self.pick_edges_button.clicked.connect(lambda: self._set_pick_mode("edges"))

        mode_group = QtWidgets.QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.pick_nodes_button)
        mode_group.addButton(self.pick_edges_button)

        segmented_control = QtWidgets.QFrame()
        segmented_control.setObjectName("PrimarySegmentedControl")
        segmented_layout = QtWidgets.QHBoxLayout(segmented_control)
        segmented_layout.setContentsMargins(2, 2, 2, 2)
        segmented_layout.setSpacing(2)
        segmented_layout.addWidget(self.pick_nodes_button)
        segmented_layout.addWidget(self.pick_edges_button)
        info_strip = QtWidgets.QFrame()
        info_strip.setObjectName("TrajectoryInfoStrip")
        info_layout = QtWidgets.QHBoxLayout(info_strip)
        info_layout.setContentsMargins(8, 3, 8, 3)
        info_layout.setSpacing(0)
        self.trajectory_view_info_label = QtWidgets.QLabel("P0 · L0 · edit")
        self.trajectory_view_info_label.setObjectName("TrajectoryInfoText")
        self.trajectory_view_info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.trajectory_view_info_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        info_layout.addWidget(self.trajectory_view_info_label)
        status_strip = QtWidgets.QFrame()
        status_layout = QtWidgets.QHBoxLayout(status_strip)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(4)
        self.working_rev_badge = QtWidgets.QLabel("R0")
        self.working_rev_badge.setObjectName("StatusBadge")
        status_layout.addWidget(self.working_rev_badge)
        self.session_state_badge = QtWidgets.QLabel("OK")
        self.session_state_badge.setObjectName("StatusBadge")
        status_layout.addWidget(self.session_state_badge)
        self.change_summary_badge = QtWidgets.QLabel("M0 D0")
        self.change_summary_badge.setObjectName("StatusBadge")
        status_layout.addWidget(self.change_summary_badge)
        header_row.addWidget(segmented_control)
        header_row.addWidget(info_strip, 1)
        header_row.addWidget(status_strip, 0, QtCore.Qt.AlignRight)
        layout.addLayout(header_row)

        self.plot_help_label = QtWidgets.QLabel("Working edits. Original compares.")
        self.plot_help_label.setObjectName("SubtleText")
        self.plot_help_label.setWordWrap(True)
        layout.addWidget(self.plot_help_label)
        self.trajectory_legend_label = QtWidgets.QLabel()
        self.trajectory_legend_label.setWordWrap(False)
        self.trajectory_legend_label.setTextFormat(QtCore.Qt.RichText)
        self.trajectory_legend_label.setObjectName("PanelLegend")
        layout.addWidget(self.trajectory_legend_label)
        layout.addWidget(self.trajectory_canvas, stretch=1)

        self.hover_label = QtWidgets.QLabel("Hover: none")
        layout.addWidget(self.hover_label)
        return group

    def _build_cloud_panel(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Point Cloud Review")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ===== BEGIN CHANGE: manual align cloud toolbar =====
        self.cloud_view = EmbeddedOpen3DWidget(group)
        self.cloud_view.manual_align_changed.connect(self._on_manual_align_changed)
        self.cloud_view.manual_align_status.connect(self._on_manual_align_status)
        self.cloud_view.camera_preset_changed.connect(self._on_camera_preset_changed)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.setSpacing(5)
        controls_row.addWidget(QtWidgets.QLabel("Display"))
        self.display_mode_combo = QtWidgets.QComboBox()
        self.display_mode_combo.addItem("Before GICP", "preview")
        self.display_mode_combo.addItem("After GICP", "final")
        self.display_mode_combo.addItem("Compare", "compare")
        self.display_mode_combo.currentIndexChanged.connect(self._update_cloud_display_controls)
        self.display_mode_combo.setMaximumWidth(125)
        controls_row.addWidget(self.display_mode_combo)
        controls_row.addWidget(QtWidgets.QLabel("Traj"))
        self.trajectory_point_size_spin = QtWidgets.QSpinBox()
        self.trajectory_point_size_spin.setRange(1, 32)
        self.trajectory_point_size_spin.setValue(8)
        self.trajectory_point_size_spin.setMaximumWidth(56)
        self.trajectory_point_size_spin.valueChanged.connect(self._update_cloud_display_controls)
        controls_row.addWidget(self.trajectory_point_size_spin)
        self.target_point_size_spin = self._make_double_spin(
            step=0.5,
            minimum=0.5,
            maximum=20.0,
            value=1.0,
        )
        self.target_point_size_spin.valueChanged.connect(self._update_cloud_display_controls)
        self.source_point_size_spin = self._make_double_spin(
            step=0.5,
            minimum=0.5,
            maximum=20.0,
            value=3.0,
        )
        self.source_point_size_spin.valueChanged.connect(self._update_cloud_display_controls)
        self.show_world_axis_check = QtWidgets.QCheckBox("World Axis")
        self.show_world_axis_check.stateChanged.connect(self._update_cloud_display_controls)
        controls_row.addWidget(self.show_world_axis_check)
        controls_row.addWidget(QtWidgets.QLabel("Edit"))
        self.cloud_interaction_mode_combo = QtWidgets.QComboBox()
        self.cloud_interaction_mode_combo.addItem("View", INTERACTION_MODE_CAMERA)
        self.cloud_interaction_mode_combo.addItem("Edit", INTERACTION_MODE_EDIT_SOURCE)
        self.cloud_interaction_mode_combo.setMaximumWidth(84)
        self.cloud_interaction_mode_combo.currentIndexChanged.connect(self._on_cloud_interaction_changed)
        controls_row.addWidget(self.cloud_interaction_mode_combo)
        controls_row.addWidget(QtWidgets.QLabel("Align"))
        self.cloud_lock_mode_combo = QtWidgets.QComboBox()
        self.cloud_lock_mode_combo.addItem("XY+Yaw", LOCK_MODE_XY_YAW)
        self.cloud_lock_mode_combo.addItem("Z", LOCK_MODE_Z_ONLY)
        self.cloud_lock_mode_combo.setMaximumWidth(96)
        self.cloud_lock_mode_combo.currentIndexChanged.connect(self._on_cloud_interaction_changed)
        controls_row.addWidget(self.cloud_lock_mode_combo)
        controls_row.addWidget(QtWidgets.QLabel("Drag"))
        self.cloud_drag_mode_combo = QtWidgets.QComboBox()
        self.cloud_drag_mode_combo.addItem("XY", EDIT_OPERATION_TRANSLATE)
        self.cloud_drag_mode_combo.addItem("Yaw", EDIT_OPERATION_ROTATE)
        self.cloud_drag_mode_combo.setMaximumWidth(72)
        self.cloud_drag_mode_combo.currentIndexChanged.connect(self._on_cloud_interaction_changed)
        controls_row.addWidget(self.cloud_drag_mode_combo)
        controls_row.addStretch(1)
        controls_row.addWidget(QtWidgets.QLabel("View"))
        self.camera_preset_buttons: dict[str, QtWidgets.QToolButton] = {}
        camera_preset_frame = QtWidgets.QFrame()
        camera_preset_frame.setObjectName("SegmentedControl")
        camera_preset_layout = QtWidgets.QHBoxLayout(camera_preset_frame)
        camera_preset_layout.setContentsMargins(2, 2, 2, 2)
        camera_preset_layout.setSpacing(2)
        for text, preset in (
            ("Top", VIEW_PRESET_TOP),
            ("Side-Y", VIEW_PRESET_SIDE_Y),
            ("Side-X", VIEW_PRESET_SIDE_X),
        ):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setCheckable(True)
            button.setProperty("segmentRole", "segmented")
            button.clicked.connect(lambda _checked=False, value=preset: self.cloud_view.set_view_preset(value))
            camera_preset_layout.addWidget(button)
            self.camera_preset_buttons[preset] = button
        controls_row.addWidget(camera_preset_frame)
        self.manual_align_translation_step_spin = self._make_double_spin(
            step=0.01,
            minimum=0.01,
            maximum=5.0,
            value=0.05,
        )
        self.manual_align_rotation_step_spin = self._make_double_spin(
            step=0.5,
            minimum=0.1,
            maximum=30.0,
            value=1.0,
        )
        self.manual_align_translation_step_spin.hide()
        self.manual_align_rotation_step_spin.hide()
        self.manual_align_translation_step_spin.valueChanged.connect(self._on_cloud_interaction_changed)
        self.manual_align_rotation_step_spin.valueChanged.connect(self._on_cloud_interaction_changed)
        self.manual_align_snap_view_check = QtWidgets.QCheckBox("Snap")
        self.manual_align_snap_view_check.setChecked(True)
        self.manual_align_snap_view_check.stateChanged.connect(self._on_cloud_interaction_changed)
        controls_row.addWidget(self.manual_align_snap_view_check)
        self.reset_manual_align_button = QtWidgets.QPushButton("Reset Align")
        self.reset_manual_align_button.clicked.connect(self._reset_manual_align)
        controls_row.addWidget(self.reset_manual_align_button)
        self.reset_camera_button = QtWidgets.QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(self.cloud_view.reset_camera)
        controls_row.addWidget(self.reset_camera_button)
        layout.addLayout(controls_row)
        self._on_camera_preset_changed(VIEW_PRESET_TOP)

        self.manual_align_status_label = QtWidgets.QLabel("View mode · wheel zoom")
        self.manual_align_status_label.setObjectName("SubtleText")
        self.manual_align_status_label.setWordWrap(False)
        layout.addWidget(self.manual_align_status_label)
        # ===== END CHANGE: manual align cloud toolbar =====

        self.cloud_legend_label = QtWidgets.QLabel(
            "Gray target · yellow selected frames · orange/cyan source · green final"
        )
        self.cloud_legend_label.setObjectName("SubtleText")
        self.cloud_legend_label.setWordWrap(False)
        layout.addWidget(self.cloud_legend_label)

        layout.addWidget(self.cloud_view, stretch=1)
        return group

    def _build_control_scroll_area(self) -> QtWidgets.QWidget:
        panel = self._build_control_panel()
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(360)
        return panel

    def _wrap_control_tab(self, content: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(content)
        return scroll_area

    def _build_summary_card(self, title: str, body_label: QtWidgets.QLabel) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("SummaryCard")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(6, 4, 6, 4)
        card_layout.setSpacing(1)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("SummaryCardTitle")
        title_label.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        card_layout.addWidget(title_label)
        card_layout.addWidget(body_label)
        return card

    def _build_action_section(
        self,
        title: str,
        widgets: list[tuple[QtWidgets.QWidget, int, int, int, int]],
    ) -> QtWidgets.QFrame:
        section = QtWidgets.QFrame()
        section.setObjectName("ActionSection")
        section_layout = QtWidgets.QVBoxLayout(section)
        section_layout.setContentsMargins(8, 6, 8, 6)
        section_layout.setSpacing(6)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("ActionSectionTitle")
        section_layout.addWidget(title_label)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        for widget, row, column, row_span, column_span in widgets:
            grid.addWidget(widget, row, column, row_span, column_span)
        section_layout.addLayout(grid)
        return section

    def _build_control_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        control_tabs = QtWidgets.QTabWidget()
        control_tabs.setDocumentMode(True)
        control_tabs.setElideMode(QtCore.Qt.ElideRight)
        summary_tab = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(6)
        advanced_tab = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)

        selection_group = QtWidgets.QGroupBox("Summary")
        selection_layout = QtWidgets.QGridLayout(selection_group)
        selection_layout.setContentsMargins(6, 6, 6, 6)
        selection_layout.setHorizontalSpacing(4)
        selection_layout.setVerticalSpacing(4)
        self.pair_label = QtWidgets.QLabel("none")
        self.pair_label.setWordWrap(True)
        self.pair_label.setTextFormat(QtCore.Qt.RichText)
        self.pair_label.setObjectName("SummaryCardBody")
        self.source_label = QtWidgets.QLabel("none")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("CompactValue")
        self.source_label.setTextFormat(QtCore.Qt.RichText)
        self.source_label.setObjectName("SummaryCardBody")
        self.target_label = QtWidgets.QLabel("none")
        self.target_label.setWordWrap(True)
        self.target_label.setTextFormat(QtCore.Qt.RichText)
        self.target_label.setObjectName("SummaryCardBody")
        self.selected_edge_label = QtWidgets.QLabel("none")
        self.selected_edge_label.setWordWrap(True)
        self.selected_edge_label.setTextFormat(QtCore.Qt.RichText)
        self.selected_edge_label.setObjectName("SummaryCardBody")
        self.working_graph_label = QtWidgets.QLabel("Rev 0 · no working graph")
        self.working_graph_label.setWordWrap(True)
        self.working_graph_label.setTextFormat(QtCore.Qt.RichText)
        self.working_graph_label.setObjectName("SummaryCardBody")
        self.target_map_label = QtWidgets.QLabel("No target submap.")
        self.target_map_label.setWordWrap(True)
        self.target_map_label.setTextFormat(QtCore.Qt.RichText)
        self.target_map_label.setObjectName("SummaryCardBody")
        self.gicp_metrics_label = QtWidgets.QLabel("No GICP result.")
        self.gicp_metrics_label.setWordWrap(True)
        self.gicp_metrics_label.setTextFormat(QtCore.Qt.RichText)
        self.gicp_metrics_label.setObjectName("SummaryCardBody")
        selection_layout.addWidget(self._build_summary_card("Graph", self.working_graph_label), 0, 0)
        selection_layout.addWidget(self._build_summary_card("State", self.gicp_metrics_label), 0, 1)
        selection_layout.addWidget(self._build_summary_card("Pair", self.pair_label), 1, 0, 1, 2)
        selection_layout.addWidget(self._build_summary_card("Edge", self.selected_edge_label), 2, 0)
        selection_layout.addWidget(self._build_summary_card("Map", self.target_map_label), 2, 1)
        selection_layout.setColumnStretch(0, 1)
        selection_layout.setColumnStretch(1, 1)
        summary_layout.addWidget(selection_group)

        delta_group = QtWidgets.QGroupBox("Delta")
        delta_group.setToolTip("Source-local seed. Viewer drag edits the same values.")
        delta_layout = QtWidgets.QGridLayout(delta_group)
        delta_layout.setContentsMargins(7, 7, 7, 7)
        delta_layout.setHorizontalSpacing(5)
        delta_layout.setVerticalSpacing(3)
        self.delta_spins = {
            "x": self._make_double_spin(step=0.1, minimum=-1000.0, maximum=1000.0, value=0.0),
            "y": self._make_double_spin(step=0.1, minimum=-1000.0, maximum=1000.0, value=0.0),
            "z": self._make_double_spin(step=0.1, minimum=-1000.0, maximum=1000.0, value=0.0),
            "roll": self._make_double_spin(step=1.0, minimum=-180.0, maximum=180.0, value=0.0),
            "pitch": self._make_double_spin(step=1.0, minimum=-180.0, maximum=180.0, value=0.0),
            "yaw": self._make_double_spin(step=1.0, minimum=-180.0, maximum=180.0, value=0.0),
        }
        for spin in self.delta_spins.values():
            spin.valueChanged.connect(self._on_delta_spin_changed)
        for column, key in enumerate(("x", "y", "z")):
            delta_layout.addWidget(QtWidgets.QLabel(key), 0, column)
            delta_layout.addWidget(self.delta_spins[key], 1, column)
        for column, (label_text, key) in enumerate((("r", "roll"), ("p", "pitch"), ("y", "yaw"))):
            delta_layout.addWidget(QtWidgets.QLabel(label_text), 2, column)
            delta_layout.addWidget(self.delta_spins[key], 3, column)
        reset_button = QtWidgets.QPushButton("Reset Delta")
        reset_button.clicked.connect(self._reset_delta)
        refresh_button = QtWidgets.QPushButton("Refresh Preview")
        refresh_button.clicked.connect(lambda: self.schedule_preview_refresh(reset_camera=False))
        self.auto_yaw_steps_spin = QtWidgets.QSpinBox()
        self.auto_yaw_steps_spin.setRange(2, 72)
        self.auto_yaw_steps_spin.setValue(12)
        self.auto_yaw_steps_spin.setToolTip("Sweep evenly-spaced yaw seeds across 0-360 degrees.")
        self.auto_yaw_button = QtWidgets.QPushButton("Auto Yaw Sweep")
        self.auto_yaw_button.clicked.connect(self.run_auto_yaw_sweep)
        delta_layout.addWidget(reset_button, 4, 0, 1, 2)
        refresh_button.setText("Preview")
        delta_layout.addWidget(refresh_button, 4, 2)
        summary_layout.addWidget(delta_group)

        registration_group = QtWidgets.QGroupBox("Registration")
        registration_layout = QtWidgets.QGridLayout(registration_group)
        registration_layout.setContentsMargins(7, 7, 7, 7)
        registration_layout.setHorizontalSpacing(5)
        registration_layout.setVerticalSpacing(4)
        self.target_cloud_mode_combo = QtWidgets.QComboBox()
        self.target_cloud_mode_combo.addItem("Temporal Window", TARGET_CLOUD_MODE_TEMPORAL_WINDOW)
        self.target_cloud_mode_combo.addItem("RS Spatial Submap", TARGET_CLOUD_MODE_RS_SPATIAL_SUBMAP)
        default_mode_index = self.target_cloud_mode_combo.findData(OFFICE_DEFAULT_TARGET_CLOUD_MODE)
        if default_mode_index >= 0:
            self.target_cloud_mode_combo.setCurrentIndex(default_mode_index)
        self.target_cloud_mode_combo.currentIndexChanged.connect(self._on_target_cloud_mode_changed)
        self.target_neighbors_spin = QtWidgets.QSpinBox()
        self.target_neighbors_spin.setRange(1, 500)
        self.target_neighbors_spin.setValue(OFFICE_DEFAULT_TARGET_NEIGHBORS)
        self.target_neighbors_spin.valueChanged.connect(self.schedule_preview_refresh)
        self.target_min_gap_spin = self._make_double_spin(
            step=1.0,
            minimum=0.0,
            maximum=10000.0,
            value=OFFICE_DEFAULT_TARGET_MIN_TIME_GAP_SEC,
        )
        self.target_min_gap_spin.valueChanged.connect(self.schedule_preview_refresh)
        self.target_map_voxel_spin = self._make_double_spin(
            step=0.05,
            minimum=0.0,
            maximum=5.0,
            value=OFFICE_DEFAULT_TARGET_MAP_VOXEL_SIZE,
        )
        self.target_map_voxel_spin.valueChanged.connect(self.schedule_preview_refresh)
        self.voxel_spin = self._make_double_spin(
            step=0.05,
            minimum=0.0,
            maximum=5.0,
            value=OFFICE_DEFAULT_VOXEL_SIZE,
        )
        self.max_corr_spin = self._make_double_spin(
            step=0.1,
            minimum=0.1,
            maximum=20.0,
            value=OFFICE_DEFAULT_MAX_CORRESPONDENCE_DISTANCE,
        )
        self.max_iter_spin = QtWidgets.QSpinBox()
        self.max_iter_spin.setRange(1, 500)
        self.max_iter_spin.setValue(OFFICE_DEFAULT_MAX_ITERATIONS)
        self.target_neighbors_label = QtWidgets.QLabel("Nbr")
        self.target_neighbors_label.setToolTip("Temporal window radius or RS spatial neighbor count.")
        self.target_min_gap_label = QtWidgets.QLabel("Gap[s]")
        self.target_min_gap_label.setToolTip("Minimum source-target time gap, only used by RS Spatial Submap.")
        mode_label = QtWidgets.QLabel("Mode")
        mode_label.setToolTip("Target cloud construction mode.")
        target_map_voxel_label = QtWidgets.QLabel("TgtVoxel")
        target_map_voxel_label.setToolTip("Downsample size for the target map preview.")
        voxel_label = QtWidgets.QLabel("Voxel")
        voxel_label.setToolTip("Source and target voxel size for GICP.")
        radius_label = QtWidgets.QLabel("Radius")
        radius_label.setToolTip("Maximum search radius for correspondence lookup.")
        iter_label = QtWidgets.QLabel("Iter")
        iter_label.setToolTip("Maximum GICP iterations.")
        registration_layout.addWidget(mode_label, 0, 0)
        registration_layout.addWidget(self.target_cloud_mode_combo, 0, 1, 1, 3)
        registration_layout.addWidget(self.target_neighbors_label, 1, 0)
        registration_layout.addWidget(self.target_neighbors_spin, 1, 1)
        registration_layout.addWidget(self.target_min_gap_label, 1, 2)
        registration_layout.addWidget(self.target_min_gap_spin, 1, 3)
        registration_layout.addWidget(target_map_voxel_label, 2, 0)
        registration_layout.addWidget(self.target_map_voxel_spin, 2, 1)
        registration_layout.addWidget(voxel_label, 2, 2)
        registration_layout.addWidget(self.voxel_spin, 2, 3)
        registration_layout.addWidget(radius_label, 3, 0)
        registration_layout.addWidget(self.max_corr_spin, 3, 1)
        registration_layout.addWidget(iter_label, 3, 2)
        registration_layout.addWidget(self.max_iter_spin, 3, 3)
        registration_layout.setColumnStretch(1, 1)
        registration_layout.setColumnStretch(3, 1)

        self._update_target_cloud_mode_controls()
        summary_layout.addWidget(registration_group)

        advanced_group = QtWidgets.QGroupBox("Advanced")
        advanced_group.setToolTip("Optional expert settings. The default values already match the validated parity runs.")
        advanced_layout_grid = QtWidgets.QGridLayout(advanced_group)
        advanced_layout_grid.setContentsMargins(8, 8, 8, 8)
        advanced_layout_grid.setHorizontalSpacing(6)
        advanced_layout_grid.setVerticalSpacing(4)
        advanced_note = QtWidgets.QLabel("Optimization runs locally with the Python GTSAM backend; ROS is not required.")
        advanced_note.setObjectName("SubtleText")
        advanced_note.setWordWrap(True)
        advanced_layout_grid.addWidget(advanced_note, 0, 0, 1, 2)
        self.variance_t_shared_spin = self._make_double_spin(
            step=0.01,
            minimum=0.0,
            maximum=100.0,
            value=OFFICE_DEFAULT_VARIANCE_T[0],
        )
        self.variance_r_shared_spin = self._make_double_spin(
            step=0.000001,
            minimum=0.0,
            maximum=100.0,
            value=production_loop_variances()[1][0],
        )
        self.variance_r_shared_spin.setDecimals(10)
        self.export_map_voxel_spin = self._make_double_spin(
            step=0.05,
            minimum=0.0,
            maximum=5.0,
            value=0.1,
        )
        self.optimize_mode_combo = QtWidgets.QComboBox()
        self.optimize_mode_combo.addItem("Robust GNC-TLS", OPTIMIZE_MODE_GNC_TLS)
        self.optimize_mode_combo.addItem("Accurate LM", OPTIMIZE_MODE_LM)
        self.optimize_mode_combo.addItem("Fast ISAM2", OPTIMIZE_MODE_ISAM2)
        self.optimize_mode_combo.setCurrentIndex(0)
        advanced_layout_grid.addWidget(QtWidgets.QLabel("Variance T [m^2]"), 1, 0)
        advanced_layout_grid.addWidget(self.variance_t_shared_spin, 1, 1)
        advanced_layout_grid.addWidget(QtWidgets.QLabel("Variance R [rad^2]"), 2, 0)
        advanced_layout_grid.addWidget(self.variance_r_shared_spin, 2, 1)
        map_voxel_label = QtWidgets.QLabel("MapVoxel")
        map_voxel_label.setToolTip("Final map export voxel size.")
        advanced_layout_grid.addWidget(map_voxel_label, 3, 0)
        advanced_layout_grid.addWidget(self.export_map_voxel_spin, 3, 1)
        optimize_mode_label = QtWidgets.QLabel("PGO Solver")
        optimize_mode_label.setToolTip(
            "GNC-TLS is the validated Repair Map default. LM and ISAM2 remain available for controlled manual comparisons."
        )
        advanced_layout_grid.addWidget(optimize_mode_label, 4, 0)
        advanced_layout_grid.addWidget(self.optimize_mode_combo, 4, 1)

        backend_label = QtWidgets.QLabel("Backend")
        backend_label.setObjectName("SubtleText")
        advanced_layout_grid.addWidget(backend_label, 5, 0)
        self.optimizer_backend_label = QtWidgets.QLabel("Python GTSAM")
        self.optimizer_backend_label.setObjectName("SubtleText")
        advanced_layout_grid.addWidget(self.optimizer_backend_label, 5, 1)
        backend_hint = QtWidgets.QLabel(
            "Install the Python GTSAM wrapper before running optimization."
        )
        backend_hint.setObjectName("SubtleText")
        backend_hint.setWordWrap(True)
        advanced_layout_grid.addWidget(backend_hint, 6, 0, 1, 2)
        # ===== BEGIN CHANGE: balm advanced settings =====
        balm_note = QtWidgets.QLabel(
            "BALM refines the last optimized trajectory with coarse-to-fine plane bundle "
            "adjustment (hku-mars/BALM): a wide-threshold stage consolidates duplicated "
            "same-surface observations, "
            "then a tight stage sharpens planes. Run Audited PGO first, then "
            "press Final Refinement."
        )
        balm_note.setObjectName("SubtleText")
        balm_note.setWordWrap(True)
        advanced_layout_grid.addWidget(balm_note, 7, 0, 1, 2)
        self.balm_voxel_spin = self._make_double_spin(
            step=0.5,
            minimum=0.2,
            maximum=10.0,
            value=1.0,
        )
        balm_voxel_label = QtWidgets.QLabel("BALM Voxel [m]")
        balm_voxel_label.setToolTip(
            "Root voxel size for adaptive plane extraction. "
            "Use ~1 m indoors and 2-4 m outdoors."
        )
        advanced_layout_grid.addWidget(balm_voxel_label, 8, 0)
        advanced_layout_grid.addWidget(self.balm_voxel_spin, 8, 1)
        self.balm_iter_spin = QtWidgets.QSpinBox()
        self.balm_iter_spin.setRange(1, 100)
        self.balm_iter_spin.setValue(20)
        balm_iter_label = QtWidgets.QLabel("BALM Iter")
        balm_iter_label.setToolTip(
            "Upper bound on BALM Gauss-Newton updates (split between the coarse and "
            "fine stages). BALM normally stops earlier once the pose update converges "
            "or the RMS plateaus."
        )
        advanced_layout_grid.addWidget(balm_iter_label, 9, 0)
        advanced_layout_grid.addWidget(self.balm_iter_spin, 9, 1)
        self.balm_merge_spin = self._make_double_spin(
            step=0.01,
            minimum=0.05,
            maximum=0.40,
            value=0.12,
        )
        balm_merge_label = QtWidgets.QLabel("BALM Merge [m]")
        balm_merge_label.setToolTip(
            "Coarse-stage plane thickness: same-face ghost layers closer than "
            "roughly this distance are merged. Opposite observed faces are kept "
            "separate when Double-sided BALM is enabled."
        )
        advanced_layout_grid.addWidget(balm_merge_label, 10, 0)
        advanced_layout_grid.addWidget(self.balm_merge_spin, 10, 1)
        self.balm_double_sided_check = QtWidgets.QCheckBox("Double-sided BALM")
        self.balm_double_sided_check.setChecked(True)
        self.balm_double_sided_check.setToolTip(
            "Separate opposite observed faces by sensor-to-surface view sign so "
            "coarse BALM does not collapse a real thin wall. Disable only for the "
            "single-surface ablation."
        )
        advanced_layout_grid.addWidget(
            self.balm_double_sided_check, 11, 0, 1, 2
        )
        self.map_diagnostics_check = QtWidgets.QCheckBox(
            "Duplicated-surface audit"
        )
        self.map_diagnostics_check.setChecked(False)
        self.map_diagnostics_check.setToolTip(
            "Optional post-refinement scan for duplicated surface layers. It "
            "is inspection only: it does not alter poses or propose production "
            "loop constraints and is disabled by default."
        )
        advanced_layout_grid.addWidget(
            self.map_diagnostics_check, 12, 0, 1, 2
        )
        self.balm_auto_check = QtWidgets.QCheckBox(
            "Auto Final Refinement after PGO"
        )
        self.balm_auto_check.setChecked(True)
        self.balm_auto_check.setToolTip(
            "Automatically run Final Map Refinement after every successful PGO "
            "(skipped when Export chained the optimization)."
        )
        # ===== END CHANGE: balm advanced settings =====
        advanced_layout.addWidget(advanced_group)

        action_group = QtWidgets.QGroupBox("Actions")
        action_layout = QtWidgets.QVBoxLayout(action_group)
        action_layout.setContentsMargins(7, 7, 7, 7)
        action_layout.setSpacing(5)
        self.run_gicp_button = QtWidgets.QPushButton("GICP")
        self.run_gicp_button.setProperty("buttonRole", "primary")
        self.run_gicp_button.clicked.connect(self.run_gicp)
        self.accept_button = QtWidgets.QPushButton("Add")
        self.accept_button.setProperty("buttonRole", "secondary")
        self.accept_button.clicked.connect(self.accept_constraint)
        self.accept_button.setEnabled(False)
        self.replace_edge_button = QtWidgets.QPushButton("Replace")
        self.replace_edge_button.setProperty("buttonRole", "secondary")
        self.replace_edge_button.clicked.connect(self.replace_selected_edge)
        self.replace_edge_button.setEnabled(False)
        self.disable_edge_button = QtWidgets.QPushButton("Disable")
        self.disable_edge_button.clicked.connect(self.disable_selected_edge)
        self.restore_edge_button = QtWidgets.QPushButton("Restore")
        self.restore_edge_button.clicked.connect(self.restore_selected_edge)
        self.remove_manual_button = QtWidgets.QPushButton("Remove")
        self.remove_manual_button.setProperty("buttonRole", "quiet")
        self.remove_manual_button.clicked.connect(self.remove_selected_manual_constraint)
        self.optimize_button = QtWidgets.QPushButton("Audited PGO")
        self.optimize_button.setToolTip(
            "Optimize the accepted active constraint set and run the "
            "post-solve consistency audit."
        )
        self.optimize_button.setProperty("buttonRole", "primary")
        self.optimize_button.clicked.connect(self.run_optimization)
        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.setProperty("buttonRole", "primary")
        self.export_button.clicked.connect(self.export_final_result)
        self.compare_pgo_balm_button = QtWidgets.QPushButton("Compare PGO/BALM")
        self.compare_pgo_balm_button.setProperty("buttonRole", "secondary")
        self.compare_pgo_balm_button.setToolTip(
            "Build two maps from the same keyframes and open them in "
            "CloudCompare: PGO is orange and Final Map Refinement is blue."
        )
        self.compare_pgo_balm_button.clicked.connect(
            self.export_pgo_balm_comparison
        )
        self.compare_pgo_balm_button.setEnabled(False)
        # ===== BEGIN CHANGE: balm refinement button =====
        self.balm_button = QtWidgets.QPushButton("Final Refinement")
        self.balm_button.setProperty("buttonRole", "secondary")
        self.balm_button.setToolTip(
            "Run Final Map Refinement using double-sided BALM plane bundle "
            "adjustment."
        )
        self.balm_button.clicked.connect(self.run_balm_refinement)
        self.balm_button.setEnabled(False)
        # ===== END CHANGE: balm refinement button =====
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setProperty("buttonRole", "quiet")
        self.undo_button.clicked.connect(self.undo_last_change)
        self.undo_button.setEnabled(False)
        yaw_steps_label = QtWidgets.QLabel("Yaw")
        yaw_steps_label.setObjectName("SubtleText")
        auto_yaw_row = QtWidgets.QWidget()
        auto_yaw_row_layout = QtWidgets.QHBoxLayout(auto_yaw_row)
        auto_yaw_row_layout.setContentsMargins(0, 0, 0, 0)
        auto_yaw_row_layout.setSpacing(6)
        auto_yaw_row_layout.addWidget(yaw_steps_label)
        auto_yaw_row_layout.addWidget(self.auto_yaw_steps_spin)
        self.auto_yaw_button.setText("Auto Yaw")
        self.auto_yaw_button.setProperty("buttonRole", "secondary")
        auto_yaw_row_layout.addWidget(self.auto_yaw_button, 1)
        self.auto_seed_button = QtWidgets.QPushButton(INITIAL_LOOP_SEARCH_LABEL)
        self.auto_seed_button.setProperty("buttonRole", "secondary")
        self.auto_seed_button.setToolTip(
            "Retrieve large-drift revisit candidates with Scan Context (yaw-aligned), "
            "verify each with GICP under the quality gate, and add the survivors as "
            "initial loop constraints. Admission uses registration and "
            "pose-graph consistency checks; optional map diagnostics do not "
            "decide whether an initial loop is kept."
        )
        self.auto_seed_button.clicked.connect(
            lambda: self.run_auto_seed_async(silent=False))
        self.env_combo = QtWidgets.QComboBox()
        self.env_combo.addItem("Indoor", "indoor")
        self.env_combo.addItem("Outdoor", "outdoor")
        self.env_combo.setPlaceholderText("Select environment")
        self.env_combo.setToolTip(
            "Environment preset. Switching it sets BOTH the seed-retrieval\n"
            "descriptor AND the registration parameters:\n"
            "  Indoor  - gravity-canonicalized dual-channel descriptor,\n"
            "            0.2 m voxel, residual bound 0.18 m, BALM root 1 m\n"
            "  Outdoor - native Scan Context (80 m radius),\n"
            "            0.4 m voxel, residual bound 0.36 m, BALM root 4 m\n"
            "The bound has to follow the voxel: a correct registration cannot\n"
            "beat its own quantisation, so an outdoor loop judged at the indoor\n"
            "bound is rejected on quantisation alone. Genuine revisits in MCD\n"
            "kth_night_01 sit at 0.25 m residual -- they pass at 0.36 and fail\n"
            "at 0.18, which is the whole difference between repairing that\n"
            "session and not touching it."
        )
        self.env_combo.currentIndexChanged.connect(self._apply_environment_preset)
        self.repair_map_button = QtWidgets.QPushButton("⟳  Repair Map")
        self.repair_map_button.setProperty("buttonRole", "primary")
        self.repair_map_button.setToolTip(
            "One click for the production repair: Initial Loop Search -> "
            "Audited PGO -> Final Map Refinement. Optional map-inconsistency "
            "diagnostics are inspection-only.\n"
            "While Final Map Refinement is running, Stop Repair cancels its "
            "isolated worker and retains the completed PGO result.\n"
            "Set the environment first -- it decides the voxel and the residual "
            "bound, and getting it wrong is the difference between finding a "
            "session's loops and finding none of them."
        )
        self.repair_map_button.clicked.connect(self.run_repair_map)
        self.gate_label = QtWidgets.QLabel("")
        self.gate_label.setToolTip(
            "The quality gate a registration must pass to be accepted. The "
            "residual bound tracks the voxel, so it changes with the "
            "environment preset."
        )
        self.gate_label.setStyleSheet("color: #888; font-size: 10px;")
        self.voxel_spin.valueChanged.connect(lambda _: self._refresh_gate_label())
        # NOT applying the preset here, deliberately. currentIndexChanged only
        # fires on a CHANGE, so a session opened and repaired without touching
        # the dropdown runs on the widgets' construction defaults instead: a 0.0
        # registration voxel (floored to 0.05 downstream) against the preset's
        # 0.20, and different correspondence, BALM and target-window values with
        # it. That is what every in-house repair has actually used, including
        # the one whose map was accepted; applying the preset at startup on
        # 2026-08-03 changed the result and the map got worse. The defaults and
        # the preset need to be reconciled deliberately, with the map judged
        # each time -- not by flipping which of the two silently wins.
        self.repair_progress = QtWidgets.QProgressBar()
        self.repair_progress.setRange(0, 100)
        self.repair_progress.setValue(0)
        self.repair_progress.setTextVisible(True)
        self.repair_progress.setFormat("Ready")
        self.repair_progress.setToolTip(
            "Progress of Initial Loop Search, candidate verification, PGO and "
            "Final Map Refinement. "
            "A moving bar means the current operation has no reliable item count.")
        self.repair_progress.hide()
        action_layout.addWidget(
            self._build_action_section(
                "Match",
                [
                    (self.run_gicp_button, 0, 0, 1, 2),
                    (auto_yaw_row, 1, 0, 1, 2),
                    (self.env_combo, 2, 0, 1, 1),
                    (self.auto_seed_button, 2, 1, 1, 1),
                    (self.gate_label, 3, 0, 1, 2),
                    (self.repair_map_button, 4, 0, 1, 2),
                    (self.repair_progress, 5, 0, 1, 2),
                ],
            )
        )
        action_layout.addWidget(
            self._build_action_section(
                "Edit Graph",
                [
                    (self.accept_button, 0, 0, 1, 1),
                    (self.replace_edge_button, 0, 1, 1, 1),
                    (self.disable_edge_button, 1, 0, 1, 1),
                    (self.restore_edge_button, 1, 1, 1, 1),
                    (self.remove_manual_button, 2, 0, 1, 2),
                ],
            )
        )
        action_layout.addWidget(
            self._build_action_section(
                "Commit",
                [
                    (self.optimize_button, 0, 0, 1, 1),
                    (self.undo_button, 0, 1, 1, 1),
                    (self.balm_button, 1, 0, 1, 1),
                    (self.balm_auto_check, 1, 1, 1, 1),
                    (self.compare_pgo_balm_button, 2, 0, 1, 2),
                    (self.export_button, 3, 0, 1, 2),
                ],
            )
        )
        summary_layout.addWidget(action_group)

        summary_layout.addStretch(1)
        advanced_layout.addStretch(1)
        control_tabs.addTab(self._wrap_control_tab(summary_tab), "Summary")
        control_tabs.addTab(self._wrap_control_tab(advanced_tab), "Advanced")
        layout.addWidget(control_tabs)
        self._update_edge_action_buttons()
        return container

    def _build_constraint_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.change_legend_label = QtWidgets.QLabel(
            "Green=Applied · Amber=Accepted · Gray=Disabled · Note supports double-click edit"
        )
        self.change_legend_label.setObjectName("SubtleText")
        self.change_legend_label.setWordWrap(True)
        layout.addWidget(self.change_legend_label)
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Status"))
        self.change_status_filter_combo = QtWidgets.QComboBox()
        self.change_status_filter_combo.addItems(
            ["All Status", "Accepted", "Applied", "Disabled"]
        )
        self.change_status_filter_combo.currentTextChanged.connect(self._on_graph_change_filter_changed)
        filter_row.addWidget(self.change_status_filter_combo)
        filter_row.addWidget(QtWidgets.QLabel("Type"))
        self.change_type_filter_combo = QtWidgets.QComboBox()
        self.change_type_filter_combo.addItems(
            ["All Types", "Manual Add", "Replace Existing Loop", "Disable Existing Loop"]
        )
        self.change_type_filter_combo.currentTextChanged.connect(self._on_graph_change_filter_changed)
        filter_row.addWidget(self.change_type_filter_combo)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)
        self.constraint_table = QtWidgets.QTableWidget(0, 13)
        self.constraint_table.setHorizontalHeaderLabels(
            [
                "Use",
                "Type",
                "Status",
                "Src",
                "Tgt",
                "Submap",
                "Range",
                "Fitness",
                "RMSE",
                "Noise",
                "Accepted@",
                "Applied@",
                "Note",
            ]
        )
        self.constraint_table.horizontalHeader().setStretchLastSection(True)
        self.constraint_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.constraint_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        self.constraint_table.itemChanged.connect(self._on_constraint_item_changed)
        self.constraint_table.itemSelectionChanged.connect(self._on_constraint_selection_changed)
        layout.addWidget(self.constraint_table)
        return widget

    def _build_log_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return widget

    def _make_double_spin(
        self,
        *,
        step: float,
        minimum: float,
        maximum: float,
        value: float,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setSingleStep(step)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setMinimumWidth(72)
        spin.setMaximumWidth(96)
        return spin

    def _pack_row(
        self,
        labels: tuple[str, ...],
        spins: tuple[QtWidgets.QDoubleSpinBox, ...],
    ) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for label, spin in zip(labels, spins):
            layout.addWidget(QtWidgets.QLabel(label))
            layout.addWidget(spin)
        layout.addStretch(1)
        return widget

    def _pack_vector_grid(
        self,
        labels: tuple[str, ...],
        spins: tuple[QtWidgets.QDoubleSpinBox, ...],
    ) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(2)
        for column, (label, spin) in enumerate(zip(labels, spins)):
            layout.addWidget(QtWidgets.QLabel(label), 0, column)
            layout.addWidget(spin, 1, column)
        return widget

    def _current_target_cloud_mode(self) -> str:
        mode = self.target_cloud_mode_combo.currentData()
        return str(mode if mode is not None else OFFICE_DEFAULT_TARGET_CLOUD_MODE)

    def _current_view_kind(self) -> str:
        return self.trajectory_view_combo.currentText().strip().lower()

    def _is_working_view(self) -> bool:
        return self._current_view_kind() == "working"

    def _capture_undo_snapshot(self) -> UndoSnapshot:
        constraints = copy.deepcopy(self.constraints)
        for constraint in constraints:
            # Preview points are reproducible from keyframe + transform and
            # are never consumed by undo restoration.  Copying them into every
            # snapshot multiplied Initial Loop Search memory by proposal count.
            constraint.source_points_world_final = np.empty(
                (0, 3), dtype=np.float64
            )
        return UndoSnapshot(
            pose_graph=copy.deepcopy(self.pose_graph),
            trajectory=copy.deepcopy(self.trajectory),
            constraints=constraints,
            disabled_loop_changes=copy.deepcopy(self.disabled_loop_changes),
            source_id=self.source_id,
            target_id=self.target_id,
            selected_edge_ref=copy.deepcopy(self.selected_edge_ref),
            candidate_replace_edge_uid=self._candidate_replace_edge_uid,
            working_revision=self._working_revision,
            session_dirty=self._session_dirty,
            last_output_dir=self._last_output_dir,
            pick_mode=self.pick_mode,
        )

    def _push_undo_snapshot(self) -> None:
        self._append_undo_snapshot(self._capture_undo_snapshot())

    def _append_undo_snapshot(self, snapshot: UndoSnapshot) -> None:
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self.MAX_UNDO_SNAPSHOTS:
            del self._undo_stack[:-self.MAX_UNDO_SNAPSHOTS]
        self.undo_button.setEnabled(True)

    def _restore_snapshot(self, snapshot: UndoSnapshot) -> None:
        self.pose_graph = snapshot.pose_graph
        self.trajectory = snapshot.trajectory
        self.constraints = copy.deepcopy(snapshot.constraints)
        self.disabled_loop_changes = copy.deepcopy(snapshot.disabled_loop_changes)
        self.source_id = snapshot.source_id
        self.target_id = snapshot.target_id
        self.selected_edge_ref = copy.deepcopy(snapshot.selected_edge_ref)
        self._candidate_replace_edge_uid = snapshot.candidate_replace_edge_uid
        self._working_revision = snapshot.working_revision
        self._session_dirty = snapshot.session_dirty
        self._last_output_dir = snapshot.last_output_dir
        self.pick_mode = snapshot.pick_mode
        self.current_preview = None
        self.last_result = None
        self._preview_scene_key = None
        self.accept_button.setEnabled(False)
        self.gicp_metrics_label.setText("Undo restored the previous working state.")
        if self.trajectory is not None and self.session_paths is not None:
            self.workspace = RegistrationWorkspace(self.session_paths.keyframe_dir, self.trajectory)
        else:
            self.workspace = None
        self.cloud_view.clear_scene()
        self._set_pick_mode(self.pick_mode)
        self._update_preview_summary(None)
        self._update_selection_labels()
        self._update_edge_action_buttons()
        self._rebuild_constraint_table()
        self._sync_constraint_table_selection()
        self._update_session_status_widgets()
        self._refresh_plot(preserve_view=False)
        self._update_cloud_interaction_controls()

    def _status_text(self, enabled: bool, applied_rev: Optional[int]) -> str:
        if not enabled:
            return "Disabled"
        if applied_rev is not None:
            return "Applied"
        return "Accepted"

    def _active_disabled_loop_changes(self) -> list[ExistingLoopChange]:
        return [change for change in self.disabled_loop_changes.values() if change.enabled]

    def _recompute_session_dirty(self) -> None:
        dirty = False
        for constraint in self.constraints:
            if constraint.enabled and constraint.applied_rev is None:
                dirty = True
                break
            if not constraint.enabled and constraint.applied_rev is not None:
                dirty = True
                break
        if not dirty:
            for change in self.disabled_loop_changes.values():
                if change.enabled and change.applied_rev is None:
                    dirty = True
                    break
                if not change.enabled and change.applied_rev is not None:
                    dirty = True
                    break
        self._session_dirty = dirty

    def _update_session_status_widgets(self) -> None:
        operation_idle = (
            self._optimizer_process is None
            and self._background_thread is None
            and self._repair_map_stage is None
        )
        self._sync_gui_compute_marker(not operation_idle)
        # Changing the environment halfway through Repair Map mixes indoor and
        # outdoor descriptor/GICP/BALM thresholds in one result.
        self.env_combo.setEnabled(operation_idle)
        manual_active = sum(1 for constraint in self.constraints if constraint.enabled)
        disabled_active = sum(1 for change in self.disabled_loop_changes.values() if change.enabled)
        self.working_rev_badge.setText(f"R{self._working_revision}")
        self.session_state_badge.setText("Dirty" if self._session_dirty else "OK")
        self.change_summary_badge.setText(f"M{manual_active} D{disabled_active}")
        if self._session_dirty:
            self.session_state_badge.setStyleSheet(
                "background:#fff4db;border:1px solid #f2c46d;border-radius:10px;padding:2px 8px;font-weight:600;color:#8a5300;"
            )
        else:
            self.session_state_badge.setStyleSheet(
                "background:#e7f8ee;border:1px solid #74c69d;border-radius:10px;padding:2px 8px;font-weight:600;color:#1f6f43;"
            )
        self.working_rev_badge.setStyleSheet(
            "background:#e8f1ff;border:1px solid #8ab4f8;border-radius:10px;padding:2px 8px;font-weight:600;color:#1e4f91;"
        )
        self.change_summary_badge.setStyleSheet(
            "background:#eef2f7;border:1px solid #d6dde6;border-radius:10px;padding:2px 8px;font-weight:600;color:#334155;"
        )
        if self.pose_graph is None or self.trajectory is None:
            self.working_graph_label.setText("No working graph loaded.")
            self.working_graph_label.setToolTip("No working graph loaded.")
        else:
            loop_count = len([edge for edge in self.pose_graph.loop_edges if edge.enabled])
            self.working_graph_label.setText(
                f"Rev {self._working_revision} · P{self.trajectory.size} · "
                f"L{loop_count} · {'Dirty' if self._session_dirty else 'Clean'}"
            )

            self.working_graph_label.setToolTip(
                f"revision={self._working_revision}\nposes={self.trajectory.size}\n"
                f"enabled_loops={loop_count}\nstate={'dirty pending changes' if self._session_dirty else 'clean'}"
            )
        export_enabled = self._last_output_dir is not None or self._session_dirty
        self.export_button.setEnabled(export_enabled)
        self.compare_pgo_balm_button.setEnabled(
            operation_idle and self._pgo_balm_comparison_sources() is not None
        )
        self.undo_button.setEnabled(bool(self._undo_stack))
        # ===== BEGIN CHANGE: balm button state =====
        balm_ready = (
            self._optimizer_process is None
            and self._background_thread is None
            and not self._session_dirty
            and self._last_output_dir is not None
            and (self._last_output_dir / "optimized_poses_tum.txt").is_file()
        )
        self.balm_button.setEnabled(balm_ready)
        self.auto_seed_button.setEnabled(
            self._optimizer_process is None
            and self._background_thread is None
            and self.workspace is not None
        )
        repair_running = self._repair_map_stage is not None
        if repair_running:
            self.repair_map_button.setEnabled(
                not self._repair_map_stop_requested)
        else:
            self.repair_map_button.setEnabled(
                self._optimizer_process is None
                and self._background_thread is None
                and self.workspace is not None
            )
        # ===== END CHANGE: balm button state =====

    def _sync_gui_compute_marker(self, busy: bool) -> None:
        """Expose only active GUI compute, without blocking on an idle window.

        The marker contains a PID and coarse stage label, never a session path.
        Consumers also validate the PID command line, so a stale marker left by
        a crash cannot block a later measured replay.
        """
        marker = getattr(self, "_gui_compute_marker", None)
        if marker is None:
            marker = _gui_compute_marker_path()
            self._gui_compute_marker = marker
        try:
            if busy:
                marker.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                temporary = marker.with_suffix(f".tmp.{os.getpid()}")
                label = (
                    getattr(self, "_background_label", "")
                    or getattr(self, "_repair_map_stage", "")
                    or "optimizer"
                )
                temporary.write_text(
                    f"{os.getpid()} {label}\n", encoding="utf-8")
                temporary.replace(marker)
            else:
                marker.unlink(missing_ok=True)
        except OSError:
            # GUI operation must remain usable if the runtime directory is
            # unavailable; the formal replay still detects child processes.
            pass

    def _update_target_cloud_mode_controls(self) -> None:
        mode = self._current_target_cloud_mode()
        temporal_mode = mode == TARGET_CLOUD_MODE_TEMPORAL_WINDOW
        self.target_neighbors_label.setText(
            "Win"
            if temporal_mode
            else "Nbr"
        )
        self.target_min_gap_spin.setEnabled(not temporal_mode)
        self.target_min_gap_label.setEnabled(not temporal_mode)
        self.target_min_gap_spin.setToolTip(
            "Only used by RS Spatial Submap mode."
            if temporal_mode
            else "Reject target candidates whose timestamps are too close to source."
        )

    def _on_target_cloud_mode_changed(self) -> None:
        self._update_target_cloud_mode_controls()
        self.schedule_preview_refresh(reset_camera=False)

    # ===== BEGIN CHANGE: manual align viewer controls =====
    def _manual_align_editable(self) -> bool:
        return bool(
            self._is_working_view()
            and self.workspace is not None
            and self.source_id is not None
            and self.target_id is not None
            and (self.selected_edge_ref is None or self.selected_edge_ref.edge_kind == "existing")
        )

    def _update_cloud_interaction_controls(self, *, apply_view_preset: bool = False) -> None:
        editable = self._manual_align_editable()
        if not editable and self.cloud_interaction_mode_combo.currentData() != INTERACTION_MODE_CAMERA:
            with QtCore.QSignalBlocker(self.cloud_interaction_mode_combo):
                self.cloud_interaction_mode_combo.setCurrentIndex(
                    self.cloud_interaction_mode_combo.findData(INTERACTION_MODE_CAMERA)
                )
        mode = str(self.cloud_interaction_mode_combo.currentData() or INTERACTION_MODE_CAMERA)
        lock_mode = str(self.cloud_lock_mode_combo.currentData() or LOCK_MODE_XY_YAW)
        drag_mode = str(self.cloud_drag_mode_combo.currentData() or EDIT_OPERATION_TRANSLATE)
        translation_step = float(self.manual_align_translation_step_spin.value())
        rotation_step = float(self.manual_align_rotation_step_spin.value())
        snap_view = self.manual_align_snap_view_check.isChecked()
        xy_lock = lock_mode == LOCK_MODE_XY_YAW
        self.cloud_drag_mode_combo.setEnabled(editable and xy_lock)
        if not xy_lock and self.cloud_drag_mode_combo.currentData() != EDIT_OPERATION_TRANSLATE:
            with QtCore.QSignalBlocker(self.cloud_drag_mode_combo):
                self.cloud_drag_mode_combo.setCurrentIndex(
                    self.cloud_drag_mode_combo.findData(EDIT_OPERATION_TRANSLATE)
                )
            drag_mode = EDIT_OPERATION_TRANSLATE
        self.cloud_view.set_interaction_mode(
            mode if editable else INTERACTION_MODE_CAMERA
        )
        self.cloud_view.set_manual_align_options(
            lock_mode=lock_mode,
            edit_operation=drag_mode,
            translation_step_m=translation_step,
            rotation_step_deg=rotation_step,
            snap_view=snap_view,
        )
        interaction_widgets = (
            self.cloud_interaction_mode_combo,
            self.cloud_lock_mode_combo,
            self.cloud_drag_mode_combo,
            self.manual_align_translation_step_spin,
            self.manual_align_rotation_step_spin,
            self.manual_align_snap_view_check,
            self.reset_manual_align_button,
        )
        for widget in interaction_widgets:
            widget.setEnabled(editable)
        if editable and mode == INTERACTION_MODE_EDIT_SOURCE and apply_view_preset and snap_view:
            self.cloud_view.apply_lock_view_preset()
        elif not editable:
            self.manual_align_status_label.setText("Camera mode")

    def _on_cloud_interaction_changed(self) -> None:
        if str(self.cloud_interaction_mode_combo.currentData() or INTERACTION_MODE_CAMERA) == INTERACTION_MODE_EDIT_SOURCE:
            self._set_display_mode("Preview")
        self._update_cloud_interaction_controls(apply_view_preset=True)

    def _on_manual_align_status(self, text: str) -> None:
        self.manual_align_status_label.setText(text)
        self.manual_align_status_label.setVisible("Edit" in text)

    def _on_camera_preset_changed(self, preset: str) -> None:
        for preset_name, button in self.camera_preset_buttons.items():
            with QtCore.QSignalBlocker(button):
                button.setChecked(preset_name == preset)

    def _set_delta_spin_values(self, values: np.ndarray) -> None:
        blockers = [QtCore.QSignalBlocker(spin) for spin in self.delta_spins.values()]
        try:
            for key, value in zip(
                ("x", "y", "z", "roll", "pitch", "yaw"),
                np.asarray(values, dtype=np.float64).tolist(),
            ):
                self.delta_spins[key].setValue(float(value))
        finally:
            del blockers

    def _clear_delta_silent(self) -> None:
        self._set_delta_spin_values(np.zeros(6, dtype=np.float64))

    def _set_delta_from_world_source_transform(self, transform_world_source: np.ndarray) -> None:
        if self.current_preview is None or self.current_preview.transform_world_source_initial is None:
            return
        base_transform = self.current_preview.transform_world_source_initial
        delta_local = np.linalg.inv(base_transform) @ np.asarray(transform_world_source, dtype=np.float64)
        self._set_delta_spin_values(matrix_to_xyz_rpy_deg(delta_local))

    def _set_delta_from_source_seed(self, source_id: int, transform_world_source: np.ndarray) -> None:
        if self.workspace is None:
            return
        base_transform = self.workspace.trajectory.transforms_world_sensor[int(source_id)]
        delta_local = np.linalg.inv(base_transform) @ np.asarray(transform_world_source, dtype=np.float64)
        self._set_delta_spin_values(matrix_to_xyz_rpy_deg(delta_local))

    def _unapplied_manual_seed_for_pair(self, source_id: int, target_id: int) -> Optional[ManualConstraint]:
        constraint = self._constraint_by_pair(source_id, target_id)
        if constraint is None or not constraint.enabled:
            return None
        if constraint.applied_rev is not None:
            return None
        return constraint

    def _prepare_delta_for_pair_preview(self, source_id: int, target_id: int) -> None:
        seed_constraint = self._unapplied_manual_seed_for_pair(source_id, target_id)
        if seed_constraint is None:
            self._clear_delta_silent()
            return
        self._set_delta_from_source_seed(source_id, seed_constraint.transform_world_source_final)
        self.append_log(
            "Using accepted manual edge as preview seed "
            f"{seed_constraint.target_id}->{seed_constraint.source_id}; run Optimize to bake it into Working."
        )

    def _invalidate_last_result_for_manual_align(self) -> None:
        if self.last_result is None and self.gicp_metrics_label.text().startswith("Manual align updated"):
            return
        self.last_result = None
        self.accept_button.setEnabled(False)
        self._set_display_mode("Preview")
        self.gicp_metrics_label.setText(
            "Manual align updated · rerun GICP."
            "<br><span style='color:#64748b;font-size:11px'>Viewer drag edits the current source seed.</span>"
        )
        self._update_edge_action_buttons()

    def _on_manual_align_changed(self, payload: ManualAlignUpdate) -> None:
        if not self._manual_align_editable():
            return
        self._set_delta_from_world_source_transform(payload.transform_world_source)
        self._invalidate_last_result_for_manual_align()
        self.schedule_preview_refresh(reset_camera=False)

    def _on_delta_spin_changed(self) -> None:
        if self.workspace is None or self.source_id is None or self.target_id is None:
            return
        self._invalidate_last_result_for_manual_align()
        self.schedule_preview_refresh(reset_camera=False)

    def _reset_manual_align(self) -> None:
        self._reset_delta()
        if self.manual_align_snap_view_check.isChecked():
            self.cloud_view.apply_lock_view_preset()
    # ===== END CHANGE: manual align viewer controls =====

    def _on_show_ghost_changed(self, *_args) -> None:
        self._update_trajectory_legend_label()
        self._refresh_plot(preserve_view=True)

    def _update_trajectory_legend_label(self) -> None:
        current_text = "Working" if self._is_working_view() else "Original"
        ghost_text = " · Overlay" if self.show_ghost_check.isChecked() else ""
        self.trajectory_legend_label.setText(
            "<span style='color:#4c78a8;font-weight:600'>Traj</span> · "
            "<span style='color:#d62728;font-weight:600'>Loop</span> · "
            "<span style='color:#8a8a8a;font-weight:600'>Disabled</span> · "
            "<span style='color:#2ca02c;font-weight:600'>Manual</span> · "
            "<span style='color:#f6d32d;font-weight:600'>Target</span> · "
            "<span style='color:#bc5090;font-weight:600'>Pair</span> · "
            "<span style='color:#ff7f0e;font-weight:600'>Src</span>/<span style='color:#17becf;font-weight:600'>Tgt</span>"
            f" <span style='color:#64748b'>| {current_text}{ghost_text}</span>"
        )

    def _update_cloud_display_controls(self) -> None:
        display_mode = str(self.display_mode_combo.currentData() or "preview")
        self.cloud_view.set_display_options(
            display_mode=display_mode,
            trajectory_point_size=float(self.trajectory_point_size_spin.value()),
            target_point_size=float(self.target_point_size_spin.value()),
            source_point_size=float(self.source_point_size_spin.value()),
            show_world_axis=self.show_world_axis_check.isChecked(),
            reset_camera=False,
        )
        self._update_cloud_interaction_controls()
        if display_mode == "preview":
            legend = "Before GICP: gray target · yellow subset · orange/cyan source seed"
        elif display_mode == "final":
            scene = getattr(self.cloud_view, "_scene", None)
            if scene is None or scene.final_source_points is None:
                legend = "After GICP: no result yet · run GICP or Auto Yaw first"
            else:
                legend = "After GICP: gray target · yellow subset · green registered source"
        else:
            legend = "Compare: gray target · yellow subset · source states"
        if self.show_world_axis_check.isChecked():
            legend += " · world axis"
        self.cloud_legend_label.setText(legend)

    def _current_cloud_point_size_summary(self) -> str:
        return (
            f"viewer_size(target={self.target_point_size_spin.value():.1f}, "
            f"source={self.source_point_size_spin.value():.1f}, "
            f"traj={self.trajectory_point_size_spin.value():.0f})"
        )

    def _gicp_log_cloud_summary(self, preview: RegistrationPreview) -> str:
        return (
            f"{self._current_cloud_point_size_summary()} | "
            f"points(target={preview.target_point_count}, "
            f"source={preview.source_points_local.shape[0]})"
        )

    def _trajectory_scene_payload(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.trajectory is None or self.trajectory.size == 0:
            return None, None
        return (
            self.trajectory.positions_xyz,
            np.arange(self.trajectory.size, dtype=np.float64),
        )

    def _make_preview_scene(
        self,
        *,
        preview: RegistrationPreview,
        initial_source_points: Optional[np.ndarray] = None,
        adjusted_source_points: Optional[np.ndarray] = None,
        final_source_points: Optional[np.ndarray] = None,
        transform_world_source_initial: Optional[np.ndarray] = None,
        transform_world_source_adjusted: Optional[np.ndarray] = None,
        transform_world_source_final: Optional[np.ndarray] = None,
    ) -> PreviewScene:
        trajectory_points, trajectory_values = self._trajectory_scene_payload()
        selected_target_trajectory_points = None
        if self.trajectory is not None and preview.target_frame_indices:
            selected_indices = np.asarray(preview.target_frame_indices, dtype=np.int64)
            selected_target_trajectory_points = self.trajectory.positions_xyz[selected_indices]
        return PreviewScene(
            target_points=_bounded_target_display_view(
                preview.target_points_world),
            trajectory_points=trajectory_points,
            trajectory_values=trajectory_values,
            selected_target_trajectory_points=selected_target_trajectory_points,
            editable_source_points_local=_bounded_display_view(
                preview.source_points_local, PREVIEW_SOURCE_DISPLAY_LIMIT),
            editable_source_transform=(
                transform_world_source_adjusted
                if transform_world_source_adjusted is not None
                else transform_world_source_initial
            ),
            initial_source_points=_bounded_display_view(
                initial_source_points, PREVIEW_SOURCE_DISPLAY_LIMIT),
            adjusted_source_points=_bounded_display_view(
                adjusted_source_points, PREVIEW_SOURCE_DISPLAY_LIMIT),
            final_source_points=_bounded_display_view(
                final_source_points, PREVIEW_SOURCE_DISPLAY_LIMIT),
            transform_world_target=preview.transform_world_target,
            transform_world_source_initial=transform_world_source_initial,
            transform_world_source_adjusted=transform_world_source_adjusted,
            transform_world_source_final=transform_world_source_final,
        )

    def _projects_root(self) -> Optional[Path]:
        if self.session_paths is None:
            return None
        return self.session_paths.session_root / "manual_loop_projects"

    def _latest_project_pointer_path(self) -> Optional[Path]:
        root = self._projects_root()
        if root is None:
            return None
        return root / "latest_project.json"

    def _new_project_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _project_payload_summary(self) -> dict:
        return {
            "project_id": self._project_id,
            "session_root": str(self.session_paths.session_root) if self.session_paths is not None else None,
            "g2o_path": str(self.session_paths.g2o_path) if self.session_paths is not None else None,
            "tum_path": str(self.session_paths.tum_path) if self.session_paths is not None else None,
            "keyframe_dir": str(self.session_paths.keyframe_dir) if self.session_paths is not None else None,
            "working_revision": self._working_revision,
            "session_dirty": self._session_dirty,
            "last_output_dir": str(self._last_output_dir) if self._last_output_dir is not None else None,
            "latest_export_dir": str(self._latest_export_dir) if self._latest_export_dir is not None else None,
            "pick_mode": self.pick_mode,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "selected_edge_ref": (
                {
                    "edge_kind": self.selected_edge_ref.edge_kind,
                    "edge_uid": self.selected_edge_ref.edge_uid,
                }
                if self.selected_edge_ref is not None
                else None
            ),
            "candidate_replace_edge_uid": self._candidate_replace_edge_uid,
            "environment": (
                str(self.env_combo.currentData())
                if self.env_combo.currentData() in self.ENV_PRESETS else None
            ),
            "constraints": [self._serialize_manual_constraint(constraint) for constraint in self.constraints],
            "disabled_loop_changes": {
                str(edge_uid): {
                    "edge_uid": change.edge_uid,
                    "enabled": change.enabled,
                    "accepted_rev": change.accepted_rev,
                    "applied_rev": change.applied_rev,
                    "note": change.note,
                }
                for edge_uid, change in self.disabled_loop_changes.items()
            },
        }

    def _serialize_manual_constraint(self, constraint: ManualConstraint) -> dict:
        return {
            "manual_uid": constraint.manual_uid,
            "enabled": constraint.enabled,
            "source_id": constraint.source_id,
            "target_id": constraint.target_id,
            "target_cloud_mode": constraint.target_cloud_mode,
            "target_neighbors": constraint.target_neighbors,
            "min_time_gap_sec": constraint.min_time_gap_sec,
            "target_map_voxel_size": constraint.target_map_voxel_size,
            "transform_world_source_final": constraint.transform_world_source_final.tolist(),
            "transform_target_source_final": constraint.transform_target_source_final.tolist(),
            "fitness": constraint.fitness,
            "inlier_rmse": constraint.inlier_rmse,
            "variance_t_m2": list(constraint.variance_t_m2),
            "variance_r_rad2": list(constraint.variance_r_rad2),
            "replaces_edge_uid": constraint.replaces_edge_uid,
            "accepted_rev": constraint.accepted_rev,
            "applied_rev": constraint.applied_rev,
            "note": constraint.note,
        }

    def _deserialize_manual_constraint(self, payload: dict) -> ManualConstraint:
        transform_world_source_final = np.asarray(
            payload["transform_world_source_final"], dtype=np.float64
        )
        transform_target_source_final = np.asarray(
            payload["transform_target_source_final"], dtype=np.float64
        )
        source_local = load_xyz_points(self.session_paths.keyframe_dir / f"{int(payload['source_id'])}.pcd")
        source_points_world_final = transform_points(source_local, transform_world_source_final)
        return ManualConstraint(
            manual_uid=int(payload["manual_uid"]),
            enabled=bool(payload["enabled"]),
            source_id=int(payload["source_id"]),
            target_id=int(payload["target_id"]),
            target_cloud_mode=str(payload["target_cloud_mode"]),
            target_neighbors=int(payload["target_neighbors"]),
            min_time_gap_sec=float(payload["min_time_gap_sec"]),
            target_map_voxel_size=float(payload["target_map_voxel_size"]),
            transform_world_source_final=transform_world_source_final,
            transform_target_source_final=transform_target_source_final,
            source_points_world_final=source_points_world_final,
            fitness=float(payload["fitness"]),
            inlier_rmse=float(payload["inlier_rmse"]),
            variance_t_m2=tuple(float(v) for v in payload["variance_t_m2"]),
            variance_r_rad2=tuple(float(v) for v in payload["variance_r_rad2"]),
            replaces_edge_uid=(
                None if payload.get("replaces_edge_uid") is None else int(payload["replaces_edge_uid"])
            ),
            accepted_rev=int(payload.get("accepted_rev", 0)),
            applied_rev=(
                None if payload.get("applied_rev") is None else int(payload["applied_rev"])
            ),
            note=str(payload.get("note", "")),
        )

    def _write_json(self, path: Optional[Path], payload: dict) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _append_operation_entry(
        self,
        *,
        message: str,
        event: str = "log",
        payload: Optional[dict] = None,
    ) -> None:
        if self._project_ops_path is None:
            return
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "message": message,
        }
        if payload:
            record["payload"] = payload
        with self._project_ops_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_project_state(self) -> None:
        if self._project_state_path is None or self.session_paths is None:
            return
        payload = {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **self._project_payload_summary(),
        }
        self._write_json(self._project_state_path, payload)
        latest_pointer = self._latest_project_pointer_path()
        if latest_pointer is not None and self._project_dir is not None:
            self._write_json(
                latest_pointer,
                {
                    "project_id": self._project_id,
                    "project_dir": str(self._project_dir),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

    def _write_run_context(self, output_dir: Path) -> None:
        payload = {
            "project_id": self._project_id,
            "project_dir": str(self._project_dir) if self._project_dir is not None else None,
            "project_state": str(self._project_state_path) if self._project_state_path is not None else None,
            "execution_log": str(self._project_log_path) if self._project_log_path is not None else None,
            "operations_log": str(self._project_ops_path) if self._project_ops_path is not None else None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "environment": (
                str(self.env_combo.currentData())
                if self.env_combo.currentData() in self.ENV_PRESETS else None
            ),
            "pgo_policy": getattr(self, "_last_optimizer_policy", None),
            "production_pgo_profile": str(PRODUCTION_PGO_PROFILE["name"]),
        }
        self._write_json(output_dir / "run_context.json", payload)

    def _write_export_manifest(self, export_dir: Path, run_dir: Path) -> None:
        manifest = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "project_id": self._project_id,
            "project_dir": str(self._project_dir) if self._project_dir is not None else None,
            "run_dir": str(run_dir),
            "run_context": str(run_dir / "run_context.json"),
            "report_json": str(run_dir / "manual_loop_report.json"),
            "pose_graph_g2o": str(run_dir / "pose_graph.g2o"),
            "optimized_tum": str(run_dir / "optimized_poses_tum.txt"),
            "pose_policy": "complete_pgo_se3",
            "ground_alignment_applied": False,
            "global_map_pcd": str(run_dir / "scans.pcd"),
            "scan_context_scd": str(run_dir / "scans.scd"),
            "trajectory_pcd": str(run_dir / "trajectory.pcd"),
        }
        self._write_json(export_dir / "export_manifest.json", manifest)
        (export_dir / "selected_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
        symlink_path = export_dir / "run"
        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(run_dir)
        except OSError:
            pass
        exports_root = export_dir.parent
        self._write_json(
            exports_root / "latest_export.json",
            {
                "export_dir": str(export_dir),
                "run_dir": str(run_dir),
                "project_id": self._project_id,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

    def _ensure_run_map_outputs(
        self,
        run_dir: Path,
        *,
        keyframe_dir: Path | None = None,
        voxel_leaf: float | None = None,
        log_fn=None,
    ) -> None:
        """Build export artifacts without requiring access to Qt widgets.

        ``export_final_result`` calls this from a worker thread, so all values
        normally read from widgets are captured on the GUI thread first and
        progress is routed through a queued signal.
        """
        log = log_fn if log_fn is not None else self.append_log
        output_map = run_dir / "scans.pcd"
        legacy_output_map = run_dir / "global_map_manual_imu.pcd"
        output_trajectory = run_dir / "trajectory.pcd"
        output_scan_context = run_dir / "scans.scd"
        optimized_tum = run_dir / "optimized_poses_tum.txt"
        policy_path = run_dir / "map_export_policy.json"
        current_policy = False
        if policy_path.is_file():
            try:
                policy = json.loads(policy_path.read_text(encoding="utf-8"))
                current_policy = (
                    policy.get("pose_policy") == "complete_pgo_se3"
                    and policy.get("ground_alignment_applied") is False
                )
            except (OSError, ValueError):
                current_policy = False
        if not output_map.is_file() and legacy_output_map.is_file():
            output_map = legacy_output_map
        if (
            current_policy
            and output_map.is_file()
            and output_trajectory.is_file()
            and output_scan_context.is_file()
        ):
            return
        if keyframe_dir is None and self.session_paths is None:
            raise RuntimeError("Session paths are unavailable for map export.")
        if keyframe_dir is None:
            keyframe_dir = self.session_paths.keyframe_dir
        if voxel_leaf is None:
            voxel_leaf = float(self.export_map_voxel_spin.value())
        if not optimized_tum.is_file():
            raise RuntimeError(f"Missing optimized TUM for export: {optimized_tum}")

        if current_policy and output_map.is_file() and output_trajectory.is_file():
            log(f"Export building corrected Scan Context database from {optimized_tum.name}")
            build_scan_context_from_tum(
                tum_path=optimized_tum,
                keyframe_dir=keyframe_dir,
                output_scan_context=output_scan_context,
                log_fn=log,
            )
            self._write_json(
                policy_path,
                {
                    "pose_policy": "complete_pgo_se3",
                    "ground_alignment_applied": False,
                    "scan_context_gravity_policy": "mapping_time_gravity_when_enabled",
                },
            )
            return

        if output_map.is_file() or output_trajectory.is_file() or output_scan_context.is_file():
            log(
                "Export rebuilding legacy map outputs so PCD, trajectory, and SCD "
                "all use the complete PGO SE(3) poses without ground alignment."
            )
        log(
            f"Export building final map from {optimized_tum.name} with voxel={voxel_leaf:.3f} m"
        )
        map_point_count, trajectory_count, elapsed = build_map_and_trajectory_from_tum(
            tum_path=optimized_tum,
            keyframe_dir=keyframe_dir,
            output_map=output_map,
            output_trajectory=output_trajectory,
            voxel_leaf=voxel_leaf,
            output_scan_context=output_scan_context,
            log_fn=log,
        )
        self._write_json(
            policy_path,
            {
                "pose_policy": "complete_pgo_se3",
                "ground_alignment_applied": False,
                "scan_context_gravity_policy": "mapping_time_gravity_when_enabled",
            },
        )
        report_path = run_dir / "manual_loop_report.json"
        update_report_map_fields(
            report_path,
            map_point_count=map_point_count,
            map_build_elapsed_sec=elapsed,
        )
        log(
            f"Export map build finished for {run_dir.name}: map_points={map_point_count}, "
            f"trajectory_points={trajectory_count}, elapsed={elapsed:.2f}s"
        )

    def _load_project_state_from_dir(self, project_dir: Path, paths: SessionPaths) -> bool:
        try:
            state_path = project_dir / "project_state.json"
            if not state_path.is_file():
                return False
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if str(payload.get("session_root", "")) != str(paths.session_root):
                return False
            self._project_dir = project_dir
            self._project_state_path = state_path
            self._project_log_path = project_dir / "execution.log"
            self._project_ops_path = project_dir / "operations.jsonl"
            self._project_id = str(payload.get("project_id", project_dir.name))
            self._remember_project_state_path(state_path)
            if self._project_log_path.is_file():
                self.log_text.setPlainText(self._project_log_path.read_text(encoding="utf-8"))
                self.log_text.moveCursor(QtGui.QTextCursor.End)
            self.constraints = [
                self._deserialize_manual_constraint(item)
                for item in payload.get("constraints", [])
            ]
            self.disabled_loop_changes = {
                int(edge_uid): ExistingLoopChange(
                    edge_uid=int(change["edge_uid"]),
                    enabled=bool(change["enabled"]),
                    accepted_rev=int(change.get("accepted_rev", 0)),
                    applied_rev=(
                        None if change.get("applied_rev") is None else int(change["applied_rev"])
                    ),
                    note=str(change.get("note", "")),
                )
                for edge_uid, change in payload.get("disabled_loop_changes", {}).items()
            }
            self._working_revision = int(payload.get("working_revision", 0))
            self._session_dirty = bool(payload.get("session_dirty", False))
            self._last_output_dir = (
                Path(payload["last_output_dir"]).expanduser()
                if payload.get("last_output_dir")
                else None
            )
            self._latest_export_dir = (
                Path(payload["latest_export_dir"]).expanduser()
                if payload.get("latest_export_dir")
                else None
            )
            self.pick_mode = str(payload.get("pick_mode", "nodes"))
            self.source_id = payload.get("source_id")
            self.target_id = payload.get("target_id")
            edge_ref_payload = payload.get("selected_edge_ref")
            self.selected_edge_ref = (
                SelectedEdgeRef(
                    edge_kind=str(edge_ref_payload["edge_kind"]),
                    edge_uid=int(edge_ref_payload["edge_uid"]),
                )
                if edge_ref_payload is not None
                else None
            )
            self._candidate_replace_edge_uid = payload.get("candidate_replace_edge_uid")
            environment = payload.get("environment")
            if environment in self.ENV_PRESETS:
                with QtCore.QSignalBlocker(self.env_combo):
                    self.env_combo.setCurrentIndex(
                        self.env_combo.findData(environment)
                    )
                self._apply_environment_preset()

            self.pose_graph = copy.deepcopy(self.original_pose_graph)
            for edge_uid, change in self.disabled_loop_changes.items():
                edge = self.pose_graph.edge_index_by_uid.get(edge_uid)
                if edge is not None:
                    edge.enabled = not change.enabled

            self._balm_ghost_suggestions = self._load_balm_ghost_suggestions()
            if self._last_output_dir is not None:
                restored_tum = self._last_output_dir / "optimized_poses_tum.txt"
                if restored_tum.is_file():
                    self.trajectory = load_tum_trajectory(restored_tum)
                    self.workspace = RegistrationWorkspace(paths.keyframe_dir, self.trajectory)
            self._next_manual_uid = (
                max((constraint.manual_uid for constraint in self.constraints), default=0) + 1
            )
            return True
        except Exception:
            return False

    def _load_latest_project_state(self, paths: SessionPaths) -> bool:
        pointer_path = paths.session_root / "manual_loop_projects" / "latest_project.json"
        if not pointer_path.is_file():
            return False
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            project_dir = Path(pointer["project_dir"]).expanduser()
        except Exception:
            return False
        return self._load_project_state_from_dir(project_dir, paths)

    def _start_or_resume_project(self, paths: SessionPaths) -> None:
        self.log_text.clear()
        if self._requested_project_dir is not None:
            requested_project_dir = self._requested_project_dir
            self._requested_project_dir = None
            if self._load_project_state_from_dir(requested_project_dir, paths):
                self.append_log(
                    f"Opened edit project {self._project_id} from {self._project_dir}",
                    event="project_open",
                )
                return
            self.append_log(
                f"Requested project {requested_project_dir} could not be restored. Falling back to latest project or new project.",
                event="project_open_failed",
            )
        if self._load_latest_project_state(paths):
            self.append_log(
                f"Resumed edit project {self._project_id} from {self._project_dir}",
                event="project_resume",
            )
            return

        project_root = paths.session_root / "manual_loop_projects"
        project_root.mkdir(parents=True, exist_ok=True)
        project_id = self._new_project_id()
        project_dir = project_root / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        self._project_dir = project_dir
        self._project_state_path = project_dir / "project_state.json"
        self._project_log_path = project_dir / "execution.log"
        self._project_ops_path = project_dir / "operations.jsonl"
        self._project_id = project_id
        self._write_json(
            project_root / "latest_project.json",
            {
                "project_id": project_id,
                "project_dir": str(project_dir),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        self._save_project_state()
        self.append_log(
            f"Started new edit project {project_id} at {project_dir}",
            event="project_start",
        )

    def append_log(
        self,
        message: str,
        *,
        event: str = "log",
        payload: Optional[dict] = None,
    ) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_text.appendPlainText(line)
        if self._project_log_path is not None:
            self._project_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._project_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        self._append_operation_entry(message=message, event=event, payload=payload)

    def _browse_session_root(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Session Root",
            self._session_root_dialog_directory(),
        )
        if directory:
            self.session_root_edit.setText(directory)
            current_g2o_text = self.g2o_edit.text().strip()
            if current_g2o_text:
                current_g2o = Path(current_g2o_text).expanduser()
                if current_g2o.exists() and not self._is_g2o_under_session_root(Path(directory), current_g2o):
                    self.g2o_edit.clear()
            self._settings.setValue("browser/last_session_root", directory)
            self._settings.sync()

    def _browse_g2o(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select pose_graph.g2o",
            self._g2o_dialog_directory(),
            filter="G2O Files (*.g2o);;All Files (*)",
        )
        if path:
            self.g2o_edit.setText(path)
            self._settings.setValue("browser/last_g2o_path", path)
            self._settings.sync()

    def _browse_project_state(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Edit Project",
            self._project_state_dialog_directory(),
            filter="Project State (project_state.json);;JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        project_state_path = Path(path).expanduser()
        try:
            payload = json.loads(project_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._show_error("Open Project Failed", f"Failed to read project state:\n{exc}")
            return
        session_root = payload.get("session_root")
        g2o_path = payload.get("g2o_path")
        if not session_root:
            self._show_error("Open Project Failed", "project_state.json does not contain session_root.")
            return
        self.session_root_edit.setText(str(session_root))
        self.g2o_edit.setText(str(g2o_path or ""))
        self._remember_project_state_path(project_state_path)
        self._requested_project_dir = project_state_path.parent
        self.load_session()

    def _set_pick_mode(self, mode: str) -> None:
        self._release_plot_toolbar_navigation()
        self.pick_mode = mode
        self.trajectory_canvas.set_interaction_mode(mode)
        self.pick_nodes_button.setChecked(mode == "nodes")
        self.pick_edges_button.setChecked(mode == "edges")
        self._update_plot_help_state()

    def _handle_plot_hover(self, kind: str, payload) -> None:
        if kind == "node" and payload is not None and self.trajectory is not None:
            pose = self.trajectory.positions_xyz[payload]
            timestamp = self.trajectory.timestamps[payload]
            self.hover_label.setText(
                f"Hover: node {payload} | t={timestamp:.3f} | xyz=({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})"
            )
            return

        if kind == "edge" and payload is not None:
            edge = self._edge_from_ref(payload)
            if edge is not None:
                self.hover_label.setText(f"Hover: {edge.summary()}")
                return

        self.hover_label.setText("Hover: none")

    def _handle_plot_selection(self, kind: str, payload) -> None:
        if self._background_thread is not None or self._optimizer_process is not None:
            self.gicp_metrics_label.setText(
                "Operation running · selection is locked until it finishes.")
            return
        if not self._is_working_view():
            return
        if kind == "node":
            self._handle_node_pick(int(payload))
            return
        if kind == "edge":
            self._select_edge(payload, reset_camera=True)

    def _handle_node_pick(self, node_id: int) -> None:
        if self.pose_graph is None:
            return

        if self.source_id is None or self.target_id is not None or self.selected_edge_ref is not None:
            self.source_id = node_id
            self.target_id = None
            self.selected_edge_ref = None
            self._candidate_replace_edge_uid = None
            self.current_preview = None
            self.last_result = None
            self.accept_button.setEnabled(False)
            self._set_display_mode("Preview")
            self.gicp_metrics_label.setText("Src selected · pick target.")
            self.append_log(f"Picked first node {node_id}")
        else:
            if node_id == self.source_id:
                self.append_log("Source and target cannot be the same node.")
                return
            first_node = self.source_id
            self.source_id = max(first_node, node_id)
            self.target_id = min(first_node, node_id)
            self.selected_edge_ref = None
            self._candidate_replace_edge_uid = None
            self.current_preview = None
            self.last_result = None
            self.accept_button.setEnabled(False)
            self._prepare_delta_for_pair_preview(self.source_id, self.target_id)
            self._set_display_mode("Preview")
            self.gicp_metrics_label.setText("Pair ready · preview refreshed.")
            self.append_log(
                f"Picked node pair target={self.target_id}, source={self.source_id}"
            )
            self.schedule_preview_refresh(reset_camera=True)

        self._update_selection_labels()
        self._update_edge_action_buttons()
        self._sync_constraint_table_selection()
        self._refresh_plot(preserve_view=True)
        self._save_project_state()

    def _select_edge(self, edge_ref: SelectedEdgeRef, *, reset_camera: bool) -> None:
        edge = self._edge_from_ref(edge_ref)
        if edge is None:
            return

        self.selected_edge_ref = edge_ref
        self.source_id = edge.source_id
        self.target_id = edge.target_id
        self._candidate_replace_edge_uid = edge.edge_uid if edge.edge_type == "loop_existing" else None
        self.last_result = None
        self.accept_button.setEnabled(False)
        self._update_selection_labels()
        self._update_edge_action_buttons()
        self._refresh_plot(preserve_view=True)
        self._sync_constraint_table_selection()

        if edge_ref.edge_kind == "manual_added":
            constraint = self._constraint_by_uid(edge_ref.edge_uid)
            if constraint is not None:
                self._preview_manual_constraint(constraint, reset_camera=reset_camera)
            return

        self._preview_existing_edge(edge, reset_camera=reset_camera)

    def _preview_existing_edge(self, edge: EdgeRecord, *, reset_camera: bool) -> None:
        if self.workspace is None:
            return
        self._set_display_mode("Preview")
        config = self._current_registration_config()
        delta_transform_local = self._current_delta_transform().copy()
        edge_uid = edge.edge_uid

        def apply(preview) -> None:
            if (
                self.selected_edge_ref is None
                or self.selected_edge_ref.edge_uid != edge_uid
            ):
                return
            self.last_result = None
            self.accept_button.setEnabled(False)
            self.current_preview = preview
            self._update_preview_summary(preview)
            if edge.edge_type == "loop_existing":
                self.gicp_metrics_label.setText(
                    f"Loop selected · {edge.target_id}->{edge.source_id}"
                    "<br><span style='color:#64748b;font-size:11px'>Adjust delta, run GICP, then replace.</span>"
                )
            else:
                self.gicp_metrics_label.setText(
                    f"Odom selected · {edge.target_id}->{edge.source_id}"
                    "<br><span style='color:#64748b;font-size:11px'>Inspect only.</span>"
                )
            self._show_scene(
                self._make_preview_scene(
                    preview=preview,
                    initial_source_points=preview.source_points_world_initial,
                    adjusted_source_points=preview.source_points_world_adjusted,
                    transform_world_source_initial=preview.transform_world_source_initial,
                    transform_world_source_adjusted=preview.transform_world_source_adjusted,
                ),
                scene_key=(
                    "existing",
                    edge.edge_uid,
                    preview.target_cloud_mode,
                    preview.target_neighbors,
                    round(preview.min_time_gap_sec, 6),
                    round(preview.target_map_voxel_size, 6),
                ),
                reset_camera=reset_camera,
            )
            self._refresh_plot(preserve_view=True)
            self.append_log(f"Selected edge {edge.summary()}")
            self._save_project_state()

        self._start_preview_build(
            source_id=edge.source_id,
            target_id=edge.target_id,
            delta_transform=delta_transform_local,
            target_cloud_mode=config.target_cloud_mode,
            target_neighbors=config.target_neighbors,
            min_time_gap_sec=config.min_time_gap_sec,
            target_map_voxel_size=config.target_map_voxel_size,
            on_finished=apply,
        )

    @staticmethod
    def _rebase_constraint_world_source(
        transform_world_target: np.ndarray,
        transform_target_source: np.ndarray,
    ) -> np.ndarray:
        return (
            np.asarray(transform_world_target, dtype=np.float64)
            @ np.asarray(transform_target_source, dtype=np.float64)
        )

    def _preview_manual_constraint(self, constraint: ManualConstraint, *, reset_camera: bool) -> None:
        if self.workspace is None:
            return
        self._set_display_mode("Final")
        manual_uid = constraint.manual_uid

        def apply(preview) -> None:
            if (
                self.selected_edge_ref is None
                or self.selected_edge_ref.edge_kind != "manual_added"
                or self.selected_edge_ref.edge_uid != manual_uid
            ):
                return
            self.current_preview = preview
            self._update_preview_summary(preview)
            # A constraint stores a target->source measurement, which remains
            # valid after PGO. Its cached world-frame source pose/points do not:
            # once the working trajectory moves, displaying those stale values
            # makes Final appear on the old side of Preview. Rebase the
            # measurement through the target's CURRENT working pose.
            transform_world_source_final = self._rebase_constraint_world_source(
                preview.transform_world_target,
                constraint.transform_target_source_final,
            )
            source_points_world_final = transform_points(
                preview.source_points_local,
                transform_world_source_final,
            )
            self.gicp_metrics_label.setText(
                f"Manual edge · fit {constraint.fitness:.4f} · rmse {constraint.inlier_rmse:.4f}"
            )
            self._show_scene(
                self._make_preview_scene(
                    preview=preview,
                    initial_source_points=preview.source_points_world_initial,
                    final_source_points=source_points_world_final,
                    transform_world_source_initial=preview.transform_world_source_initial,
                    transform_world_source_final=transform_world_source_final,
                ),
                scene_key=(
                    "manual",
                    constraint.manual_uid,
                    constraint.target_cloud_mode,
                    constraint.target_neighbors,
                    round(constraint.min_time_gap_sec, 6),
                    round(constraint.target_map_voxel_size, 6),
                ),
                reset_camera=reset_camera,
            )
            self._refresh_plot(preserve_view=True)
            self.append_log(
                f"Selected manual constraint {constraint.target_id}->{constraint.source_id}"
            )
            self._save_project_state()

        self._start_preview_build(
            source_id=constraint.source_id,
            target_id=constraint.target_id,
            delta_transform=np.eye(4, dtype=np.float64),
            target_cloud_mode=constraint.target_cloud_mode,
            target_neighbors=constraint.target_neighbors,
            min_time_gap_sec=constraint.min_time_gap_sec,
            target_map_voxel_size=constraint.target_map_voxel_size,
            on_finished=apply,
        )

    def _show_scene(
        self,
        scene: PreviewScene,
        *,
        scene_key: tuple,
        reset_camera: bool,
        force_preserve_camera: bool = False,
    ) -> None:
        if force_preserve_camera:
            should_reset = bool(reset_camera)
        else:
            should_reset = reset_camera or scene_key != self._preview_scene_key
        previous_camera = None if should_reset else self.cloud_view.capture_camera_state()
        self.cloud_view.update_scene(
            scene,
            reset_camera=should_reset,
            camera_state=previous_camera,
        )
        self._preview_scene_key = scene_key

    def _update_selection_labels(self) -> None:
        self.source_label.setText(self._format_node_info(self.source_id))
        self.source_label.setToolTip(
            "none" if self.source_id is None or self.trajectory is None else
            f"node={self.source_id}\nt={self.trajectory.timestamps[self.source_id]:.6f}s\n"
            f"xyz={self.trajectory.positions_xyz[self.source_id, 0]:.3f}, "
            f"{self.trajectory.positions_xyz[self.source_id, 1]:.3f}, "
            f"{self.trajectory.positions_xyz[self.source_id, 2]:.3f}"
        )
        self.target_label.setText(self._format_node_info(self.target_id))
        self.target_label.setToolTip(
            "none" if self.target_id is None or self.trajectory is None else
            f"node={self.target_id}\nt={self.trajectory.timestamps[self.target_id]:.6f}s\n"
            f"xyz={self.trajectory.positions_xyz[self.target_id, 0]:.3f}, "
            f"{self.trajectory.positions_xyz[self.target_id, 1]:.3f}, "
            f"{self.trajectory.positions_xyz[self.target_id, 2]:.3f}"
        )
        if self.source_id is None or self.target_id is None or self.trajectory is None:
            self.pair_label.setText("none")
            self.pair_label.setToolTip("none")
        else:
            self.pair_label.setText(
                f"T {self.target_id} · S {self.source_id}"
                f"<br><span style='color:#64748b;font-size:11px'>"
                f"{self.trajectory.positions_xyz[self.target_id, 0]:.2f}, {self.trajectory.positions_xyz[self.target_id, 1]:.2f} → "
                f"{self.trajectory.positions_xyz[self.source_id, 0]:.2f}, {self.trajectory.positions_xyz[self.source_id, 1]:.2f}"
                "</span>"
            )
            self.pair_label.setToolTip(
                "target:\n"
                f"node={self.target_id}\nt={self.trajectory.timestamps[self.target_id]:.6f}s\n"
                f"xyz={self.trajectory.positions_xyz[self.target_id, 0]:.3f}, "
                f"{self.trajectory.positions_xyz[self.target_id, 1]:.3f}, "
                f"{self.trajectory.positions_xyz[self.target_id, 2]:.3f}\n\n"
                "source:\n"
                f"node={self.source_id}\nt={self.trajectory.timestamps[self.source_id]:.6f}s\n"
                f"xyz={self.trajectory.positions_xyz[self.source_id, 0]:.3f}, "
                f"{self.trajectory.positions_xyz[self.source_id, 1]:.3f}, "
                f"{self.trajectory.positions_xyz[self.source_id, 2]:.3f}"
            )
        if self.selected_edge_ref is None:
            self.selected_edge_label.setText("none")
            self.selected_edge_label.setToolTip("none")
        else:
            edge = self._edge_from_ref(self.selected_edge_ref)
            if edge is None:
                self.selected_edge_label.setText("none")
                self.selected_edge_label.setToolTip("none")
            else:
                state = "on" if edge.enabled else "off"
                type_text = (
                    "Loop" if edge.edge_type == "loop_existing"
                    else "Odom" if edge.edge_type == "odom"
                    else "Manual"
                )
                self.selected_edge_label.setText(f"{type_text} · {edge.target_id}->{edge.source_id} · {state}")
                self.selected_edge_label.setToolTip(edge.summary())

    def _update_preview_summary(self, preview: Optional[RegistrationPreview]) -> None:
        self.current_preview = preview
        if preview is None:
            self.target_map_label.setText("No target submap.")
            return

        if preview.target_cloud_mode == TARGET_CLOUD_MODE_TEMPORAL_WINDOW:
            mode_text = f"TW±{preview.target_neighbors}"
            filter_text = "off"
        elif preview.time_gap_filter_enabled:
            mode_text = f"RS{preview.target_neighbors}"
            filter_text = "on" if preview.time_gap_filter_applied else "idle"
        else:
            mode_text = f"RS{preview.target_neighbors}"
            filter_text = "off"
        frame_range = preview.target_frame_range
        range_text = "n/a" if frame_range is None else f"{frame_range[0]}..{frame_range[1]}"
        if preview.target_cloud_mode == TARGET_CLOUD_MODE_TEMPORAL_WINDOW:
            clip_text = "clipped" if preview.target_window_clipped else "full"
        else:
            clip_text = "n/a"
        # Report the same hybrid cap that _make_preview_scene actually uses.
        # Using the raw display limit here made compact outdoor targets look
        # truncated in the UI even though the renderer kept every point.
        display_count = int(_bounded_target_display_view(
            preview.target_points_world,
        ).shape[0])
        point_text = f"{preview.target_point_count} pts"
        if display_count < preview.target_point_count:
            point_text += f" · view {display_count}"
        self.target_map_label.setText(
            f"{mode_text} · {preview.target_frame_count}f · {point_text}"
            f"<br><span style='color:#64748b;font-size:11px'>r {range_text} · b {clip_text} · g {filter_text}</span>"
        )
        self.target_map_label.setToolTip(
            f"mode={mode_text}\nframes={preview.target_frame_count}\nrange={range_text}\n"
            f"boundary={clip_text}\ntime_gap={filter_text}\n"
            f"registration_points={preview.target_point_count}\n"
            f"display_points={display_count}"
        )

    def _set_display_mode(self, text: str) -> None:
        requested = text.strip().lower()
        aliases = {
            "preview": "preview",
            "before": "preview",
            "before gicp": "preview",
            "final": "final",
            "after": "final",
            "after gicp": "final",
            "compare": "compare",
        }
        mode = aliases.get(requested, "preview")
        if self.display_mode_combo.currentData() == mode:
            return
        index = self.display_mode_combo.findData(mode)
        if index < 0:
            return
        with QtCore.QSignalBlocker(self.display_mode_combo):
            self.display_mode_combo.setCurrentIndex(index)
        self._update_cloud_display_controls()

    def _format_node_info(self, node_id: Optional[int]) -> str:
        if node_id is None or self.trajectory is None:
            return "none"
        pose = self.trajectory.positions_xyz[node_id]
        timestamp = self.trajectory.timestamps[node_id]
        return (
            f"{node_id} · ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) · t={timestamp:.3f}s"
        )

    def _refresh_plot(self, *, preserve_view: bool) -> None:
        plot_pose_graph = self.pose_graph if self._is_working_view() else self.original_pose_graph
        plot_trajectory = self.trajectory if self._is_working_view() else self.original_trajectory
        plot_constraints = self.constraints if self._is_working_view() else []
        selected_edge_ref = self.selected_edge_ref if self._is_working_view() else None

        positions_xy = (
            plot_trajectory.positions_xyz[:, :2]
            if plot_trajectory is not None
            else None
        )
        ghost_positions_xy = None
        if self.show_ghost_check.isChecked():
            ghost_trajectory = self.original_trajectory if self._is_working_view() else self.trajectory
            if ghost_trajectory is not None:
                ghost_positions_xy = ghost_trajectory.positions_xyz[:, :2]
        self.trajectory_canvas.set_interaction_mode(self.pick_mode)
        self.trajectory_canvas.set_plot_data(
            positions_xy=positions_xy,
            ghost_positions_xy=ghost_positions_xy,
            selected_target_positions_xy=self._selected_target_positions_xy(),
            pose_graph=plot_pose_graph,
            constraints=plot_constraints,
            source_id=self.source_id,
            target_id=self.target_id,
            selected_edge_ref=selected_edge_ref,
            preserve_view=preserve_view,
        )

    def _on_trajectory_view_changed(self, *_args) -> None:
        if self._is_working_view():
            loop_count = len(self.pose_graph.loop_edges) if self.pose_graph is not None else 0
            pose_count = self.trajectory.size if self.trajectory is not None else 0
            self.trajectory_view_info_label.setText(
                f"P{pose_count} · L{loop_count} · edit"
            )
            self.pick_nodes_button.setEnabled(True)
            self.pick_edges_button.setEnabled(True)
        else:
            loop_count = len(self.original_pose_graph.loop_edges) if self.original_pose_graph is not None else 0
            pose_count = self.original_trajectory.size if self.original_trajectory is not None else 0
            self.trajectory_view_info_label.setText(
                f"P{pose_count} · L{loop_count} · read only"
            )
            self.pick_nodes_button.setEnabled(False)
            self.pick_edges_button.setEnabled(False)
            self._release_plot_toolbar_navigation()
        self._update_plot_help_state()
        self._update_trajectory_legend_label()
        self._update_edge_action_buttons()
        self._refresh_plot(preserve_view=False)

    def _edge_from_ref(self, edge_ref: Optional[SelectedEdgeRef]) -> Optional[EdgeRecord]:
        if edge_ref is None:
            return None
        if edge_ref.edge_kind == "existing" and self.pose_graph is not None:
            return self.pose_graph.edge_index_by_uid.get(edge_ref.edge_uid)
        constraint = self._constraint_by_uid(edge_ref.edge_uid)
        if constraint is not None:
            return constraint.as_edge_record()
        return None

    def _constraint_by_uid(self, manual_uid: int) -> Optional[ManualConstraint]:
        for constraint in self.constraints:
            if constraint.manual_uid == manual_uid:
                return constraint
        return None

    def _constraint_by_pair(self, source_id: int, target_id: int) -> Optional[ManualConstraint]:
        for constraint in self.constraints:
            if constraint.source_id == source_id and constraint.target_id == target_id:
                return constraint
        return None

    def _disabled_loop_change(self, edge_uid: int) -> Optional[ExistingLoopChange]:
        return self.disabled_loop_changes.get(edge_uid)

    def _matches_graph_change_filters(self, type_text: str, status_text: str) -> bool:
        status_filter = self._graph_change_status_filter
        type_filter = self._graph_change_type_filter
        if status_filter not in {"", "All Status"} and status_text != status_filter:
            return False
        if type_filter not in {"", "All Types"} and type_text != type_filter:
            return False
        return True

    def _graph_change_table_rows(self) -> list[tuple[str, int, dict[str, str]]]:
        rows: list[tuple[str, int, dict[str, str]]] = []
        replaced_edge_uids = {
            constraint.replaces_edge_uid
            for constraint in self.constraints
            if constraint.replaces_edge_uid is not None
        }
        for constraint in self.constraints:
            status = self._status_text(constraint.enabled, constraint.applied_rev)
            frame_range = "-"
            if constraint.target_cloud_mode == TARGET_CLOUD_MODE_TEMPORAL_WINDOW:
                frame_range = f"+/-{constraint.target_neighbors}"
            type_text = (
                "Replace Existing Loop"
                if constraint.replaces_edge_uid is not None
                else "Manual Add"
            )
            if not self._matches_graph_change_filters(type_text, status):
                continue
            noise_text = (
                f"T:{'/'.join(f'{value:.2f}' for value in constraint.variance_t_m2)} "
                f"R:{'/'.join(f'{value:.2f}' for value in constraint.variance_r_rad2)}"
            )
            rows.append(
                (
                    "manual",
                    constraint.manual_uid,
                    {
                        "use": "1" if constraint.enabled else "0",
                        "type": type_text,
                        "status": status,
                        "src": str(constraint.source_id),
                        "tgt": str(constraint.target_id),
                        "submap": (
                            f"TW±{constraint.target_neighbors}"
                            if constraint.target_cloud_mode == TARGET_CLOUD_MODE_TEMPORAL_WINDOW
                            else f"RS{constraint.target_neighbors}"
                        ),
                        "range": frame_range,
                        "fitness": f"{constraint.fitness:.4f}",
                        "rmse": f"{constraint.inlier_rmse:.4f}",
                        "noise": noise_text,
                        "accepted": f"Rev {constraint.accepted_rev}",
                        "applied": "-" if constraint.applied_rev is None else f"Rev {constraint.applied_rev}",
                        "note": constraint.note,
                    },
                )
            )

        for edge_uid, change in sorted(self.disabled_loop_changes.items()):
            if edge_uid in replaced_edge_uids:
                continue
            edge = self.pose_graph.edge_index_by_uid.get(edge_uid) if self.pose_graph is not None else None
            if edge is None:
                continue
            status = self._status_text(change.enabled, change.applied_rev)
            type_text = "Disable Existing Loop"
            if not self._matches_graph_change_filters(type_text, status):
                continue
            rows.append(
                (
                    "disable_loop",
                    edge_uid,
                    {
                        "use": "1" if change.enabled else "0",
                        "type": type_text,
                        "status": status,
                        "src": str(edge.source_id),
                        "tgt": str(edge.target_id),
                        "submap": "-",
                        "range": "-",
                        "fitness": "-",
                        "rmse": "-",
                        "noise": "-",
                        "accepted": f"Rev {change.accepted_rev}",
                        "applied": "-" if change.applied_rev is None else f"Rev {change.applied_rev}",
                        "note": change.note,
                    },
                )
            )
        return rows

    def _status_brushes(self, status: str) -> tuple[QtGui.QBrush, QtGui.QBrush]:
        if status == "Applied":
            return (
                QtGui.QBrush(QtGui.QColor("#e7f8ee")),
                QtGui.QBrush(QtGui.QColor("#1f6f43")),
            )
        if status == "Accepted":
            return (
                QtGui.QBrush(QtGui.QColor("#fff4db")),
                QtGui.QBrush(QtGui.QColor("#8a5300")),
            )
        return (
            QtGui.QBrush(QtGui.QColor("#f1f5f9")),
            QtGui.QBrush(QtGui.QColor("#64748b")),
        )

    def _current_delta_transform(self) -> np.ndarray:
        return build_delta_transform(
            self.delta_spins["x"].value(),
            self.delta_spins["y"].value(),
            self.delta_spins["z"].value(),
            self.delta_spins["roll"].value(),
            self.delta_spins["pitch"].value(),
            self.delta_spins["yaw"].value(),
        )

    # Registration presets per environment, mirroring the headless pipeline.
    # Selecting the environment used to change only which descriptor seeded
    # retrieval, leaving the voxel at its indoor value, so an outdoor session
    # was silently registered and judged at indoor settings.
    ENV_PRESETS = REPAIR_ENV_PRESETS

    def _recorded_environment(self):
        """The preset this session was last repaired with, if it has been.

        ``auto_repair_summary.json`` already records it; nothing read it back,
        so the choice had to be remade by hand every time a session was
        reopened -- and getting it wrong is not a cosmetic mistake. Outdoor
        doubles the residual bound the quality gate enforces, from 0.18 m to
        0.36 m, which is precisely how a registration too poor to trust becomes
        an accepted loop.
        """
        paths = getattr(self, 'session_paths', None)
        root = getattr(paths, 'session_root', None) if paths else None
        if root is None:
            return None
        for name in ('auto_repair_summary.json', 'input_manifest.json'):
            try:
                with (Path(root) / name).open(encoding='utf-8') as fh:
                    recorded = json.load(fh).get('environment')
            except (OSError, ValueError):
                continue
            if recorded in self.ENV_PRESETS:
                return recorded
        return None

    def _restore_recorded_environment(self) -> bool:
        recorded = self._recorded_environment()
        if recorded is None:
            with QtCore.QSignalBlocker(self.env_combo):
                self.env_combo.setCurrentIndex(-1)
            return False
        with QtCore.QSignalBlocker(self.env_combo):
            self.env_combo.setCurrentIndex(self.env_combo.findData(recorded))
        self._apply_environment_preset()
        return True

    def _warn_environment_mismatch(self) -> None:
        recorded = self._recorded_environment()
        chosen = str(self.env_combo.currentData())
        if recorded is None or recorded == chosen:
            return
        bound = max(self.GATE_MAX_INLIER_RMSE,
                    self.GATE_RMSE_VOXEL_RATIO * self.ENV_PRESETS[chosen]['voxel'])
        was = max(self.GATE_MAX_INLIER_RMSE,
                  self.GATE_RMSE_VOXEL_RATIO * self.ENV_PRESETS[recorded]['voxel'])
        self.append_log(
            f"[environment] this session was last repaired as '{recorded}', "
            f"now set to '{chosen}': the gate's residual bound moves "
            f"{was:.2f} m → {bound:.2f} m."
        )

    def _apply_environment_preset(self) -> None:
        preset = self.ENV_PRESETS.get(str(self.env_combo.currentData()))
        if not preset:
            return
        self._warn_environment_mismatch()
        self.voxel_spin.setValue(preset['voxel'])
        self.max_corr_spin.setValue(preset['max_corr'])
        self.balm_voxel_spin.setValue(preset['balm_voxel'])
        self.balm_iter_spin.setValue(preset['balm_iterations'])
        self.balm_double_sided_check.setChecked(
            bool(preset['balm_double_sided'])
        )
        self._balm_downsample_leaf = float(preset['balm_downsample'])
        self._balm_max_range = float(preset['balm_max_range'])
        self.target_neighbors_spin.setValue(preset['target_neighbors'])
        self.target_map_voxel_spin.setValue(preset['target_map_voxel'])
        self._refresh_gate_label()

    def _refresh_gate_label(self) -> None:
        """Show the residual bound the gate is actually using right now."""
        bound = max(self.GATE_MAX_INLIER_RMSE,
                    self.GATE_RMSE_VOXEL_RATIO * self.voxel_spin.value())
        if hasattr(self, 'gate_label'):
            self.gate_label.setText(
                f"gate: overlap ≥ {self.GATE_MIN_FITNESS:.2f}, "
                f"residual ≤ {bound:.2f} m")

    def _current_registration_config(self) -> RegistrationConfig:
        return RegistrationConfig(
            target_cloud_mode=self._current_target_cloud_mode(),
            target_neighbors=self.target_neighbors_spin.value(),
            min_time_gap_sec=self.target_min_gap_spin.value(),
            target_map_voxel_size=self.target_map_voxel_spin.value(),
            voxel_size=self.voxel_spin.value(),
            max_correspondence_distance=self.max_corr_spin.value(),
            max_iterations=self.max_iter_spin.value(),
        )

    def _current_variances(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        translation_value = float(self.variance_t_shared_spin.value())
        rotation_value = float(self.variance_r_shared_spin.value())
        return (
            (translation_value, translation_value, translation_value),
            (rotation_value, rotation_value, rotation_value),
        )

    def _selected_target_positions_xy(self) -> Optional[np.ndarray]:
        if (
            not self._is_working_view()
            or self.current_preview is None
            or self.trajectory is None
            or not self.current_preview.target_frame_indices
        ):
            return None
        indices = np.asarray(self.current_preview.target_frame_indices, dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < self.trajectory.size)]
        if indices.size == 0:
            return None
        return self.trajectory.positions_xyz[indices, :2]

    def _on_graph_change_filter_changed(self) -> None:
        self._graph_change_status_filter = self.change_status_filter_combo.currentText().strip()
        self._graph_change_type_filter = self.change_type_filter_combo.currentText().strip()
        self._rebuild_constraint_table()
        self._sync_constraint_table_selection()

    def _sync_constraint_table_selection(self) -> None:
        if not hasattr(self, "constraint_table"):
            return
        with QtCore.QSignalBlocker(self.constraint_table.selectionModel()):
            self.constraint_table.clearSelection()
            if self.selected_edge_ref is None:
                return
            if self.selected_edge_ref.edge_kind == "existing":
                for row, (row_kind, row_uid) in enumerate(self._graph_change_rows):
                    if row_kind != "manual":
                        continue
                    constraint = self._constraint_by_uid(row_uid)
                    if constraint is not None and constraint.replaces_edge_uid == self.selected_edge_ref.edge_uid:
                        self.constraint_table.selectRow(row)
                        return
            for row, (row_kind, row_uid) in enumerate(self._graph_change_rows):
                if self.selected_edge_ref.edge_kind == "manual_added" and row_kind == "manual" and row_uid == self.selected_edge_ref.edge_uid:
                    self.constraint_table.selectRow(row)
                    break
                if self.selected_edge_ref.edge_kind == "existing" and row_kind == "disable_loop" and row_uid == self.selected_edge_ref.edge_uid:
                    self.constraint_table.selectRow(row)
                    break

    def _update_edge_action_buttons(self) -> None:
        edge = self._edge_from_ref(self.selected_edge_ref)
        editable = self._is_working_view() and self._background_thread is None
        self.disable_edge_button.setEnabled(bool(editable and edge and edge.edge_type == "loop_existing" and edge.enabled))
        self.restore_edge_button.setEnabled(bool(editable and edge and edge.edge_type == "loop_existing" and not edge.enabled))
        self.remove_manual_button.setEnabled(bool(editable and edge and edge.edge_type == "manual_added"))
        pair_ready = bool(editable and self.workspace is not None and self.source_id is not None and self.target_id is not None)
        self.run_gicp_button.setEnabled(pair_ready)
        self.auto_yaw_button.setEnabled(pair_ready)
        self.accept_button.setEnabled(bool(editable and self.last_result is not None))
        replacement_edge = (
            self.pose_graph.edge_index_by_uid.get(self._candidate_replace_edge_uid)
            if editable and self.pose_graph is not None and self._candidate_replace_edge_uid is not None
            else None
        )
        self.replace_edge_button.setEnabled(
            bool(
                editable
                and self.last_result is not None
                and replacement_edge is not None
                and replacement_edge.edge_type == "loop_existing"
            )
        )
        self._update_cloud_interaction_controls()

    def _clear_selection(self) -> None:
        had_selection = (
            self.source_id is not None
            or self.target_id is not None
            or self.selected_edge_ref is not None
        )
        self.source_id = None
        self.target_id = None
        self.selected_edge_ref = None
        self._candidate_replace_edge_uid = None
        self.current_preview = None
        self.last_result = None
        self.accept_button.setEnabled(False)
        self.gicp_metrics_label.setText("No GICP result.")
        self._preview_scene_key = None
        self._update_preview_summary(None)
        self.cloud_view.clear_scene()
        self._update_selection_labels()
        self._update_edge_action_buttons()
        self._sync_constraint_table_selection()
        self._refresh_plot(preserve_view=True)
        self._save_project_state()
        if had_selection:
            self.append_log("Cleared node/edge selection.")

    def _reset_delta(self) -> None:
        blockers = [QtCore.QSignalBlocker(spin) for spin in self.delta_spins.values()]
        try:
            for spin in self.delta_spins.values():
                spin.setValue(0.0)
        finally:
            del blockers
        self._invalidate_last_result_for_manual_align()
        self.schedule_preview_refresh(reset_camera=False)

    def load_session(self) -> None:
        """Resolve and validate a session without blocking the Qt event loop."""
        if self._background_thread is not None or self._optimizer_process is not None:
            self._show_error(
                "Load Session",
                "Wait for the current background operation to finish before loading another session.",
            )
            return
        session_root_text = self.session_root_edit.text().strip()
        g2o_text = self.g2o_edit.text().strip()
        session_root_path = Path(session_root_text) if session_root_text else None
        g2o_path = Path(g2o_text) if g2o_text else None

        def work(report_progress):
            report_progress("Load Session · resolving files", 0, 4)
            paths = None
            stale_g2o_note = None
            if session_root_path is not None and g2o_path is not None:
                resolved_root = session_root_path.expanduser().resolve()
                resolved_g2o = g2o_path.expanduser().resolve()
                try:
                    resolved_g2o.relative_to(resolved_root)
                    g2o_under_root = True
                except ValueError:
                    g2o_under_root = False
                if resolved_root.is_dir() and resolved_g2o.is_file() and not g2o_under_root:
                    paths = resolve_session_paths(session_root=session_root_path)
                    stale_g2o_note = (
                        f"Ignored stale g2o outside session root: {resolved_g2o}. "
                        f"Using {paths.g2o_path} from {paths.session_root}."
                    )
            if paths is None:
                paths = resolve_session_paths(
                    session_root=session_root_path,
                    g2o_path=g2o_path,
                )
            report_progress("Load Session · reading pose graph", 1, 4)
            pose_graph = load_pose_graph(paths.g2o_path)
            report_progress("Load Session · reading trajectory", 2, 4)
            trajectory = load_tum_trajectory(paths.tum_path)
            report_progress("Load Session · validating keyframes", 3, 4)
            keyframe_paths = list_numbered_pcds(paths.keyframe_dir)
            if trajectory.size != len(keyframe_paths):
                raise PcdValidationError(
                    "Keyframe PCD count does not match TUM pose count: "
                    f"{len(keyframe_paths)} vs {trajectory.size}"
                )
            pose_graph, trim_note = align_pose_graph_to_frame_count(
                pose_graph, trajectory.size)
            validate_keyframe_numbering(keyframe_paths, len(pose_graph.vertex_ids))
            report_progress("Load Session · validation complete", 4, 4)
            return (
                paths,
                pose_graph,
                trajectory,
                keyframe_paths,
                trim_note,
                stale_g2o_note,
            )

        def apply(payload) -> None:
            self._preloaded_session_payload = payload
            self._load_session_preloaded()

        self._start_background_task("Load Session", work, apply)

    def _load_session_preloaded(self) -> None:
        """Install worker-validated session data and update Qt widgets."""
        if self._background_thread is not None or self._optimizer_process is not None:
            self._show_error(
                "Load Session",
                "Wait for the current background operation to finish before loading another session.",
            )
            return
        try:
            session_root_text = self.session_root_edit.text().strip()
            g2o_text = self.g2o_edit.text().strip()
            session_root_path = Path(session_root_text) if session_root_text else None
            g2o_path = Path(g2o_text) if g2o_text else None
            paths = None
            stale_g2o_note = None
            if session_root_path is not None and g2o_path is not None:
                resolved_root = session_root_path.expanduser().resolve()
                resolved_g2o = g2o_path.expanduser().resolve()
                if (
                    resolved_root.is_dir()
                    and resolved_g2o.is_file()
                    and not self._is_g2o_under_session_root(resolved_root, resolved_g2o)
                ):
                    paths = resolve_session_paths(session_root=session_root_path)
                    stale_g2o_note = (
                        f"Ignored stale g2o outside session root: {resolved_g2o}. "
                        f"Using {paths.g2o_path} from {paths.session_root}."
                    )
            if paths is None:
                paths = resolve_session_paths(
                    session_root=session_root_path,
                    g2o_path=g2o_path,
                )
            preloaded = self._preloaded_session_payload
            self._preloaded_session_payload = None
            if preloaded is not None and preloaded[0] == paths:
                (
                    _loaded_paths,
                    pose_graph,
                    trajectory,
                    keyframe_paths,
                    trim_note,
                    loaded_stale_g2o_note,
                ) = preloaded
                if loaded_stale_g2o_note is not None:
                    stale_g2o_note = loaded_stale_g2o_note
            else:
                # Retain a safe synchronous fallback for callers that invoke
                # this private installer directly during recovery/tests.
                pose_graph = load_pose_graph(paths.g2o_path)
                trajectory = load_tum_trajectory(paths.tum_path)
                keyframe_paths = list_numbered_pcds(paths.keyframe_dir)
                if trajectory.size != len(keyframe_paths):
                    raise PcdValidationError(
                        "Keyframe PCD count does not match TUM pose count: "
                        f"{len(keyframe_paths)} vs {trajectory.size}"
                    )
                pose_graph, trim_note = align_pose_graph_to_frame_count(
                    pose_graph, trajectory.size)
                validate_keyframe_numbering(
                    keyframe_paths, len(pose_graph.vertex_ids))

            previous_workspace = self.workspace
            if previous_workspace is not None:
                previous_workspace.clear_caches(include_shared_frames=True)
            self.session_paths = paths
            self.session_root_edit.setText(str(paths.session_root))
            self.g2o_edit.setText(str(paths.g2o_path))
            environment_restored = self._restore_recorded_environment()
            self._remember_loaded_paths(paths)
            self.original_pose_graph = pose_graph
            self.original_trajectory = trajectory
            self.pose_graph = copy.deepcopy(pose_graph)
            self.trajectory = copy.deepcopy(trajectory)
            self.workspace = RegistrationWorkspace(paths.keyframe_dir, self.trajectory)
            self.keyframe_paths = keyframe_paths
            self.source_id = None
            self.target_id = None
            self.selected_edge_ref = None
            self._candidate_replace_edge_uid = None
            self.constraints.clear()
            self.disabled_loop_changes.clear()
            self.current_preview = None
            self.last_result = None
            self._next_manual_uid = 1
            self._preview_scene_key = None
            self._working_revision = 0
            self._session_dirty = False
            self._pending_export_after_optimize = False
            self._undo_stack.clear()
            self._last_output_dir = None
            self._latest_export_dir = None
            # A new session re-indexes every keyframe; the cached Scan Context
            # stack describes the previous one and its indices are meaningless
            # here, including when the two happen to share a keyframe count.
            # Same reason: a badness score from the previous session would gate
            # this one's first BALM pass on a map it never saw.
            self._prev_ghost_badness = None
            self._balm_source_run_dir = None
            self.accept_button.setEnabled(False)
            self.gicp_metrics_label.setText("No GICP result.")
            self._set_pick_mode("nodes")
            self._set_display_mode("Preview")
            with QtCore.QSignalBlocker(self.show_world_axis_check):
                self.show_world_axis_check.setChecked(False)
            with QtCore.QSignalBlocker(self.cloud_interaction_mode_combo):
                self.cloud_interaction_mode_combo.setCurrentIndex(
                    self.cloud_interaction_mode_combo.findData(INTERACTION_MODE_CAMERA)
                )
            with QtCore.QSignalBlocker(self.cloud_lock_mode_combo):
                self.cloud_lock_mode_combo.setCurrentIndex(
                    self.cloud_lock_mode_combo.findData(LOCK_MODE_XY_YAW)
                )
            with QtCore.QSignalBlocker(self.cloud_drag_mode_combo):
                self.cloud_drag_mode_combo.setCurrentIndex(
                    self.cloud_drag_mode_combo.findData(EDIT_OPERATION_TRANSLATE)
                )
            with QtCore.QSignalBlocker(self.manual_align_snap_view_check):
                self.manual_align_snap_view_check.setChecked(True)
            with QtCore.QSignalBlocker(self.trajectory_view_combo):
                self.trajectory_view_combo.clear()
                self.trajectory_view_combo.addItems(["Working", "Original"])
                self.trajectory_view_combo.setCurrentText("Working")
            with QtCore.QSignalBlocker(self.change_status_filter_combo):
                self.change_status_filter_combo.setCurrentText("All Status")
            with QtCore.QSignalBlocker(self.change_type_filter_combo):
                self.change_type_filter_combo.setCurrentText("All Types")
            self._graph_change_status_filter = "All Status"
            self._graph_change_type_filter = "All Types"
            self._update_cloud_display_controls()
            self._start_or_resume_project(paths)
            environment_restored = (
                environment_restored
                or self.env_combo.currentData() in self.ENV_PRESETS
            )
            self._update_trajectory_legend_label()
            self.trajectory_view_info_label.setText(
                f"P{self.trajectory.size} · L{len(self.pose_graph.loop_edges)} · edit"
            )
            self.pick_edges_button.setEnabled(True)
            self._release_plot_toolbar_navigation()
            self._update_plot_help_state()
            self.cloud_view.clear_scene()
            self._update_preview_summary(None)
            self._update_selection_labels()
            self._update_edge_action_buttons()
            self._rebuild_constraint_table()
            self._update_session_status_widgets()
            self._refresh_plot(preserve_view=False)
            self._update_cloud_interaction_controls()
            self._save_project_state()

            self.session_info_label.setText(
                f"Loaded session: {paths.session_root} | "
                f"g2o={paths.g2o_path.name} | tum={paths.tum_path.name} | "
                f"keyframes={len(self.keyframe_paths)} | loops={len(self.pose_graph.loop_edges)} | "
                f"project={self._project_id}"
            )
            if stale_g2o_note:
                self.append_log(stale_g2o_note)
            if trim_note:
                self.append_log(trim_note)
            self.append_log(
                f"Loaded session root={paths.session_root}, g2o={paths.g2o_path}, "
                f"tum={paths.tum_path}, keyframes={len(self.keyframe_paths)}, "
                f"existing_loops={len(self.pose_graph.loop_edges)}"
            )
            if not environment_restored:
                self.append_log(
                    "[environment] no manifest/project environment was found; "
                    "choose Indoor or Outdoor before Initial Loop Search or Repair Map."
                )
            self._save_project_state()
        except (
            SessionResolutionError,
            PoseGraphValidationError,
            TrajectoryValidationError,
            PcdValidationError,
            OSError,
        ) as exc:
            self._show_error("Load Session Failed", str(exc))
        except Exception as exc:
            self._show_error("Load Session Failed", f"{exc}\n\n{traceback.format_exc()}")

    @property
    def _automatic_flow_running(self) -> bool:
        """Is canonical Repair Map driving the session right now?"""
        return self._repair_map_stage is not None

    def _flush_queued_preview(self) -> None:
        """Build the one preview the automatic flow deferred, now that it ended.

        Skipping previews mid-flow would otherwise leave the viewport showing
        whatever was on screen when Repair Map started, which no longer matches
        the trajectory the run produced.
        """
        if not self._preview_refresh_queued or self._automatic_flow_running:
            return
        self._preview_refresh_queued = False
        self.schedule_preview_refresh(reset_camera=False)

    def schedule_preview_refresh(self, reset_camera: bool = False) -> None:
        if (
            self.workspace is None
            or self.source_id is None
            or self.target_id is None
        ):
            return
        self._pending_preview_reset_camera = self._pending_preview_reset_camera or reset_camera
        if self._automatic_flow_running:
            # Nobody is reading a per-constraint preview during a multi-round
            # automatic run, and the headless pipeline has no preview at all.
            # Queue one for when the flow ends and skip the work.
            self._preview_refresh_queued = True
            return
        if self._background_thread is not None:
            if self._background_label == "Build Preview":
                # A slider/config change arrived while an older point-cloud
                # preview was being built. Keep just one coalesced refresh;
                # the finished callback will start it with the latest values.
                self._preview_refresh_queued = True
            return
        self._preview_timer.start()

    def refresh_preview(self) -> None:
        if self.workspace is None or self.source_id is None or self.target_id is None:
            return
        reset_camera = self._pending_preview_reset_camera
        self._pending_preview_reset_camera = False
        if self.selected_edge_ref is not None:
            edge = self._edge_from_ref(self.selected_edge_ref)
            if edge is None:
                return
            if self.selected_edge_ref.edge_kind == "manual_added":
                constraint = self._constraint_by_uid(self.selected_edge_ref.edge_uid)
                if constraint is not None:
                    self._preview_manual_constraint(constraint, reset_camera=reset_camera)
            else:
                self._preview_existing_edge(edge, reset_camera=reset_camera)
            return
        self._set_display_mode("Preview")
        config = self._current_registration_config()
        source_id = int(self.source_id)
        target_id = int(self.target_id)
        delta_transform = self._current_delta_transform().copy()

        def apply(preview) -> None:
            if self.source_id != source_id or self.target_id != target_id:
                return
            self.current_preview = preview
            self._update_preview_summary(preview)
            self.last_result = None
            self.accept_button.setEnabled(False)
            self.gicp_metrics_label.setText("Preview ready · run GICP.")
            self._show_scene(
                self._make_preview_scene(
                    preview=preview,
                    initial_source_points=preview.source_points_world_initial,
                    adjusted_source_points=preview.source_points_world_adjusted,
                    transform_world_source_initial=preview.transform_world_source_initial,
                    transform_world_source_adjusted=preview.transform_world_source_adjusted,
                ),
                scene_key=(
                    "pair",
                    source_id,
                    target_id,
                    preview.target_cloud_mode,
                    preview.target_neighbors,
                    round(preview.min_time_gap_sec, 6),
                    round(preview.target_map_voxel_size, 6),
                ),
                reset_camera=reset_camera,
            )
            self._refresh_plot(preserve_view=True)

        self._start_preview_build(
            source_id=source_id,
            target_id=target_id,
            delta_transform=delta_transform,
            target_cloud_mode=config.target_cloud_mode,
            target_neighbors=config.target_neighbors,
            min_time_gap_sec=config.min_time_gap_sec,
            target_map_voxel_size=config.target_map_voxel_size,
            on_finished=apply,
        )

    def _start_preview_build(
        self,
        *,
        source_id: int,
        target_id: int,
        delta_transform: np.ndarray,
        target_cloud_mode: str,
        target_neighbors: int,
        min_time_gap_sec: float,
        target_map_voxel_size: float,
        on_finished,
    ) -> bool:
        """Build a potentially multi-million-point preview off the GUI thread.

        Refused outright while Repair Map is running. Two reasons,
        and either alone would be enough:

        * Nobody is looking. The automatic flow re-solves, runs BALM and moves
          on; a several-million-point preview of one candidate is built and
          discarded, which is pure cost. The headless pipeline never builds one.
        * It races the optimizer. This worker enters Open3D/numpy parallel
          regions while the main thread is forking BALM out through QProcess,
          and on 2026-08-05 the indoor ih session deadlocked exactly there:
          main thread parked in ``futex_do_wait``, the whole process at 0% CPU,
          zero optimizer heartbeats, balm_cli's stdout never read even though
          the child ran to completion. The process image carried two OpenMP
          runtimes (numpy's bundled ``libgomp-<hash>.so`` plus the system one)
          alongside Open3D's TBB. Not building the preview removes the race
          from the automatic flow rather than trying to referee it.
        """
        if self.workspace is None:
            return False
        if self._automatic_flow_running:
            self._preview_refresh_queued = True
            return False
        workspace = self.workspace

        def work(report_progress):
            report_progress(
                f"Preview · {target_id}->{source_id} · {target_neighbors} neighbors",
                0,
                0,
            )
            return workspace.build_preview(
                source_id=source_id,
                target_id=target_id,
                delta_transform_local=delta_transform,
                target_cloud_mode=target_cloud_mode,
                target_neighbors=target_neighbors,
                min_time_gap_sec=min_time_gap_sec,
                target_map_voxel_size=target_map_voxel_size,
            )

        return self._start_background_task("Build Preview", work, on_finished)

    def run_gicp(self) -> None:
        if self.workspace is None or self.source_id is None or self.target_id is None:
            self._show_error("Run GICP", "Select source and target nodes first.")
            return
        if self._background_thread is not None or self._optimizer_process is not None:
            self._show_error("Run GICP", "Another operation is already running.")
            return

        replace_edge_uid = None
        if self.selected_edge_ref is not None and self.selected_edge_ref.edge_kind == "existing":
            edge = self._edge_from_ref(self.selected_edge_ref)
            if edge is not None and edge.edge_type == "loop_existing":
                replace_edge_uid = edge.edge_uid

        source_id = int(self.source_id)
        target_id = int(self.target_id)
        workspace = self.workspace
        delta_transform = self._current_delta_transform().copy()
        config = self._current_registration_config()
        self.selected_edge_ref = None
        self._candidate_replace_edge_uid = replace_edge_uid
        self._update_selection_labels()
        self._sync_constraint_table_selection()
        self.accept_button.setEnabled(False)
        if self.current_preview is not None:
            self.append_log(
                f"Run GICP target={target_id}, source={source_id} | "
                f"{self._gicp_log_cloud_summary(self.current_preview)}"
            )

        def work(report_progress):
            report_progress(f"GICP · {target_id}->{source_id}", 0, 1)
            result = workspace.run_gicp(
                source_id=source_id,
                target_id=target_id,
                delta_transform_local=delta_transform,
                config=config,
            )
            report_progress(f"GICP · {target_id}->{source_id}", 1, 1)
            return result

        def apply(result) -> None:
            self.source_id = source_id
            self.target_id = target_id
            self._set_display_mode("Final")
            self.current_preview = result.preview
            self._update_preview_summary(result.preview)
            self.last_result = result
            self._show_scene(
                self._make_preview_scene(
                    preview=result.preview,
                    initial_source_points=result.preview.source_points_world_initial,
                    adjusted_source_points=result.preview.source_points_world_adjusted,
                    final_source_points=result.source_points_world_final,
                    transform_world_source_initial=result.preview.transform_world_source_initial,
                    transform_world_source_adjusted=result.preview.transform_world_source_adjusted,
                    transform_world_source_final=result.transform_world_source_final,
                ),
                scene_key=(
                    "pair",
                    source_id,
                    target_id,
                    result.preview.target_cloud_mode,
                    result.preview.target_neighbors,
                    round(result.preview.min_time_gap_sec, 6),
                    round(result.preview.target_map_voxel_size, 6),
                    "gicp",
                ),
                reset_camera=False,
            )
            self.gicp_metrics_label.setText(
                (
                    f"GICP ready · fit {result.fitness:.4f} · rmse {result.inlier_rmse:.4f}"
                    "<br><span style='color:#64748b;font-size:11px'>Replace selected loop when confirmed.</span>"
                )
                if replace_edge_uid is not None
                else f"GICP ready · fit {result.fitness:.4f} · rmse {result.inlier_rmse:.4f}"
            )
            self.accept_button.setEnabled(True)
            self._update_edge_action_buttons()
            self._refresh_plot(preserve_view=True)
            self.append_log(
                f"GICP completed for target={target_id}, source={source_id}, "
                f"fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f} | "
                f"{self._gicp_log_cloud_summary(result.preview)}"
            )

        self._start_background_task("Run GICP", work, apply)

    def run_auto_yaw_sweep(self) -> None:
        if self.workspace is None or self.source_id is None or self.target_id is None:
            self._show_error("Auto Yaw Sweep", "Select source and target nodes first.")
            return
        if self._background_thread is not None or self._optimizer_process is not None:
            self._show_error("Auto Yaw Sweep", "Another operation is already running.")
            return

        replace_edge_uid = None
        if self.selected_edge_ref is not None and self.selected_edge_ref.edge_kind == "existing":
            edge = self._edge_from_ref(self.selected_edge_ref)
            if edge is not None and edge.edge_type == "loop_existing":
                replace_edge_uid = edge.edge_uid

        base_delta = {key: spin.value() for key, spin in self.delta_spins.items()}
        step_count = max(int(self.auto_yaw_steps_spin.value()), 2)
        yaw_candidates = [360.0 * float(index) / float(step_count) for index in range(step_count)]
        config = self._current_registration_config()
        source_id = int(self.source_id)
        target_id = int(self.target_id)
        workspace = self.workspace
        self.selected_edge_ref = None
        self._candidate_replace_edge_uid = replace_edge_uid
        self._update_selection_labels()
        self._sync_constraint_table_selection()
        self._set_display_mode("Preview")
        self.accept_button.setEnabled(False)

        def work(report_progress):
            candidate_results: list[tuple[float, RegistrationResult]] = []
            for yaw_index, yaw_seed in enumerate(yaw_candidates, start=1):
                report_progress(
                    f"Auto Yaw · {yaw_index}/{step_count} · seed {yaw_seed:.1f}°",
                    yaw_index - 1,
                    step_count,
                )
                delta_transform = build_delta_transform(
                    base_delta["x"],
                    base_delta["y"],
                    base_delta["z"],
                    base_delta["roll"],
                    base_delta["pitch"],
                    yaw_seed,
                )
                result = workspace.run_gicp(
                    source_id=source_id,
                    target_id=target_id,
                    delta_transform_local=delta_transform,
                    config=config,
                )
                candidate_results.append((yaw_seed, result))
                report_progress(
                    f"Auto Yaw · {yaw_index}/{step_count} complete",
                    yaw_index,
                    step_count,
                )

            if not candidate_results:
                raise RuntimeError("Auto yaw sweep produced no valid GICP results.")

            max_fitness = max(result.fitness for _, result in candidate_results)
            shortlist = [
                (yaw_seed, result)
                for yaw_seed, result in candidate_results
                if result.fitness >= max(0.0, max_fitness - 0.05)
            ]
            best_yaw_seed, best_result = min(
                shortlist,
                key=lambda item: (item[1].inlier_rmse, -item[1].fitness, abs(item[0])),
            )
            return best_yaw_seed, best_result, candidate_results

        def apply(payload) -> None:
            best_yaw_seed, best_result, candidate_results = payload
            self.source_id = source_id
            self.target_id = target_id
            blockers = [QtCore.QSignalBlocker(spin) for spin in self.delta_spins.values()]
            try:
                self.delta_spins["x"].setValue(base_delta["x"])
                self.delta_spins["y"].setValue(base_delta["y"])
                self.delta_spins["z"].setValue(base_delta["z"])
                self.delta_spins["roll"].setValue(base_delta["roll"])
                self.delta_spins["pitch"].setValue(base_delta["pitch"])
                self.delta_spins["yaw"].setValue(best_yaw_seed)
            finally:
                del blockers

            self._set_display_mode("Final")
            self.current_preview = best_result.preview
            self._update_preview_summary(best_result.preview)
            self.last_result = best_result
            self._show_scene(
                self._make_preview_scene(
                    preview=best_result.preview,
                    initial_source_points=best_result.preview.source_points_world_initial,
                    adjusted_source_points=best_result.preview.source_points_world_adjusted,
                    final_source_points=best_result.source_points_world_final,
                    transform_world_source_initial=best_result.preview.transform_world_source_initial,
                    transform_world_source_adjusted=best_result.preview.transform_world_source_adjusted,
                    transform_world_source_final=best_result.transform_world_source_final,
                ),
                scene_key=(
                    "pair",
                    source_id,
                    target_id,
                    best_result.preview.target_cloud_mode,
                    best_result.preview.target_neighbors,
                    round(best_result.preview.min_time_gap_sec, 6),
                    round(best_result.preview.target_map_voxel_size, 6),
                    "auto_yaw",
                    round(best_yaw_seed, 6),
                ),
                reset_camera=False,
                force_preserve_camera=True,
            )
            self.gicp_metrics_label.setText(
                (
                    f"Auto yaw · {best_yaw_seed:.1f}° · fit {best_result.fitness:.4f} · "
                    f"rmse {best_result.inlier_rmse:.4f}"
                    "<br><span style='color:#64748b;font-size:11px'>Replace selected loop when confirmed.</span>"
                )
                if replace_edge_uid is not None
                else f"Auto yaw · {best_yaw_seed:.1f}° · fit {best_result.fitness:.4f} · rmse {best_result.inlier_rmse:.4f}"
            )
            self.accept_button.setEnabled(True)
            self._update_edge_action_buttons()
            self._refresh_plot(preserve_view=True)
            for yaw_seed, result in candidate_results:
                self.append_log(
                    f"Auto yaw seed {yaw_seed:.1f} deg -> "
                    f"fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.4f} | "
                    f"{self._gicp_log_cloud_summary(result.preview)}"
                )
            self.append_log(
                f"Auto yaw selected seed {best_yaw_seed:.1f} deg for "
                f"target={target_id}, source={source_id}."
            )

        self._start_background_task("Auto Yaw", work, apply)

    def accept_constraint(
        self,
        *,
        variance_t: tuple[float, float, float] | None = None,
        variance_r: tuple[float, float, float] | None = None,
        note: str | None = None,
        record_undo: bool = True,
        refresh_ui: bool = True,
        persist: bool = True,
    ) -> ManualConstraint | None:
        if self.last_result is None:
            self._show_error("Accept Constraint", "Run GICP and review the preview before accepting.")
            return None

        if record_undo:
            self._push_undo_snapshot()
        if variance_t is None or variance_r is None:
            variance_t, variance_r = self._current_variances()
        constraint, replaced = self._upsert_manual_constraint(
            self.last_result,
            variance_t=variance_t,
            variance_r=variance_r,
            replaces_edge_uid=None,
        )
        if note is not None:
            constraint.note = str(note)

        self.selected_edge_ref = SelectedEdgeRef("manual_added", constraint.manual_uid)
        result_preview = self.last_result.preview
        self._consume_gicp_candidate(result_preview)
        self._recompute_session_dirty()
        if refresh_ui:
            self._update_preview_summary(self.current_preview)
            self._rebuild_constraint_table()
            self._update_selection_labels()
            self._sync_constraint_table_selection()
            self._update_session_status_widgets()
            self._refresh_plot(preserve_view=True)
        self.append_log(
            f"{'Updated' if replaced else 'Accepted'} new manual edge "
            f"{constraint.target_id}->{constraint.source_id}"
        )
        if persist:
            self._save_project_state()
        return constraint

    def _upsert_manual_constraint(
        self,
        result: RegistrationResult,
        *,
        variance_t: tuple[float, float, float],
        variance_r: tuple[float, float, float],
        replaces_edge_uid: Optional[int],
    ) -> tuple[ManualConstraint, bool]:
        existing_constraint = self._constraint_by_pair(
            result.preview.source_id,
            result.preview.target_id,
        )
        manual_uid = existing_constraint.manual_uid if existing_constraint is not None else self._next_manual_uid
        preserved_note = existing_constraint.note if existing_constraint is not None else ""
        constraint = ManualConstraint(
            manual_uid=manual_uid,
            enabled=True,
            source_id=result.preview.source_id,
            target_id=result.preview.target_id,
            target_cloud_mode=result.preview.target_cloud_mode,
            target_neighbors=result.preview.target_neighbors,
            min_time_gap_sec=result.preview.min_time_gap_sec,
            target_map_voxel_size=result.preview.target_map_voxel_size,
            transform_world_source_final=result.transform_world_source_final.copy(),
            transform_target_source_final=result.transform_target_source_final.copy(),
            source_points_world_final=result.source_points_world_final.copy(),
            fitness=result.fitness,
            inlier_rmse=result.inlier_rmse,
            variance_t_m2=variance_t,
            variance_r_rad2=variance_r,
            replaces_edge_uid=replaces_edge_uid,
            accepted_rev=self._working_revision,
            applied_rev=None,
            note=preserved_note,
        )

        replaced = False
        for index, existing in enumerate(self.constraints):
            if existing.manual_uid == manual_uid:
                self.constraints[index] = constraint
                replaced = True
                break
        if not replaced:
            self.constraints.append(constraint)
            self._next_manual_uid += 1
        return constraint, replaced

    def _consume_gicp_candidate(self, preview: RegistrationPreview) -> None:
        self.current_preview = preview
        self.last_result = None
        self._candidate_replace_edge_uid = None
        self._update_edge_action_buttons()

    def replace_selected_edge(self) -> None:
        if self.last_result is None or self._candidate_replace_edge_uid is None or self.pose_graph is None:
            self._show_error(
                "Replace Selected Edge",
                "Run GICP from a selected existing loop edge first.",
            )
            return

        edge = self.pose_graph.edge_index_by_uid.get(self._candidate_replace_edge_uid)
        if edge is None or edge.edge_type != "loop_existing":
            self._show_error(
                "Replace Selected Edge",
                "The current GICP result is not tied to an existing loop edge.",
            )
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "Replace Selected Edge",
            "This will disable the selected existing loop edge in the working graph "
            "and replace it with the current manual GICP result. "
            "Original graph will not be modified.\n\nContinue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        self._push_undo_snapshot()
        variance_t, variance_r = self._current_variances()
        edge.enabled = False
        change = self.disabled_loop_changes.get(edge.edge_uid)
        if change is None:
            change = ExistingLoopChange(
                edge_uid=edge.edge_uid,
                enabled=True,
                accepted_rev=self._working_revision,
                applied_rev=None,
            )
            self.disabled_loop_changes[edge.edge_uid] = change
        else:
            change.enabled = True
            change.accepted_rev = self._working_revision
            change.applied_rev = None

        constraint, replaced = self._upsert_manual_constraint(
            self.last_result,
            variance_t=variance_t,
            variance_r=variance_r,
            replaces_edge_uid=edge.edge_uid,
        )

        self.selected_edge_ref = SelectedEdgeRef("manual_added", constraint.manual_uid)
        result_preview = self.last_result.preview
        self._consume_gicp_candidate(result_preview)
        self._update_preview_summary(self.current_preview)
        self._recompute_session_dirty()
        self._rebuild_constraint_table()
        self._update_selection_labels()
        self._sync_constraint_table_selection()
        self._update_session_status_widgets()
        self._refresh_plot(preserve_view=True)
        self.append_log(
            f"{'Updated' if replaced else 'Created'} replacement for existing loop "
            f"{edge.target_id}->{edge.source_id}; working graph now uses the new manual result."
        )
        self._save_project_state()

    def _rebuild_constraint_table(self) -> None:
        self._table_updating = True
        try:
            rows = self._graph_change_table_rows()
            self._graph_change_rows = [(kind, uid) for kind, uid, _ in rows]
            self.constraint_table.setRowCount(len(rows))
            for row, (_kind, _uid, values_map) in enumerate(rows):
                enabled_item = QtWidgets.QTableWidgetItem()
                enabled_item.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsSelectable
                    | QtCore.Qt.ItemIsUserCheckable
                )
                enabled_item.setCheckState(QtCore.Qt.Checked if values_map["use"] == "1" else QtCore.Qt.Unchecked)
                self.constraint_table.setItem(row, 0, enabled_item)

                values = [
                    values_map["type"],
                    values_map["status"],
                    values_map["src"],
                    values_map["tgt"],
                    values_map["submap"],
                    values_map["range"],
                    values_map["fitness"],
                    values_map["rmse"],
                    values_map["noise"],
                    values_map["accepted"],
                    values_map["applied"],
                    values_map["note"],
                ]
                background, foreground = self._status_brushes(values_map["status"])
                enabled_item.setBackground(background)
                enabled_item.setForeground(foreground)
                for column, value in enumerate(values, start=1):
                    item = QtWidgets.QTableWidgetItem(value)
                    flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                    if column == 12:
                        flags |= QtCore.Qt.ItemIsEditable
                        item.setToolTip("Double-click to edit the note for this graph change.")
                    item.setFlags(flags)
                    item.setBackground(background)
                    item.setForeground(foreground)
                    if column == 12:
                        item.setData(QtCore.Qt.UserRole, value)
                    self.constraint_table.setItem(row, column, item)
        finally:
            self._table_updating = False

    def _on_constraint_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._table_updating:
            return
        row = item.row()
        if row < 0 or row >= len(self._graph_change_rows):
            return
        row_kind, row_uid = self._graph_change_rows[row]
        if item.column() == 0:
            self._push_undo_snapshot()
            enabled = item.checkState() == QtCore.Qt.Checked
            if row_kind == "manual":
                constraint = self._constraint_by_uid(row_uid)
                if constraint is not None:
                    constraint.enabled = enabled
                    if constraint.replaces_edge_uid is not None and self.pose_graph is not None:
                        change = self._disabled_loop_change(constraint.replaces_edge_uid)
                        edge = self.pose_graph.edge_index_by_uid.get(constraint.replaces_edge_uid)
                        if change is not None and edge is not None:
                            change.enabled = enabled
                            edge.enabled = not enabled
            elif row_kind == "disable_loop":
                change = self._disabled_loop_change(row_uid)
                edge = self.pose_graph.edge_index_by_uid.get(row_uid) if self.pose_graph is not None else None
                if change is not None and edge is not None:
                    change.enabled = enabled
                    edge.enabled = not enabled
                    replacement_constraint = next(
                        (
                            constraint
                            for constraint in self.constraints
                            if constraint.replaces_edge_uid == row_uid
                        ),
                        None,
                    )
                    if replacement_constraint is not None:
                        replacement_constraint.enabled = enabled
            self._recompute_session_dirty()
            self._update_edge_action_buttons()
            self._update_session_status_widgets()
            self._rebuild_constraint_table()
            self._refresh_plot(preserve_view=True)
            self._save_project_state()
            return

        if item.column() != 12:
            return

        new_note = item.text().strip()
        previous_note = str(item.data(QtCore.Qt.UserRole) or "").strip()
        if new_note == previous_note:
            return
        self._push_undo_snapshot()
        if row_kind == "manual":
            constraint = self._constraint_by_uid(row_uid)
            if constraint is not None:
                constraint.note = new_note
        elif row_kind == "disable_loop":
            change = self._disabled_loop_change(row_uid)
            if change is not None:
                change.note = new_note
        self.append_log(
            f"Updated graph change note for {row_kind}:{row_uid} to "
            f"'{new_note or '(empty)'}'."
        )
        with QtCore.QSignalBlocker(self.constraint_table):
            item.setData(QtCore.Qt.UserRole, new_note)
        self._save_project_state()

    def _on_constraint_selection_changed(self) -> None:
        if self._table_updating:
            return
        row = self.constraint_table.currentRow()
        if row < 0 or row >= len(self._graph_change_rows):
            return
        row_kind, row_uid = self._graph_change_rows[row]
        if row_kind == "manual":
            self._select_edge(SelectedEdgeRef("manual_added", row_uid), reset_camera=True)
        else:
            self._select_edge(SelectedEdgeRef("existing", row_uid), reset_camera=True)

    def disable_selected_edge(self) -> None:
        edge = self._edge_from_ref(self.selected_edge_ref)
        if edge is None or edge.edge_type != "loop_existing" or not edge.enabled:
            self._show_error("Disable Edge", "Select an enabled existing loop edge first.")
            return
        self._push_undo_snapshot()
        edge.enabled = False
        self.disabled_loop_changes[edge.edge_uid] = ExistingLoopChange(
            edge_uid=edge.edge_uid,
            enabled=True,
            accepted_rev=self._working_revision,
            applied_rev=None,
        )
        self._recompute_session_dirty()
        self._update_selection_labels()
        self._update_edge_action_buttons()
        self._rebuild_constraint_table()
        self._update_session_status_widgets()
        self._refresh_plot(preserve_view=True)
        self.append_log(f"Disabled existing loop edge {edge.target_id}->{edge.source_id}")
        self._save_project_state()

    def restore_selected_edge(self) -> None:
        edge = self._edge_from_ref(self.selected_edge_ref)
        if edge is None or edge.edge_type != "loop_existing" or edge.enabled:
            self._show_error("Restore Edge", "Select a disabled existing loop edge first.")
            return
        self._push_undo_snapshot()
        replacement_constraint = None
        for constraint in self.constraints:
            if constraint.replaces_edge_uid == edge.edge_uid and constraint.enabled:
                replacement_constraint = constraint
                break
        edge.enabled = True
        change = self.disabled_loop_changes.get(edge.edge_uid)
        if change is not None:
            if change.applied_rev is None:
                self.disabled_loop_changes.pop(edge.edge_uid, None)
            else:
                change.enabled = False
        if replacement_constraint is not None:
            if replacement_constraint.applied_rev is None:
                self.constraints = [
                    constraint
                    for constraint in self.constraints
                    if constraint.manual_uid != replacement_constraint.manual_uid
                ]
            else:
                replacement_constraint.enabled = False
        self._recompute_session_dirty()
        self._update_selection_labels()
        self._update_edge_action_buttons()
        self._rebuild_constraint_table()
        self._update_session_status_widgets()
        self._refresh_plot(preserve_view=True)
        if replacement_constraint is not None:
            self.append_log(
                f"Restored existing loop edge {edge.target_id}->{edge.source_id} and removed its replacement manual edge."
            )
        else:
            self.append_log(f"Restored existing loop edge {edge.target_id}->{edge.source_id}")
        self._save_project_state()

    def remove_selected_manual_constraint(self) -> None:
        edge = self._edge_from_ref(self.selected_edge_ref)
        if edge is None or edge.edge_type != "manual_added":
            self._show_error("Remove Manual Constraint", "Select a manual constraint first.")
            return
        self._push_undo_snapshot()

        removed = None
        for index, constraint in enumerate(self.constraints):
            if constraint.manual_uid == edge.edge_uid:
                if constraint.applied_rev is None:
                    removed = self.constraints.pop(index)
                else:
                    constraint.enabled = False
                    removed = constraint
                break
        if removed is None:
            return

        if removed.replaces_edge_uid is not None and self.pose_graph is not None:
            restored_edge = self.pose_graph.edge_index_by_uid.get(removed.replaces_edge_uid)
            change = self.disabled_loop_changes.get(removed.replaces_edge_uid)
            if restored_edge is not None:
                restored_edge.enabled = True
            if change is not None:
                if change.applied_rev is None:
                    self.disabled_loop_changes.pop(removed.replaces_edge_uid, None)
                else:
                    change.enabled = False

        self.selected_edge_ref = None
        self._candidate_replace_edge_uid = None
        self._recompute_session_dirty()
        self._rebuild_constraint_table()
        self._update_selection_labels()
        self._update_edge_action_buttons()
        self._update_session_status_widgets()
        self._refresh_plot(preserve_view=True)
        if removed.replaces_edge_uid is not None:
            self.append_log(
                f"Removed replacement manual constraint {removed.target_id}->{removed.source_id} and restored the original loop edge."
            )
        else:
            self.append_log(f"Removed manual constraint {removed.target_id}->{removed.source_id}")
        self._save_project_state()

        if self.workspace is not None and self.source_id is not None and self.target_id is not None:
            self.schedule_preview_refresh(reset_camera=False)
        else:
            self.cloud_view.clear_scene()
            self._preview_scene_key = None

    # ===== BEGIN CHANGE: optimizer backend resolution =====
    def _resolve_optimizer_backend(self):
        return resolve_python_optimizer_backend(
            script_dir=SCRIPT_DIR,
            project_root=PROJECT_ROOT,
        )
    # ===== END CHANGE: optimizer backend resolution =====

    def _start_optimizer_backend(
        self,
        backend,
        options: OptimizerRunOptions,
    ) -> None:
        program = backend.program
        args = backend.build_process_args(options)
        backend_name = backend.display_name

        self._optimizer_process = QtCore.QProcess(self)
        process_env = QtCore.QProcessEnvironment.systemEnvironment()
        process_env.insert("PYTHONUNBUFFERED", "1")
        self._optimizer_process.setProcessEnvironment(process_env)
        self._optimizer_process.setProgram(program)
        self._optimizer_process.setArguments(args)
        self._optimizer_process.readyReadStandardOutput.connect(self._read_optimizer_stdout)
        self._optimizer_process.readyReadStandardError.connect(self._read_optimizer_stderr)
        self._optimizer_process.finished.connect(self._optimizer_finished)
        self.optimize_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.balm_button.setEnabled(False)
        self._active_optimizer_backend_name = backend_name
        self._optimizer_started_at = time.perf_counter()
        self._optimizer_heartbeat_timer.start()
        self.append_log(
            "Starting Optimize Working Graph with "
            f"{sum(1 for constraint in self.constraints if constraint.enabled)} manual constraints, "
            f"{len(self._active_disabled_loop_changes())} disabled loop edges, "
            f"export_map_voxel={self.export_map_voxel_spin.value():.3f} m, "
            f"optimize_mode={options.optimize_mode}, "
            f"loop_information_scale={options.manual_information_scale:g}, "
            f"correlation_window={options.loop_correlation_window_keyframes}, "
            f"cluster_budget={options.loop_cluster_information_budget:g}, "
            f"optimizer backend={backend_name}. "
            "Full map rebuild is deferred until Export."
        )
        self._optimizer_process.start()
        self._update_session_status_widgets()
        if self._repair_map_stage is not None:
            self._set_repair_progress(
                "Pose-graph optimization", busy=True)

    def _on_optimizer_heartbeat(self) -> None:
        if self._optimizer_process is None or self._optimizer_started_at is None:
            return
        elapsed = time.perf_counter() - self._optimizer_started_at
        self.append_log(
            f"Optimizer running ({self._active_optimizer_backend_name}) · elapsed={elapsed:.1f}s"
        )

    FACTOR_CONSISTENCY_LIMIT = 1.0
    FACTOR_CHI2_9973_DOF6 = 20.061901972375512

    def _factor_residuals(self, trajectory):
        """Each enabled constraint's uncertainty-normalized solve residual.

        Translation and rotation are evaluated together against the factor's
        stored covariance and the global PGO information scale.  Dividing NIS
        by the 99.73% chi-square threshold gives a unitless gate at one.
        """
        transforms = trajectory.transforms_world_sensor
        out = []
        policy = getattr(self, "_last_optimizer_policy", None) or {}
        information_scale = float(policy.get("manual_information_scale", 1.0))
        for index, constraint in enumerate(self.constraints):
            if not constraint.enabled:
                continue
            sid, tid = constraint.source_id, constraint.target_id
            if not (0 <= sid < len(transforms) and 0 <= tid < len(transforms)):
                continue
            gap = (np.linalg.inv(constraint.transform_target_source_final)
                   @ (np.linalg.inv(transforms[tid]) @ transforms[sid]))
            translation = np.asarray(gap[:3, 3], dtype=np.float64)
            rotation = Rotation.from_matrix(gap[:3, :3]).as_rotvec()
            variance_t = np.maximum(
                np.asarray(constraint.variance_t_m2, dtype=np.float64), 1e-12
            )
            variance_r = np.maximum(
                np.asarray(constraint.variance_r_rad2, dtype=np.float64), 1e-12
            )
            nis = information_scale * (
                float(np.sum(np.square(translation) / variance_t))
                + float(np.sum(np.square(rotation) / variance_r))
            )
            out.append((nis / self.FACTOR_CHI2_9973_DOF6, index))
        return out

    def _leave_one_out_worst(self, index: int):
        """Re-solve without constraint ``index`` and return the worst residual.

        The graph has to be solved again for the answer to mean anything.
        Recomputing residuals against the SAME trajectory only drops that row
        from the list, so the minimum over trials is always "remove the worst
        one" and its value is always the second-worst -- the test degenerates
        into `worst > 2 x second-worst`. Measured on the indoor ih session:
        a false loop demanding a 39 m correction scored 2.452 m against a
        second-worst of 1.161 m, i.e. it cleared that accidental test by 13 cm.
        Re-solving is what the headless evaluator does, and it answers the
        question actually being asked -- there the same constraint went
        2.45 -> 0.03 m, which is not a near miss.

        Returns None when the trial could not be run, so the caller skips it
        rather than treating a failed solve as evidence.
        """
        if self.session_paths is None or self._last_output_dir is None:
            return None
        if self.original_trajectory is None:
            return None
        backend = self._resolve_optimizer_backend()
        g2o_path = self._last_output_dir / "edited_input_pose_graph.g2o"
        if backend is None or not g2o_path.is_file():
            return None
        constraint = self.constraints[index]
        trial_dir = self._last_output_dir.parent / (
            self._last_output_dir.name + f"_loo{index}")
        trial_csv = self._last_output_dir / f"loo{index}.csv"
        was = constraint.enabled
        constraint.enabled = False
        try:
            with trial_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(CONSTRAINT_CSV_HEADER)
                for other in self.constraints:
                    if other.enabled:
                        writer.writerow(other.csv_row())
            policy = getattr(self, "_last_optimizer_policy", None) or {
                "optimize_mode": str(self.optimize_mode_combo.currentData()),
                "manual_information_scale": 1.0,
                "loop_correlation_window_keyframes": 0,
                "loop_cluster_information_budget": 0.0,
                "loop_correlation_policy": "pair",
                "loop_cluster_allocation": "equal",
            }
            options = OptimizerRunOptions(
                session_root=self.session_paths.session_root,
                g2o_path=g2o_path,
                tum_path=self.original_trajectory.path,
                keyframe_dir=self.session_paths.keyframe_dir,
                constraints_csv=trial_csv,
                output_dir=trial_dir,
                map_voxel_leaf=float(self.export_map_voxel_spin.value()),
                optimize_mode=policy["optimize_mode"],
                skip_map_build=True,
                manual_information_scale=policy["manual_information_scale"],
                loop_correlation_window_keyframes=policy[
                    "loop_correlation_window_keyframes"
                ],
                loop_cluster_information_budget=policy[
                    "loop_cluster_information_budget"
                ],
                loop_correlation_policy=policy["loop_correlation_policy"],
                loop_cluster_allocation=policy["loop_cluster_allocation"],
            )
            trial_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [backend.program, *backend.build_process_args(options)],
                capture_output=True, text=True, check=False)
            trial_tum = trial_dir / "optimized_poses_tum.txt"
            if completed.returncode != 0 or not trial_tum.is_file():
                return None
            residuals = self._factor_residuals(load_tum_trajectory(trial_tum))
            if not residuals:
                return 0.0
            return max(value for value, _ in residuals)
        except (OSError, ValueError):
            return None
        finally:
            constraint.enabled = was

    def _retract_inconsistent_constraint(self, trajectory) -> bool:
        """Disable the one constraint the solution cannot accommodate.

        Ported from the headless evaluator, which is the only place either
        implementation could retract a loop AFTER the graph was solved. Without
        it a constraint that cleared the per-pair gate is permanent, and it goes
        on shaping every later round's diagnosis and measurements.

        When two constraints contradict, the optimizer splits the difference and
        BOTH show a large residual, so the flagged set cannot be read directly.
        Leave-one-out resolves it: the constraint whose removal most reduces the
        worst residual is the one that was wrong. Requiring the removal to more
        than halve the worst residual keeps "removing anything helps a little"
        from retracting a good loop.

        Returns True when something was disabled and a re-solve was started.
        """

        if self._optimizer_process is not None:
            return False
        flagged = [(value, index) for value, index in self._factor_residuals(trajectory)
                   if value > self.FACTOR_CONSISTENCY_LIMIT]
        if not flagged:
            return False
        worst = max(flagged)[0]
        self.append_log(
            f"[consistency] {len(flagged)} constraint(s) disagree after "
            f"optimization (worst normalized NIS {worst:.2f}) -> leave-one-out")
        best = None
        trials = sorted(flagged, reverse=True)[:4]
        for order, (_, index) in enumerate(trials, start=1):
            # Each trial is a full re-solve, so say so -- otherwise the window
            # goes quiet for several seconds with no explanation.
            self._set_repair_progress(
                f"Consistency leave-one-out · re-solving {order}/{len(trials)}",
                busy=True)
            trial_worst = self._leave_one_out_worst(index)
            if trial_worst is None:
                continue
            if best is None or trial_worst < best[0]:
                best = (trial_worst, index)
        if best is None or best[0] >= worst * 0.5:
            return False
        index = best[1]
        target = self.constraints[index]
        self.constraints[index].enabled = False
        self._rebuild_constraint_table()
        return self._reoptimize_after_retraction(
            f"[consistency] disabled {target.target_id}->{target.source_id}: "
            f"worst normalized NIS {worst:.2f} -> {best[0]:.2f} without it")

    def _reoptimize_after_retraction(self, what: str) -> bool:
        """Re-solve the graph after a safeguard disabled one or more constraints.

        Retraction and re-optimization have to be one step. Disabling a
        constraint without re-solving leaves the poses still shaped by it --
        the worst of both, since the map keeps the damage and the graph no
        longer records the cause. Every mechanism ported from the headless
        evaluator ends this way: consistency leave-one-out, seed probation, and
        the repair check all disable and then `continue` into a fresh solve.

        ``run_optimization`` cannot be called directly for this. It refuses when
        the working graph is not dirty and when no enabled constraint remains,
        and it reports the refusal through a modal dialog -- so a retraction
        that empties the enabled set would stall Repair Map behind a message box
        with no one to dismiss it. Both refusals are wrong here by construction:
        the graph IS dirty (we just changed it), and an empty enabled set is a
        legitimate outcome that still needs solving, because the answer is then
        the odometry.

        Returns True when a solve was started; the caller must not advance its
        own stage in that case, because ``_optimizer_finished`` will.
        """
        if self._optimizer_process is not None or self.session_paths is None:
            return False
        self._session_dirty = True
        self.append_log(f"[retract] {what}; re-optimizing without them.")
        previous_stage = self._repair_map_stage
        # Rewind the old in-process one-click stage machine when it is driving
        # the solve. Canonical Repair runs in the headless process and never
        # enters this branch.
        if previous_stage is not None:
            self._repair_map_stage = "optimize"
        try:
            self.run_optimization()
        except Exception:
            self.append_log(f"[retract] re-optimization failed:\n{traceback.format_exc()}")
            self._repair_map_stage = previous_stage
            return False
        if self._optimizer_process is None:
            # run_optimization refused after all; do not leave the caller
            # believing a solve is on its way.
            self._repair_map_stage = previous_stage
            self.append_log("[retract] re-optimization was refused; constraints stay disabled.")
            return False
        return True

    def run_optimization(self) -> None:
        if self._optimizer_process is not None:
            self._show_error("Optimize Working Graph", "An optimization process is already running.")
            return
        if self.session_paths is None or self.pose_graph is None or self.original_pose_graph is None or self.original_trajectory is None:
            self._show_error("Optimize Working Graph", "Load a session first.")
            return
        baseline_repair = (
            self._repair_map_stage == "optimize" and not self._session_dirty
        )
        if not self._session_dirty and not baseline_repair:
            self._show_error("Optimize Working Graph", "Working graph is already clean. Add, remove, or toggle a change first.")
            return

        enabled_constraints = [constraint for constraint in self.constraints if constraint.enabled]
        active_disabled_changes = self._active_disabled_loop_changes()
        # An empty enabled set is a legitimate state to solve when a safeguard
        # has just retracted the last constraint: the answer is the odometry,
        # and refusing here would strand Repair Map mid-flow.
        retraction_solve = self._repair_map_stage is not None
        if (not enabled_constraints and not active_disabled_changes
                and not baseline_repair and not retraction_solve):
            self._show_error(
                "Optimize Working Graph",
                "No enabled manual constraints or disabled existing loop edges to export.",
            )
            return
        if baseline_repair:
            self.append_log(
                "[RepairMap] Running a zero-change optimizer baseline so BALM "
                "has an auditable run directory and trajectory input."
            )

        backend = self._resolve_optimizer_backend()
        if backend is None:
            self._show_error(
                "Optimize Working Graph",
                "The Python optimizer is unavailable. Install the Python GTSAM wrapper and try again.",
            )
            return

        output_dir = (
            self.session_paths.session_root
            / "manual_loop_runs"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        edited_g2o_path = output_dir / "edited_input_pose_graph.g2o"
        baseline_pose_graph = copy.deepcopy(self.original_pose_graph)
        for edge in baseline_pose_graph.loop_edges:
            if edge.edge_uid in self.disabled_loop_changes:
                edge.enabled = not self.disabled_loop_changes[edge.edge_uid].enabled
        write_filtered_pose_graph(baseline_pose_graph, edited_g2o_path)

        csv_path = output_dir / "manual_loop_constraints.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CONSTRAINT_CSV_HEADER)
            for constraint in enabled_constraints:
                writer.writerow(constraint.csv_row())

        production_run = self._repair_map_stage is not None
        optimize_mode = (
            str(PRODUCTION_PGO_PROFILE["optimize_mode"])
            if production_run else str(self.optimize_mode_combo.currentData())
        )
        use_calibrated_weight = optimize_mode == OPTIMIZE_MODE_GNC_TLS
        optimizer_policy = {
            "optimize_mode": optimize_mode,
            "manual_information_scale": (
                float(PRODUCTION_PGO_PROFILE["manual_information_scale"])
                if use_calibrated_weight else 1.0
            ),
            "loop_correlation_window_keyframes": (
                int(PRODUCTION_PGO_PROFILE["correlation_window_keyframes"])
                if use_calibrated_weight else 0
            ),
            "loop_cluster_information_budget": (
                float(PRODUCTION_PGO_PROFILE["cluster_information_budget"])
                if use_calibrated_weight else 0.0
            ),
            "loop_correlation_policy": str(
                PRODUCTION_PGO_PROFILE["correlation_policy"]
            ),
            "loop_cluster_allocation": str(
                PRODUCTION_PGO_PROFILE["cluster_allocation"]
            ),
        }

        options = OptimizerRunOptions(
            session_root=self.session_paths.session_root,
            g2o_path=edited_g2o_path,
            tum_path=self.original_trajectory.path,
            keyframe_dir=self.session_paths.keyframe_dir,
            constraints_csv=csv_path,
            output_dir=output_dir,
            map_voxel_leaf=float(self.export_map_voxel_spin.value()),
            optimize_mode=optimizer_policy["optimize_mode"],
            skip_map_build=True,
            manual_information_scale=optimizer_policy[
                "manual_information_scale"
            ],
            loop_correlation_window_keyframes=optimizer_policy[
                "loop_correlation_window_keyframes"
            ],
            loop_cluster_information_budget=optimizer_policy[
                "loop_cluster_information_budget"
            ],
            loop_correlation_policy=optimizer_policy[
                "loop_correlation_policy"
            ],
            loop_cluster_allocation=optimizer_policy[
                "loop_cluster_allocation"
            ],
        )
        self._last_optimizer_policy = dict(optimizer_policy)
        # Capture the previously accepted run *before* redirecting
        # ``_last_output_dir``.  The old order snapshotted the new, still-empty
        # directory, so a failed optimizer could become the apparent export
        # source instead of rolling back to the last valid result.
        self._pre_optimize_snapshot = self._capture_undo_snapshot()
        self._last_output_dir = output_dir
        self._write_run_context(output_dir)
        self._start_optimizer_backend(backend, options)

    def _read_optimizer_stdout(self) -> None:
        if self._optimizer_process is None:
            return
        data = bytes(self._optimizer_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for line in data.splitlines():
            self.append_log(line)
            self._consume_balm_progress_line(line)

    # ===== BEGIN CHANGE: balm progress display =====
    def _set_repair_progress(self, text: str, value: int | None = None,
                             busy: bool = False, visible: bool = True) -> None:
        if not hasattr(self, "repair_progress"):
            return
        self.repair_progress.setVisible(visible)
        if busy:
            self.repair_progress.setRange(0, 0)
        else:
            self.repair_progress.setRange(0, 100)
            if value is not None:
                self.repair_progress.setValue(max(0, min(100, int(value))))
        self.repair_progress.setFormat(text)

    def _consume_balm_progress_line(self, line: str) -> None:
        if getattr(self, "_balm_progress_total", 0) <= 0:
            return
        match = re.search(r"\[BALM\] iter=\d+ .*?rms=([0-9.]+)", line)
        if match is None:
            return
        self._balm_progress_seen += 1
        fraction = self._balm_progress_seen / max(1, self._balm_progress_total)
        if self._repair_map_stage is not None:
            value = min(94, 20 + round(6 * fraction))
            self._set_repair_progress(
                f"BALM · final refinement · {self._balm_progress_seen}/"
                f"{self._balm_progress_total}",
                value=value)
        else:
            self._set_repair_progress(
                f"BALM · {self._balm_progress_seen}/{self._balm_progress_total}",
                value=round(100 * fraction))
        update_match = re.search(
            r"max_trans=([0-9.]+) m max_rot=([0-9.]+) deg", line
        )
        update_text = ""
        if update_match is not None:
            update_text = (
                f" · max Δt={float(update_match.group(1)) * 100:.1f} cm"
                f" · max ΔR={float(update_match.group(2)):.3f}°"
            )
        self.gicp_metrics_label.setText(
            f"Final Map Refinement · iteration {self._balm_progress_seen}/"
            f"{self._balm_progress_total} · rms={float(match.group(1)) * 100:.1f} cm"
            + update_text
        )
    # ===== END CHANGE: balm progress display =====

    def _read_optimizer_stderr(self) -> None:
        if self._optimizer_process is None:
            return
        data = bytes(self._optimizer_process.readAllStandardError()).decode("utf-8", errors="ignore")
        for line in data.splitlines():
            self.append_log(f"[stderr] {line}")

    def _optimizer_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        self.optimize_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._update_session_status_widgets()
        self._optimizer_heartbeat_timer.stop()
        elapsed_text = ""
        if self._optimizer_started_at is not None:
            elapsed_text = f", elapsed={time.perf_counter() - self._optimizer_started_at:.2f}s"
        # 这个进程已经退出了，句柄必须在任何后续步骤之前清掉。留着它的代价是
        # 静默的：_retract_inconsistent_constraint 第一句就是 "句柄非空则返回
        # False"，于是一致性检查在 GUI 里**一次也没跑过**——2026-08-05 室内 ih
        # 会话上，一条要求把图挪 39 m 的假回环（533->1765，残差 2.45 m）就是这样
        # 混过去的，无头同一批约束在第一次优化后立刻撤掉了它。
        self._optimizer_process = None
        if exit_status == QtCore.QProcess.NormalExit and exit_code == 0:
            self.append_log(
                f"Optimizer finished successfully ({self._active_optimizer_backend_name}). "
                f"Output: {self._last_output_dir}{elapsed_text}"
            )
            if not self._apply_working_optimization_result():
                self.append_log(
                    "Optimizer output could not be loaded; treating the run "
                    "as failed instead of advancing to BALM."
                )
                if self._pre_optimize_snapshot is not None:
                    self._last_output_dir = (
                        self._pre_optimize_snapshot.last_output_dir
                    )
                self._pending_export_after_optimize = False
                self._repair_map_advance("optimize", False)
                self._pre_optimize_snapshot = None
                self._optimizer_started_at = None
                self._update_session_status_widgets()
                return
            self.append_log(
                "Working trajectory updated. Final map rebuild will run "
                "during Export."
            )
            if self._pre_optimize_snapshot is not None:
                self._append_undo_snapshot(self._pre_optimize_snapshot)
            self._save_project_state()
            # Retract before advancing: a constraint the solution cannot
            # accommodate must not reach BALM or seed the next round's
            # diagnosis. When one is retracted this starts a fresh solve and
            # _optimizer_finished runs again, so the stage is advanced by that
            # pass instead of this one.
            if (self._repair_map_stage is not None
                    and self.trajectory is not None
                    and self._retract_inconsistent_constraint(self.trajectory)):
                self._pre_optimize_snapshot = None
                self._optimizer_started_at = None
                return
            if self._pending_export_after_optimize:
                self._pending_export_after_optimize = False
                self.export_final_result()
            elif self._repair_map_stage == "optimize":
                self._repair_map_advance("optimize", True)
            elif self.balm_auto_check.isChecked():
                self.append_log("Auto BALM is enabled; starting BALM refinement.")
                QtCore.QTimer.singleShot(0, self.run_balm_refinement)
        else:
            self.append_log(
                f"Optimizer failed ({self._active_optimizer_backend_name}). "
                f"exit_code={exit_code}, exit_status={int(exit_status)}{elapsed_text}"
            )
            self._pending_export_after_optimize = False
            if self._pre_optimize_snapshot is not None:
                self._last_output_dir = self._pre_optimize_snapshot.last_output_dir
            self._repair_map_advance("optimize", False)
        self._pre_optimize_snapshot = None
        self._optimizer_started_at = None
        self._optimizer_process = None
        self._update_session_status_widgets()

    def _apply_working_optimization_result(self) -> bool:
        if self._last_output_dir is None:
            return False
        tum_path = self._last_output_dir / "optimized_poses_tum.txt"
        if not tum_path.is_file():
            self.append_log("Working graph update skipped: optimized TUM output is missing.")
            return False
        try:
            candidate = load_tum_trajectory(tum_path)
            reference = self.original_trajectory or self.trajectory
            if reference is None:
                raise TrajectoryValidationError(
                    "No loaded trajectory is available for output validation."
                )
            if candidate.size != reference.size:
                raise TrajectoryValidationError(
                    "Optimized trajectory pose count changed from "
                    f"{reference.size} to {candidate.size}."
                )
            if not np.allclose(
                candidate.timestamps, reference.timestamps,
                rtol=0.0, atol=1e-6,
            ):
                raise TrajectoryValidationError(
                    "Optimized trajectory timestamps do not match the "
                    "loaded keyframe sequence."
                )
            if not np.isfinite(candidate.transforms_world_sensor).all():
                raise TrajectoryValidationError(
                    "Optimized trajectory contains non-finite poses."
                )
            candidate_workspace = RegistrationWorkspace(
                self.session_paths.keyframe_dir, candidate
            )
            self.trajectory = candidate
            self.workspace = candidate_workspace
            self._working_revision += 1
            for constraint in self.constraints:
                if constraint.enabled:
                    constraint.applied_rev = self._working_revision
            for change in self.disabled_loop_changes.values():
                if change.enabled:
                    change.applied_rev = self._working_revision
            self._session_dirty = False
            self._clear_delta_silent()
            self.current_preview = None
            self.last_result = None
            self._candidate_replace_edge_uid = None
            self.accept_button.setEnabled(False)
            self.gicp_metrics_label.setText("Working graph updated · continue from the new trajectory.")
            with QtCore.QSignalBlocker(self.trajectory_view_combo):
                self.trajectory_view_combo.setCurrentText("Working")
            self._rebuild_constraint_table()
            self._update_selection_labels()
            self._update_edge_action_buttons()
            self._update_session_status_widgets()
            self._refresh_plot(preserve_view=False)
            if self.source_id is not None and self.target_id is not None:
                self.schedule_preview_refresh(reset_camera=False)
            self._save_project_state()
            return True
        except Exception as exc:
            self.append_log(f"Failed to update working graph after optimization: {exc}")
            return False

    # ===== BEGIN CHANGE: balm refinement workflow =====
    def _resolve_balm_backend(self):
        return resolve_python_balm_backend(
            script_dir=SCRIPT_DIR,
            project_root=PROJECT_ROOT,
        )

    def run_balm_refinement(self) -> None:
        if self._optimizer_process is not None:
            self._show_error("BALM Refinement", "An optimization process is already running.")
            return
        if self.session_paths is None or self.trajectory is None:
            self._show_error("BALM Refinement", "Load a session first.")
            return
        if self._session_dirty:
            self._show_error(
                "BALM Refinement",
                "Working graph has pending changes. Run Optimize Working Graph first so "
                "BALM refines the committed result.",
            )
            return
        if self._last_output_dir is None or not (
            self._last_output_dir / "optimized_poses_tum.txt"
        ).is_file():
            self._show_error(
                "BALM Refinement",
                "No optimized working result is available yet. Run Optimize Working Graph first.",
            )
            return

        backend = self._resolve_balm_backend()
        if backend is None:
            self._show_error(
                "BALM Refinement",
                "The BALM backend is unavailable. It needs a Python interpreter with "
                "numpy and scipy installed.",
            )
            return

        source_run_dir = self._last_output_dir
        # Kept so the sanity gate below can measure how far this pass moved the
        # poses, and can hand them back if the answer is "too far".
        self._balm_source_run_dir = source_run_dir
        output_dir = (
            self.session_paths.session_root
            / "manual_loop_runs"
            / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_balm")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        options = BalmRunOptions(
            tum_path=source_run_dir / "optimized_poses_tum.txt",
            keyframe_dir=self.session_paths.keyframe_dir,
            output_dir=output_dir,
            source_run_dir=source_run_dir,
            root_voxel_size=float(self.balm_voxel_spin.value()),
            max_iterations=int(self.balm_iter_spin.value()),
            coarse_plane_thickness=float(self.balm_merge_spin.value()),
            downsample_leaf=float(self._balm_downsample_leaf),
            max_range=float(self._balm_max_range),
            # The same value the pre-seed baseline diagnosed at, so every round
            # is measured against the map the baseline described. It used to run
            # at balm_cli's hardcoded 20 m, which outdoors is a tighter filter
            # than the 30 m the baseline used.
            max_observation_range=float(REPAIR_POLICY["ghost_observation_range_m"]),
            double_sided_enable=bool(self.balm_double_sided_check.isChecked()),
            diagnose_map_inconsistency=bool(
                self.map_diagnostics_check.isChecked()
            ),
        )
        self._pre_optimize_snapshot = self._capture_undo_snapshot()
        self._balm_cancel_requested = False
        self._last_output_dir = output_dir
        self._write_run_context(output_dir)
        self._start_balm_backend(backend, options)

    def _start_balm_backend(self, backend, options: "BalmRunOptions") -> None:
        self._balm_progress_seen = 0
        self._balm_progress_total = int(options.max_iterations)
        self._optimizer_process = QtCore.QProcess(self)
        process_env = QtCore.QProcessEnvironment.systemEnvironment()
        process_env.insert("PYTHONUNBUFFERED", "1")
        self._optimizer_process.setProcessEnvironment(process_env)
        self._optimizer_process.setProgram(backend.program)
        self._optimizer_process.setArguments(backend.build_process_args(options))
        self._optimizer_process.readyReadStandardOutput.connect(self._read_optimizer_stdout)
        self._optimizer_process.readyReadStandardError.connect(self._read_optimizer_stderr)
        self._optimizer_process.finished.connect(self._balm_finished)
        self.optimize_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.balm_button.setEnabled(False)
        self._active_optimizer_backend_name = backend.display_name
        self._optimizer_started_at = time.perf_counter()
        self._optimizer_heartbeat_timer.start()
        self.append_log(
            "Starting BALM plane bundle adjustment on "
            f"{options.tum_path} with root_voxel={options.root_voxel_size:.2f} m, "
            f"max_iterations={options.max_iterations}, backend={backend.display_name}."
        )
        self._set_repair_progress("BALM · final refinement", value=20)
        self._optimizer_process.start()
        self._update_session_status_widgets()

    BALM_SANITY_LIMIT_M = REPAIR_POLICY["balm_sanity_limit_m"]
    BALM_SANITY_LIMIT_ROTATION_DEG = REPAIR_POLICY[
        "balm_sanity_limit_rotation_deg"
    ]

    def _balm_report_badness(self) -> float:
        """Map inconsistency scored from the BALM report that just landed."""
        if self._last_output_dir is None:
            return 0.0
        try:
            report = json.loads(
                (self._last_output_dir / "balm_report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0.0
        return ghost_badness(report.get("ghost_regions") or [])

    def _balm_output_is_sane(self) -> bool:
        """Should this BALM pass be adopted, or should its input be kept?

        Plane refinement should move poses centimetres. Metre-scale motion
        means the BA geometry is degenerate -- no horizontal support in
        narrow-FOV data, say -- and its output has to be dropped in favour of
        the poses that went in. Map-inconsistency scores are diagnostic-only;
        they never accept or reject the production BALM result.
        """
        source, output = self._balm_source_run_dir, self._last_output_dir
        self._last_balm_motion_stats = {}
        if source is None or output is None:
            self.append_log(
                "[balm-gate] missing source/output run directory -> rejecting "
                "BALM output"
            )
            return False
        try:
            before = load_tum_trajectory(source / "optimized_poses_tum.txt")
            after = load_tum_trajectory(output / "optimized_poses_tum.txt")
        except Exception as exc:
            self.append_log(
                f"[balm-gate] cannot validate BALM trajectory ({exc}) -> "
                "rejecting BA output, keeping PGO poses"
            )
            return False
        a, b = before.positions_xyz, after.positions_xyz
        ta = before.transforms_world_sensor
        tb = after.transforms_world_sensor
        if (
            len(a) == 0 or len(a) != len(b)
            or not np.isfinite(ta).all() or not np.isfinite(tb).all()
        ):
            self.append_log(
                f"[balm-gate] invalid trajectory shape/values "
                f"({len(a)} input vs {len(b)} output) -> rejecting BA output, "
                "keeping PGO poses"
            )
            return False
        moved = float(np.linalg.norm(a - b, axis=1).mean())
        relative_rotation = np.einsum(
            "nji,njk->nik", ta[:, :3, :3], tb[:, :3, :3]
        )
        rotated = float(np.degrees(
            Rotation.from_matrix(relative_rotation).magnitude()
        ).mean())
        self._last_balm_motion_stats = {
            "measured_mean_pose_motion_m": moved,
            "measured_mean_pose_rotation_deg": rotated,
        }
        if moved > self.BALM_SANITY_LIMIT_M:
            self.append_log(
                f"[balm-gate] refinement moved poses {moved:.2f} m on average "
                f"(> {self.BALM_SANITY_LIMIT_M}) -> rejecting BA output, "
                "keeping PGO poses")
            return False
        if rotated > self.BALM_SANITY_LIMIT_ROTATION_DEG:
            self.append_log(
                f"[balm-gate] refinement rotated poses {rotated:.2f}° on "
                f"average (> {self.BALM_SANITY_LIMIT_ROTATION_DEG}°) -> "
                "rejecting BA output, keeping PGO poses"
            )
            return False
        return True

    def _balm_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        self.optimize_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._optimizer_heartbeat_timer.stop()
        elapsed_text = ""
        if self._optimizer_started_at is not None:
            elapsed_text = f", elapsed={time.perf_counter() - self._optimizer_started_at:.2f}s"
        working_result_applied = False
        cancelled = bool(getattr(self, "_balm_cancel_requested", False))
        # A cancellation always means "retain completed PGO", even if a worker
        # catches SIGTERM and happens to exit with status zero.
        if (
            not cancelled
            and exit_status == QtCore.QProcess.NormalExit
            and exit_code == 0
        ):
            self.append_log(
                f"BALM refinement finished successfully ({self._active_optimizer_backend_name}). "
                f"Output: {self._last_output_dir}{elapsed_text}"
            )
            if self._pre_optimize_snapshot is not None:
                self._append_undo_snapshot(self._pre_optimize_snapshot)
            self._balm_ghost_suggestions = self._load_balm_ghost_suggestions()
            badness = self._balm_report_badness()
            balm_adopted = self._balm_output_is_sane()
            self._last_balm_adopted = bool(balm_adopted)
            if not balm_adopted and self._balm_source_run_dir is not None:
                # Copy the trajectory and matching graph that went in over the
                # ones that came out, rather than pointing the tool back at the
                # source directory:
                # every downstream consumer -- the working trajectory, the
                # report summary, the next round's BALM source -- keeps reading
                # this run directory, and now finds the accepted poses in it.
                # The suggestions loaded above still stand; they describe the
                # map, not this pass's motion.
                for name in ("optimized_poses_tum.txt", "pose_graph.g2o"):
                    source = self._balm_source_run_dir / name
                    if source.is_file():
                        shutil.copy2(source, self._last_output_dir / name)
                self._balm_ghost_suggestions = []
            report_path = self._last_output_dir / "balm_report.json"
            try:
                adoption_report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                adoption_report = {}
            adoption_report["production_adoption"] = {
                "adopted": bool(balm_adopted),
                "sanity_limit_mean_pose_motion_m": float(
                    self.BALM_SANITY_LIMIT_M
                ),
                "sanity_limit_mean_pose_rotation_deg": float(
                    self.BALM_SANITY_LIMIT_ROTATION_DEG
                ),
                **getattr(self, "_last_balm_motion_stats", {}),
                "fallback": (
                    None if balm_adopted
                    else "restored_pgo_trajectory_and_graph"
                ),
            }
            self._write_json(report_path, adoption_report)
            # Next pass's gate reads this round's map, exactly as the headless
            # loop's `prev_badness = badness` does.
            self._prev_ghost_badness = badness
            working_result_applied = self._apply_working_optimization_result()
            self.gicp_metrics_label.setText(
                (
                    "BALM refinement applied · working trajectory updated."
                    if balm_adopted else
                    "BALM rejected by sanity gate · PGO trajectory restored."
                )
                + (self._balm_report_summary() if balm_adopted else "")
            )
            self._save_project_state()
        else:
            self._last_balm_adopted = False
            if cancelled:
                self.append_log(
                    "Final Map Refinement cancelled; restoring the completed "
                    f"PGO result{elapsed_text}."
                )
            else:
                self.append_log(
                    f"BALM refinement failed ({self._active_optimizer_backend_name}). "
                    f"exit_code={exit_code}, exit_status={int(exit_status)}{elapsed_text}"
                )
            if self._pre_optimize_snapshot is not None:
                self._last_output_dir = self._pre_optimize_snapshot.last_output_dir
        self._balm_progress_total = 0
        self._pre_optimize_snapshot = None
        self._optimizer_started_at = None
        self._optimizer_process = None
        self._balm_cancel_requested = False
        self._update_session_status_widgets()
        balm_succeeded = (
            exit_status == QtCore.QProcess.NormalExit
            and exit_code == 0 and working_result_applied
        )
        if cancelled and self._repair_map_stage is not None:
            self._repair_map_finish(
                "stopped during Final Map Refinement; retained PGO"
            )
            return
        if self._repair_map_advance("balm", balm_succeeded):
            return

    # ===== BEGIN CHANGE: one-click balm suggestions =====
    def _start_background_task(self, label: str, fn, on_finished) -> bool:
        """Start a single serialized Python worker and keep GUI state coherent."""
        if self._background_thread is not None:
            self._show_error(label, f"{self._background_label} is still running.")
            return False
        thread = QtCore.QThread(self)
        worker = BackgroundTask(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            self._on_background_task_result, QtCore.Qt.QueuedConnection)
        worker.failed.connect(
            self._on_background_task_failed, QtCore.Qt.QueuedConnection)
        worker.progress.connect(
            self._on_background_task_progress, QtCore.Qt.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(
            self._finalize_background_task, QtCore.Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater)
        self._background_thread = thread
        self._background_worker = worker
        self._background_label = label
        self._background_result = None
        self._background_error = None
        self._background_on_finished = on_finished
        self._preview_timer.stop()
        self.optimize_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.run_gicp_button.setEnabled(False)
        self.auto_yaw_button.setEnabled(False)
        self.accept_button.setEnabled(False)
        self.gicp_metrics_label.setText(
            f"{label} is running in background · the window remains responsive.")
        progress_labels = {
            INITIAL_LOOP_SEARCH_LABEL: (
                "Initial Loop Search · retrieval, registration and audit"
            ),
            "Build Preview": "Building point-cloud preview",
            "Run GICP": "Running GICP registration",
            "Auto Yaw": "Sweeping yaw candidates",
            "Load Session": "Loading and validating session",
            "Export Final Result": "Export · preparing final map",
            "Compare PGO/BALM": "Compare PGO/BALM · preparing two maps",
        }
        if label in progress_labels:
            progress_text = progress_labels[label]
        else:
            progress_text = "Verifying loop candidates"
        self._set_repair_progress(progress_text, busy=True)
        self.append_log(f"[{label}] Running in background; the window remains responsive.")
        self._update_session_status_widgets()
        thread.start()
        return True

    @QtCore.pyqtSlot(str, int, int)
    def _on_background_task_progress(
        self, text: str, current: int, total: int
    ) -> None:
        if self._background_thread is None:
            return
        if total > 0:
            fraction = max(0, min(current, total)) / total
            # Initial Loop Search is only the first part of Repair Map.
            # pipeline.  Showing its local 0..100 progress made the bar jump
            # backwards from 100 to 12 when Optimize started.  Reserve 0..12
            # for seeding so the full pipeline remains monotonic.
            if (self._repair_map_stage == "seed"
                    and self._background_label == INITIAL_LOOP_SEARCH_LABEL):
                value = 2 + int(round(9.0 * fraction))
                text = f"Repair Map · {text}"
            else:
                value = int(round(100.0 * fraction))
            self._set_repair_progress(text, value=value)
        else:
            self._set_repair_progress(text, busy=True)
        self.gicp_metrics_label.setText(
            f"{text}<br><span style='color:#64748b;font-size:11px'>"
            "Running in background · the window remains responsive.</span>"
        )

    @QtCore.pyqtSlot(object)
    def _on_background_task_result(self, value) -> None:
        self._background_result = value
        thread = self._background_thread
        if thread is not None:
            thread.quit()

    @QtCore.pyqtSlot(str)
    def _on_background_task_failed(self, details: str) -> None:
        self._background_error = details
        thread = self._background_thread
        if thread is not None:
            thread.quit()

    @QtCore.pyqtSlot()
    def _finalize_background_task(self) -> None:
        """Apply worker output only after its event loop has stopped.

        This intentionally contains no ``QThread.wait()``. Waiting from a Qt
        completion callback can freeze the main event loop and, depending on
        PyQt's callable proxy, can even attempt to wait on the worker itself.
        """
        label = self._background_label
        value = self._background_result
        details = self._background_error
        on_finished = self._background_on_finished
        restart_preview = label == "Build Preview" and self._preview_refresh_queued
        self._preview_refresh_queued = False
        self._background_worker = None
        self._background_thread = None
        self._background_label = ""
        self._background_result = None
        self._background_error = None
        self._background_on_finished = None
        self.optimize_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.export_button.setEnabled(
            self._last_output_dir is not None or self._session_dirty)
        self._update_edge_action_buttons()
        self._update_session_status_widgets()

        if details is not None:
            self._set_repair_progress(f"{label} failed", value=100)
            self.append_log(f"[{label}] Background task failed:\n{details}")
            if self._repair_map_stage is not None:
                self._repair_map_finish(f"{label} failed")
            self._show_error(
                label, "The background operation failed. See Execution Log.")
            return

        self.gicp_metrics_label.setText(
            "Background operation finished · results applied.")
        if self._repair_map_stage is None:
            self._set_repair_progress(f"{label} finished", value=100)
        try:
            on_finished(value)
        except Exception:
            details = traceback.format_exc()
            self.append_log(f"[{label}] Failed while applying result:\n{details}")
            self._set_repair_progress(f"{label} apply failed", value=100)
            if self._repair_map_stage is not None:
                self._repair_map_finish(f"{label} apply failed")
            self._show_error(
                label, "The result was computed but could not be applied. See Execution Log.")
            return
        if restart_preview:
            # Coalesce all edits made while the previous cloud was loading and
            # render exactly one new preview from the latest widget values.
            self._preview_timer.start()

    def _load_balm_ghost_suggestions(self) -> list:
        if self._last_output_dir is None:
            return []
        try:
            report = json.loads(
                (self._last_output_dir / "balm_report.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return []
        suggestions = []
        for region in report.get("ghost_regions") or []:
            pair = region.get("suggested_pair") or []
            if len(pair) != 2:
                continue
            suggestions.append(
                {
                    "source_id": int(max(pair)),
                    "target_id": int(min(pair)),
                    # Which way along the normal the source has to move. The
                    # detector knows: offsets = centered @ normal is a SIGNED
                    # projection, _two_means_1d splits it into center_low <
                    # center_high, and suggested_pair[0] is always drawn from
                    # the low layer, [1] from the high one (balm.py:778-781).
                    # Re-ordering the pair by keyframe index below throws that
                    # away, which is why this used to try both signs -- it was
                    # lost information, not geometric ambiguity.
                    "sep_sign": 1.0 if int(max(pair)) == int(pair[0]) else -1.0,
                    "separation_m": float(region.get("separation_m", 0.0)),
                    # The surface normal the separation was measured along, so a
                    # proposal can be checked against the defect that produced
                    # it rather than only against the quality gate.
                    "normal": region.get("normal"),
                    "center_xyz": region.get("center_xyz"),
                    # Carried through because seed probation scores these with
                    # ghost_badness(), which weights each region by its size.
                    # Dropping it here crashed the GUI with KeyError the moment
                    # a BALM run finished while any seed was on probation --
                    # i.e. on every automatic repair.
                    "point_count": int(region.get("point_count", 0)),
                }
            )
        suggestions.sort(key=lambda item: -item["separation_m"])
        return suggestions

    def _retry_when_idle(self, retry, on_finished) -> None:
        """Retry an in-process repair continuation after transient GUI work."""
        if self._repair_map_stage is not None:
            QtCore.QTimer.singleShot(250, retry)
        else:
            on_finished(0)              # loop already stopped; let it finish

    # ===== BEGIN CHANGE: automatic seed loops =====
    # How many of a pair's initial guesses register at once; see concurrency.py.
    _HYPOTHESIS_WORKERS = hypothesis_workers()
    # The acceptance gate lives in repair_presets so this window and the
    # headless evaluator cannot disagree; see the note there for the session
    # that diverged when they did.
    GATE_MIN_FITNESS = REPAIR_GATE["min_overlap"]
    GATE_RMSE_VOXEL_RATIO = REPAIR_GATE["rmse_voxel_ratio"]
    GATE_MAX_INLIER_RMSE = REPAIR_GATE["rmse_floor"]

    def _odometry_budget_ok(self, result, environment: str) -> bool:
        """Reject a correction implausibly large for its odometry-chain length."""
        reference = self.original_trajectory or self.trajectory
        transforms = reference.transforms_world_sensor
        sid, tid = result.preview.source_id, result.preview.target_id
        odo = np.linalg.inv(transforms[tid]) @ transforms[sid]
        correction = np.linalg.norm(
            (np.linalg.inv(odo)
             @ result.transform_target_source_final)[:3, 3])
        positions = transforms[:, :3, 3]
        cumulative = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
        base = REPAIR_POLICY["budget_base_outdoor_m" if environment == "outdoor"
                             else "budget_base_indoor_m"]
        budget = base + REPAIR_POLICY["budget_rate"] * abs(
            cumulative[sid] - cumulative[tid])
        return correction <= budget

    def _gated_gicp(self, source_id: int, target_id: int, delta,
                    near_config=None, seed_stage: bool = False):
        """GICP under the quality gate over one or more initial guesses.

        ``delta`` may be a single 4x4 or a list of them. Four tiers are tried in
        increasing cost, and within a tier every hypothesis is scored and the
        smallest residual wins. Returns the best gate-passing result, or None.

        The first two tiers differ only in correspondence radius, and neither
        subsumes the other. A collocation seed zeroes the translation error, so
        what remains is the descriptor's yaw quantisation (one sector, 6 deg),
        which becomes point displacement in proportion to range: 1.0 m at 10 m,
        3.1 m at 30 m. The near radius resolves the close case cleanly but
        cannot reach the far one; the wide radius reaches it but pairs points
        across faces when the seed was already good. GT-checked on MCD
        kth_night_01: two of the three genuine revisits pass at 2 m and fail at
        8 m, the third does the reverse.

        The last two are different failures again. ``submap`` accumulates ten
        frames on the SOURCE side, for scans too sparse for a single frame to
        reach the overlap floor at any radius; ``anneal`` drops the orientation
        gate for a warm start only, for seeds too far off for normals to agree
        before the clouds are aligned.

        ``seed_stage`` drops ``submap``. Retrieval proposes many more pairs than
        ghost diagnosis does and most of them fail, and this is the tier that
        makes failing expensive: 5.1 s of the 8.3 s a hopeless candidate spends
        in the cascade (61%), because twenty-one accumulated source frames carry
        twenty times the points. Against that it has won zero initial-loop
        constraints on every real sequence logged (277 near, 27 annealed,
        22 wide, 0 submap); its only seed wins are on the 16-ring Gazebo
        simulation, which is the sparsity case it was added for. The ghost
        rounds keep all four -- there it wins real constraints (MCD ntu
        795->1751, 1609->1867) and proposes few enough pairs that the same
        per-failure cost buys far less waste.
        """
        deltas = delta if isinstance(delta, list) else [delta]

        def attempt(delta_try, config=None):
            cfg = config or self._current_registration_config()
            try:
                result = self.workspace.run_gicp(
                    source_id=source_id,
                    target_id=target_id,
                    delta_transform_local=delta_try,
                    config=cfg,
                )
            except Exception:
                return None
            rmse_bound = max(self.GATE_MAX_INLIER_RMSE,
                             self.GATE_RMSE_VOXEL_RATIO * cfg.voxel_size)
            if (
                result.fitness < self.GATE_MIN_FITNESS
                or result.inlier_rmse > rmse_bound
            ):
                return None
            return result

        near = near_config or self._current_registration_config()
        # 环境取下拉框，不从体素反推。反推有两个坑，今晚都撞上了：预设里的
        # wide_max_corr 被复制成了 8.0/3.0 两个字面量（改了预设 GUI 不跟），
        # 而 `voxel >= 0.3` 这个判据在体素被改动时会静默把环境判成另一个——
        # 2026-08-05 一次室内会话的体素被滚轮蹭成 0.000，宽档半径和里程计
        # 预算都会跟着翻。无头直接读 ENV 和 repair_preset，这里也照做。
        environment = str(self.env_combo.currentData())
        preset = self.ENV_PRESETS.get(environment) or self.ENV_PRESETS['indoor']
        wide_distance = float(preset['wide_max_corr'])
        wide = replace(
            near,
            max_correspondence_distance=wide_distance,
            max_iterations=max(near.max_iterations, 100),
        )
        submap = replace(wide, source_window=10)
        # anneal 档的热启动配置。2026-08-06 之前它与 wide 的差别是关掉法向门；
        # 该机制已移出本版本（留给期刊版），所以现在两者相同，只剩两段式调度。
        warm_config = wide

        # Taking the smallest residual within a tier, rather than the first
        # hypothesis to clear the gate, is what removed the coarse-ICP prescreen
        # that used to rank them: ordering only mattered because the search
        # stopped at its first success. It cost several seconds per candidate
        # and decided nothing the residual does not decide better.
        # A tier's hypotheses are independent -- same pair, different starting
        # transform -- and each KD query is single-threaded (see concurrency.py),
        # so they register concurrently with bounded fan-out. Native kernels may
        # still use internal threads. Picking the smallest residual keeps the
        # answer identical to serial order; full Repair measured 1.71-2.34x
        # faster with byte-identical terminal constraints and trajectories.
        def _fan_out(fn):
            if len(deltas) < 2 or self._HYPOTHESIS_WORKERS < 2:
                return [fn(d) for d in deltas]
            workers = min(self._HYPOTHESIS_WORKERS, len(deltas))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return [f.result() for f in [pool.submit(fn, d) for d in deltas]]

        def _lowest(results):
            out = None
            for result in results:
                if result is not None and (out is None
                                           or result.inlier_rmse < out.inlier_rmse):
                    out = result
            return out

        tiers = (near, wide) if seed_stage else (near, wide, submap)
        for config in tiers:
            best = _lowest(_fan_out(lambda d, c=config: attempt(d, c)))
            if best is not None:
                return best if self._odometry_budget_ok(best, environment) else None

        transforms = self.trajectory.transforms_world_sensor

        def _annealed(delta_try):
            try:
                warm = self.workspace.run_gicp(
                    source_id=source_id, target_id=target_id,
                    delta_transform_local=delta_try, config=warm_config)
            except Exception:
                return None
            seed = (np.linalg.inv(transforms[source_id])
                    @ transforms[target_id]
                    @ warm.transform_target_source_final)
            return attempt(seed)

        best = _lowest(_fan_out(_annealed))
        if best is not None and not self._odometry_budget_ok(best, environment):
            return None
        return best

    def _balm_default_params(self):
        """Return the environment-scaled inspection/refinement parameters."""
        return BalmParams(
            root_voxel_size=float(self.balm_voxel_spin.value()),
            max_iterations=int(self.balm_iter_spin.value()),
            downsample_leaf=float(self._balm_downsample_leaf),
            max_range=float(self._balm_max_range),
            double_sided_enable=bool(self.balm_double_sided_check.isChecked()),
        )

    def run_auto_seed_async(self, silent: bool = False, on_finished=None) -> None:
        """Retrieve, register and audit initial loops outside the GUI thread."""
        if self._optimizer_process is not None or self._background_thread is not None:
            if on_finished is not None:
                self._retry_when_idle(
                    lambda: self.run_auto_seed_async(
                        silent=silent, on_finished=on_finished),
                    on_finished)
                return
            if not silent:
                self._show_error(INITIAL_LOOP_SEARCH_LABEL, "Another operation is running.")
            return
        if self.workspace is None or self.trajectory is None or self.session_paths is None:
            if not silent:
                self._show_error(INITIAL_LOOP_SEARCH_LABEL, "Load a session first.")
            return

        environment_value = self.env_combo.currentData()
        if environment_value not in self.ENV_PRESETS:
            if not silent:
                self._show_error(
                    INITIAL_LOOP_SEARCH_LABEL,
                    "Choose Indoor or Outdoor before Initial Loop Search."
                )
            if on_finished is not None:
                on_finished(0)
            return

        session_root = self.session_paths.session_root
        environment = str(environment_value)
        trajectory = self.trajectory
        transforms = trajectory.transforms_world_sensor.copy()
        near = self._current_registration_config()

        def work(report_progress):
            messages = []
            report_progress("Initial Loop Search · loading descriptor configuration")
            config, weights = descriptor_env_setup(
                load_scan_context_config(session_root), environment)
            if environment == "indoor" and not config.dual_z_layer_enable:
                messages.append(
                    "Indoor session recorded dual_z=false; using protected "
                    "single-layer retrieval (not UpDownSC).")
            gravity = None
            if config.gravity_canonicalization_enable:
                try:
                    gravity = load_scan_context_gravity(session_root, trajectory)
                except Exception as exc:
                    messages.append(f"Gravity sidecar unavailable: {exc}")
            report_progress(
                f"Initial Loop Search · preparing {trajectory.size} keyframes")
            descriptors, masks = build_descriptor_stack(
                self.workspace.load_local_points_uncached,
                trajectory.size,
                config,
                gravity,
                progress_fn=lambda current, total: report_progress(
                    f"Initial Loop Search · preparing keyframes {current}/{total}",
                    current,
                    total,
                ),
            )
            # From the preset, not a second copy of the same pair of numbers;
            # see the note there for why outdoor is 8 rather than the 5 it was.
            _preset = REPAIR_ENV_PRESETS[environment]
            seed_distance = _preset["seed_max_distance"]
            seed_count = _preset["seed_max_candidates"]
            def _seed_deltas(sid: int, tid: int):
                relative = np.linalg.inv(transforms[tid]) @ transforms[sid]
                out = []
                for _, yaw_deg in pair_yaw_peaks(
                        descriptors, masks, sid, tid, weights, config.num_rings,
                        min_joint_rings=config.min_joint_rings,
                        retrieval_height_offset=config.retrieval_height_offset,
                        sector_support_exponent=config.sector_support_exponent):
                    yaw = math.radians(yaw_deg)
                    spin = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                                     [math.sin(yaw), math.cos(yaw), 0.0],
                                     [0.0, 0.0, 1.0]])
                    if gravity is not None:
                        desired = (_gravity_canonical_rotation(gravity[tid]).T
                                   @ spin @ _gravity_canonical_rotation(gravity[sid]))
                    else:
                        desired = spin
                    if not out:
                        odom_seed = np.eye(4)
                        odom_seed[:3, :3] = relative[:3, :3].T @ desired
                        out.append(odom_seed)
                    desired4 = np.eye(4)
                    desired4[:3, :3] = desired
                    out.append(np.linalg.inv(transforms[sid])
                               @ transforms[tid] @ desired4)
                return out

            seeds = find_seed_candidates(
                descriptors, masks,
                max_seeds=seed_count,
                max_distance=seed_distance,
                min_index_gap=min(150, trajectory.size // 4),
                segment_radius=min(100, max(10, trajectory.size // 8)),
                channel_weights=weights, num_rings=config.num_rings,
                min_joint_rings=config.min_joint_rings,
                retrieval_height_offset=config.retrieval_height_offset,
                sector_support_exponent=config.sector_support_exponent)
            verified = []
            total = len(seeds)
            for index, seed in enumerate(seeds, start=1):
                report_progress(
                    f"Initial Loop Search · verifying {index}/{total} · "
                    f"{seed.target_id}->{seed.source_id}", index - 1, total)
                result = self._gated_gicp(
                    seed.source_id, seed.target_id,
                    _seed_deltas(seed.source_id, seed.target_id),
                    near_config=near, seed_stage=True)
                gravity_error = None
                if result is not None and gravity is not None:
                    gravity_error = _gravity_error_deg(
                        result.transform_target_source_final,
                        gravity[seed.source_id], gravity[seed.target_id],
                    )
                verified.append((
                    seed, _compact_registration_result(result), gravity_error
                ))
            return messages, verified

        def apply(payload) -> None:
            messages, verified = payload
            for message in messages:
                self.append_log(f"[InitialLoops] {message}")
            added_uids = []
            skipped = []
            batch_snapshot = self._capture_undo_snapshot()
            variance_t, variance_r = production_loop_variances()
            gravity_limit = float(
                PRODUCTION_PGO_PROFILE["gravity_sanity_limit_deg"]
            )
            for seed, result, gravity_error in verified:
                sid, tid = seed.source_id, seed.target_id
                if self._constraint_by_pair(sid, tid) is not None:
                    continue
                if result is None:
                    skipped.append(f"{tid}->{sid}: quality gate")
                    continue
                if gravity_error is not None and gravity_error > gravity_limit:
                    skipped.append(
                        f"{tid}->{sid}: gravity sanity {gravity_error:.1f}° "
                        f"> {gravity_limit:.1f}°"
                    )
                    continue
                self.source_id, self.target_id = sid, tid
                self.selected_edge_ref = None
                self._candidate_replace_edge_uid = None
                self.current_preview = result.preview
                self.last_result = result
                constraint = self.accept_constraint(
                    variance_t=variance_t,
                    variance_r=variance_r,
                    note="auto-seed",
                    record_undo=False,
                    refresh_ui=False,
                    persist=False,
                )
                if constraint is None:
                    skipped.append(f"{tid}->{sid}: could not persist constraint")
                    continue
                added_uids.append(constraint.manual_uid)
                if gravity_error is not None:
                    self.append_log(
                        f"[InitialLoops] {tid}->{sid} gravity error "
                        f"{gravity_error:.2f}° (sanity limit {gravity_limit:.1f}°)."
                    )
            if added_uids:
                self._append_undo_snapshot(batch_snapshot)
                self._save_project_state()
            self._rebuild_constraint_table()
            self._update_selection_labels()
            self._update_session_status_widgets()
            self._refresh_plot(preserve_view=False)
            count = len(added_uids)
            self.append_log(
                f"[InitialLoops] Done: added={count}, skipped={len(skipped)}. "
                "Map-inconsistency diagnosis was not used for admission.")
            if on_finished is not None:
                on_finished(count)
                return
            summary = f"Added {count} audited initial loop constraint(s)."
            if skipped:
                summary += "\n\nSkipped:\n" + "\n".join(skipped)
            QtWidgets.QMessageBox.information(
                self, INITIAL_LOOP_SEARCH_LABEL, summary
            )

        self._start_background_task(INITIAL_LOOP_SEARCH_LABEL, work, apply)

    def run_auto_seed(self, silent: bool = False) -> int:
        """Retrieve and verify initial loops. Returns how many were added.

        ``silent`` suppresses every dialog, including the follow-up question
        that would otherwise start a SECOND repair loop of its own. A caller
        that is already sequencing the pipeline must not have the steps it
        invokes popping modal boxes at it or starting rival state machines.
        """
        MIN_FITNESS = self.GATE_MIN_FITNESS
        # follow the voxel, exactly as _gated_gicp does -- a fixed 0.15 rejects
        # every genuine outdoor loop on quantisation alone
        MAX_INLIER_RMSE = max(self.GATE_MAX_INLIER_RMSE,
                              self.GATE_RMSE_VOXEL_RATIO * self.voxel_spin.value())
        if self._optimizer_process is not None:
            if not silent:
                self._show_error(
                    INITIAL_LOOP_SEARCH_LABEL, "An optimization process is running."
                )
            return 0
        if self.workspace is None or self.trajectory is None or self.session_paths is None:
            if not silent:
                self._show_error(INITIAL_LOOP_SEARCH_LABEL, "Load a session first.")
            return 0
        if self.env_combo.currentData() not in self.ENV_PRESETS:
            if not silent:
                self._show_error(
                    INITIAL_LOOP_SEARCH_LABEL,
                    "Choose Indoor or Outdoor before Initial Loop Search."
                )
            return 0
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            session_root = self.session_paths.session_root
            environment = str(self.env_combo.currentData())
            config, channel_weights = descriptor_env_setup(
                load_scan_context_config(session_root), environment
            )
            self.append_log(
                f"[InitialLoops] Environment preset: {environment} "
                f"(dual_z={config.dual_z_layer_enable}, weights={channel_weights}, "
                f"radius={config.max_radius:.0f} m, "
                f"min_joint_rings={config.min_joint_rings}, "
                f"sector_support_exp={config.sector_support_exponent:.2f})"
            )
            if environment == "indoor" and not config.dual_z_layer_enable:
                self.append_log(
                    "[InitialLoops] Indoor session recorded dual_z=false; using "
                    "protected single-layer retrieval (not UpDownSC)."
                )
            gravity = None
            if config.gravity_canonicalization_enable:
                try:
                    gravity = load_scan_context_gravity(session_root, self.trajectory)
                except Exception as exc:
                    self.append_log(
                        f"[InitialLoops] Gravity sidecar unavailable ({exc}); matching "
                        "without gravity canonicalization."
                    )
            self.append_log(
                f"[InitialLoops] Building {self.trajectory.size} Scan Context descriptors ..."
            )
            descriptors, masks = build_descriptor_stack(
                self.workspace.load_local_points,
                self.trajectory.size,
                config,
                gravity,
                log_fn=self.append_log,
            )
            seeds = find_seed_candidates(
                descriptors,
                masks,
                channel_weights=channel_weights,
                num_rings=config.num_rings,
                min_joint_rings=config.min_joint_rings,
                retrieval_height_offset=config.retrieval_height_offset,
                sector_support_exponent=config.sector_support_exponent,
                log_fn=self.append_log,
            )
            if not seeds:
                QtWidgets.QApplication.restoreOverrideCursor()
                self.append_log("[InitialLoops] No confident revisit candidates found.")
                if not silent:
                    QtWidgets.QMessageBox.information(
                        self,
                        INITIAL_LOOP_SEARCH_LABEL,
                        "No confident revisit candidates found. Add an initial loop manually.",
                    )
                return 0
            self.append_log(
                "[InitialLoops] Candidates: "
                + ", ".join(
                    f"{s.target_id}->{s.source_id} (d={s.distance:.2f}, yaw={s.yaw_deg:.0f}°)"
                    for s in seeds
                )
            )

            added_uids: list[int] = []
            skipped: list[str] = []
            batch_snapshot = self._capture_undo_snapshot()
            for seed in seeds:
                if self._constraint_by_pair(seed.source_id, seed.target_id) is not None:
                    continue
                transforms = self.trajectory.transforms_world_sensor
                relative = np.linalg.inv(transforms[seed.target_id]) @ transforms[seed.source_id]
                # Multi-hypothesis initialization. The descriptor correlation's
                # top peaks are tried rather than only its argmax (symmetric
                # places put the true yaw in a secondary peak), and each peak
                # also seeds a COLLOCATION guess that places the source at the
                # target: a place-recognition hit means "same place", so that
                # guess is independent of how far the odometry has drifted.
                # Attitude comes from the per-frame gravity estimate, since a
                # bare yaw would inherit the source's drifted roll and pitch.
                peaks = pair_yaw_peaks(
                    descriptors, masks, seed.source_id, seed.target_id,
                    channel_weights, config.num_rings,
                    min_joint_rings=config.min_joint_rings,
                    retrieval_height_offset=config.retrieval_height_offset,
                    sector_support_exponent=config.sector_support_exponent)
                delta = []
                for _, yaw_deg in peaks:
                    yaw_rad = math.radians(yaw_deg)
                    spin = np.array(
                        [
                            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
                            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    )
                    if gravity is not None:
                        rot_s = _gravity_canonical_rotation(gravity[seed.source_id])
                        rot_t = _gravity_canonical_rotation(gravity[seed.target_id])
                        desired_rotation = rot_t.T @ spin @ rot_s
                    else:
                        desired_rotation = spin
                    if not delta:
                        odom_seed = np.eye(4)
                        odom_seed[:3, :3] = relative[:3, :3].T @ desired_rotation
                        delta.append(odom_seed)
                    desired4 = np.eye(4)
                    desired4[:3, :3] = desired_rotation
                    delta.append(
                        np.linalg.inv(transforms[seed.source_id])
                        @ transforms[seed.target_id]
                        @ desired4
                    )
                self.source_id = seed.source_id
                self.target_id = seed.target_id
                self.selected_edge_ref = None
                self._candidate_replace_edge_uid = None
                self.current_preview = None
                self.last_result = None
                self._clear_delta_silent()
                result = self._gated_gicp(seed.source_id, seed.target_id, delta,
                                          seed_stage=True)
                if result is None:
                    skipped.append(
                        f"{seed.target_id}->{seed.source_id}: rejected by quality gate"
                    )
                    continue
                gravity_error = None
                if gravity is not None:
                    gravity_error = _gravity_error_deg(
                        result.transform_target_source_final,
                        gravity[seed.source_id], gravity[seed.target_id],
                    )
                gravity_limit = float(
                    PRODUCTION_PGO_PROFILE["gravity_sanity_limit_deg"]
                )
                if gravity_error is not None and gravity_error > gravity_limit:
                    skipped.append(
                        f"{seed.target_id}->{seed.source_id}: gravity sanity "
                        f"{gravity_error:.1f}° > {gravity_limit:.1f}°"
                    )
                    continue
                self.current_preview = result.preview
                self.last_result = result
                variance_t, variance_r = production_loop_variances()
                constraint = self.accept_constraint(
                    variance_t=variance_t,
                    variance_r=variance_r,
                    note="auto-seed",
                    record_undo=False,
                    refresh_ui=False,
                    persist=False,
                )
                if constraint is None:
                    skipped.append(
                        f"{seed.target_id}->{seed.source_id}: could not persist constraint"
                    )
                    continue
                added_uids.append(constraint.manual_uid)
                self.append_log(
                    f"[InitialLoops] Initial loop {seed.target_id}->{seed.source_id} accepted "
                    f"(fit {result.fitness:.2f}, rmse {result.inlier_rmse:.3f})."
                )
            if added_uids:
                self._append_undo_snapshot(batch_snapshot)
                self._save_project_state()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._update_session_status_widgets()
        self._rebuild_constraint_table()
        self._update_selection_labels()
        self._refresh_plot(preserve_view=False)
        summary = (
            f"Added {len(added_uids)} audited initial loop constraint(s)."
            + ("" if not skipped else "\n\nSkipped:\n" + "\n".join(skipped))
        )
        self.append_log(f"[InitialLoops] Done: added={len(added_uids)}, skipped={len(skipped)}.")
        if silent:
            return len(added_uids)
        if not added_uids:
            QtWidgets.QMessageBox.information(
                self, INITIAL_LOOP_SEARCH_LABEL, summary
            )
            return 0
        answer = QtWidgets.QMessageBox.question(
            self,
            INITIAL_LOOP_SEARCH_LABEL,
            summary + "\n\nRun PGO followed by the configured final BALM pass now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.append_log(
                "[InitialLoops] Starting audited PGO; Final Map Refinement "
                "will run once if enabled."
            )
            self.run_optimization()

    # The repair-effectiveness check -- "did constraint N close the ghost it was
    # added to close?" -- was removed on 2026-08-05. Six retractions across the
    # measured runs, six of them wrong, none right.
    #
    # Its criterion used map consistency as a proxy for trajectory correctness.
    # The same evening's data refutes that proxy: on MCD ntu, round 4 sat at
    # 232.8 cm ATE with ghost badness 6378, round 6 at 272.4 cm with badness
    # 483. The map grew steadily more consistent while the trajectory grew
    # steadily worse.
    #
    # Every constraint it retracted, labelled against ground truth (true
    # separation = the two keyframes' actual distance in GT):
    #     475->685  4.50 m    788->831  1.83 m
    #     795->1751 2.61 m    392->769  4.46 m
    # All four are the same place -- real loops. Two more on the in-house
    # session (no ground truth there) were inspected in the detector's own 1 m
    # voxel: the constraint had pulled that voxel's two layers from 30.0 cm to
    # 19.7 cm, minority layer 105 -> 258 points, 51 -> 86 supporting keyframes.
    # It was working, and was judged ineffective.
    #
    # The premise does not hold either. A correct loop routinely fails to close
    # a particular ghost: the ghost may come from drift spread over the whole
    # circuit, which one constraint only partly repairs, and several ghosts may
    # share one cause. "Did not close it" is not "was not useful".
    #
    # The cost was real -- one extra PGO+BALM round per firing -- and the best
    # run in the paper (2026-07-29, 16 constraints) never had this check, while
    # tonight's runs carrying it reached only 10-11 and produced a visibly worse
    # map at the wall the teaser figure is about.
    #
    # The tools remain: scratchpad/label_constraints.py labels each constraint
    # true or false against ground truth, scratchpad/inspect_voxel.py shows one
    # ghost in the detector's own voxel. Anything reintroduced here should first
    # be shown, with those, to actually catch a false loop.

    def _handle_seed_probation(self) -> bool:
        """Disable seeds that a NEW ghost region implicates by segment.

        The obvious test -- did total map inconsistency go up? -- does not
        work, and measuring it is how that was learned. Global badness is not
        monotone in true error: a repair that pulls a badly drifted arm back
        into place moves its error from "too far apart to diagnose" into the
        0.08-0.35 m band the detector reads, so the score rises precisely
        because the map got better. Judged globally, the best seeds of a run
        are the ones that look worst.

        So each seed is judged only by ghosts at its own two segments, and only
        by ones that were not already there before seeding. Two of them, jointly
        heavy, is the bar: one region is within the noise of where a repair
        leaves a residual seam.

        Returns True when seeds were rolled back and a re-optimization was
        scheduled (callers should then skip their normal continuation).
        """
        if self._seed_probation is None:
            return False
        probation = self._seed_probation
        self._seed_probation = None
        uids = set(probation["uids"])
        baseline_pairs = probation.get("baseline_pairs") or []

        def implicated(pair, sid: int, tid: int,
                       radius: int = REPAIR_POLICY["probation_radius_kf"]) -> bool:
            pa, pb = pair
            return (min(abs(pa - sid), abs(pb - sid)) <= radius
                    and min(abs(pa - tid), abs(pb - tid)) <= radius)

        current_pairs = [
            (int(item["source_id"]), int(item["target_id"]),
             float(item.get("point_count") or 0) * float(item.get("separation_m") or 0.0))
            for item in self._balm_ghost_suggestions
        ]
        suspects = []
        for constraint in self.constraints:
            if constraint.manual_uid not in uids or not constraint.enabled:
                continue
            sid, tid = int(constraint.source_id), int(constraint.target_id)
            fresh = [w for pa, pb, w in current_pairs
                     if implicated((pa, pb), sid, tid)
                     and not any(implicated(bp, sid, tid) for bp in baseline_pairs)]
            if (len(fresh) >= REPAIR_POLICY["probation_min_regions"]
                    and sum(fresh) > REPAIR_POLICY["probation_min_badness"]):
                suspects.append((constraint, len(fresh), sum(fresh)))
        if not suspects:
            self.append_log("[InitialLoops] Probation passed (segment-attributed check).")
            return False
        for constraint, count, badness in suspects:
            constraint.enabled = False
            self.append_log(
                f"[InitialLoops] Probation: initial loop {constraint.target_id}->"
                f"{constraint.source_id} SUSPECT: {count} new ghost regions at "
                f"its segments (badness {badness:.0f}); disabled")
        self._rebuild_constraint_table()
        self._update_session_status_widgets()
        if self._reoptimize_after_retraction(
                f"probation disabled {len(suspects)} seed constraint(s) "
                "implicated by new ghosts at their own segments"):
            return True
        QtWidgets.QMessageBox.warning(
            self,
            INITIAL_LOOP_SEARCH_LABEL,
            "Initial loop constraints put new inconsistency regions at the segments they "
            "connect (likely perceptual aliasing). They were disabled; press "
            "Optimize to restore the graph.",
        )
        return False
    # ===== END CHANGE: automatic seed loops =====

    # ===== BEGIN CHANGE: one-click Repair Map =====
    def run_repair_map(self) -> None:
        """Run the canonical headless production engine and adopt its result.

        Automatic repair intentionally has one implementation.  The GUI owns
        interaction, progress, cancellation and result loading; candidate
        retrieval, per-candidate trial PGO, batch PGO, NIS/leave-one-out and
        the terminal BALM pass all run in ``auto_repair_headless.py``.
        """
        if self._repair_map_stage == "canonical":
            self._stop_canonical_repair()
            return
        if self._repair_map_stage is not None:
            self._repair_map_stop_requested = True
            self.repair_map_button.setText("Stopping Repair…")
            self.repair_map_button.setEnabled(False)
            if (
                self._repair_map_stage == "balm"
                and self._optimizer_process is not None
            ):
                process = self._optimizer_process
                self._balm_cancel_requested = True
                self._set_repair_progress(
                    "Stopping Final Map Refinement · retaining PGO", busy=True
                )
                self.append_log(
                    "[RepairMap] Cancelling the active Final Map Refinement; "
                    "its isolated partial output will not be adopted."
                )
                process.terminate()
                QtCore.QTimer.singleShot(
                    5000,
                    lambda active=process: self._force_kill_balm_process(active),
                )
                self._update_session_status_widgets()
                return
            self._set_repair_progress(
                "Stop requested · finishing current stage", busy=True)
            self.append_log(
                "[RepairMap] Stop requested; preserving the current stage output."
            )
            if (self._optimizer_process is None
                    and self._background_thread is None):
                self._repair_map_finish("stopped by user between stages")
                return
            self._update_session_status_widgets()
            return
        if self._optimizer_process is not None or self._background_thread is not None:
            self._show_error("Repair Map", "Another operation is already running.")
            return

        if self.workspace is None or self.trajectory is None or self.session_paths is None:
            self._show_error("Repair Map", "Load a session first.")
            return
        if self._session_dirty:
            self._show_error(
                "Repair Map",
                "The canonical automatic repair always starts from the immutable "
                "session TUM and pose graph. Optimize, undo, or export the pending "
                "manual edits before running it.",
            )
            return
        if self.env_combo.currentData() not in self.ENV_PRESETS:
            self._show_error(
                "Repair Map",
                "Choose Indoor or Outdoor first. The choice controls both "
                "registration and BALM scales and cannot be guessed safely.",
            )
            return
        env = str(self.env_combo.currentData())
        # The environment dropdown is this flow's declared intent, so re-apply
        # its preset before starting. The spin boxes are editable and a mouse
        # wheel passing over one is enough to move it: on 2026-08-05 an indoor
        # session ran at registration voxel 0.000 instead of 0.200, which
        # collapsed the residual bound from max(0.15, 1.1 x voxel) = 0.220 m to
        # the 0.150 m floor and silently cost two seed constraints (11 instead
        # of 13) over a 27-minute run. Nothing else in the log looked wrong.
        #
        # This also removes a way for the two implementations to disagree: the
        # headless evaluator always reads repair_presets, with no widget in
        # between, so a GUI repair that used drifted values was not comparable
        # with it. Hand-tuning still works for per-constraint Run GICP, which
        # reads the widgets directly; it is the one-click automatic flow that
        # follows the dropdown.
        _before = (self.voxel_spin.value(), self.max_corr_spin.value(),
                   self.target_map_voxel_spin.value(), self.balm_voxel_spin.value(),
                   self.target_neighbors_spin.value(), self.balm_iter_spin.value(),
                   self.balm_double_sided_check.isChecked())
        self._apply_environment_preset()
        _after = (self.voxel_spin.value(), self.max_corr_spin.value(),
                  self.target_map_voxel_spin.value(), self.balm_voxel_spin.value(),
                  self.target_neighbors_spin.value(), self.balm_iter_spin.value(),
                  self.balm_double_sided_check.isChecked())
        if _before != _after:
            self.append_log(
                f"[RepairMap] settings had drifted from the '{env}' preset and "
                f"were restored: voxel {_before[0]:.3f}->{_after[0]:.3f}, "
                f"max_corr {_before[1]:.2f}->{_after[1]:.2f}, "
                f"target_map_voxel {_before[2]:.3f}->{_after[2]:.3f}, "
                f"balm_voxel {_before[3]:.2f}->{_after[3]:.2f}, "
                f"target_neighbors {_before[4]}->{_after[4]}, "
                f"balm_iters {_before[5]}->{_after[5]}, "
                f"double_sided {_before[6]}->{_after[6]}")
        # Record what this repair actually runs with. Three sessions repaired
        # from the same odometry on 2026-08-03 gave 13, 31 and 12 constraints,
        # and nothing on disk said which parameters each had used -- the
        # operations log captured the clicks and the outcomes but not the
        # settings, so every diagnosis of the difference had to be guessed.
        rmse_bound = max(self.GATE_MAX_INLIER_RMSE,
                         self.GATE_RMSE_VOXEL_RATIO * self.voxel_spin.value())
        self.append_log(
            f"[RepairMap] effective settings: environment={env}, "
            f"registration voxel={self.voxel_spin.value():.3f} m, "
            f"max_corr={self.max_corr_spin.value():.2f} m, "
            f"gate overlap>={self.GATE_MIN_FITNESS:.2f}, residual<={rmse_bound:.3f} m, "
            f"target_neighbors={self.target_neighbors_spin.value()}, "
            f"target_map_voxel={self.target_map_voxel_spin.value():.3f} m, "
            f"BALM root={self.balm_voxel_spin.value():.2f} m/"
            f"leaf={float(self._balm_downsample_leaf):.3f} m/"
            f"iters={self.balm_iter_spin.value()}, "
            f"double_sided={self.balm_double_sided_check.isChecked()}, "
            f"map_diagnostics={self.map_diagnostics_check.isChecked()}, "
            f"PGO={PRODUCTION_PGO_PROFILE['optimize_mode']}/"
            f"scale={PRODUCTION_PGO_PROFILE['manual_information_scale']}/"
            f"sigma_t={math.sqrt(float(PRODUCTION_PGO_PROFILE['translation_variance_m2'])):.3f}m/"
            f"sigma_r={PRODUCTION_PGO_PROFILE['rotation_sigma_deg']:.1f}deg, "
            f"near_duplicate={NEAR_DUPLICATE_KEYFRAMES} kf, "
            f"nn_workers={_registration._NN_WORKERS}, "
            f"hypothesis_workers={self._HYPOTHESIS_WORKERS}"
        )
        reply = QtWidgets.QMessageBox.question(
            self, "Repair Map",
            f"Run the full automatic repair on this session?\n\n"
            f"  environment : {env}  (voxel {self.voxel_spin.value():.2f} m)\n"
            f"  steps       : Initial Loop Search -> Audited PGO -> Final Map Refinement\n\n"
            f"This takes minutes on a large session. Every proposed loop is "
            f"trial-optimized before admission, then audited again after the "
            f"batch solve. Stop Repair cancels the isolated run and leaves the "
            f"currently loaded result unchanged.\n"
            + (
                "Map-inconsistency regions will be reported for inspection only."
                if self.map_diagnostics_check.isChecked()
                else "Map-inconsistency diagnostics are disabled."
            ),
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        if reply != QtWidgets.QMessageBox.Ok:
            return
        self._start_canonical_repair(env)

    @staticmethod
    def _sha256_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _start_canonical_repair(self, env: str) -> None:
        """Launch the one production implementation in an isolated process group."""
        assert self.session_paths is not None
        headless_script = (
            PROJECT_ROOT / "scripts" / "ghostloop_eval"
            / "auto_repair_headless.py"
        )
        if not headless_script.is_file():
            self._show_error(
                "Repair Map",
                f"Canonical repair engine is missing: {headless_script}",
            )
            return

        python_candidates = [Path(sys.executable).absolute()]
        optimizer_backend = self._resolve_optimizer_backend()
        if optimizer_backend is not None:
            candidate = Path(optimizer_backend.program).absolute()
            if candidate not in python_candidates:
                python_candidates.append(candidate)
        canonical_python = next(
            (candidate for candidate in python_candidates
             if _python_supports_canonical_repair(candidate)),
            None,
        )
        if canonical_python is None:
            self._show_error(
                "Repair Map",
                "No Python environment provides the complete canonical engine "
                "dependencies (Open3D, NumPy, SciPy, and GTSAM).",
            )
            return

        runs_root = self.session_paths.session_root / "manual_loop_runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        contract = runs_root / f".{token}_gui_headless_result.json"
        args = [
            "-u", str(headless_script),
            str(self.session_paths.session_root),
            "production", env,
            "--result-json", str(contract),
            "--balm-root-voxel-size", f"{self.balm_voxel_spin.value():.9g}",
            "--pgo-profile", "calibrated",
        ]
        if self.map_diagnostics_check.isChecked():
            args.append("--diagnose-map-inconsistency")

        self._canonical_repair_snapshot = self._capture_undo_snapshot()
        self._canonical_repair_contract = contract
        self._canonical_repair_environment = env
        self._canonical_repair_cancelled = False
        self._repair_map_stage = "canonical"
        self._repair_map_stop_requested = False
        self._repair_map_seeds = 0
        self._repair_map_balm_passes = 0
        self._repair_map_balm_succeeded = False
        self._repair_map_balm_adopted = None
        self._repair_map_started_at = time.perf_counter()
        self._balm_progress_seen = 0
        self._balm_progress_total = int(self.balm_iter_spin.value())

        process = QtCore.QProcess(self)
        process_env = QtCore.QProcessEnvironment.systemEnvironment()
        process_env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(process_env)
        setsid = shutil.which("setsid") if os.name == "posix" else None
        if setsid:
            process.setProgram(setsid)
            process.setArguments([str(canonical_python), *args])
            self._canonical_repair_process_grouped = True
        else:
            process.setProgram(str(canonical_python))
            process.setArguments(args)
            self._canonical_repair_process_grouped = False
        process.readyReadStandardOutput.connect(self._read_canonical_repair_stdout)
        process.readyReadStandardError.connect(self._read_canonical_repair_stderr)
        process.finished.connect(self._canonical_repair_finished)
        process.errorOccurred.connect(self._canonical_repair_error)
        self._optimizer_process = process
        self._active_optimizer_backend_name = "canonical-headless"
        self._optimizer_started_at = time.perf_counter()
        self._optimizer_heartbeat_timer.start()

        self.optimize_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.balm_button.setEnabled(False)
        self.repair_map_button.setText("Stop Repair")
        self.repair_map_button.setEnabled(True)
        self._set_repair_progress(
            "Canonical repair · Initial Loop Search", busy=True)
        self.append_log(
            f"[RepairMap] Started canonical headless production engine ({env}); "
            "each loop must pass trial-PGO before batch PGO."
        )
        process.start()
        self._update_session_status_widgets()

    def _read_canonical_repair_stdout(self) -> None:
        process = self._optimizer_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode(
            "utf-8", errors="ignore")
        for line in data.splitlines():
            self.append_log(line)
            self._consume_balm_progress_line(line)
            match = re.search(
                r"Initial Loop Search done: (\d+) accepted", line)
            if match is not None:
                self._repair_map_seeds = int(match.group(1))
                self._set_repair_progress(
                    f"Canonical repair · trial-PGO accepted "
                    f"{self._repair_map_seeds} loops",
                    value=45,
                )
            elif "Round 1: optimizing" in line:
                self._set_repair_progress(
                    "Canonical repair · batch PGO and post-solve audit",
                    busy=True,
                )
            elif "production complete" in line:
                self._set_repair_progress(
                    "Canonical repair · validating terminal artifacts",
                    value=95,
                )

    def _read_canonical_repair_stderr(self) -> None:
        process = self._optimizer_process
        if process is None:
            return
        data = bytes(process.readAllStandardError()).decode(
            "utf-8", errors="ignore")
        for line in data.splitlines():
            self.append_log(f"[headless stderr] {line}")

    def _stop_canonical_repair(self) -> None:
        process = self._optimizer_process
        if process is None:
            self._repair_map_finish("stopped before the engine started")
            return
        self._canonical_repair_cancelled = True
        self._repair_map_stop_requested = True
        self.repair_map_button.setText("Stopping Repair…")
        self.repair_map_button.setEnabled(False)
        self._set_repair_progress(
            "Stopping canonical repair · loaded result unchanged", busy=True)
        self.append_log(
            "[RepairMap] Cancelling the isolated canonical engine and all of "
            "its optimizer/refinement children; partial output will not be adopted."
        )
        self._signal_canonical_repair(process, signal.SIGTERM)
        QtCore.QTimer.singleShot(
            5000,
            lambda active=process: self._force_kill_canonical_repair(active),
        )

    def _signal_canonical_repair(self, process, sig: signal.Signals) -> None:
        if process.state() == QtCore.QProcess.NotRunning:
            return
        pid = int(process.processId())
        if self._canonical_repair_process_grouped and pid > 0:
            try:
                os.killpg(pid, sig)
                return
            except (OSError, ProcessLookupError):
                pass
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()

    def _force_kill_canonical_repair(self, process) -> None:
        if (
            self._canonical_repair_cancelled
            and self._optimizer_process is process
            and process.state() != QtCore.QProcess.NotRunning
        ):
            self.append_log(
                "[RepairMap] Canonical engine did not stop within 5 s; "
                "killing its isolated process group."
            )
            self._signal_canonical_repair(process, signal.SIGKILL)

    def _canonical_repair_error(self, error) -> None:
        if error != QtCore.QProcess.FailedToStart:
            return
        self.append_log("[RepairMap] Canonical headless engine failed to start.")
        self._optimizer_process = None
        self._optimizer_heartbeat_timer.stop()
        self._canonical_repair_snapshot = None
        self._canonical_repair_environment = None
        self._canonical_repair_process_grouped = False
        self._repair_map_finish("canonical engine failed to start")
        self._update_session_status_widgets()

    def _validated_canonical_contract(self) -> dict:
        if self._canonical_repair_contract is None or self.session_paths is None:
            raise ValueError("canonical result contract path is unavailable")
        payload = json.loads(
            self._canonical_repair_contract.read_text(encoding="utf-8"))
        if payload.get("schema") != "lidar-map-refiner/headless-result":
            raise ValueError("unexpected canonical result schema")
        if payload.get("schema_version") != 1 or payload.get("status") != "complete":
            raise ValueError("canonical result is not a complete v1 contract")
        if payload.get("mode") != "production":
            raise ValueError("canonical result was not produced in production mode")
        if payload.get("environment") != self._canonical_repair_environment:
            raise ValueError("canonical result environment does not match the GUI run")
        expected_session = self.session_paths.session_root.resolve()
        if Path(payload.get("session", "")).resolve() != expected_session:
            raise ValueError("canonical result belongs to a different session")

        terminal = Path(payload["terminal_output_dir"]).resolve()
        if terminal.parent != (expected_session / "manual_loop_runs").resolve():
            raise ValueError("canonical terminal output is outside this session")
        artifact_keys = (
            "optimized_tum", "pose_graph_g2o", "constraints_csv")
        for key in artifact_keys:
            path = Path(payload[key]).resolve()
            if path.parent != terminal or not path.is_file():
                raise ValueError(f"invalid canonical artifact: {key}")
            expected_hash = payload.get("output_sha256", {}).get(key)
            if expected_hash != self._sha256_path(path):
                raise ValueError(f"canonical artifact hash mismatch: {key}")
        input_paths = {
            "optimized_tum": expected_session / "optimized_poses_tum.txt",
            "pose_graph_g2o": expected_session / "pose_graph.g2o",
        }
        for key, path in input_paths.items():
            expected_hash = payload.get("input_sha256", {}).get(key)
            if expected_hash != self._sha256_path(path):
                raise ValueError(
                    f"session input changed while canonical repair ran: {key}")
        ledger = Path(payload.get("proposal_ledger_jsonl", "")).resolve()
        ledgers_root = (expected_session / "proposal_ledgers").resolve()
        if ledger.parent != ledgers_root or not ledger.is_file():
            raise ValueError("canonical proposal ledger is missing or misplaced")
        if payload.get("proposal_ledger_sha256") != self._sha256_path(ledger):
            raise ValueError("canonical proposal ledger hash mismatch")
        return payload

    def _canonical_constraints_from_csv(
        self, csv_path: Path, terminal_trajectory: TrajectoryData,
        ledger_path: Path,
    ) -> list[ManualConstraint]:
        metrics_by_pair: dict[tuple[int, int], dict] = {}
        try:
            for raw_line in ledger_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(raw_line)
                gicp = record.get("gicp")
                if isinstance(gicp, dict):
                    metrics_by_pair[(
                        int(record["source_id"]), int(record["target_id"])
                    )] = gicp
        except (OSError, ValueError, KeyError):
            metrics_by_pair = {}

        constraints: list[ManualConstraint] = []
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row_number, row in enumerate(csv.DictReader(stream), start=1):
                source_id = int(row["source_id"])
                target_id = int(row["target_id"])
                if not (
                    0 <= source_id < terminal_trajectory.size
                    and 0 <= target_id < terminal_trajectory.size
                ):
                    raise ValueError(
                        f"constraint pose id outside terminal trajectory at row {row_number}")
                transform = np.eye(4, dtype=np.float64)
                transform[:3, 3] = [
                    float(row["tx"]), float(row["ty"]), float(row["tz"])
                ]
                transform[:3, :3] = Rotation.from_quat([
                    float(row["qx"]), float(row["qy"]),
                    float(row["qz"]), float(row["qw"]),
                ]).as_matrix()
                sigma_t = tuple(float(row[name]) for name in (
                    "sigma_tx", "sigma_ty", "sigma_tz"))
                sigma_r_deg = tuple(float(row[name]) for name in (
                    "sigma_roll_deg", "sigma_pitch_deg", "sigma_yaw_deg"))
                metrics = metrics_by_pair.get((source_id, target_id), {})
                constraints.append(ManualConstraint(
                    manual_uid=row_number,
                    enabled=str(row["enabled"]).strip() == "1",
                    source_id=source_id,
                    target_id=target_id,
                    target_cloud_mode="canonical_headless",
                    target_neighbors=int(self.target_neighbors_spin.value()),
                    min_time_gap_sec=OFFICE_DEFAULT_TARGET_MIN_TIME_GAP_SEC,
                    target_map_voxel_size=float(
                        self.target_map_voxel_spin.value()),
                    transform_world_source_final=(
                        terminal_trajectory.transforms_world_sensor[source_id].copy()),
                    transform_target_source_final=transform,
                    source_points_world_final=np.empty((0, 3), dtype=np.float64),
                    fitness=float(metrics.get("fitness", float("nan"))),
                    inlier_rmse=float(
                        metrics.get("inlier_rmse_m", float("nan"))),
                    variance_t_m2=tuple(value * value for value in sigma_t),
                    variance_r_rad2=tuple(
                        math.radians(value) ** 2 for value in sigma_r_deg),
                    accepted_rev=self._working_revision,
                    applied_rev=None,
                    note="canonical headless production audit",
                ))
        return constraints

    def _adopt_canonical_repair_result(self, payload: dict) -> None:
        terminal = Path(payload["terminal_output_dir"])
        terminal_trajectory = load_tum_trajectory(
            Path(payload["optimized_tum"]))
        reference = self.original_trajectory
        if reference is None or terminal_trajectory.size != reference.size:
            raise TrajectoryValidationError(
                "canonical terminal trajectory pose count does not match the session")
        if not np.allclose(
            terminal_trajectory.timestamps, reference.timestamps,
            rtol=0.0, atol=1e-6,
        ):
            raise TrajectoryValidationError(
                "canonical terminal trajectory timestamps do not match the session")
        ledger_path = Path(payload["proposal_ledger_jsonl"])
        self.constraints = self._canonical_constraints_from_csv(
            Path(payload["constraints_csv"]), terminal_trajectory, ledger_path)
        self.disabled_loop_changes.clear()
        self.pose_graph = copy.deepcopy(self.original_pose_graph)
        self._last_output_dir = terminal
        if not self._apply_working_optimization_result():
            raise TrajectoryValidationError(
                "canonical terminal trajectory failed GUI adoption validation")
        self._next_manual_uid = len(self.constraints) + 1
        self._balm_ghost_suggestions = self._load_balm_ghost_suggestions()
        self._repair_map_seeds = int(payload["active_constraint_count"])
        self._repair_map_balm_succeeded = bool(payload["balm_completed"])
        self._repair_map_balm_passes = int(bool(payload["balm_completed"]))
        self._repair_map_balm_adopted = payload.get("balm_adopted")
        self._last_balm_adopted = payload.get("balm_adopted")
        if self._canonical_repair_snapshot is not None:
            self._append_undo_snapshot(self._canonical_repair_snapshot)
        self._save_project_state()

    def _canonical_repair_finished(
        self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus,
    ) -> None:
        process = self._optimizer_process
        if process is not None:
            self._read_canonical_repair_stdout()
            self._read_canonical_repair_stderr()
        self._optimizer_process = None
        self._optimizer_heartbeat_timer.stop()
        elapsed = time.perf_counter() - getattr(
            self, "_repair_map_started_at", time.perf_counter())
        cancelled = self._canonical_repair_cancelled
        succeeded = (
            not cancelled
            and exit_status == QtCore.QProcess.NormalExit
            and exit_code == 0
        )
        reason = ""
        if succeeded:
            try:
                payload = self._validated_canonical_contract()
                self._adopt_canonical_repair_result(payload)
                reason = "canonical headless result adopted"
                self.append_log(
                    f"[RepairMap] Canonical result verified and adopted in "
                    f"{elapsed:.1f} s: {payload['terminal_output_dir']}"
                )
            except Exception as exc:
                succeeded = False
                reason = f"terminal result validation failed: {exc}"
                if self._canonical_repair_snapshot is not None:
                    self._restore_snapshot(self._canonical_repair_snapshot)
        elif cancelled:
            reason = "stopped by user; loaded result unchanged"
        else:
            reason = (
                f"canonical engine failed (exit_code={exit_code}, "
                f"exit_status={int(exit_status)}); loaded result unchanged"
            )
        self._canonical_repair_snapshot = None
        self._canonical_repair_environment = None
        self._canonical_repair_process_grouped = False
        self._canonical_repair_cancelled = False
        self._optimizer_started_at = None
        self._repair_map_finish(reason)
        self._update_session_status_widgets()

    def _force_kill_balm_process(self, process) -> None:
        """Finish a requested BALM cancellation without touching accepted data."""
        if (
            self._balm_cancel_requested
            and self._optimizer_process is process
            and process.state() != QtCore.QProcess.NotRunning
        ):
            self.append_log(
                "[RepairMap] Final Map Refinement did not terminate within 5 s; "
                "forcing the isolated worker to stop."
            )
            process.kill()

    def _repair_map_after_seed(self, count: int) -> None:
        if self._repair_map_stage != "seed":
            return
        self._repair_map_seeds = count
        self._set_repair_progress(
            f"Initial Loop Search finished · {count} accepted", value=12)
        self.append_log(
            f"[RepairMap] Initial loop search done: "
            f"{self._repair_map_seeds} accepted."
        )
        if self._repair_map_stop_requested:
            self._repair_map_finish("stopped by user after Initial Loop Search")
            return
        # Initial loop registration is complete. BALM streams the raw frames again
        # in its worker process, so keeping GICP target maps and shared frame
        # arrays here only overlaps their memory with PGO/BALM.  Manual preview
        # remains correct because RegistrationWorkspace reloads on demand.
        workspace = self.__dict__.get("workspace")
        if workspace is not None:
            workspace.clear_caches(include_shared_frames=True)
        self.current_preview = None
        self.last_result = None
        cloud_view = self.__dict__.get("cloud_view")
        if cloud_view is not None:
            cloud_view.clear_scene()
        gc.collect()
        self.append_log(
            "[RepairMap] released Initial Loop Search descriptor/GICP caches before "
            "PGO and BALM."
        )
        # BALM consumes an optimizer run directory.  Even with zero new seeds,
        # create that clean baseline first; a freshly loaded session has no
        # ``_last_output_dir`` and cannot legitimately jump straight to BALM.
        if self._repair_map_seeds <= 0 and not self._balm_ghost_suggestions:
            self.append_log(
                "[RepairMap] No initial loops accepted; creating the audited "
                "PGO baseline before Final Map Refinement."
            )
        self._repair_map_stage = "optimize"
        self.append_log("[RepairMap] Running audited PGO.")
        QtCore.QTimer.singleShot(0, self._repair_map_start_optimization)

    def _repair_map_start_optimization(self) -> None:
        if self._repair_map_stage != "optimize":
            return
        if self._repair_map_stop_requested:
            self._repair_map_finish("stopped by user after Initial Loop Search")
            return
        self.run_optimization()

    def _repair_map_start_balm(self) -> None:
        if self._repair_map_stage != "balm":
            return
        if self._repair_map_stop_requested:
            self._repair_map_finish("stopped by user after audited PGO")
            return
        self.run_balm_refinement()

    def _repair_map_advance(self, after: str, ok: bool) -> bool:
        """Move the one-click flow on. Returns True if it handled the event."""
        if self._repair_map_stage is None:
            return False
        phase_label = {
            "seed": INITIAL_LOOP_SEARCH_LABEL,
            "optimize": "audited PGO",
            "balm": "Final Map Refinement",
        }.get(after, after)
        if not ok:
            self.append_log(f"[RepairMap] Aborted: {phase_label} failed.")
            if after == "balm":
                self._repair_map_balm_succeeded = False
                self._repair_map_finish(
                    "Final Map Refinement failed; retained PGO"
                )
            else:
                self._repair_map_finish(f"{phase_label} failed")
            return True
        if self._repair_map_stop_requested:
            self._repair_map_finish(f"stopped by user after {phase_label}")
            return True
        if after == "optimize" and self._repair_map_stage == "optimize":
            self._repair_map_stage = "balm"
            self.append_log(
                "[RepairMap] Audited PGO complete; running one Final Map "
                "Refinement pass."
            )
            QtCore.QTimer.singleShot(0, self._repair_map_start_balm)
            return True
        if after == "balm" and self._repair_map_stage == "balm":
            self._repair_map_balm_passes = 1
            self._repair_map_balm_succeeded = True
            self._repair_map_balm_adopted = bool(
                getattr(self, "_last_balm_adopted", False)
            )
            if self._repair_map_balm_adopted:
                self.append_log(
                    "[RepairMap] Final Map Refinement complete and adopted."
                )
                reason = "Final Map Refinement adopted"
            else:
                self.append_log(
                    "[RepairMap] Final Map Refinement completed but failed the adoption "
                    "gate; restored the PGO trajectory and graph."
                )
                reason = "Final Map Refinement rejected by sanity gate; PGO retained"
            self._repair_map_finish(reason)
            return True
        return False

    def _repair_map_finish(self, reason: str = "") -> None:
        was_running = (self._repair_map_stage is not None
                       or getattr(self, "_repair_map_reporting", False))
        self._repair_map_stage = None
        self._repair_map_reporting = False
        self._repair_map_stop_requested = False
        self.repair_map_button.setText("⟳  Repair Map")
        self.repair_map_button.setEnabled(True)
        self._preview_refresh_queued = False
        if not was_running:
            return
        elapsed = time.perf_counter() - getattr(self, "_repair_map_started_at",
                                                time.perf_counter())
        balm_succeeded = getattr(self, "_repair_map_balm_succeeded", False)
        diagnostics_enabled = bool(self.map_diagnostics_check.isChecked())
        ghosts = len(self._balm_ghost_suggestions) if balm_succeeded else 0
        worst = max((g["separation_m"] for g in self._balm_ghost_suggestions),
                    default=0.0) if balm_succeeded else 0.0
        balm_adopted = getattr(self, '_repair_map_balm_adopted', None)
        residual_text = (
            "unavailable — BALM was rejected and PGO was retained"
            if balm_succeeded and balm_adopted is False else
            "disabled (inspection-only option)"
            if balm_succeeded and not diagnostics_enabled else
            f"{ghosts}" + (f"  (worst {worst*100:.0f} cm)"
                           if ghosts else "  — map is clean")
            if balm_succeeded
            else "unavailable — BALM did not complete"
        )
        lines = [
            f"Initial loops accepted: {getattr(self, '_repair_map_seeds', 0)}",
            f"Final refinements   : {getattr(self, '_repair_map_balm_passes', 0)}",
            f"Refinement adopted  : {getattr(self, '_repair_map_balm_adopted', None)}",
            f"Residual diagnostics: {residual_text}",
            f"Elapsed             : {elapsed:.0f} s",
        ]
        if reason:
            lines.append(f"Finished because    : {reason}")
        body = "\n".join(lines)
        self._set_repair_progress(
            "Repair Map finished" if not reason else f"Finished · {reason}",
            value=100)
        self.append_log("[RepairMap] " + body.replace("\n", " | "))
        # Initial Loop Search target maps and the shared raw-frame cache are useful only
        # while proposals are being registered.  Keeping them after the single
        # terminal BALM pass used several GiB on NTU and reduced the headroom
        # of the next run.  A later manual preview transparently reloads what it
        # needs.
        if self.workspace is not None:
            self.workspace.clear_caches(include_shared_frames=True)
        self.current_preview = None
        self.last_result = None
        for constraint in self.constraints:
            constraint.source_points_world_final = np.empty(
                (0, 3), dtype=np.float64
            )
        self.cloud_view.clear_scene()
        gc.collect()
        self.append_log(
            "[RepairMap] released Initial Loop Search descriptors and GICP caches; "
            "manual previews will reload on demand."
        )
        QtWidgets.QMessageBox.information(
            self, "Repair Map — finished",
            body + "\n\nExport the result to write the repaired map to disk.")
    # ===== END CHANGE: one-click Repair Map =====

    def _balm_report_summary(self) -> str:
        if self._last_output_dir is None:
            return ""
        report_path = self._last_output_dir / "balm_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        parts = []
        termination = report.get("termination") or {}
        reason = str(
            termination.get("reason") or report.get("stop_reason") or ""
        )
        completed = termination.get("completed_updates")
        requested = termination.get("requested_updates")
        count_text = (
            f" after {completed}/{requested} updates"
            if completed is not None and requested is not None else ""
        )
        stages = report.get("stages") or []
        iterations = (stages[-1].get("iterations") or []) if stages else []
        last = iterations[-1] if iterations else {}
        motion_text = ""
        if "max_translation_update_m" in last:
            motion_text = (
                f" (last max Δt={100 * float(last['max_translation_update_m']):.2f} cm, "
                f"ΔR={float(last.get('max_rotation_update_deg', 0.0)):.3f}°)"
            )
        if reason in {"final_stage_pose_converged", "pose_converged"}:
            parts.append("pose-converged" + count_text + motion_text)
        elif reason in {
            "final_stage_objective_plateau", "objective_plateau", "plateau",
        }:
            parts.append(
                "stopped on re-associated RMS plateau; not certified as pose "
                "convergence" + count_text + motion_text
            )
        elif reason == "configured_update_cap_reached":
            parts.append(
                "reached the configured update cap without pose convergence"
                + count_text + motion_text
            )
        elif reason:
            parts.append(reason.replace("_", " ") + count_text + motion_text)

        diagnostics_enabled = bool(
            (report.get("params") or {}).get(
                "map_inconsistency_diagnostics_enabled", True
            )
        )
        ghost_regions = report.get("ghost_regions") or []
        if not diagnostics_enabled:
            parts.append("duplicated-surface audit disabled")
        elif ghost_regions:
            parts.append(
                f"{len(ghost_regions)} duplicated-surface region(s) reported "
                "for inspection only"
            )
        else:
            parts.append("no duplicated-surface regions detected")
        return " " + " · ".join(parts) + "."
    # ===== END CHANGE: balm refinement workflow =====

    def _pgo_balm_comparison_sources(self):
        """Return the PGO/BALM trajectories for the current accepted BALM run."""
        if self._last_output_dir is None:
            return None
        report_path = self._last_output_dir / "balm_report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            source_run = Path(report["source_run_dir"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        adoption = report.get("production_adoption") or {}
        if adoption.get("adopted") is False:
            return None
        pgo_tum = source_run / "optimized_poses_tum.txt"
        balm_tum = self._last_output_dir / "optimized_poses_tum.txt"
        if not pgo_tum.is_file() or not balm_tum.is_file():
            return None
        return pgo_tum, balm_tum

    def export_pgo_balm_comparison(self) -> None:
        """Export and open an honest same-keyframe PGO/BALM color overlay."""
        if self.session_paths is None:
            self._show_error("Compare PGO/BALM", "Load a session first.")
            return
        sources = self._pgo_balm_comparison_sources()
        if sources is None:
            self._show_error(
                "Compare PGO/BALM",
                "The selected result is not an accepted BALM run with a "
                "recorded PGO input.",
            )
            return
        pgo_tum, balm_tum = sources
        keyframe_dir = self.session_paths.keyframe_dir
        voxel_leaf = float(self.export_map_voxel_spin.value())
        output_dir = (
            self.session_paths.session_root
            / "manual_loop_exports"
            / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_pgo_balm_compare")
        )

        def build_comparison(report_progress):
            output_dir.mkdir(parents=True, exist_ok=False)
            items = (
                ("after_pgo", pgo_tum, (249, 115, 22)),
                ("after_final_map_refinement", balm_tum, (59, 130, 246)),
            )
            manifest = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "voxel_size_m": voxel_leaf,
                "alignment": "none; identical keyframes in the session frame",
                "colors": {
                    "after_pgo": [249, 115, 22],
                    "after_final_map_refinement": [59, 130, 246],
                },
                "maps": {},
            }
            for index, (label, trajectory, color) in enumerate(items, start=1):
                report_progress(
                    f"Compare PGO/BALM · building {label} ({index}/2)", 0, 0
                )
                xyzi = output_dir / f"{label}_xyzi.pcd"
                rgb = output_dir / f"{label}_rgb.pcd"
                trajectory_pcd = output_dir / f"{label}_trajectory.pcd"
                points, poses, elapsed = build_map_and_trajectory_from_tum(
                    tum_path=trajectory,
                    keyframe_dir=keyframe_dir,
                    output_map=xyzi,
                    output_trajectory=trajectory_pcd,
                    voxel_leaf=voxel_leaf,
                    log_fn=lambda message, name=label: report_progress(
                        f"Compare · {name} · {message}", 0, 0
                    ),
                )
                colorize_binary_xyzi_pcd(xyzi, rgb, color)
                manifest["maps"][label] = {
                    "trajectory": str(trajectory),
                    "xyzi_pcd": xyzi.name,
                    "rgb_pcd": rgb.name,
                    "trajectory_pcd": trajectory_pcd.name,
                    "points": points,
                    "poses": poses,
                    "elapsed_sec": elapsed,
                }
            self._write_json(output_dir / "comparison_manifest.json", manifest)
            return output_dir

        self._start_background_task(
            "Compare PGO/BALM", build_comparison,
            self._finish_pgo_balm_comparison,
        )

    def _finish_pgo_balm_comparison(self, output_dir: Path) -> None:
        pgo_rgb = output_dir / "after_pgo_rgb.pcd"
        balm_rgb = output_dir / "after_final_map_refinement_rgb.pcd"
        candidates = (
            shutil.which("CloudCompare"),
            shutil.which("cloudcompare"),
            str(Path.home() / "cloudcompare-pcl-install/bin/CloudCompare"),
        )
        executable = next(
            (item for item in candidates if item and Path(item).is_file()), None
        )
        opened = False
        if executable is not None:
            detached = QtCore.QProcess.startDetached(
                executable, [str(pgo_rgb), str(balm_rgb)]
            )
            opened = bool(detached[0] if isinstance(detached, tuple) else detached)
        self._latest_export_dir = output_dir
        self._save_project_state()
        self.append_log(
            "[Compare PGO/BALM] Exported orange PGO and blue Final Map "
            f"Refinement maps to {output_dir}; CloudCompare opened={opened}."
        )
        QtWidgets.QMessageBox.information(
            self,
            "Compare PGO/BALM",
            "Orange: after audited PGO\n"
            "Blue: after Final Map Refinement\n"
            "No CloudCompare ICP alignment was applied.\n\n"
            f"Output: {output_dir}\n"
            + ("CloudCompare was opened." if opened else
               "CloudCompare was not found; open the two RGB PCDs manually."),
        )

    def export_final_result(self) -> None:
        if self.session_paths is None:
            self._show_error("Export Final Result", "Load a session first.")
            return
        if self._session_dirty:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Export Final Result",
                "Working graph has unapplied changes. Optimize before export?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if answer == QtWidgets.QMessageBox.Yes:
                self._pending_export_after_optimize = True
                self.run_optimization()
            return
        if self._last_output_dir is None or not self._last_output_dir.exists():
            self._show_error(
                "Export Final Result",
                "No optimized working result is available yet. Run Optimize Working Graph first.",
            )
            return
        run_dir = self._last_output_dir
        keyframe_dir = self.session_paths.keyframe_dir
        voxel_leaf = float(self.export_map_voxel_spin.value())

        def build_export(report_progress):
            report_progress("Export · preparing final map", 0, 0)
            self._ensure_run_map_outputs(
                run_dir,
                keyframe_dir=keyframe_dir,
                voxel_leaf=voxel_leaf,
                log_fn=lambda message: report_progress(
                    f"Export · {message}", 0, 0),
            )
            return run_dir

        self._start_background_task(
            "Export Final Result", build_export, self._finish_export_final_result)

    def _finish_export_final_result(self, run_dir: Path) -> None:
        """Create the lightweight export manifest after map build completes."""
        if self.session_paths is None:
            self._show_error(
                "Export Final Result",
                "The session changed while export was running; no manifest was created.",
            )
            return

        export_dir = (
            self.session_paths.session_root
            / "manual_loop_exports"
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        export_dir.parent.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        self._write_export_manifest(export_dir, run_dir)
        self._latest_export_dir = export_dir
        self.append_log(
            f"Exported final clean working manifest to {export_dir} -> {run_dir}",
            event="export_final",
            payload={"export_dir": str(export_dir), "run_dir": str(run_dir)},
        )
        self._save_project_state()
        QtWidgets.QMessageBox.information(
            self,
            "Export Final Result",
            f"Exported final clean working manifest to:\n{export_dir}\n\nSelected run:\n{run_dir}",
        )

    def undo_last_change(self) -> None:
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)
        self.append_log("Undo restored the previous working-session state.")
        self._save_project_state()

    def _show_error(self, title: str, message: str) -> None:
        self.append_log(f"{title}: {message}")
        QtWidgets.QMessageBox.critical(self, title, message)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._background_thread is not None:
            if self._background_label == "Export Final Result":
                detail = (
                    "The final map export is still writing verified artifacts. "
                    "Keep this window open until the progress bar finishes."
                )
            elif self._background_label == "Build Preview":
                detail = (
                    "The point-cloud preview is still loading in the background. "
                    "Keep this window open until the progress bar finishes."
                )
            else:
                detail = (
                    f"{self._background_label} is still running. Click Stop Repair "
                    "to stop safely after the current stage, then close the window."
                )
            QtWidgets.QMessageBox.information(
                self,
                "Operation in progress",
                detail,
            )
            event.ignore()
            return
        if self._optimizer_process is not None:
            if self._repair_map_stage is not None:
                detail = (
                    "The canonical repair engine is still running. Click Stop "
                    "Repair, wait for its isolated PGO/BALM process group to "
                    "exit, then close the window. The currently loaded result "
                    "will remain unchanged."
                    if self._repair_map_stage == "canonical" else
                    "Optimize or BALM is still writing the current repair "
                    "stage. Click Stop Repair and wait for that stage to "
                    "finish before closing the window."
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "Repair in progress",
                    detail,
                )
                event.ignore()
                return
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Optimization in progress",
                "An optimizer is still writing its run directory. Force-stop "
                "it and close? The current run may be incomplete.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self._optimizer_process.kill()
            self._optimizer_process.waitForFinished(1000)
            self._optimizer_process = None
        self._sync_gui_compute_marker(False)
        self.cloud_view.shutdown()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiDAR Map Refiner: offline loop closure and map refinement GUI."
    )
    parser.add_argument("--session-root", type=Path, help="Session root directory")
    parser.add_argument("--g2o", type=Path, help="Explicit pose_graph.g2o path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = ManualLoopClosureWindow(
        initial_session_root=args.session_root,
        initial_g2o_path=args.g2o,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
