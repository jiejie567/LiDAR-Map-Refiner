from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class SessionResolutionError(RuntimeError):
    """Raised when the session layout cannot be resolved."""


@dataclass(frozen=True)
class SessionPaths:
    session_root: Path
    g2o_path: Path
    tum_path: Path
    keyframe_dir: Path


def _existing_paths(candidates: Iterable[Path]) -> list[Path]:
    return [path for path in candidates if path.exists()]


def _sort_timestamp_candidates(paths: list[Path]) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: (
            path.parent.name,
            path.stat().st_mtime,
        ),
        reverse=True,
    )


def _resolve_g2o_from_root(session_root: Path) -> Path:
    direct_g2o = session_root / "pose_graph.g2o"
    if direct_g2o.is_file():
        return direct_g2o

    timestamp_candidates = _sort_timestamp_candidates(
        _existing_paths(
            child / "pose_graph.g2o"
            for child in session_root.iterdir()
            if child.is_dir()
        )
    )
    if timestamp_candidates:
        return timestamp_candidates[0]

    raise SessionResolutionError(
        f"Failed to find pose_graph.g2o under session root: {session_root}"
    )


def _resolve_tum_from_g2o(session_root: Path, g2o_path: Path) -> Path:
    candidates = _existing_paths(
        [
            g2o_path.parent / "optimized_poses_tum.txt",
            g2o_path.parent / "optimized_odom_tum.txt",
            session_root / "optimized_poses_tum.txt",
            session_root / "optimized_odom_tum.txt",
        ]
    )
    if candidates:
        return candidates[0]

    timestamp_candidates = _sort_timestamp_candidates(
        _existing_paths(
            child / "optimized_poses_tum.txt"
            for child in session_root.iterdir()
            if child.is_dir()
        )
    )
    if timestamp_candidates:
        return timestamp_candidates[0]

    timestamp_candidates = _sort_timestamp_candidates(
        _existing_paths(
            child / "optimized_odom_tum.txt"
            for child in session_root.iterdir()
            if child.is_dir()
        )
    )
    if timestamp_candidates:
        return timestamp_candidates[0]

    raise SessionResolutionError(
        "Failed to find optimized_poses_tum.txt next to the selected g2o "
        f"or under the session root: {session_root}"
    )


def _infer_session_root_from_g2o(g2o_path: Path) -> Path:
    candidates = [g2o_path.parent, g2o_path.parent.parent]
    for candidate in candidates:
        if candidate and (candidate / "key_point_frame").is_dir():
            return candidate

    for ancestor in g2o_path.parents:
        if (ancestor / "key_point_frame").is_dir():
            return ancestor

    raise SessionResolutionError(
        "Failed to infer session root from g2o path. Expected key_point_frame "
        f"next to {g2o_path} or one level above it."
    )


def resolve_session_paths(
    *,
    session_root: Optional[Path] = None,
    g2o_path: Optional[Path] = None,
) -> SessionPaths:
    if session_root is None and g2o_path is None:
        raise SessionResolutionError("Either session_root or g2o_path must be provided.")

    resolved_root = session_root.expanduser().resolve() if session_root else None
    resolved_g2o = g2o_path.expanduser().resolve() if g2o_path else None

    if resolved_g2o is not None:
        if not resolved_g2o.is_file():
            raise SessionResolutionError(f"g2o path does not exist: {resolved_g2o}")
        if resolved_g2o.suffix.lower() != ".g2o":
            raise SessionResolutionError(f"Expected a .g2o file, got: {resolved_g2o}")
        if resolved_root is None:
            resolved_root = _infer_session_root_from_g2o(resolved_g2o)
    else:
        if resolved_root is None or not resolved_root.is_dir():
            raise SessionResolutionError(f"Session root does not exist: {resolved_root}")
        resolved_g2o = _resolve_g2o_from_root(resolved_root)

    if resolved_root is None:
        raise SessionResolutionError("Failed to resolve the session root directory.")

    keyframe_dir = resolved_root / "key_point_frame"
    if not keyframe_dir.is_dir():
        raise SessionResolutionError(
            f"Expected key_point_frame under session root, missing: {keyframe_dir}"
        )

    tum_path = _resolve_tum_from_g2o(resolved_root, resolved_g2o)

    return SessionPaths(
        session_root=resolved_root,
        g2o_path=resolved_g2o,
        tum_path=tum_path,
        keyframe_dir=keyframe_dir,
    )
