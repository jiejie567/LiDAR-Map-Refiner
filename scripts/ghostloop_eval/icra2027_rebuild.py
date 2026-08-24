#!/usr/bin/env python3
"""Prepare and execute the frozen ICRA-2027 public-data rebuild.

Every destination receives only immutable inputs, a content-hashed manifest,
and a symlink to the read-only keyframe store. Historical experiment outputs
are never copied into the rebuild. The headless driver then writes all mutable
artifacts below that destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
RUNTIME = Path("/home/anyverse/icra2027_runtime")
DRIVER = HERE / "auto_repair_headless.py"
PACKAGE_ROOT = REPO / "tools/manual_loop_closure/gui/manual_loop_closure"
# Hash the complete imported package instead of maintaining a fragile manual
# list of modules.  This deliberately includes a few GUI-only files: an extra
# hash is harmless, while omitting optimizer_backend.py or a transitive helper
# makes a supposedly frozen run impossible to reconstruct exactly.
CODE_INPUTS = tuple(dict.fromkeys((
    DRIVER,
    HERE / "causal_audit.py",
    HERE / "proposal_ledger.py",
    HERE / "experiment_io.py",
    *sorted(PACKAGE_ROOT.rglob("*.py")),
)))
INPUT_FILES = (
    "optimized_poses_tum.txt",
    "pose_graph.g2o",
    "runtime_params.yaml",
    "scan_context_gravity.csv",
)
OPTIONAL_INPUT_FILES = ("keyframe_frame_contract.json",)


@dataclass(frozen=True)
class DatasetSpec:
    source: Path
    environment: str
    truth: Path | None
    truth_kind: str | None
    legacy_keyframe_frame_role: str = "lidar_imu"


DATASETS = {
    "ntu": DatasetSpec(
        REPO / "fast_lio/PCD/manual_loop_session_mcd_ntu_day01",
        "outdoor",
        RUNTIME / "datasets/mcd_ntu/ntu_day_01_gt_tum.txt",
        "se3",
    ),
    "kth": DatasetSpec(
        REPO / "fast_lio/PCD/manual_loop_session_mcd_kth_night01",
        "outdoor",
        RUNTIME / "datasets/mcd_kth/kth_night_01_gt_tum.txt",
        "se3",
    ),
    "hall02": DatasetSpec(
        RUNTIME / "experiments/ghostloop_eval/hall_02_clean",
        "indoor",
        RUNTIME / "datasets/m2dgr/hall_02_gt.txt",
        "position",
    ),
    "hall04": DatasetSpec(
        RUNTIME / "experiments/ghostloop_eval/hall_04_clean",
        "indoor",
        RUNTIME / "datasets/m2dgr/hall_04_gt.txt",
        "position",
    ),
    "building_day": DatasetSpec(
        RUNTIME / "experiments/ghostloop_eval/fp_building_day",
        "outdoor",
        RUNTIME / "datasets/fusionportable/building_day_gt_tum.txt",
        "se3",
    ),
    "spires": DatasetSpec(
        REPO / "fast_lio/PCD/manual_loop_session_spires",
        "outdoor",
        RUNTIME / "datasets/spires_gt/2024-03-20-christ-church-05.txt",
        "se3",
    ),
    "mulran": DatasetSpec(
        REPO / "fast_lio/PCD/manual_loop_session_sejong01_real",
        "outdoor",
        None,
        None,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def keyframe_inventory(path: Path) -> dict:
    digest = hashlib.sha256()
    files = sorted(path.glob("*.pcd"), key=lambda item: item.name)
    total_bytes = 0
    for item in files:
        file_digest = sha256_file(item)
        size = item.stat().st_size
        total_bytes += size
        digest.update(f"{item.name}\0{size}\0{file_digest}\n".encode())
    return {
        "resolved_path": str(path.resolve()),
        "count": len(files),
        "total_bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def prepare_dataset(tag: str, spec: DatasetSpec, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite rebuild: {destination}")
    source = spec.source.resolve()
    keyframes = (source / "key_point_frame").resolve()
    if not keyframes.is_dir():
        raise FileNotFoundError(f"Missing keyframes: {keyframes}")
    destination.mkdir(parents=True)
    hashes = {}
    for name in INPUT_FILES:
        src = source / name
        if not src.is_file():
            if name == "scan_context_gravity.csv":
                continue
            raise FileNotFoundError(src)
        dst = destination / name
        shutil.copy2(src, dst)
        hashes[name] = sha256_file(dst)
    for name in OPTIONAL_INPUT_FILES:
        src = source / name
        if not src.is_file():
            continue
        dst = destination / name
        shutil.copy2(src, dst)
        hashes[name] = sha256_file(dst)
    frame_contract_path = destination / "keyframe_frame_contract.json"
    if frame_contract_path.is_file():
        frame_contract = json.loads(frame_contract_path.read_text(encoding="utf-8"))
        if frame_contract.get("schema_version") != 1:
            raise ValueError(f"Unsupported frame contract: {frame_contract_path}")
        if frame_contract.get("point_frame_equals_pose_frame") is not True:
            raise ValueError(
                "Loop evaluation requires PCD points and exported poses in the "
                f"same local frame: {frame_contract_path}"
            )
        if frame_contract.get("keyframe_frame_role") not in {
            "lidar_imu", "base_link"
        }:
            raise ValueError(f"Invalid keyframe frame role: {frame_contract_path}")
        frame_contract["provenance"] = "exported_sidecar"
        frame_contract["sha256"] = hashes[frame_contract_path.name]
    else:
        frame_contract = {
            "schema_version": 0,
            "keyframe_frame_role": spec.legacy_keyframe_frame_role,
            "point_frame_equals_pose_frame": True,
            "provenance": "explicit_legacy_dataset_spec",
            "note": (
                "Pre-sidecar session; role is frozen by DatasetSpec and must "
                "not be inferred from the current ROS configuration."
            ),
        }
    (destination / "key_point_frame").symlink_to(keyframes, target_is_directory=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": tag,
        "source": str(source),
        "environment": spec.environment,
        "truth": None if spec.truth is None else str(spec.truth.resolve()),
        "truth_kind": spec.truth_kind,
        "keyframe_export": frame_contract,
        "input_sha256": hashes,
        "keyframes": keyframe_inventory(keyframes),
        "code_sha256": {
            str(path.relative_to(REPO)): sha256_file(path) for path in CODE_INPUTS
        },
    }
    (destination / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run_dataset(
    destination: Path,
    environment: str,
    *,
    repair_mode: str,
    max_rounds: int,
    seed_cap: int,
    candidate_pool: int,
    candidate_distance: float | None,
    stage0_audit_only: bool,
    candidate_segment_radius: int | None,
    ring_key_top_k: int | None,
    loop_factor_mode: str,
    factor_information_scale: float,
    gravity_max_error_deg: float,
    pgo_profile: str,
) -> int:
    env = os.environ.copy()
    env["GHOSTLOOP_MAX_ROUNDS"] = str(max_rounds)
    if seed_cap:
        env["GHOSTLOOP_SEED_CAP"] = str(seed_cap)
    if candidate_pool:
        env["GHOSTLOOP_CANDIDATE_POOL"] = str(candidate_pool)
    if candidate_distance is not None:
        env["GHOSTLOOP_CANDIDATE_DISTANCE"] = str(candidate_distance)
    if stage0_audit_only:
        env["GHOSTLOOP_STAGE0_AUDIT_ONLY"] = "1"
    if candidate_segment_radius is not None:
        env["GHOSTLOOP_CANDIDATE_SEGMENT_RADIUS"] = str(candidate_segment_radius)
    if ring_key_top_k is not None:
        env["GHOSTLOOP_RING_KEY_TOP_K"] = str(ring_key_top_k)
    command = [
        sys.executable,
        "-u",
        str(DRIVER),
        str(destination),
        repair_mode,
        environment,
        "--loop-factor-mode", loop_factor_mode,
        "--factor-information-scale", str(factor_information_scale),
        "--gravity-max-error-deg", str(gravity_max_error_deg),
        "--pgo-profile", pgo_profile,
    ]
    with (destination.parent / f"{destination.name}.log").open(
        "w", encoding="utf-8"
    ) as log:
        completed = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument(
        "--repair-mode",
        choices=("production",),
        default="production",
        help=(
            "Default production runs Initial Loop Search, Audited PGO, and "
            "one Final Map Refinement. "
            "Select ghost explicitly only to reproduce the legacy proposal loop."
        ),
    )
    parser.add_argument("--seed-cap", type=int, default=0)
    parser.add_argument("--candidate-pool", type=int, default=0)
    parser.add_argument("--candidate-distance", type=float)
    parser.add_argument(
        "--initial-loops-audit-only", "--stage0-audit-only",
        dest="stage0_audit_only", action="store_true",
        help=(
            "Stop after auditing the Initial Loop Search constraints. The "
            "stage0 spelling remains as a compatibility alias."
        ),
    )
    parser.add_argument("--candidate-segment-radius", type=int)
    parser.add_argument("--ring-key-top-k", type=int)
    parser.add_argument(
        "--loop-factor-mode",
        choices=("diagonal", "oriented_anisotropic"),
        default="diagonal",
    )
    parser.add_argument(
        "--factor-information-scale", type=float,
        default=0.00016358831377740368,
    )
    # Keep the dependency-free rebuild launcher aligned with
    # repair_presets.PRODUCTION_PGO_PROFILE.  This is a catastrophic sanity
    # limit, not the retired one-degree accuracy gate.
    parser.add_argument("--gravity-max-error-deg", type=float, default=10.0)
    parser.add_argument(
        "--pgo-profile", choices=("legacy", "calibrated"),
        default="calibrated",
        help=(
            "Frozen calibrated is the paper/Occam path; legacy is retained "
            "only as a reproducibility control."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (args.max_rounds < 1 or args.seed_cap < 0 or args.candidate_pool < 0
            or (args.candidate_distance is not None
                and args.candidate_distance <= 0.0)
            or (args.candidate_segment_radius is not None
                and args.candidate_segment_radius < 0)
            or (args.ring_key_top_k is not None and args.ring_key_top_k < 1)
            or args.factor_information_scale <= 0.0
            or args.gravity_max_error_deg <= 0.0):
        parser.error("max-rounds must be positive and seed-cap non-negative")
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = root / "rebuild_run_manifest.json"
    if run_manifest_path.exists():
        raise FileExistsError(f"Refusing to reuse rebuild root: {root}")
    run_manifest_path.write_text(
        json.dumps({
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "arguments": {
                "datasets": list(args.datasets),
                "repair_mode": args.repair_mode,
                "max_rounds": args.max_rounds,
                "seed_cap": args.seed_cap,
                "candidate_pool": args.candidate_pool,
                "candidate_distance": args.candidate_distance,
                "stage0_audit_only": args.stage0_audit_only,
                "candidate_segment_radius": args.candidate_segment_radius,
                "ring_key_top_k": args.ring_key_top_k,
                "loop_factor_mode": args.loop_factor_mode,
                "factor_information_scale": args.factor_information_scale,
                "gravity_max_error_deg": args.gravity_max_error_deg,
                "pgo_profile": args.pgo_profile,
                "prepare_only": args.prepare_only,
            },
            "driver_sha256": sha256_file(Path(__file__).resolve()),
            "headless_driver_sha256": sha256_file(DRIVER),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    registry_path = root / "rebuild_registry.json"
    registry = (
        json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    )
    for tag in args.datasets:
        spec = DATASETS[tag]
        destination = root / tag
        manifest = prepare_dataset(tag, spec, destination)
        returncode = None
        if not args.prepare_only:
            returncode = run_dataset(
                destination,
                spec.environment,
                repair_mode=args.repair_mode,
                max_rounds=args.max_rounds,
                seed_cap=args.seed_cap,
                candidate_pool=args.candidate_pool,
                candidate_distance=args.candidate_distance,
                stage0_audit_only=args.stage0_audit_only,
                candidate_segment_radius=args.candidate_segment_radius,
                ring_key_top_k=args.ring_key_top_k,
                loop_factor_mode=args.loop_factor_mode,
                factor_information_scale=args.factor_information_scale,
                gravity_max_error_deg=args.gravity_max_error_deg,
                pgo_profile=args.pgo_profile,
            )
        registry[tag] = {
            "destination": str(destination),
            "input_manifest_sha256": sha256_file(
                destination / "input_manifest.json"
            ),
            "pose_graph_sha256": manifest["input_sha256"]["pose_graph.g2o"],
            "trajectory_sha256": manifest["input_sha256"][
                "optimized_poses_tum.txt"
            ],
            "returncode": returncode,
        }
        if returncode not in (None, 0):
            break
    registry_path.write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    if any(item["returncode"] not in (None, 0) for item in registry.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
