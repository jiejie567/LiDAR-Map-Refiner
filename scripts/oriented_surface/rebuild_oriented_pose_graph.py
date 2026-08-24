#!/usr/bin/env python3
"""Rebuild sequential and loop factors from one oriented-surface estimator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve()
GUI = HERE.parents[2] / "gui"
if str(GUI) not in sys.path:
    sys.path.insert(0, str(GUI))

from manual_loop_closure.pcd_io import load_xyz_points  # noqa: E402
from manual_loop_closure.python_optimizer.oriented_surface_factors import (  # noqa: E402
    OrientedFactorConfig,
    PreparedOrientedCloud,
    estimate_oriented_surface_factor,
    prepare_oriented_cloud,
)


VARIANTS = (
    "original_graph",
    "icp_isotropic",
    "icp_anisotropic",
    "oriented_anisotropic",
    "icp_anisotropic_reweight",
    "oriented_anisotropic_reweight",
    "loop_icp_isotropic",
    "loop_icp_anisotropic",
    "loop_oriented_anisotropic",
)

REWEIGHT_VARIANTS = {
    "icp_anisotropic_reweight",
    "oriented_anisotropic_reweight",
}

LOOP_ONLY_VARIANTS = {
    "loop_icp_isotropic",
    "loop_icp_anisotropic",
    "loop_oriented_anisotropic",
}


def isotropize_information(information_g2o: np.ndarray) -> np.ndarray:
    """Remove directional shape while preserving each tangent block's scale."""
    information = np.asarray(information_g2o, dtype=np.float64)
    if information.shape != (6, 6):
        raise ValueError("information must be 6x6")
    translation = max(float(np.median(np.diag(information)[:3])), 1e-12)
    rotation = max(float(np.median(np.diag(information)[3:])), 1e-12)
    return np.diag([translation] * 3 + [rotation] * 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_information(tokens: list[str], offset: int = 0) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=np.float64)
    cursor = offset
    for row in range(6):
        for column in range(row, 6):
            matrix[row, column] = matrix[column, row] = float(tokens[cursor])
            cursor += 1
    return matrix


def format_information(matrix: np.ndarray) -> str:
    return " ".join(
        f"{float(matrix[row, column]):.15e}"
        for row in range(6) for column in range(row, 6)
    )


def transform_from_fields(translation, quaternion_xyzw) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    transform[:3, 3] = translation
    return transform


def edge_line(
    target_id: int, source_id: int, transform_target_source: np.ndarray,
    information_g2o: np.ndarray,
) -> str:
    quaternion = Rotation.from_matrix(transform_target_source[:3, :3]).as_quat()
    translation = transform_target_source[:3, 3]
    return (
        f"EDGE_SE3:QUAT {target_id} {source_id} "
        + " ".join(f"{value:.15e}" for value in translation)
        + " " + " ".join(f"{value:.15e}" for value in quaternion)
        + " " + format_information(information_g2o) + "\n"
    )


def parse_g2o_edge(line: str) -> dict | None:
    tokens = line.split()
    if not tokens or tokens[0] != "EDGE_SE3:QUAT" or len(tokens) < 31:
        return None
    target_id, source_id = int(tokens[1]), int(tokens[2])
    transform = transform_from_fields(
        [float(value) for value in tokens[3:6]],
        [float(value) for value in tokens[6:10]],
    )
    return {
        "target_id": target_id,
        "source_id": source_id,
        "transform": transform,
        "information": parse_information(tokens, 10),
        "sequential": source_id == target_id + 1,
    }


def load_loop_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    output = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        for line_number, row in enumerate(reader, start=1):
            if not row or row[0].strip().lower() in {"enabled", "#enabled"}:
                continue
            if row[0].strip().startswith("#") or row[0].strip() in {"0", "false"}:
                continue
            if len(row) < 16:
                raise ValueError(f"{path}:{line_number}: expected 16 columns")
            source_id, target_id = int(row[1]), int(row[2])
            transform = transform_from_fields(
                [float(value) for value in row[3:6]],
                [float(value) for value in row[6:10]],
            )
            sigma_translation = np.asarray(row[10:13], dtype=np.float64)
            sigma_rotation = np.deg2rad(np.asarray(row[13:16], dtype=np.float64))
            sigmas_g2o = np.concatenate([sigma_translation, sigma_rotation])
            information = np.diag(1.0 / np.square(sigmas_g2o))
            output.append({
                "source_id": source_id,
                "target_id": target_id,
                "transform": transform,
                "information": information,
                "proposal_line": line_number,
            })
    return output


def prepare_frame(
    keyframe_dir: Path, frame_id: int, config: OrientedFactorConfig,
    cache: dict[int, PreparedOrientedCloud],
) -> PreparedOrientedCloud:
    if frame_id not in cache:
        path = keyframe_dir / f"{frame_id}.pcd"
        if not path.is_file():
            raise FileNotFoundError(path)
        cache[frame_id] = prepare_oriented_cloud(load_xyz_points(path), config)
    return cache[frame_id]


def remeasure(
    target_id: int,
    source_id: int,
    initial: np.ndarray,
    keyframe_dir: Path,
    config: OrientedFactorConfig,
    cache: dict[int, PreparedOrientedCloud],
):
    source = prepare_frame(keyframe_dir, source_id, config, cache)
    target = prepare_frame(keyframe_dir, target_id, config, cache)
    no_downsample = replace(config, voxel_size_m=0.0)
    return estimate_oriented_surface_factor(
        source.points, target.points, initial, no_downsample,
        source_normals_local=source.normals,
        target_normals_local=target.normals,
    )


def factor_record(
    *, origin: str, target_id: int, source_id: int, action: str,
    reason: str, transform: np.ndarray, information: np.ndarray,
    result=None,
) -> dict:
    eigenvalues = np.linalg.eigvalsh(information)
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    information_upper = [
        float(information[row, column])
        for row in range(6) for column in range(row, 6)
    ]
    record = {
        "origin": origin,
        "target_id": target_id,
        "source_id": source_id,
        "action": action,
        "reason": reason,
        "tx": float(transform[0, 3]),
        "ty": float(transform[1, 3]),
        "tz": float(transform[2, 3]),
        "qx": float(quaternion[0]),
        "qy": float(quaternion[1]),
        "qz": float(quaternion[2]),
        "qw": float(quaternion[3]),
        "information_upper_json": json.dumps(information_upper),
        "information_min_eigenvalue": float(eigenvalues[0]),
        "information_max_eigenvalue": float(eigenvalues[-1]),
        "valid_registration": bool(result.valid) if result is not None else None,
        "overlap": float(result.overlap) if result is not None else None,
        "spatial_overlap": (
            float(result.spatial_overlap) if result is not None else None
        ),
        "rmse_m": float(result.inlier_rmse_m) if result is not None else None,
        "residual_scale_m": (
            float(result.residual_scale_m) if result is not None else None
        ),
        "oriented_correspondences": (
            int(result.orientation_consistent_count) if result is not None else None
        ),
        "observable_rank": (
            int(result.observable_rank) if result is not None else None
        ),
        "condition_number": (
            float(result.scaled_condition_number) if result is not None else None
        ),
    }
    return record


def rebuild(
    input_g2o: Path,
    keyframe_dir: Path,
    output_dir: Path,
    variant: str,
    config: OrientedFactorConfig,
    loop_csv: Path | None = None,
    fallback_information_scale: float = 0.05,
    max_sequential_edges: int | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    lines = input_g2o.read_text(encoding="utf-8").splitlines(keepends=True)
    cache: dict[int, PreparedOrientedCloud] = {}
    records: list[dict] = []
    output_lines = []
    sequential_seen = 0
    for line in lines:
        edge = parse_g2o_edge(line)
        if edge is None or not edge["sequential"]:
            output_lines.append(line)
            continue
        sequential_seen += 1
        if max_sequential_edges is not None and sequential_seen > max_sequential_edges:
            output_lines.append(line)
            records.append(factor_record(
                origin="sequential", target_id=edge["target_id"],
                source_id=edge["source_id"], action="preserved_after_smoke_limit",
                reason="max_sequential_edges", transform=edge["transform"],
                information=edge["information"],
            ))
            continue
        if variant == "original_graph":
            output_lines.append(line)
            records.append(factor_record(
                origin="sequential", target_id=edge["target_id"],
                source_id=edge["source_id"], action="preserved_original",
                reason="original_graph_arm", transform=edge["transform"],
                information=edge["information"],
            ))
            continue
        if variant in LOOP_ONLY_VARIANTS:
            output_lines.append(line)
            records.append(factor_record(
                origin="sequential", target_id=edge["target_id"],
                source_id=edge["source_id"], action="preserved_original",
                reason="loop_only_ablation_arm", transform=edge["transform"],
                information=edge["information"],
            ))
            continue
        result = remeasure(
            edge["target_id"], edge["source_id"], edge["transform"],
            keyframe_dir, config, cache,
        )
        if result.valid:
            information = (
                isotropize_information(result.information_g2o)
                if variant == "icp_isotropic"
                else result.information_g2o
            )
            if variant in REWEIGHT_VARIANTS:
                transform = edge["transform"]
                action, reason = "reweighted", "accepted_information_shape"
            else:
                transform = result.transform_target_source
                action, reason = "replaced", "accepted_registration"
        else:
            information = edge["information"] * fallback_information_scale
            transform = edge["transform"]
            action, reason = "weakened_fallback", result.reason
        output_lines.append(edge_line(
            edge["target_id"], edge["source_id"], transform, information
        ))
        records.append(factor_record(
            origin="sequential", target_id=edge["target_id"],
            source_id=edge["source_id"], action=action, reason=reason,
            transform=transform, information=information, result=result,
        ))

    loop_rows = load_loop_rows(loop_csv)
    for loop in loop_rows:
        result = None
        transform = loop["transform"]
        information = loop["information"]
        action, reason = "appended_original", "original_graph_arm"
        if variant != "original_graph":
            result = remeasure(
                loop["target_id"], loop["source_id"], transform,
                keyframe_dir, config, cache,
            )
            if not result.valid:
                records.append(factor_record(
                    origin="loop", target_id=loop["target_id"],
                    source_id=loop["source_id"], action="rejected",
                    reason=result.reason, transform=transform,
                    information=information, result=result,
                ))
                continue
            transform = result.transform_target_source
            if variant in {"icp_isotropic", "loop_icp_isotropic"}:
                information = isotropize_information(result.information_g2o)
            else:
                information = result.information_g2o
            action, reason = "appended_remeasured", "accepted_registration"
        output_lines.append(edge_line(
            loop["target_id"], loop["source_id"], transform, information
        ))
        records.append(factor_record(
            origin="loop", target_id=loop["target_id"],
            source_id=loop["source_id"], action=action, reason=reason,
            transform=transform, information=information, result=result,
        ))

    output_g2o = output_dir / "pose_graph.g2o"
    output_g2o.write_text("".join(output_lines), encoding="utf-8")
    ledger = output_dir / "factor_ledger.csv"
    fields = list(records[0]) if records else ["origin", "target_id", "source_id"]
    with ledger.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variant": variant,
        "measurement_convention": "target_to_source",
        "information_order": "g2o_translation_rotation",
        "sequential_policy": (
            "preserve_frontend_measurement_rebuild_information"
            if variant in REWEIGHT_VARIANTS
            else "preserve_original_for_loop_only_ablation"
            if variant in LOOP_ONLY_VARIANTS
            else "replace_not_augment"
        ),
        "invalid_sequential_policy": "original_measurement_weakened_information",
        "fallback_information_scale": fallback_information_scale,
        "config": asdict(config),
        "inputs": {
            "g2o": str(input_g2o.resolve()),
            "g2o_sha256": sha256_file(input_g2o),
            "keyframe_dir": str(keyframe_dir.resolve()),
            "loop_csv": str(loop_csv.resolve()) if loop_csv else None,
            "loop_csv_sha256": sha256_file(loop_csv) if loop_csv else None,
        },
        "counts": {
            "sequential": sum(row["origin"] == "sequential" for row in records),
            "sequential_replaced": sum(
                row["origin"] == "sequential" and row["action"] == "replaced"
                for row in records
            ),
            "sequential_reweighted": sum(
                row["origin"] == "sequential" and row["action"] == "reweighted"
                for row in records
            ),
            "sequential_fallback": sum(
                row["origin"] == "sequential"
                and row["action"] == "weakened_fallback" for row in records
            ),
            "loop_proposed": len(loop_rows),
            "loop_added": sum(
                row["origin"] == "loop" and row["action"].startswith("appended")
                for row in records
            ),
        },
        "outputs": {
            "g2o_sha256": sha256_file(output_g2o),
            "ledger_sha256": sha256_file(ledger),
        },
    }
    manifest_path = output_dir / "factor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-g2o", type=Path, required=True)
    parser.add_argument("--keyframe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--loop-csv", type=Path)
    parser.add_argument("--voxel", type=float, default=0.20)
    parser.add_argument("--normal-radius", type=float, default=0.60)
    parser.add_argument("--max-correspondence", type=float, default=0.60)
    parser.add_argument("--normal-gate-deg", type=float, default=45.0)
    parser.add_argument("--min-overlap", type=float, default=0.20)
    parser.add_argument("--min-correspondences", type=int, default=80)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--fallback-information-scale", type=float, default=0.05)
    parser.add_argument("--information-scale", type=float, default=1.0)
    parser.add_argument("--translation-sigma-floor-m", type=float, default=0.0)
    parser.add_argument("--rotation-sigma-floor-deg", type=float, default=0.0)
    parser.add_argument("--max-sequential-edges", type=int)
    args = parser.parse_args()
    normal_gate = (
        180.0 if args.variant in {
            "icp_isotropic", "icp_anisotropic", "icp_anisotropic_reweight",
            "loop_icp_isotropic", "loop_icp_anisotropic",
        }
        else float(args.normal_gate_deg)
    )
    config = OrientedFactorConfig(
        voxel_size_m=float(args.voxel),
        normal_radius_m=float(args.normal_radius),
        max_correspondence_m=float(args.max_correspondence),
        normal_gate_deg=normal_gate,
        min_overlap=float(args.min_overlap),
        min_correspondences=int(args.min_correspondences),
        max_iterations=int(args.max_iterations),
        information_scale=float(args.information_scale),
        translation_sigma_floor_m=float(args.translation_sigma_floor_m),
        rotation_sigma_floor_rad=math.radians(
            float(args.rotation_sigma_floor_deg)
        ),
    )
    manifest = rebuild(
        args.input_g2o.expanduser().resolve(),
        args.keyframe_dir.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.variant,
        config,
        args.loop_csv.expanduser().resolve() if args.loop_csv else None,
        float(args.fallback_information_scale),
        args.max_sequential_edges,
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
