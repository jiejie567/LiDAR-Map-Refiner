#!/usr/bin/env python3
"""Diagnose why adding a place-correct loop changes trajectory accuracy.

The reported diagnostics are available without trajectory ground truth.  ATE
is read only as an offline label for testing whether any diagnostic predicts a
bad trial-PGO update; it is never used to compute the diagnostic itself.

This script targets ``run_correct_loop_prefix_ablation.py`` outputs.  Each
prefix is compared with the immediately preceding prefix while keeping the
original pose graph, proposal order, and loop measurements fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr


ANCHOR_VERTEX_ID = 2147483646


@dataclass(frozen=True)
class EdgeRecord:
    node_i: int
    node_j: int
    measurement: np.ndarray
    information_g2o: np.ndarray


def _transform(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    result[:3, 3] = translation
    return result


def load_tum_transforms(path: Path) -> np.ndarray:
    transforms = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, raw_line in enumerate(stream, start=1):
            tokens = raw_line.strip().split()
            if not tokens or tokens[0].startswith("#"):
                continue
            if len(tokens) < 8:
                raise ValueError(f"{path}:{line_no}: expected at least 8 columns")
            translation = np.asarray(tokens[1:4], dtype=np.float64)
            quaternion = np.asarray(tokens[4:8], dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if norm <= 1e-12:
                raise ValueError(f"{path}:{line_no}: zero quaternion")
            transforms.append(_transform(translation, quaternion / norm))
    if not transforms:
        raise ValueError(f"no TUM poses in {path}")
    return np.asarray(transforms, dtype=np.float64)


def _parse_information(tokens: list[str], offset: int) -> np.ndarray:
    information = np.zeros((6, 6), dtype=np.float64)
    cursor = int(offset)
    for row in range(6):
        for col in range(row, 6):
            information[row, col] = float(tokens[cursor])
            information[col, row] = information[row, col]
            cursor += 1
    return information


def load_g2o_edges(path: Path) -> tuple[int, list[EdgeRecord]]:
    """Load the same deduplicated non-anchor edges as the Python optimizer."""
    vertex_ids: list[int] = []
    raw_edges: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            tokens = raw_line.strip().split()
            if not tokens:
                continue
            cursor = 1
            tag = tokens[0]
            if tag.startswith("#"):
                if len(tokens) < 2:
                    continue
                tag = tokens[1]
                cursor = 2
            if tag == "VERTEX_SE3:QUAT":
                vertex_id = int(tokens[cursor])
                if vertex_id != ANCHOR_VERTEX_ID:
                    vertex_ids.append(vertex_id)
            elif tag == "EDGE_SE3:QUAT":
                node_i, node_j = int(tokens[cursor]), int(tokens[cursor + 1])
                translation = np.asarray(
                    tokens[cursor + 2 : cursor + 5], dtype=np.float64
                )
                quaternion = np.asarray(
                    tokens[cursor + 5 : cursor + 9], dtype=np.float64
                )
                raw_edges.append((
                    node_i,
                    node_j,
                    _transform(translation, quaternion),
                    _parse_information(tokens, cursor + 9),
                ))

    id_map = {original_id: index for index, original_id in enumerate(vertex_ids)}
    seen: set[tuple[int, int]] = set()
    edges: list[EdgeRecord] = []
    for original_i, original_j, measurement, information in raw_edges:
        if ANCHOR_VERTEX_ID in {original_i, original_j}:
            continue
        if original_i not in id_map or original_j not in id_map:
            continue
        pair = (original_i, original_j)
        if pair in seen:
            continue
        seen.add(pair)
        edges.append(EdgeRecord(
            node_i=id_map[original_i],
            node_j=id_map[original_j],
            measurement=measurement,
            information_g2o=information,
        ))
    return len(vertex_ids), edges


def pose_residual(
    pose_i: np.ndarray, pose_j: np.ndarray, measurement_i_j: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation and rotation-vector residuals in g2o channel order."""
    predicted = np.linalg.inv(pose_i) @ pose_j
    error = np.linalg.inv(measurement_i_j) @ predicted
    return (
        np.asarray(error[:3, 3], dtype=np.float64),
        Rotation.from_matrix(error[:3, :3]).as_rotvec(),
    )


def residual_energy(
    translation: np.ndarray,
    rotation: np.ndarray,
    information_g2o: np.ndarray,
) -> tuple[float, float, float]:
    """Split a quadratic factor cost into translation, rotation, and total."""
    residual = np.concatenate([translation, rotation])
    translation_energy = float(
        translation @ information_g2o[:3, :3] @ translation
    )
    rotation_energy = float(rotation @ information_g2o[3:, 3:] @ rotation)
    total_energy = float(residual @ information_g2o @ residual)
    return translation_energy, rotation_energy, total_energy


def trajectory_motion(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    if before.shape != after.shape or before.ndim != 3 or before.shape[1:] != (4, 4):
        raise ValueError("before and after trajectories must have matching Nx4x4 shape")
    translation = np.linalg.norm(after[:, :3, 3] - before[:, :3, 3], axis=1)
    relative_rotation = np.einsum(
        "nij,njk->nik", np.transpose(before[:, :3, :3], (0, 2, 1)), after[:, :3, :3]
    )
    rotation_deg = np.degrees(Rotation.from_matrix(relative_rotation).magnitude())

    def summarize(prefix: str, values: np.ndarray) -> dict[str, float]:
        return {
            f"{prefix}_median": float(np.median(values)),
            f"{prefix}_rms": float(np.sqrt(np.mean(np.square(values)))),
            f"{prefix}_p95": float(np.percentile(values, 95.0)),
            f"{prefix}_max": float(np.max(values)),
        }

    return {
        **summarize("trajectory_translation_delta_m", translation),
        **summarize("trajectory_rotation_delta_deg", rotation_deg),
    }


def graph_energy(poses: np.ndarray, edges: list[EdgeRecord]) -> dict[str, float]:
    translation_energy = 0.0
    rotation_energy = 0.0
    total_energy = 0.0
    for edge in edges:
        translation, rotation = pose_residual(
            poses[edge.node_i], poses[edge.node_j], edge.measurement
        )
        t_value, r_value, total_value = residual_energy(
            translation, rotation, edge.information_g2o
        )
        translation_energy += t_value
        rotation_energy += r_value
        total_energy += total_value
    return {
        "translation": translation_energy,
        "rotation": rotation_energy,
        "total": total_energy,
    }


def chain_channel_leverage(
    source_id: int,
    target_id: int,
    edges: list[EdgeRecord],
    loop_information_g2o: np.ndarray,
) -> dict[str, float | None]:
    """Approximate scalar channel gain from the contiguous odometry chain.

    The proxy sums mean marginal variances of adjacent edges between the two
    endpoints.  It deliberately does not claim to be a full SE(3) marginal;
    its purpose is to reveal gross channel imbalance before a trial solve.
    """
    variances = chain_channel_variances(source_id, target_id, edges)
    if variances is None:
        return {
            "chain_translation_variance": None,
            "chain_rotation_variance": None,
            "loop_translation_variance": None,
            "loop_rotation_variance": None,
            "translation_channel_gain": None,
            "rotation_channel_gain": None,
            "rotation_to_translation_gain_ratio": None,
        }
    chain_translation_variance, chain_rotation_variance = variances
    loop_covariance = np.linalg.inv(loop_information_g2o)
    loop_translation_variance = float(np.mean(np.diag(loop_covariance[:3, :3])))
    loop_rotation_variance = float(np.mean(np.diag(loop_covariance[3:, 3:])))
    translation_gain = chain_translation_variance / (
        chain_translation_variance + loop_translation_variance
    )
    rotation_gain = chain_rotation_variance / (
        chain_rotation_variance + loop_rotation_variance
    )
    return {
        "chain_translation_variance": chain_translation_variance,
        "chain_rotation_variance": chain_rotation_variance,
        "loop_translation_variance": loop_translation_variance,
        "loop_rotation_variance": loop_rotation_variance,
        "translation_channel_gain": translation_gain,
        "rotation_channel_gain": rotation_gain,
        "rotation_to_translation_gain_ratio": (
            rotation_gain / max(translation_gain, 1e-15)
        ),
    }


def chain_channel_variances(
    source_id: int,
    target_id: int,
    edges: list[EdgeRecord],
) -> tuple[float, float] | None:
    """Sum scalar translation/rotation variances along adjacent graph edges."""
    adjacent: dict[int, EdgeRecord] = {}
    for edge in edges:
        lower, upper = sorted((edge.node_i, edge.node_j))
        if upper == lower + 1 and lower not in adjacent:
            adjacent[lower] = edge
    lower, upper = sorted((int(source_id), int(target_id)))
    path = [adjacent.get(index) for index in range(lower, upper)]
    if not path or any(edge is None for edge in path):
        return None

    chain_translation_variance = 0.0
    chain_rotation_variance = 0.0
    for edge in path:
        assert edge is not None
        covariance = np.linalg.inv(edge.information_g2o)
        chain_translation_variance += float(np.mean(np.diag(covariance[:3, :3])))
        chain_rotation_variance += float(np.mean(np.diag(covariance[3:, 3:])))
    return chain_translation_variance, chain_rotation_variance


def _last_constraint(path: Path, information_scale: float) -> tuple[dict, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("enabled") == "1"]
    if not rows:
        raise ValueError(f"no enabled constraints in {path}")
    row = rows[-1]
    measurement = _transform(
        np.asarray([row[axis] for axis in ("tx", "ty", "tz")], dtype=np.float64),
        np.asarray([row[axis] for axis in ("qx", "qy", "qz", "qw")], dtype=np.float64),
    )
    encoded_information = row.get("information_upper_json")
    if encoded_information:
        values = json.loads(encoded_information)
        if not isinstance(values, list) or len(values) != 21:
            raise ValueError("information_upper_json must contain 21 values")
        information = np.zeros((6, 6), dtype=np.float64)
        cursor = 0
        for matrix_row in range(6):
            for matrix_col in range(matrix_row, 6):
                information[matrix_row, matrix_col] = float(values[cursor])
                information[matrix_col, matrix_row] = float(values[cursor])
                cursor += 1
        information *= float(information_scale)
    else:
        sigma_t = np.asarray(
            [row[axis] for axis in ("sigma_tx", "sigma_ty", "sigma_tz")],
            dtype=np.float64,
        )
        sigma_r = np.asarray(
            [row[axis] for axis in ("sigma_roll_deg", "sigma_pitch_deg", "sigma_yaw_deg")],
            dtype=np.float64,
        )
        unit = (row.get("sigma_rotation_unit") or "deg").strip().lower()
        if unit == "deg":
            sigma_r = np.deg2rad(sigma_r)
        elif unit != "rad":
            raise ValueError(f"unsupported sigma_rotation_unit {unit!r}")
        sigmas = np.concatenate([sigma_t, sigma_r])
        information = np.diag(1.0 / np.square(sigmas)) * float(information_scale)
    return row, measurement, information


def _auc(values: list[float], labels: list[bool]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = rankdata(values, method="average")
    positive_rank_sum = float(sum(
        rank for rank, label in zip(ranks, labels) if label
    ))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def diagnostic_associations(rows: list[dict]) -> list[dict]:
    excluded = {
        "prefix_count", "source_id", "target_id", "ate_before_m", "ate_after_m",
        "ate_delta_m", "ate_worsened", "candidate_weight",
    }
    labels = [bool(row["ate_worsened"]) for row in rows]
    output = []
    for name in sorted(set.intersection(*(
        {key for key, value in row.items() if isinstance(value, (int, float)) and key not in excluded}
        for row in rows
    ))):
        values = [float(row[name]) for row in rows]
        if not np.isfinite(values).all() or np.ptp(values) <= 1e-15:
            continue
        correlation = spearmanr(values, [float(row["ate_delta_m"]) for row in rows])
        auc = _auc(values, labels)
        output.append({
            "metric": name,
            "spearman_rho_vs_ate_delta": float(correlation.statistic),
            "spearman_pvalue": float(correlation.pvalue),
            "auc_for_ate_worsened_high_value": auc,
            "best_orientation_auc": None if auc is None else max(auc, 1.0 - auc),
        })
    return sorted(
        output,
        key=lambda row: (
            -(row["best_orientation_auc"] or 0.0),
            -abs(row["spearman_rho_vs_ate_delta"]),
            row["metric"],
        ),
    )


def analyze(report_path: Path, rebuild_root: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    all_rows: list[dict] = []
    for profile_name, profile in report["profiles"].items():
        information_scale = float(profile["config"]["information_scale"])
        profile_root = report_path.parent / "runs" / profile_name
        for dataset, dataset_report in profile["datasets"].items():
            session = rebuild_root / dataset
            pose_count, edges = load_g2o_edges(session / "pose_graph.g2o")
            before_path = session / "optimized_poses_tum.txt"
            ate_curve = dataset_report["ate_curve_m"]
            for prefix in dataset_report["prefixes"]:
                count = int(prefix["count"])
                run_root = profile_root / dataset / f"prefix_{count:02d}"
                after_path = run_root / "optimized_poses_tum.txt"
                if count > 1:
                    before_path = (
                        profile_root / dataset / f"prefix_{count - 1:02d}"
                        / "optimized_poses_tum.txt"
                    )
                before = load_tum_transforms(before_path)
                after = load_tum_transforms(after_path)
                if before.shape[0] != pose_count or after.shape[0] != pose_count:
                    raise ValueError(f"pose count mismatch for {dataset} prefix {count}")
                constraint, measurement, loop_information = _last_constraint(
                    run_root / "candidate_constraints.csv", information_scale
                )
                source_id = int(constraint["source_id"])
                target_id = int(constraint["target_id"])
                pre_t, pre_r = pose_residual(
                    before[target_id], before[source_id], measurement
                )
                post_t, post_r = pose_residual(
                    after[target_id], after[source_id], measurement
                )
                pre_t_e, pre_r_e, pre_e = residual_energy(
                    pre_t, pre_r, loop_information
                )
                post_t_e, post_r_e, post_e = residual_energy(
                    post_t, post_r, loop_information
                )
                before_graph = graph_energy(before, edges)
                after_graph = graph_energy(after, edges)
                candidate_weights = prefix.get("candidate_weights")
                row = {
                    "profile": profile_name,
                    "dataset": dataset,
                    "prefix_count": count,
                    "proposal_id": prefix["added_proposal_id"],
                    "source_id": source_id,
                    "target_id": target_id,
                    "keyframe_span": abs(source_id - target_id),
                    "ate_before_m": float(ate_curve[count - 1]),
                    "ate_after_m": float(ate_curve[count]),
                    "ate_delta_m": float(ate_curve[count] - ate_curve[count - 1]),
                    "ate_worsened": bool(ate_curve[count] - ate_curve[count - 1] > 1e-6),
                    "candidate_weight": (
                        None if not candidate_weights else float(candidate_weights[-1])
                    ),
                    "pre_loop_translation_residual_m": float(np.linalg.norm(pre_t)),
                    "pre_loop_rotation_residual_deg": float(np.degrees(np.linalg.norm(pre_r))),
                    "post_loop_translation_residual_m": float(np.linalg.norm(post_t)),
                    "post_loop_rotation_residual_deg": float(np.degrees(np.linalg.norm(post_r))),
                    "pre_loop_translation_chi2": pre_t_e,
                    "pre_loop_rotation_chi2": pre_r_e,
                    "pre_loop_total_chi2": pre_e,
                    "post_loop_translation_chi2": post_t_e,
                    "post_loop_rotation_chi2": post_r_e,
                    "post_loop_total_chi2": post_e,
                    "loop_chi2_reduction": pre_e - post_e,
                    "original_graph_chi2_before": before_graph["total"],
                    "original_graph_chi2_after": after_graph["total"],
                    "original_graph_chi2_increase": (
                        after_graph["total"] - before_graph["total"]
                    ),
                    "original_graph_translation_chi2_increase": (
                        after_graph["translation"] - before_graph["translation"]
                    ),
                    "original_graph_rotation_chi2_increase": (
                        after_graph["rotation"] - before_graph["rotation"]
                    ),
                    "source_translation_delta_m": float(np.linalg.norm(
                        after[source_id, :3, 3] - before[source_id, :3, 3]
                    )),
                    "target_translation_delta_m": float(np.linalg.norm(
                        after[target_id, :3, 3] - before[target_id, :3, 3]
                    )),
                    **trajectory_motion(before, after),
                    **chain_channel_leverage(
                        source_id, target_id, edges, loop_information
                    ),
                }
                all_rows.append(row)

    associations = diagnostic_associations(all_rows)
    return {
        "schema_version": 1,
        "question": (
            "Can a no-ground-truth trial-PGO diagnostic identify individual "
            "place-correct loop additions that increase ATE?"
        ),
        "important": (
            "ATE is an offline label only. Every metric in diagnostics is "
            "computed from the pose graph, loop factor, and trial solutions."
        ),
        "source": {
            "prefix_report": str(report_path.resolve()),
            "prefix_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "rebuild_root": str(rebuild_root.resolve()),
            "measurement_source": report["protocol"].get("measurement_source"),
        },
        "row_count": len(all_rows),
        "worsened_count": sum(bool(row["ate_worsened"]) for row in all_rows),
        "rows": all_rows,
        "associations": associations,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix_report", type=Path)
    parser.add_argument("rebuild_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    output = analyze(
        args.prefix_report.expanduser().resolve(),
        args.rebuild_root.expanduser().resolve(),
    )
    output_json = args.output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {output_json}")
    output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else output_json.with_suffix(".csv")
    )
    if output_csv.exists():
        raise FileExistsError(f"Refusing to overwrite {output_csv}")
    _write_csv(output_csv, output["rows"])
    digest = hashlib.sha256(output_json.read_bytes()).hexdigest()
    output_json.with_suffix(output_json.suffix + ".sha256").write_text(
        f"{digest}  {output_json.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "row_count": output["row_count"],
        "worsened_count": output["worsened_count"],
        "top_associations": output["associations"][:8],
    }, indent=2))


if __name__ == "__main__":
    main()
