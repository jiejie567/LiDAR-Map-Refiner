#!/usr/bin/env python3
"""Create two immutable-input session views for a 4 m / 5 m ablation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from experiment_io import CONSTRAINT_HEADER, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-session", type=Path, required=True)
    parser.add_argument("--source-constraints", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage0-count", type=int, default=5)
    args = parser.parse_args()
    if args.stage0_count <= 0:
        raise SystemExit("--stage0-count must be positive")
    source = args.source_session.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with args.source_constraints.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [row for row in reader if row][: args.stage0_count]
    if header != CONSTRAINT_HEADER or len(rows) != args.stage0_count:
        raise SystemExit("Source constraints do not contain the requested checkpoint")
    checkpoint = output / f"stage0_first{args.stage0_count}_constraints.csv"
    if checkpoint.exists():
        raise SystemExit(f"Refusing to overwrite existing checkpoint: {checkpoint}")
    with checkpoint.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)

    immutable_names = (
        "optimized_poses_tum.txt", "pose_graph.g2o", "runtime_params.yaml",
        "scan_context_gravity.csv",
    )
    branches = {}
    for voxel in (4, 5):
        branch = output / f"mcd_building_root{voxel}m"
        if branch.exists():
            raise SystemExit(f"Refusing to overwrite existing branch: {branch}")
        branch.mkdir()
        for name in immutable_names:
            source_path = source / name
            if source_path.exists():
                shutil.copy2(source_path, branch / name)
        os.symlink((source / "key_point_frame").resolve(), branch / "key_point_frame")
        branches[f"{voxel}m"] = str(branch)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": "MCD Building",
        "source_session": str(source),
        "stage0_policy": f"first {args.stage0_count} rows in original CSV order",
        "stage0_constraints_csv": str(checkpoint),
        "stage0_constraints_sha256": sha256_file(checkpoint),
        "shared_input_sha256": {
            name: sha256_file(source / name)
            for name in immutable_names if (source / name).is_file()
        },
        "branches": branches,
        "root_voxel_sizes_m": [4.0, 5.0],
        "outdoor_product_default_changed": False,
    }
    (output / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
