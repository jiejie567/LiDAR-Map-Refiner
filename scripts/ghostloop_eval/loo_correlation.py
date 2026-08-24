#!/usr/bin/env python3
"""每条约束到底帮了还是害了，与它自己的配准指标有没有关系。

这是论文 C2 的核心证据：如果配准质量能预测一条约束的好坏，那么按质量过滤就够了，
准入就不是难点。要证伪它，必须对每条已接受的约束做留一重解，量出它对 ATE 的真实
贡献，再去和它的 per-pair 指标求相关。

对每条约束：从原始里程计图出发，用「全集减去这一条」重解一次，与全集的 ATE 相比。
差值为正说明这条约束是有益的（去掉它变差），为负说明它有害。

用法：loo_correlation.py <run_dir> <env>
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GUI = HERE.parents[1] / 'gui'
CLI = GUI / 'manual_loop_closure' / 'python_optimizer' / 'cli.py'
sys.path.insert(0, str(HERE))
from paper_table import metric  # noqa: E402


def solve(session: Path, rows: list[dict], header: list[str], work: Path) -> Path:
    """Re-solve the pose graph from raw odometry with exactly these rows."""
    work.mkdir(parents=True, exist_ok=True)
    csv_path = work / 'manual_loop_constraints.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    g2o = work / 'edited_input_pose_graph.g2o'
    g2o.write_bytes((session / 'pose_graph.g2o').read_bytes())
    r = subprocess.run(
        [sys.executable, '-u', str(CLI),
         '--session-root', str(session), '--g2o', str(g2o),
         '--tum', str(session / 'optimized_poses_tum.txt'),
         '--keyframe-dir', str(session / 'key_point_frame'),
         '--constraints-csv', str(csv_path),
         '--output-dir', str(work), '--optimize-mode', 'isam2', '--skip-map-build'],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
    return work / 'optimized_poses_tum.txt'


def main() -> None:
    session = Path(sys.argv[1]).expanduser()
    s = json.loads((session / 'auto_repair_summary.json').read_text())
    balm = Path(s['terminal_output_dir'])
    pgo = balm.parent / balm.name.replace('_balm', '')
    src = pgo / 'manual_loop_constraints.csv'
    with src.open() as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames
        rows = list(rd)
    on = [i for i, r in enumerate(rows) if r['enabled'] == '1']

    tmp = Path(tempfile.mkdtemp(prefix='loo_'))
    try:
        full = metric(session, solve(session, rows, header, tmp / 'full'))
        print(f'# {session.name}: {len(on)} constraints, full-set metric {full:.3f}')
        print('src\tdst\tdelta_if_removed')
        for k, i in enumerate(on):
            trimmed = [dict(r) for r in rows]
            trimmed[i]['enabled'] = '0'
            v = metric(session, solve(session, trimmed, header, tmp / f'loo{k}'))
            # positive => removing it hurt => the constraint was earning its place
            print(f"{rows[i]['source_id']}\t{rows[i]['target_id']}\t{v - full:+.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
