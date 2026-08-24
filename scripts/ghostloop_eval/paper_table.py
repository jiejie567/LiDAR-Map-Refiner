#!/usr/bin/env python3
"""论文表 I / 表 II 的唯一数字来源。

每个数据集家族的评测口径不同，这是这个项目里最容易串的地方，所以在一个文件里
写死一次：

  MCD (ntu/kth) 与 FusionPortable (esc/bd)  真值 TUM，SE(3) 对齐，位置 RMSE
  M2DGR (hall_02/hall_04)                  真值四元数全零，必须用 ate_eval.py
                                           联合拟合棱镜杆臂与每序列时钟偏移
  Oxford Spires (cc05)                     真值 TUM，同 MCD
  自采 IH / BIG                            无外部参考，用起止闭合误差
                                           （这两条序列可证地回到原点）

同时报告 PGO（平面精化前）与终态两个数：平面精化不降低轨迹误差，两者分开报，
互不背书。

用法：paper_table.py <实验目录> [<实验目录> ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MCD_EVAL = REPO / 'experiments/mcd_new_sequences/evaluate_trajectory.py'
M2DGR_EVAL = HERE / 'ate_eval.py'
DATA = Path('/home/anyverse/icra2027_runtime/datasets')

# 序列 -> (评测方式, 真值路径或 None, ate_eval 的序列名)
SPEC = {
    'ntu':    ('tum', DATA / 'mcd_ntu/ntu_day_01_gt_tum.txt', None),
    'kth':    ('tum', DATA / 'mcd_kth/kth_night_01_gt_tum.txt', None),
    'esc':    ('tum', DATA / 'fusionportable/escalator00_gt_tum.txt', None),
    'bd':     ('tum', DATA / 'fusionportable/building_day_gt_tum.txt', None),
    'cc05':   ('tum', DATA / 'spires_gt/2024-03-20-christ-church-05.txt', None),
    'hall02': ('m2dgr', DATA / 'm2dgr/hall_02_gt.txt', 'hall_02'),
    'hall04': ('m2dgr', DATA / 'm2dgr/hall_04_gt.txt', 'hall_04'),
    'ih':     ('closure', None, None),
    'big':    ('closure', None, None),
}
ARM_SUFFIX = ('_sc', '_rad', '_ba', '_widest', '_2nd', '_wide', '_loose')


def base_name(run: str) -> str:
    for s in ARM_SUFFIX:
        if run.endswith(s):
            return run[:-len(s)]
    return run


def metric(run_dir: Path, tum: Path) -> float | None:
    """Return the sequence's paper metric in cm, or None if not measurable."""
    kind, gt, seq = SPEC[base_name(run_dir.name)]
    if kind == 'closure':
        p = np.loadtxt(tum)[:, 1:4]
        return float(np.linalg.norm(p[0] - p[-1])) * 100.0
    if kind == 'tum':
        r = subprocess.run([sys.executable, str(MCD_EVAL), '--estimate', str(tum),
                            '--ground-truth', str(gt)],
                           capture_output=True, text=True)
        try:
            return json.loads(r.stdout)['position_error_m']['rmse'] * 100.0
        except Exception:
            return None
    r = subprocess.run([sys.executable, str(M2DGR_EVAL), str(tum), str(gt), seq],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.split('=')[1].split('cm')[0])
    except Exception:
        return None


def report(run_dir: Path) -> dict | None:
    j = run_dir / 'auto_repair_summary.json'
    if not j.is_file():
        return None
    s = json.loads(j.read_text())
    if s.get('status') != 'complete':
        return None
    balm = Path(s['terminal_output_dir'])
    pgo = balm.parent / balm.name.replace('_balm', '')
    return {
        'run': run_dir.name,
        'odom': metric(run_dir, run_dir / 'optimized_poses_tum.txt'),
        'pgo': metric(run_dir, pgo / 'optimized_poses_tum.txt'),
        'balm': metric(run_dir, balm / 'optimized_poses_tum.txt'),
        'n': s['active_constraint_count'],
        'rounds': s['rounds_completed'],
    }


def main() -> None:
    for root in (Path(a).expanduser() for a in sys.argv[1:]):
        print(f'\n=== {root}')
        print(f'  {"run":<12}{"odom":>9}{"PGO":>9}{"terminal":>10}{"n":>5}{"r":>4}')
        for d in sorted(root.iterdir()):
            if not d.is_dir() or base_name(d.name) not in SPEC:
                continue
            r = report(d)
            if r is None:
                print(f'  {d.name:<12}   (未完成)')
                continue
            f = lambda v: f'{v:9.1f}' if v is not None else '        -'
            print(f'  {r["run"]:<12}{f(r["odom"])}{f(r["pgo"])}{f(r["balm"])[1:]}'
                  f'{r["n"]:>5}{r["rounds"]:>4}')


if __name__ == '__main__':
    main()
