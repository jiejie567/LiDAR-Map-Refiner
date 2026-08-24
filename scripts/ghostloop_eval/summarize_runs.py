#!/usr/bin/env python3
"""把一批 auto_repair_headless 的跑汇总成论文主表要的形式。

放在仓库里而不是临时目录：这份脚本在 2026-08-06 因为 scratchpad 被清而丢过一次，
主表的每个数字都要靠它复现。

真值与评测器的对应关系不是可选项：
  * MCD ntu/kth、FusionPortable esc/bd 有完整 SE(3) 真值 -> evaluate_trajectory.py
  * M2DGR hall_02/hall_04 的真值是徕卡棱镜轨迹，四元数全零，只有位置。
    evaluate_trajectory.py 会直接崩，必须用 ate_eval.py —— 它还带每序列时钟偏移
    和棱镜杆臂拟合。
  * IH 与 big 没有外部真值。

用法：summarize_runs.py <实验根目录> [<对照根目录>]
      summarize_runs.py ~/icra2027_runtime/experiments/sigfix_20260806
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]
SLAM = TOOLS.parents[1]
PY = sys.executable
EVAL_SE3 = SLAM / 'experiments/mcd_new_sequences/evaluate_trajectory.py'
EVAL_PRISM = HERE / 'ate_eval.py'
DATA = Path.home() / 'icra2027_runtime/datasets'

# tag -> (真值文件, 评测器, ate_eval 需要的序列名)
GT = {
    'ntu':    (DATA / 'mcd_ntu/ntu_day_01_gt_tum.txt',            'se3',   None),
    'kth':    (DATA / 'mcd_kth/kth_night_01_gt_tum.txt',          'se3',   None),
    'esc':    (DATA / 'fusionportable/escalator00_gt_tum.txt',    'se3',   None),
    'bd':     (DATA / 'fusionportable/building_day_gt_tum.txt',   'se3',   None),
    'hall04': (DATA / 'm2dgr/hall_04_gt.txt',                     'prism', 'hall_04'),
    'hall02': (DATA / 'm2dgr/hall_02_gt.txt',                     'prism', 'hall_02'),
}


def ate_cm(traj: Path, tag: str) -> float | None:
    """ATE RMSE，单位 cm。没有真值或评测失败返回 None。"""
    if tag not in GT or not Path(traj).exists():
        return None
    gt, kind, seq = GT[tag]
    if not gt.exists():
        return None
    try:
        if kind == 'se3':
            out = subprocess.run([PY, str(EVAL_SE3), '--estimate', str(traj),
                                  '--ground-truth', str(gt)],
                                 capture_output=True, text=True, timeout=1800)
            return json.loads(out.stdout)['position_error_m']['rmse'] * 100
        out = subprocess.run([PY, str(EVAL_PRISM), str(traj), str(gt), seq],
                             capture_output=True, text=True, timeout=1800)
        # "ATE RMSE = 4.3 cm over 234 associated keyframes"
        return float(out.stdout.split('=')[1].split('cm')[0])
    except Exception:
        return None


def summary(session: Path):
    j = session / 'auto_repair_summary.json'
    if not j.exists():
        return None
    d = json.loads(j.read_text())
    if 'terminal_output_dir' not in d:        # 还在跑
        return None
    return d


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    tags = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / 'auto_repair_summary.json').exists())
    print(f'{"序列":10} {"里程计":>9} {"修复后":>9} {"削减":>9}  {"约束":>5} {"轮":>4} '
          f'{"提案":>5} {"配准":>5}  终止')
    print('-' * 84)
    for tag in tags:
        d = summary(root / tag)
        if d is None:
            print(f'{tag:10}   (未完成)')
            continue
        st = d.get('statistics', {})
        odo = ate_cm(root / tag / 'optimized_poses_tum.txt', tag)
        new = ate_cm(Path(d['terminal_output_dir']) / 'optimized_poses_tum.txt', tag)
        cut = f'{(odo-new)/odo*100:+.0f}%' if (odo and new) else '—'
        f = lambda v: f'{v:.1f}' if v is not None else '—'
        # 撞上限还是自然收敛：日志里最后一行说了算
        log = root / f'{tag}.log'
        end = '—'
        if log.exists():
            txt = log.read_text()
            end = 'round limit' if 'stopped at round limit' in txt else (
                'converged' if 'CONVERGED' in txt else '—')
        print(f'{tag:10} {f(odo):>9} {f(new):>9} {cut:>9}  '
              f'{d["active_constraint_count"]:>5} {d["rounds_completed"]:>4} '
              f'{st.get("proposed","—"):>5} {st.get("gicp_pairs","—"):>5}  {end}')


if __name__ == '__main__':
    main()
