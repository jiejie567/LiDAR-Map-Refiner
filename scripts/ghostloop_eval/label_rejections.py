#!/usr/bin/env python3
"""给被安全机制拦下的约束打真伪标签。

审稿人会问：说某条约束"声称 91 m 修正而链条只允许 28 m"，凭什么断定它是假的？
里程计链本身也可能错。只有在有测绘级真值的序列上，把约束声称的相对位姿与真值
给出的相对位姿直接比，才能下结论。

判据取的是外参无关量 —— 两端真值位置的距离。约束声称两帧相距 d_claim，真值说
它们相距 d_true；若二者差得远超真值噪声，这条约束就是假的，与标定无关。

用法：label_rejections.py <跑日志> <真值tum> <会话目录>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

BUDGET = re.compile(r'\[budget\] (\d+)->(\d+): correction ([\d.]+) m exceeds '
                    r'odometry budget ([\d.]+) m \(chain (\d+) m\)')
ACCEPT = re.compile(r'^\[(?:AutoSeed|Round\d+)\] (\d+)->(\d+): fit=([\d.]+) '
                    r'rmse=([\d.]+)')


def load_tum(path: Path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1:4]


def main() -> None:
    log, gt_path, session = (Path(sys.argv[1]), Path(sys.argv[2]),
                             Path(sys.argv[3]))
    est_t, est_p = load_tum(session / 'optimized_poses_tum.txt')
    gt_t, gt_p = load_tum(gt_path)

    # 关键帧索引 -> 真值位置：按时间戳最近邻关联
    idx = np.searchsorted(gt_t, est_t)
    idx = np.clip(idx, 1, len(gt_t) - 1)
    left = np.abs(gt_t[idx - 1] - est_t) < np.abs(gt_t[idx] - est_t)
    idx[left] -= 1
    dt = np.abs(gt_t[idx] - est_t)
    ok = dt < 0.05
    gt_at = np.where(ok[:, None], gt_p[idx], np.nan)

    def true_gap(a, b):
        if a >= len(gt_at) or b >= len(gt_at):
            return None
        pa, pb = gt_at[a], gt_at[b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            return None
        return float(np.linalg.norm(pa - pb))

    text = log.read_text(errors='ignore')
    rej = [(int(a), int(b), float(c), float(bd), float(ch))
           for a, b, c, bd, ch in BUDGET.findall(text)]
    acc = [(int(a), int(b), float(f), float(r))
           for a, b, f, r in ACCEPT.findall(text) if True]

    print(f'{"约束":>14} {"声称修正":>9} {"预算":>8} {"真值间距":>9}  判定')
    n_false = n_check = 0
    for a, b, corr, bd, ch in rej:
        g = true_gap(a, b)
        if g is None:
            print(f'{a:>6}->{b:<7} {corr:9.1f} {bd:8.1f} {"无真值":>9}')
            continue
        n_check += 1
        # 一条真回环两端在真值里应当靠得很近；相距几十米的不可能是同一地点
        verdict = 'FALSE (真值相距 %.0f m)' % g if g > 5.0 else 'plausible'
        if g > 5.0:
            n_false += 1
        print(f'{a:>6}->{b:<7} {corr:9.1f} {bd:8.1f} {g:9.1f}  {verdict}')
    if n_check:
        print(f'\n  预算拦下的约束中，{n_false}/{n_check} 条经真值确认为假 '
              f'（两端真实相距 > 5 m）')

    # 对照：被接受的约束在真值里应当都靠得很近
    good = [true_gap(a, b) for a, b, _, _ in acc]
    good = [g for g in good if g is not None]
    if good:
        arr = np.array(good)
        print(f'  被接受的 {len(arr)} 条：真值间距中位 {np.median(arr):.2f} m，'
              f'最大 {arr.max():.2f} m，超过 5 m 的 {(arr > 5).sum()} 条')


if __name__ == '__main__':
    main()
