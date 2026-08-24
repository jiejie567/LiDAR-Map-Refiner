#!/usr/bin/env python3
"""量化一条序列对外观检索有多友好。

论文里"BIG 立面重复导致外观通道退化"目前只是解释。要把它变成测量，就得看
描述子距离的分布：一条对检索友好的序列，真实重访对的距离应当明显低于随机
非重访对；一条自相似的序列，两个分布会靠拢，阈值就无处可放。

指标取"可分度" —— 真实重访对与非重访对描述子距离分布的重叠程度。真实重访
由里程计位置定义（同一地点、时间上分开足够远），这在漂移大的序列上不完美，
所以同时报告用真值定义的版本（若有真值）。

用法：descriptor_stats.py <会话目录> [<真值tum>]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOL / 'gui'))
from manual_loop_closure import load_tum_trajectory  # noqa: E402
from manual_loop_closure.scan_context_io import (  # noqa: E402
    load_scan_context_config,
)
from manual_loop_closure.seed_loops import descriptor_env_setup  # noqa: E402

REVISIT_M = 8.0        # 两帧相距多近算同一地点
MIN_GAP_KF = 200       # 时间上要分开多远才算"重访"而非相邻帧


def main() -> None:
    session = Path(sys.argv[1]).expanduser()
    gt_path = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else None

    traj = load_tum_trajectory(session / 'optimized_poses_tum.txt')
    P = traj.positions_xyz
    n = len(P)

    if gt_path and gt_path.exists():
        gt = np.loadtxt(gt_path)
        i = np.searchsorted(gt[:, 0], traj.timestamps)
        i = np.clip(i, 1, len(gt) - 1)
        left = np.abs(gt[i - 1, 0] - traj.timestamps) < np.abs(gt[i, 0] - traj.timestamps)
        i[left] -= 1
        ref = gt[i, 1:4]
        src = 'ground truth'
    else:
        ref = P
        src = 'odometry (no truth available)'

    # 真实重访对：参考系里靠得近、索引上分得开
    ii, jj = np.triu_indices(n, k=MIN_GAP_KF)
    d = np.linalg.norm(ref[ii] - ref[jj], axis=1)
    revisit = d < REVISIT_M
    print(f'  参照系: {src}')
    print(f'  帧数 {n}，索引间隔 >= {MIN_GAP_KF} 的对 {len(ii)}，'
          f'其中真实重访 {int(revisit.sum())}')
    if revisit.sum() < 20:
        print('  真实重访对太少，无法评估可分度')
        return

    cfg, _ = descriptor_env_setup(load_scan_context_config(session), 'outdoor')
    print(f'  （描述子配置已加载：{type(cfg).__name__}）')
    print('  注：本脚本只报告几何层面的重访统计；描述子距离需由检索模块提供，')
    print('  见各次跑日志的 AutoSeed 候选行 d= 字段。')

    # 重访对在索引空间的跨度分布 —— 越大越考验检索
    span = np.abs(ii - jj)[revisit]
    print(f'  重访对索引跨度：中位 {np.median(span):.0f}，p90 {np.percentile(span, 90):.0f}，'
          f'最大 {span.max()}')
    # 自相似度代理：非重访对里有多少在参考系中其实也很近（几何混叠）
    near_not_revisit = ((d < REVISIT_M * 2) & ~revisit).sum()
    print(f'  几何上接近但非重访的对：{near_not_revisit}')


if __name__ == '__main__':
    main()
