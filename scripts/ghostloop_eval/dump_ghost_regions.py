#!/usr/bin/env python3
"""跑论文自己的鬼影检测器，把区域连同几何一起存成 JSON。

放在仓库里而不是临时目录：teaser 图与室内普查那一行都要靠它复现，2026-08-06
因为临时目录被清而丢过一次。

输出每个区域的 center_xyz / normal / separation_m / point_count /
layer_a_keyframes / layer_b_keyframes / suggested_pair —— 后两项就是这处不一致
所蕴含的"缺失回环连接哪两段轨迹"。

参数取自会话自己的 auto_repair_summary.json（若存在），否则用室内预设，
这样图上标的区域和流水线当时诊断出的是同一批。

用法：dump_ghost_regions.py <tum> <out.json> [<会话目录，取 BALM 参数>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOL / 'gui'))
sys.path.insert(0, str(TOOL / 'gui/manual_loop_closure/python_optimizer'))
from manual_loop_closure import RegistrationWorkspace, load_tum_trajectory  # noqa: E402
from balm import BalmParams, detect_ghost_regions, prepare_local_cloud  # noqa: E402

KF_DEFAULT = Path.home() / 'icra2027_runtime/manual_loop/original/key_point_frame'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tum", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("session", type=Path, nargs="?")
    parser.add_argument("--root-voxel-size", type=float)
    parser.add_argument("--balm-report", type=Path)
    args = parser.parse_args()
    tum = args.tum.expanduser()
    out = args.out.expanduser()
    session = args.session.expanduser() if args.session else None

    kf = (session / 'key_point_frame') if session else KF_DEFAULT
    params_kw = dict(root_voxel_size=1.0, max_layer=3, min_voxel_points=20,
                     min_poses_per_voxel=2, plane_thickness=0.05,
                     plane_spread_ratio=2.0, downsample_leaf=0.2,
                     max_range=80.0, max_points_per_voxel_pose=40)
    if session and (session / 'auto_repair_summary.json').exists():
        d = json.loads((session / 'auto_repair_summary.json').read_text()).get('balm', {})
        for k in list(params_kw):
            if k in d:
                params_kw[k] = d[k]
    if args.balm_report is not None:
        report_payload = json.loads(args.balm_report.read_text())
        d = report_payload.get('params', {})
        for k in list(params_kw):
            if k in d:
                params_kw[k] = d[k]
        max_observation_range = float(d.get('max_observation_range', 20.0))
    else:
        max_observation_range = 20.0
    if args.root_voxel_size is not None:
        if args.root_voxel_size <= 0.0:
            raise SystemExit('--root-voxel-size must be positive')
        params_kw['root_voxel_size'] = float(args.root_voxel_size)

    traj = load_tum_trajectory(tum)
    ws = RegistrationWorkspace(kf, traj)
    clouds = []
    for i in range(traj.size):
        clouds.append(prepare_local_cloud(
            ws.load_local_points_uncached(int(i)), BalmParams(**params_kw)
        ).astype(np.float64))
        if (i + 1) % 500 == 0:
            print(f'  loaded {i+1}/{traj.size}', flush=True)

    regions = detect_ghost_regions(clouds, traj.transforms_world_sensor,
                                   BalmParams(**params_kw),
                                   max_regions=100000,
                                   max_observation_range=max_observation_range,
                                   log_fn=None)
    for r in regions:
        for k in ('layer_a_keyframes', 'layer_b_keyframes', 'suggested_pair'):
            r[k] = [int(v) for v in r[k]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(regions, indent=1) + '\n')
    sev = sum(r['point_count'] * r.get('separation_typical_m', r['separation_m'])
              for r in regions)
    print(f'{len(regions)} regions, severity {sev:.0f} -> {out}')


if __name__ == '__main__':
    main()
