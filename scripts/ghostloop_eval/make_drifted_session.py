"""Build a corrupted copy of a keyframe session: inject calibrated odometry
drift (yaw-rate bias + translation scale), reintegrate, write TUM + odometry
g2o. Keyframes/gravity are shared via symlink/copy."""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
YAW_BIAS_DEG_PER_M = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
TRANS_SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 1.01

data = np.loadtxt(SRC / 'optimized_poses_tum.txt')
ts, pos, quat = data[:, 0], data[:, 1:4], data[:, 4:8]
rots = Rotation.from_quat(quat).as_matrix()
n = len(ts)

T = np.tile(np.eye(4), (n, 1, 1))
T[:, :3, :3] = rots
T[:, :3, 3] = pos

# corrupted chain: start at same first pose, integrate biased increments
Tc = [T[0]]
for i in range(1, n):
    inc = np.linalg.inv(T[i - 1]) @ T[i]
    step = float(np.linalg.norm(inc[:3, 3]))
    bias_yaw = math.radians(YAW_BIAS_DEG_PER_M) * step
    Rb = Rotation.from_euler('z', bias_yaw).as_matrix()
    inc_c = inc.copy()
    inc_c[:3, :3] = Rb @ inc[:3, :3]
    inc_c[:3, 3] = TRANS_SCALE * (Rb @ inc[:3, 3])
    Tc.append(Tc[-1] @ inc_c)
Tc = np.stack(Tc)

DST.mkdir(parents=True, exist_ok=True)
kf_link = DST / 'key_point_frame'
if not kf_link.exists():
    kf_link.symlink_to((SRC / 'key_point_frame').resolve())
for name in ('runtime_params.yaml', 'scan_context_gravity.csv'):
    if (SRC / name).exists():
        shutil.copy2(SRC / name, DST / name)

quat_c = Rotation.from_matrix(Tc[:, :3, :3]).as_quat()
with (DST / 'optimized_poses_tum.txt').open('w') as f:
    for i in range(n):
        x, y, z = Tc[i, :3, 3]
        qx, qy, qz, qw = quat_c[i]
        f.write(f'{ts[i]:.9f} {x:.9f} {y:.9f} {z:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n')

INFO = '1000 0 0 0 0 0 1000 0 0 0 0 100 0 0 0 4000 0 0 4000 0 4000'
with (DST / 'pose_graph.g2o').open('w') as f:
    for i in range(n):
        x, y, z = Tc[i, :3, 3]
        qx, qy, qz, qw = quat_c[i]
        f.write(f'VERTEX_SE3:QUAT {i} {x:.12f} {y:.12f} {z:.12f} '
                f'{qx:.12f} {qy:.12f} {qz:.12f} {qw:.12f}\n')
    for i in range(1, n):
        rel = np.linalg.inv(Tc[i - 1]) @ Tc[i]
        q = Rotation.from_matrix(rel[:3, :3]).as_quat()
        x, y, z = rel[:3, 3]
        f.write(f'EDGE_SE3:QUAT {i-1} {i} {x:.12f} {y:.12f} {z:.12f} '
                f'{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} {INFO}\n')

drift = np.linalg.norm(Tc[:, :3, 3] - pos, axis=1)
print(f'built {DST}: {n} kf, injected drift max={drift.max():.2f} m, end={drift[-1]:.2f} m')
