#!/usr/bin/env python3
"""Is the lidar-IMU extrinsic in this session's config self-consistent?

Read the gravity estimate in the SENSOR frame and it will mostly tell you how
the rig was carried -- a handheld unit held at 25 degrees looks alarming and is
perfectly fine. Rotate the same vector into the world frame with the estimated
attitude and it must land on [0, 0, 1]; a fixed offset there is an extrinsic
error, and the spread says whether the offset is systematic or noise.

    ./check_extrinsic.py <session_dir>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

session = Path(sys.argv[1])
grav = np.genfromtxt(session / 'scan_context_gravity.csv', delimiter=',', names=True)
up_body = np.stack([grav['up_x'], grav['up_y'], grav['up_z']], axis=1)
odo = np.loadtxt(session / 'optimized_poses_tum.txt')
n = min(len(up_body), len(odo))
rot = Rotation.from_quat(odo[:n, 4:8]).as_matrix()
up_world = np.einsum('nij,nj->ni', rot, up_body[:n])
up_world /= np.linalg.norm(up_world, axis=1, keepdims=True)

body_tilt = np.degrees(np.arccos(np.clip(abs(np.median(up_body[:n, 2])), -1, 1)))
world = np.median(up_world, axis=0)
world_tilt = np.degrees(np.arccos(np.clip(world[2], -1, 1)))
spread = np.degrees(np.std(np.arccos(np.clip(up_world[:, 2], -1, 1))))
print(f'sensor-frame tilt : {body_tilt:5.1f} deg   (how the rig was carried/mounted)')
print(f'world-frame tilt  : {world_tilt:5.1f} deg   spread {spread:4.1f} deg')
# A real extrinsic error bends the trajectory out of plane, so the honest test
# is its consequence: how much z the solution gains per metre travelled. Read
# that first -- the tilt alone over-reports. A rig whose accelerometer carries
# a genuine bias shows a steady several-degree tilt (tight spread) while the
# filter estimates the bias away and the odometry stays healthy; both
# mapping_big (5.2 deg) and Oxford Spires christ-church-05 (5.4 deg, spread
# 0.2) tripped the old 5 deg bound while drifting 0.17 deg and 0.02% of path
# respectively. A tilt that matters comes with a climb that matters.
p = odo[:n, 1:4]
step = np.linalg.norm(np.diff(p, axis=0), axis=1)
arc = np.concatenate([[0.0], np.cumsum(step)])
path = arc[-1]
slope = np.polyfit(arc, p[:, 2], 1)[0]          # metres of z per metre travelled
climb = np.degrees(np.arcsin(np.clip(slope, -1, 1)))
print(f'implied climb     : {climb:5.2f} deg   ({path:.0f} m of path)')
ok = abs(climb) < 1.0 and spread < 3
print('verdict           :', 'extrinsic consistent' if ok
      else 'SUSPECT -- the solution climbs; gravity and attitude disagree')
if not ok:
    print()
    print('Next step: find the UP direction in each sensor frame and compare.')
    print('  IMU frame   : mean the accelerometer over the run. At rest it')
    print('                reads the SUPPORT force, so its mean points UP,')
    print('                not down -- this sign is easy to get backwards.')
    print('  lidar frame : fit the ground plane in a few raw scans and take')
    print('                the normal that points away from the ground.')
    print('Both vectors must denote the same physical direction before the')
    print('rotation between them means anything. Comparing one frame\'s up')
    print('against the other\'s down yields a spurious 180 degrees; acting on')
    print('that reading once made a 23 km sequence 46 m worse.')
    print('If the two directions agree, the extrinsic is not the problem and')
    print('the tilt is a symptom -- look at per-point timestamps and de-skew.')
