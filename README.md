# LiDAR Map Refiner

[English](README.md) | [中文](README.zh.md)

LiDAR Map Refiner repairs existing LiDAR maps through automatic loop search,
audited pose-graph optimization, and double-sided surface refinement.

This project continues development from **Xiangcheng Hu (Hu Xiangcheng /
[JokerJohn](https://github.com/JokerJohn))**'s
[Manual Loop Closure Tools](https://github.com/JokerJohn/Mannual-Loop-Closure-Tools).
The extensions in this repository focus on automatic map repair, BALM refinement,
and preserving the two sides of thin structures.

Maintained in this repository by [jiejie567](https://github.com/jiejie567).

## Repair workflow

```text
Mapping session
    -> Initial Loop Search
    -> Audited PGO
    -> Final Map Refinement
    -> Export optimized trajectory and map
```

1. **Initial Loop Search** retrieves candidate loops with descriptors and verifies
   them with GICP. Candidates enter the active constraint set after admission checks.
2. **Audited PGO** applies robust pose-graph optimization, with trial optimization
   and post-solve checks to detect and withdraw inconsistent loop constraints.
3. **Final Map Refinement** runs double-sided BALM once on the accepted trajectory.
   Observation-side association helps preserve distinct surfaces on opposite
   sides of thin structures.

The GUI's **Repair Map** action and the headless entry point share the production
engine. Repair can be stopped; failed results are rejected before adoption.
The optional duplicated-surface audit provides diagnostic information.

## Features

- Automatic loop retrieval, GICP verification, and constraint admission audit.
- Robust PGO with constraint checks and withdrawal.
- Double-sided BALM with observation-side surface association.
- GUI inspection, manual loop editing, undo, and session resume.
- Input/output hash checks and an atomic result contract.
- Export of optimized `g2o`, TUM trajectories, map PCD, and trajectory PCD.

## Input

Provide a mapping-session directory containing:

```text
/path/to/mapping_session/
├── pose_graph.g2o
├── optimized_poses_tum.txt
└── key_point_frame/
    └── *.pcd
```

Choose the appropriate indoor or outdoor environment preset before repair.

## Install and launch

The software uses Python 3.10 or newer, PyQt5, Open3D, NumPy, SciPy, and GTSAM.
See [requirements.txt](requirements.txt) for dependency pins.

```bash
git clone https://github.com/jiejie567/LiDAR-Map-Refiner.git
cd LiDAR-Map-Refiner
make venv
source .venv/bin/activate
make gtsam-python
python launch_gui.py --session-root /path/to/mapping_session
```

`make gtsam-python` builds and installs the Python GTSAM wrapper; see the
[installation guide](docs/INSTALL_GTSAM_PYTHON.md) for prerequisites.

For automatic repair from the command line:

```bash
python scripts/ghostloop_eval/auto_repair_headless.py \
  /path/to/mapping_session production indoor
```

Use `outdoor` in place of `indoor` for an outdoor session.

## GUI use and outputs

1. Load the mapping session and select its environment.
2. Inspect the original trajectory and point clouds.
3. Run **Repair Map** and review the resulting trajectory and constraints.
4. Use **Working / Original** to compare results; edit individual loops when needed.
5. Use **Export** to rebuild and save the final map.

Edit projects, optimization runs, and export manifests are stored under
`manual_loop_projects/`, `manual_loop_runs/`, and `manual_loop_exports/`
within the session. Check the run report before adopting an output.

## Documentation

- [Installation](docs/INSTALL.md)
- [Python GTSAM setup](docs/INSTALL_GTSAM_PYTHON.md)
- [Docker setup](docs/DOCKER.md)
- [Tool manual](docs/TOOL_README.md)
- [Oriented-surface method](docs/ORIENTED_SURFACE_METHOD.md)

## Upstream acknowledgment and license

The original manual loop-editing interface and supporting infrastructure come
from Xiangcheng Hu's Manual Loop Closure Tools. We acknowledge its original
authors Xiangcheng Hu, Jin Wu, Xieyuanli Chen, and other upstream contributors.

This repository is distributed under **GNU GPL v3.0**; see [LICENSE](LICENSE).
Upstream copyright and license notices are retained.
