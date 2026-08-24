# Optimization Backends

[English](https://github.com/JokerJohn/Manual-Loop-Closure-Tools/wiki/Optimization-Backends) | [中文](https://github.com/JokerJohn/Manual-Loop-Closure-Tools/wiki/Optimization-Backends-zh)

## Python Optimizer

The standalone project uses the Python optimizer exclusively.

Why:

- simpler installation
- no ROS / catkin dependency
- consistent integration with the PyQt + Open3D GUI
- validated parity against the legacy C++ backend

## Optimize Modes

The Python backend now exposes two solve modes:

- `Fast ISAM2`
  - default in the GUI
  - intended for repeated working-graph updates during manual editing
- `Accurate LM`
  - batch-style reference solve
  - useful for parity checks or final confirmation before export

In the current GUI:

- `Fast ISAM2` lives in `Advanced -> Optimize`
- `MapVoxel` also lives in `Advanced` and defaults to `0.1 m`
- `TgtVoxel` lives in `Registration` and defaults to `0.1 m`

## Parameter Consistency

The Python backend uses this runtime-parameter precedence:

1. explicit CLI / GUI options
2. `runtime_params.yaml`
3. validated offline defaults

## Output Files

The Python backend exports:

- `pose_graph.g2o`
- `optimized_poses_tum.txt`
- `pose_graph.png`
- `manual_loop_report.json`

The full output layout is:

- `manual_loop_projects/<project_id>/`
  - edit-state and resume files
- `manual_loop_runs/<run_id>/`
  - actual optimizer outputs
- `manual_loop_exports/<export_id>/`
  - final-export manifest pointing to one selected run

`scans.pcd` and `trajectory.pcd` are built during `Export` by default so iterative graph editing stays responsive.
