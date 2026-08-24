# 优化后端

[English](https://github.com/JokerJohn/Manual-Loop-Closure-Tools/wiki/Optimization-Backends) | [中文](https://github.com/JokerJohn/Manual-Loop-Closure-Tools/wiki/Optimization-Backends-zh)

## Python 优化器

独立项目只使用 Python 优化后端。

原因：

- 安装更简单
- 不依赖 ROS / catkin
- 与 PyQt + Open3D GUI 的集成更自然
- 已与 legacy C++ backend 做过一致性验证

## 优化模式

Python backend 现在提供两种求解模式：

- `Fast ISAM2`
  - GUI 默认模式
  - 适合多轮手工编辑时快速更新 working graph
- `Accurate LM`
  - batch 风格的参考求解
  - 适合 parity 对照或导出前最终确认

当前 GUI 中：

- `Fast ISAM2` 位于 `Advanced -> Optimize`
- `MapVoxel` 也位于 `Advanced`，默认值 `0.1 m`
- `TgtVoxel` 位于 `Registration`，默认值 `0.1 m`

## 参数一致性

Python backend 遵循以下参数优先级：

1. 显式 CLI / GUI 参数
2. `runtime_params.yaml`
3. 已验证的离线默认值

## 导出文件

Python backend 会导出：

- `pose_graph.g2o`
- `optimized_poses_tum.txt`
- `pose_graph.png`
- `manual_loop_report.json`

完整目录结构是：

- `manual_loop_projects/<project_id>/`
  - 编辑状态与恢复文件
- `manual_loop_runs/<run_id>/`
  - 真正的优化输出
- `manual_loop_exports/<export_id>/`
  - 指向某次 run 的最终导出清单

默认情况下，`scans.pcd` 和 `trajectory.pcd` 会在 `Export` 阶段生成，这样多轮图编辑时更流畅。
