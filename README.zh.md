# LiDAR Map Refiner

[English](README.md) | [中文](README.zh.md)

LiDAR Map Refiner 是面向已有激光点云地图的离线自动修复与精修软件，串联自动回环搜索、约束审计、位姿图优化和双面表面精修。

本项目在 **Hu Xiangcheng（Xiangcheng Hu / [JokerJohn](https://github.com/JokerJohn)）** 的
[Manual Loop Closure Tools](https://github.com/JokerJohn/Mannual-Loop-Closure-Tools)
基础上继续修改，主要扩展了自动地图修复、BALM 精修及薄结构双面表面保持功能。

本仓库由 [jiejie567](https://github.com/jiejie567) 继续维护。

## 自动修复流程

```text
已有建图会话
    -> Initial Loop Search 自动回环搜索
    -> Audited PGO 带约束审计的位姿图优化
    -> Final Map Refinement 双面 BALM 精修
    -> 导出优化轨迹与地图
```

1. **自动回环搜索**：通过描述子检索候选回环，再用 GICP 验证配准结果；候选通过准入检查后进入有效约束集合。
2. **带审计的位姿图优化**：执行鲁棒 PGO，通过试验性优化和求解后检查识别、撤回不一致的回环约束。
3. **双面地图精修**：在接受的轨迹上执行一次双面 BALM，按观测侧关联表面，保留薄结构两侧不同的表面身份。

图形界面的 **Repair Map** 和无头入口共用生产引擎。修复过程支持停止，失败结果在采用前被拒绝。可选的重影表面检查用于诊断。

## 主要功能

- 自动回环检索、GICP 验证与约束准入审计。
- 鲁棒位姿图优化、异常约束检查与撤回。
- 双面 BALM 和观测侧表面关联。
- 轨迹与点云检查、手工闭环编辑、撤销及会话恢复。
- 输入输出哈希校验与原子结果协议。
- 优化位姿图、TUM 轨迹、地图 PCD 和轨迹 PCD 导出。

## 输入数据

建图会话目录应包含：

```text
/path/to/mapping_session/
├── pose_graph.g2o
├── optimized_poses_tum.txt
└── key_point_frame/
    └── *.pcd
```

运行修复前，根据数据选择 Indoor 或 Outdoor 环境预设。

## 安装与启动

运行依赖包括 Python 3.10 及以上、PyQt5、Open3D、NumPy、SciPy 和 GTSAM。
依赖版本见 [requirements.txt](requirements.txt)。

```bash
git clone https://github.com/jiejie567/LiDAR-Map-Refiner.git
cd LiDAR-Map-Refiner
make venv
source .venv/bin/activate
make gtsam-python
python launch_gui.py --session-root /path/to/mapping_session
```

`make gtsam-python` 会编译并安装 Python GTSAM 包，所需编译环境见
[GTSAM 安装说明](docs/INSTALL_GTSAM_PYTHON.md)。

命令行自动修复入口：

```bash
python scripts/ghostloop_eval/auto_repair_headless.py \
  /path/to/mapping_session production indoor
```

处理室外会话时，将 `indoor` 改为 `outdoor`。

## 图形界面操作与输出

1. 加载建图会话并选择环境。
2. 检查原始轨迹和点云。
3. 点击 **Repair Map**，检查修复后的轨迹与约束。
4. 使用 **Working / Original** 对比结果，需要时编辑个别闭环。
5. 点击 **Export** 重建并导出最终地图。

编辑工程、优化运行结果和导出清单分别保存在会话目录下的
`manual_loop_projects/`、`manual_loop_runs/` 和 `manual_loop_exports/`。
采用输出前应检查对应运行报告。

## 文档

- [安装说明](docs/INSTALL.md)
- [Python GTSAM 安装](docs/INSTALL_GTSAM_PYTHON.md)
- [Docker 部署](docs/DOCKER.md)
- [工具使用说明](docs/TOOL_README.md)
- [定向表面关联方法](docs/ORIENTED_SURFACE_METHOD.md)

## 原项目致谢与许可

原手工闭环编辑界面及基础设施来自 Hu Xiangcheng 的 Manual Loop Closure Tools。
感谢原项目作者 Xiangcheng Hu、Jin Wu、Xieyuanli Chen 及其他贡献者。

本仓库采用 **GNU GPL v3.0**，许可全文见 [LICENSE](LICENSE)，原项目的版权与许可声明保留。
