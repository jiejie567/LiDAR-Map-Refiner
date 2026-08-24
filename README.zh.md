# LiDAR Map Refiner

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh.md"><strong>中文</strong></a>
</p>

<p align="center">
  <img src="assets/hero.svg" alt="LiDAR Map Refiner hero" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/JokerJohn/Mannual-Loop-Closure-Tools/actions/workflows/ci.yml">
    <img src="https://github.com/JokerJohn/Mannual-Loop-Closure-Tools/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/optimizer-Python-2E8B57.svg" alt="Python optimizer" />
  <a href="https://youtu.be/lemd4XfPSYY">
    <img src="https://img.shields.io/badge/YouTube-demo-FF0000?logo=youtube&logoColor=white" alt="YouTube demo" />
  </a>
  <a href="https://www.bilibili.com/video/BV1B6R4BGEvo/">
    <img src="https://img.shields.io/badge/Bilibili-demo-00A1D6?logo=bilibili&logoColor=white" alt="Bilibili demo" />
  </a>
  <img src="https://img.shields.io/badge/license-GPLv3-blue.svg" alt="GNU GPL v3.0 License" />
</p>

<p align="center">
  <strong>作者</strong>：
  <a href="https://github.com/JokerJohn">Xiangcheng HU</a>、
  <a href="https://github.com/zarathustr">Jin Wu</a>、
  <a href="https://github.com/Chen-Xieyuanli">Xieyuanli Chen</a><br/>
  <strong>联系邮箱</strong>：<a href="mailto:xhubd@connect.ust.hk">xhubd@connect.ust.hk</a>
</p>

面向激光雷达建图会话的离线回环搜索、鲁棒位姿图优化与表面保持精修工具。

## 视频教程

<p align="center">
  <a href="https://youtu.be/lemd4XfPSYY">
    <img src="https://img.youtube.com/vi/lemd4XfPSYY/hqdefault.jpg" alt="Manual Loop Closure Tools YouTube demo" width="80%" />
  </a>
</p>

## 项目简介

LiDAR Map Refiner 将自动与手动地图修复工作流整理为一个独立项目，包含：

- 用于轨迹检查和点云辅助闭环编辑的 PyQt 图形界面
- `Initial Loop Search -> Audited PGO -> Final Map Refinement` 生产流程
- 用于导出新位姿图和地图的 Python 优先离线优化后端
- 用于虚拟环境、Python GTSAM 检查、环境检查和截图生成的辅助脚本

它面向已经导出以下结果的建图任务：

- `pose_graph.g2o`
- `optimized_poses_tum.txt`
- `key_point_frame/*.pcd`

## 界面截图

<p align="center">
  <img src="assets/screenshots/edge-selected.png" alt="Edge selected screenshot" width="82%" />
</p>

## 工作流

<p align="center">
  <img src="assets/workflow.png" alt="Workflow diagram" width="100%" />
</p>

该图形界面支持轨迹检查、节点对和已有闭环边选择、source/target 点云预览、GICP 配准、手工新增或替换闭环约束、工作态位姿图管理，以及新优化地图导出。

## 功能动图

### 新增闭环边

<p align="center">
  <img src="assets/add_loopsx3.gif" alt="Add loop demo" width="82%" />
</p>

在确认 GICP 结果后新增一条手工闭环边。

### 替换已有闭环边

<p align="center">
  <img src="assets/replace_loopx3.gif" alt="Replace loop demo" width="82%" />
</p>

用更可靠的手工配准结果替换已有闭环边。

### 禁用已有闭环边

<p align="center">
  <img src="assets/disable_loop.gif" alt="Disable loop demo" width="82%" />
</p>

在重新优化前临时禁用已有闭环边。

## 主要功能

| 功能 | 说明 |
|---|---|
| 内嵌式 PyQt + Open3D viewer | 在一个工作流中查看轨迹与点云 |
| `Working` / `Original` 双轨迹对比 | 对比编辑图与基线图 |
| 手工新增、替换、禁用、恢复闭环边 | 显式控制 working session 中的闭环约束 |
| 点云视图中的 source 初值交互调整 | 在 GICP 前修正初始位姿 |
| 面向地面机器人的 Auto Yaw Sweep | 在最终配准前搜索 yaw 初值 |
| 导出新的 `g2o`、`TUM`、地图和轨迹 PCD | 在验证完成后输出优化结果 |
| 带撤销和改动跟踪的会话式图编辑 | 更可控地管理闭环边修改流程 |

## 快速开始

### 最快方式：Docker

```bash
cd ~/my_git/Mannual-Loop-Closure-Tools
make docker-build
xhost +local:docker
docker run --rm -it \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /path/to/mapping_session:/data/session \
  manual-loop-closure-tools:latest \
  python launch_gui.py --session-root /data/session
```

面向首次使用者的 Docker FAQ：

- 先执行显示权限放通：`xhost +local:docker`
- 把宿主机 session 目录挂载到 `/data/session`
- 宿主机输出会落在 `manual_loop_projects/`、`manual_loop_runs/` 和 `manual_loop_exports/`
- 无图形界面环境可以直接使用 Python 优化器 CLI
- 详细说明见：[docs/DOCKER.md](docs/DOCKER.md)

### 推荐本地方式：`requirements.txt` + `.venv`

```bash
cd ~/my_git/Mannual-Loop-Closure-Tools
make venv
source .venv/bin/activate
make gtsam-python
python launch_gui.py --session-root /path/to/mapping_session
```

对于大多数用户，到这里就够了。GUI 和优化器均为纯 Python，不依赖 ROS 或 catkin。

Python GTSAM 4.3 wrapper 现在可以通过仓库辅助脚本一键安装：

- `make gtsam-python`
- 或 `bash scripts/install_gtsam_python.sh`
- 详细说明见：[docs/INSTALL_GTSAM_PYTHON.md](docs/INSTALL_GTSAM_PYTHON.md)

也可以直接指定某个 `g2o` 文件：

```bash
python launch_gui.py --g2o /path/to/pose_graph.g2o
```

### 备选方式：conda

```bash
cd ~/my_git/Mannual-Loop-Closure-Tools
conda env create -f environment.yml
conda activate manual-loop-closure
python launch_gui.py --session-root /path/to/mapping_session
```

## Python 优先优化后端

当前验证过的手动闭环工作流只使用 Python GTSAM 后端。原 ROS1/catkin C++ fallback 已删除，避免这个独立工具被 ROS 工作区发现或参与编译。

## 关键模式与参数

生产流程不再使用编号式研究阶段名，统一定义为：

1. `Initial Loop Search（初始回环搜索）`：先做描述子检索，再经 GICP
   注册与准入审计；未通过审计的只是候选，不是图中约束。
2. `Audited PGO（经审计的位姿图优化）`：只优化已接受的活跃约束集。
3. `Final Map Refinement（最终地图精修）`：运行一次收敛驱动的双面 BALM。

`Duplicated-surface audit（重复表面审计）`是默认关闭的可选检查工具，
只检测同侧观测的重复平面支持，不是通用地图一致性证明、第四个修复阶段或回环提议器。
历史文件和研究脚本中的
`stage0`、`seed`、`auto_seed` 仅作兼容别名，均指初始回环搜索。

当前最重要的几个运行时选项如下：

- `Advanced` 中的 `Optimize`
  - 默认模式是 `Robust GNC-TLS`，一键 `Repair Map` 也固定使用该求解器。
  - `Accurate LM` 仍保留为 batch 风格的精确参考求解。
  - `Fast ISAM2` 仅保留给显式的旧版/延迟对照。
- `Registration` 中的 `TgtVoxel`
  - 环境预设：室内 `0.2 m`，室外 `0.4 m`
  - 时间窗默认值：`target_id ± 40` 帧
  - 影响预览和 GICP 使用的 target 子图密度。
  - 渲染只使用确定性限点视图；GICP 仍保留完整 target 点云。
- `Advanced` 中的 `MapVoxel`
  - 默认值：`0.1 m`
  - 只影响 `Export` 阶段最终全局地图的重建体素大小。

建议用法：

- GUI 的一键 `Repair Map` 直接调用同一个无头生产引擎：每条候选先做 trial-PGO，
  通过后才进入批量 PGO，并继续接受求解后 NIS/leave-one-out 审计；GUI 不再维护
  另一套自动准入顺序。
- 运行后同一按钮会变为 `Stop Repair`。引擎只在完整结束后原子发布结果协议，GUI
  校验原始输入与终态文件哈希后才采用；停止会终止隔离进程组及其 PGO/BALM 子进程，
  当前已加载结果保持不变。
- 生产版 Repair 在一次最终双面 BALM 后结束；地图不自洽区域只作可选诊断。旧的迭代提议流程
  已删除，不再提供从诊断区域生成回环的入口。
- 最终 BALM 按位姿更新量收敛或目标平台启发式停止，总更新上限为 20，运行时不读取真值；
  只有复现历史固定预算消融时才把 `BALM Iter` 设为 `2`。
- PGO/BALM 非法输出会 fail-closed 回到上一个有效结果；BALM 的 0.5 m/10° 平均运动
  阈值只是回滚保护，不是优化中的位姿先验。
- 若 session 没有记录环境，Initial Loop Search/Repair Map 会要求先明确选择 Indoor 或
  Outdoor；manifest 与编辑工程会在下次打开时恢复该选择。
- Initial Loop Search 候选验证会在后台对独立 GICP 初值做最多 8 路受控并发；完整 A/B 实测
  提速 1.71×–2.34×，约束和终端轨迹逐字节一致，Qt 主线程仍只负责界面更新。
- Optimize/BALM 写盘期间关闭窗口时，默认继续等待；Repair 流程可用
  `Stop Repair` 取消整次隔离运行，独立优化只有明确确认后才会强制终止。
- 修复与导出默认使用 `Robust GNC-TLS`
- `Accurate LM` 与 `Fast ISAM2` 仅用于受控对照
- 点击 `Add` 或 `Replace` 后，当前 GICP 候选会被消费，按钮会重新灰掉
- 如果你修改了初值、delta 或 target map 参数，需要重新运行 GICP 才能再次接受图改动

## 测试数据

你可以通过下面的链接下载示例建图结果，快速验证工具流程：

- Google Drive: https://drive.google.com/file/d/1iu3wO5YsiIl9ZuWlSlXz2fmu6ujTBw5J/view?usp=drive_link

## 当前测试环境

当前仓库内容在 Ubuntu 20.04 环境下使用如下依赖版本进行了测试。

| Ubuntu | Python | CMake | GCC / G++ |
|---|---|---|---|
| 20.04 | 3.10.16 | 3.25.0 | 9.4.0 |

| Open3D | PyQt5 | Qt | NumPy | SciPy | Matplotlib | GTSAM |
|---|---|---|---|---|---|---|
| 0.19.0 | 5.15.10 | 5.15.2 | 1.24.4 | 1.14.1 | 3.10.8 | 4.3.0 |

## 输出文件

工具现在把编辑状态、优化运行结果和最终导出清单分开保存：

| 位置 | 作用 | 主要文件 |
|---|---|---|
| `manual_loop_projects/<project_id>/` | 持久化编辑项目，便于恢复和复盘 | `project_state.json`, `execution.log`, `operations.jsonl` |
| `manual_loop_runs/<run_id>/` | 一次 `Optimize` 的真实输出 | `edited_input_pose_graph.g2o`, `manual_loop_constraints.csv`, `pose_graph.g2o`, `optimized_poses_tum.txt`, `pose_graph.png`, `manual_loop_report.json`, `run_context.json` |
| `manual_loop_exports/<export_id>/` | 轻量级最终导出清单，不再重复复制整包 run 数据 | `export_manifest.json`, `selected_run.txt`, `run` 软链接 |

恢复逻辑：

- `Load Session` 会恢复该 session 当前最新的编辑项目。
- `Resume Project` 可以通过选择某个 `project_state.json` 打开历史编辑项目。
- `Optimize` 会立即更新 working graph 和优化后的 TUM，但默认不重建整张地图。
- `Export` 会在写最终清单前按需生成 `scans.pcd` 和 `trajectory.pcd`。
- `Export` 不再复制整包优化结果，而是写一个指向目标 run 的清单。

## Python 与 C++ 一致性验证

Python backend 已在多组真实 session 上与原 C++ 实现做过一致性验证。这里保留历史验证数据，但 C++ 运行时已不再随仓库提供，也不会作为 fallback 使用。

| Session | Constraints | Python time [s] | C++ time [s] | TUM max t err [m] | TUM max r err [rad] | g2o max t err [m] | g2o max r err [rad] | Map points Py / C++ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `office_fs_fastlio_saved` | 1 | 10.193 | 2.096 | 5.20e-09 | 3.11e-06 | 6.85e-05 | 3.27e-06 | 4,850,749 / 4,850,749 |
| `floor34-1_fs_fastlio_saved` | 1 | 16.335 | 3.827 | 1.00e-09 | 3.56e-06 | 6.83e-05 | 4.50e-06 | 13,771,605 / 13,771,605 |
| `dr_tunnel_2026_01_24_145439` | 0 | 12.112 | 2.909 | 2.49e-08 | 2.87e-09 | 4.99e-04 | 1.05e-06 | 2,139,789 / 2,139,789 |

说明：

- `optimized_poses_tum.txt` 已在 `1e-9 m` 到 `1e-8 m` 的平移量级和 `1e-6 rad` 的旋转量级上与 C++ 对齐。
- `pose_graph.g2o` 的残余差异主要来自导出文本精度和四元数符号等价表示，而不是优化结果失配。

## 仓库结构

```text
Mannual-Loop-Closure-Tools/
├── README.md
├── README.zh.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── Dockerfile
├── requirements.txt
├── environment.yml
├── launch_gui.py
├── assets/
├── docs/
├── gui/
├── scripts/
└── wiki/
```

## 文档

- [English README](README.md)
- [安装说明](docs/INSTALL.md)
- [Python GTSAM 4.3 安装](docs/INSTALL_GTSAM_PYTHON.md)
- [Docker 使用说明](docs/DOCKER.md)
- [工具说明](docs/TOOL_README.md)
- [GitHub Wiki](https://github.com/JokerJohn/Mannual-Loop-Closure-Tools/wiki)
- [贡献说明](CONTRIBUTING.md)
- [版本记录](CHANGELOG.md)

## 开发辅助

```bash
make help
make gtsam-python
make check
make env-check
make optimizer-help
make docker-build
make assets SESSION_ROOT=/path/to/session
```

## 开源说明

本仓库聚焦于独立的手动闭环工具链，不包含完整的在线建图系统。

本项目源自并服务于更完整的 **MS-Mapping** 研究与代码体系：

- 项目地址: https://github.com/JokerJohn/MS-Mapping

如果你在学术工作中使用了本仓库，也请同时引用 MS-Mapping 论文：

```bibtex
@misc{hu2024msmapping,
      title={MS-Mapping: An Uncertainty-Aware Large-Scale Multi-Session LiDAR Mapping System},
      author={Xiangcheng Hu, Jin Wu, Jianhao Jiao, Binqian Jiang, Wei Zhang, Wenshuo Wang and Ping Tan},
      year={2024},
      eprint={2408.03723},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2408.03723},
}
```

## 许可

本独立仓库采用 GNU General Public License v3.0（GPLv3）。
