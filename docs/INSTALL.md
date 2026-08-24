# Installation Guide | 安装说明

## Scope | 适用范围

This guide targets Ubuntu 20.04 and matches the versions currently used to test the Python-only GUI and optimizer.

本说明面向 Ubuntu 20.04，并与当前仓库已经验证过的纯 Python GUI 和优化器版本保持一致。

ROS and catkin are not used. Install the Python GTSAM 4.3 wrapper to enable optimization.

本工具不使用 ROS 或 catkin。安装 Python GTSAM 4.3 wrapper 后即可使用优化功能。

## Tested Versions | 已测试版本

| Dependency | Version | Notes |
|---|---:|---|
| Python | 3.10.16 | GUI environment |
| Open3D | 0.19.0 | GUI point-cloud viewer |
| PyQt5 | 5.15.10 | GUI |
| NumPy | 1.24.4 | GUI |
| SciPy | 1.14.1 | GUI / transforms |
| Matplotlib | 3.10.8 | Trajectory view |
| GCC | 9.4.0 | GTSAM wrapper build |
| CMake | 3.25.0 | GTSAM wrapper build |
| GTSAM | 4.3.0 | Python optimizer |

## 1. Clone the Repository | 1. 克隆仓库

```bash
cd ~/my_git
git clone git@github.com:JokerJohn/Mannual-Loop-Closure-Tools.git
cd Mannual-Loop-Closure-Tools
```

## 2. Install System Dependencies | 2. 安装系统依赖

Run the helper script to install the Python GUI and GTSAM wrapper build prerequisites.

运行辅助脚本安装 Python GUI 和 GTSAM wrapper 的系统构建依赖。

```bash
bash scripts/install_ubuntu20.sh
```

What the script installs:

脚本会安装以下依赖：

- build tools: `build-essential`, `cmake`, `git`, `pkg-config`
- system libraries: `libboost-all-dev`, `libtbb-dev`
- utilities: `python3-dev`, `python3-pip`, `python3-venv`

## 3. Fastest Path: Docker | 3. 最快路径：Docker

If you want the most reproducible setup with the fewest host-side dependency issues, use Docker.

如果你希望尽量避免宿主机环境问题，Docker 是最省事的方式。

```bash
cd ~/my_git/Mannual-Loop-Closure-Tools
docker build -t manual-loop-closure-tools:latest .
```

For GUI usage with X11:

图形界面可以这样启动：

```bash
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

More details:

更多细节：

- [DOCKER.md](DOCKER.md)

## 4. Create the Python Environment | 4. 创建 Python 环境

Recommended default path:

默认推荐方式：

```bash
make venv
source .venv/bin/activate
```

This uses the version-pinned [requirements.txt](../requirements.txt) and creates a local virtual environment under `.venv/`.

这条路径会使用带版本号约束的 [requirements.txt](../requirements.txt)，并在仓库根目录创建 `.venv/` 本地虚拟环境。

Manual equivalent:

手动等价命令：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Alternative with conda:

也可使用 conda：

```bash
conda env create -f environment.yml
conda activate manual-loop-closure
```

## 5. Install Python GTSAM 4.3 | 5. 安装 Python GTSAM 4.3

The Python optimizer was validated with the `GTSAM 4.3` line.

Python 优化器已使用 `GTSAM 4.3` 版本线完成验证。

Recommended helper path:

推荐的一键方式：

```bash
make gtsam-python
```

Or:

或者：

```bash
bash scripts/install_gtsam_python.sh
```

Dedicated installation notes:

请参考这里的专用安装说明：

- [INSTALL_GTSAM_PYTHON.md](INSTALL_GTSAM_PYTHON.md)

## 6. Launch the GUI (Python-first path) | 6. 启动 GUI（Python 主路径）

```bash
cd ~/my_git/Mannual-Loop-Closure-Tools
source .venv/bin/activate
python launch_gui.py --session-root /path/to/mapping_session
```

Or:

或者：

```bash
python launch_gui.py --g2o /path/to/pose_graph.g2o
```

## 7. Verify the Environment | 7. 检查环境

```bash
make env-check
```

This script prints Python package versions, Python GTSAM availability, common GTSAM CMake paths, and optimizer launch hints.

该脚本会打印 Python 包版本、Python GTSAM 可用性、常见 GTSAM CMake 路径以及优化器启动提示。

## 8. Expected Input Layout | 8. 输入目录结构

```text
mapping_session/
├── key_point_frame/
│   ├── 0.pcd
│   ├── 1.pcd
│   └── ...
├── pose_graph.g2o
└── optimized_poses_tum.txt
```

The tool also supports sessions where `pose_graph.g2o` and `optimized_poses_tum.txt` are stored under the latest timestamp subdirectory, while `key_point_frame/` remains at the session root.

工具也支持这样的 session：`pose_graph.g2o` 和 `optimized_poses_tum.txt` 位于最新时间戳子目录下，而 `key_point_frame/` 仍位于 session 根目录。

## Troubleshooting | 故障排查

### Open3D cannot be imported | 无法导入 Open3D

- Make sure you launched the GUI from the tested conda or venv environment.
- The GUI will try to re-launch itself with a compatible Python interpreter if possible.

- 请确认 GUI 在已安装依赖的 conda 或 venv 环境中启动。
- GUI 会尽量自动切换到可导入依赖的 Python 解释器。

### Python optimizer not found | 找不到 Python 优化器

- Install the Python GTSAM 4.3 wrapper with `make gtsam-python`.
- If needed, set `MANUAL_LOOP_OPTIMIZER_PYTHON=/absolute/path/to/python` to select an interpreter that can import GTSAM.

- 使用 `make gtsam-python` 安装 Python GTSAM 4.3 wrapper。
- 如有需要，可设置 `MANUAL_LOOP_OPTIMIZER_PYTHON=/absolute/path/to/python`，指定能够导入 GTSAM 的解释器。

### GTSAM not found by CMake | CMake 找不到 GTSAM

Export a CMake prefix before building:

构建前先导出 CMake 前缀：

```bash
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
```

## Related Documentation | 相关文档

- [Tool Manual / 工具说明](TOOL_README.md)
- [Python GTSAM 4.3 安装 / Python GTSAM 4.3](INSTALL_GTSAM_PYTHON.md)
- [Docker Guide / Docker 使用说明](DOCKER.md)
- [Project Overview / 项目总览](../README.md)
