# LiDAR Map Refiner Manual | LiDAR Map Refiner 工具说明

## What This Tool Does | 工具作用

LiDAR Map Refiner provides an offline workflow for loop search, robust pose-graph
optimization, surface-preserving refinement, manual inspection, and map export.

LiDAR Map Refiner 提供离线回环搜索、鲁棒位姿图优化、表面保持精修、人工检查与地图导出工作流。

Typical use cases:

典型用途包括：

- difficult automatic loop closures in repetitive indoor scenes
- replacing a weak existing loop edge with a manually validated one
- adding several manual loop edges and optimizing them together

- 室内重复结构场景中的自动闭环困难
- 用手工验证后的结果替换质量较差的已有闭环边
- 连续添加多条手工闭环边并统一优化

## Main UI Areas | 主要界面区域

| Area | Description | 中文说明 |
|---|---|---|
| Trajectory panel | 2D graph view for node and edge selection | 2D 位姿图，用于选节点和选边 |
| Point Cloud Review | Embedded point-cloud viewer for preview, manual alignment, and GICP inspection | 内嵌点云窗口，用于预览、手动初值调整和 GICP 检查 |
| Control panel | Summary, delta, registration settings, and actions | 参数、摘要、手工增量和操作按钮 |
| Graph Changes | Session-based list of accepted graph edits | 工作会话中的图改动列表 |
| Execution Log | Runtime log panel | 运行日志面板 |

## Workflow | 基本流程

### A. Add a new manual loop edge | A. 新增手工闭环边

1. Load a session.
2. In the trajectory panel, switch to `Pick Nodes`.
3. Select two nodes. The tool normalizes them to `source_id > target_id`.
4. Preview the point clouds.
5. Adjust the source pose if needed.
6. Run `GICP` or `Auto Yaw Sweep`.
7. If the result is good, click `Add Manual`.
8. After collecting the desired changes, click `Optimize`.
9. Optionally click `BALM` to refine the optimized trajectory with plane bundle
   adjustment ([hku-mars/BALM](https://github.com/hku-mars/BALM) style). It extracts
   planar patches from the aggregated map via adaptive voxelization and tightens all
   keyframe poses jointly. Root voxel size (~1 m indoor, 2-4 m outdoor) and iteration
   count live in the Advanced tab. `Double-sided BALM` is enabled by default and
   separates oppositely observed faces before coarse fitting; disable it only for
   the single-surface ablation. The refined result is written to a new
   `manual_loop_runs/<timestamp>_balm/` directory and becomes the working trajectory;
   `Undo` restores the pre-BALM state and `Export` picks up the refined run.

1. 加载 session。
2. 在轨迹图切换到 `Pick Nodes`。
3. 选择两个节点，工具会自动规范为 `source_id > target_id`。
4. 查看点云预览。
5. 如有需要，手动调整 source 初值。
6. 运行 `GICP` 或 `Auto Yaw Sweep`。
7. 若结果满意，点击 `Add Manual`。
8. 收集完需要的改动后，点击 `Optimize`。
9. 可选：点击 `BALM`，用平面束调整（参考 [hku-mars/BALM](https://github.com/hku-mars/BALM)）
   进一步精化优化后的轨迹。它对聚合地图做自适应体素平面提取，并联合收紧所有关键帧位姿。
   根体素尺寸（室内约 1 m，室外 2-4 m）与迭代次数在 Advanced 页可调。默认开启
   `Double-sided BALM`，在粗拟合前分离反向观测的两个物理表面；仅在单面模型消融中关闭。
   精化结果写入新的
   `manual_loop_runs/<时间戳>_balm/` 目录并成为工作轨迹；`Undo` 可回退到 BALM 之前的状态，
   `Export` 会直接使用精化后的 run。

### B. Replace an existing loop edge | B. 替换已有闭环边

1. Switch to `Pick Edges`.
2. Select an existing loop edge.
3. Inspect or manually align the source cloud.
4. Run `GICP`.
5. Click `Replace Edge`.
6. The working graph disables the old edge and uses the new manual result as its replacement.

1. 切换到 `Pick Edges`。
2. 选择一条已有闭环边。
3. 检查或手动调整 source 点云。
4. 运行 `GICP`。
5. 点击 `Replace Edge`。
6. 工作图会禁用旧边，并使用新的手工结果作为替换约束。

### C. One-click repair | C. 一键修复

`Repair Map` runs Initial Loop Search → Audited PGO → Final Map Refinement,
then stops. Initial Loop Search means descriptor retrieval plus GICP admission
audit. Every candidate must pass an isolated trial-PGO before joining the batch
PGO, and the batch result is audited again with NIS/leave-one-out. The GUI
invokes this canonical headless production engine instead of maintaining a
second automatic admission implementation. Final Map Refinement is one
double-sided BALM run. Optional residual
duplicated-surface regions are inspection-only; they do not propose loops or
decide whether an initial loop or BALM output is retained.
Its PGO path is fixed to the cross-dataset production profile (diagonal GICP
factors, GNC-TLS, and conservative global information scaling), independent of
the manual expert widgets. A session without a recorded environment must choose
Indoor or Outdoor explicitly before the run starts.
Optimizer output is adopted only after pose-count, timestamp, and finite-value
validation. The engine atomically publishes a run-ID result contract; the GUI
also verifies hashes of the immutable input TUM/G2O, terminal TUM/G2O/constraint
CSV, and proposal ledger before adoption. Cancelling `Repair Map` terminates
the isolated process group and leaves the currently loaded result unchanged.
The terminal BALM pass additionally has catastrophic mean-motion
guards (0.5 m translation, 10° rotation); these are output rollback checks,
not pose priors inside BALM.
A large session may take several minutes;
the progress bar advances across stages without resetting while point-cloud Preview,
session loading, GICP, optimization, BALM, and final Export run outside the GUI event loop.
Independent GICP initial guesses use a bounded worker pool (up to eight) inside
that background stage; two complete Repair A/B runs measured 1.71x and 2.34x
speedups with byte-identical constraints and terminal trajectories.
Repeated Preview parameter edits are coalesced so only the newest settings are
rendered. Manual source-cloud dragging keeps the target submap resident in the
renderer, and high-frequency camera/resize events are frame-coalesced to avoid
apparent GUI hangs on large point clouds. If Initial Loop Search finds no
candidate, the tool still creates a zero-change optimizer baseline before BALM,
so a freshly loaded clean session can still be refined. The historical
diagnosis-driven loop-proposal cycle has been removed; map-inconsistency output
is inspection-only.

The final BALM pass stops on pose-update convergence or an objective-plateau
heuristic, with 20 total updates as a runtime cap. It does not inspect ground
truth when deciding when to stop. Only simultaneous translation and rotation
update convergence is reported as `pose_converged`; `objective_plateau` is a
practical stop, not a numerical convergence certificate. Process completion,
termination details, and the independent production adoption sanity gate remain
separate. The historical two-update run is reproducible with
`BALM Iter`/`--max-iterations=2`.

`Repair Map` 会依次运行初始回环搜索 → 经审计的位姿图优化 → 最终地图精修，然后结束。
初始回环搜索包含描述子检索和 GICP 准入审计；每条候选必须先通过隔离的 trial-PGO，
才会进入批量 PGO，批量解随后还会接受 NIS/leave-one-out 审计。GUI 直接调用这一份
无头生产引擎，不再维护第二套自动准入实现。最终地图精修是一次双面 BALM。
可选的残余地图不自洽区域只作检查输出，不会提议回环，也不决定初始回环或 BALM 结果是否保留。
其 PGO 固定使用跨数据集生产配置（GICP 对角因子、GNC-TLS 与保守全局信息缩放），
不受手动高级控件漂移影响；没有记录环境的 session 必须先明确选择 Indoor/Outdoor。
引擎完整结束后才会原子发布带 run ID 的结果协议；GUI 采用前会校验原始 TUM/G2O、
终态 TUM/G2O/约束 CSV 与 proposal ledger 哈希。取消 `Repair Map` 会终止整个隔离
进程组，并保持当前已加载结果不变。优化输出还必须通过位姿数、时间戳与有限值校验；
最终 BALM 另有平均运动灾难门
（平移 0.5 m、旋转 10°），只负责失败回滚，不是加入 BALM 的位姿先验。
大规模 session 可能需要数分钟；Session 加载、点云 Preview、GICP、优化、BALM 与最终 Export 均不会阻塞
GUI 事件循环，连续修改 Preview 参数时只会补算最新一次，进度条会按阶段单调前进，不再回退。
同一候选的独立 GICP 初值会在后台阶段使用最多 8 个受控 worker；两套完整 Repair A/B
实测提速 1.71× 和 2.34×，终端约束与轨迹均逐字节一致。
手动拖动源点云时目标子图会留在显存中，不再逐鼠标事件整幅重传；相机与窗口缩放渲染也会
按帧合并，降低大点云下的假死现象。
即使初始回环搜索没找到候选，也会先生成零改动优化基线再运行
BALM，因此刚加载的干净 session 也可以直接优化。历史重影提议循环已经删除；地图
不自洽输出仅供检查。

最终 BALM 在位姿更新量收敛或目标平台启发式触发时停止，20 次总更新
只作为运行时间兜底，停止过程不读取真值。只有平移和旋转更新同时过阈值才记为
`pose_converged`；`objective_plateau` 只是工程停止条件，不是数值收敛证明。
历史两次更新仍可通过把 `BALM Iter`/`--max-iterations` 设为 `2` 复现。

## Working vs Original | Working 与 Original

- `Original`: read-only baseline graph.
- `Working`: editable graph used for subsequent preview, matching, and optimization.

- `Original`：只读基线图。
- `Working`：后续预览、匹配和优化实际使用的可编辑工作图。

Every successful optimization updates the `Working` trajectory revision.

每次优化成功后，`Working` 轨迹版本都会更新。

## Target Cloud Modes | Target 点云构造模式

### 1. Temporal Window

- default mode
- target cloud = `target_id ± N` frames
- all frames are transformed to the map frame and merged directly
- the environment preset uses `N=40`; `TgtVoxel` is `0.2 m` indoors and
  `0.4 m` outdoors
- only the renderer is point-capped; registration keeps the complete cloud

- 默认模式
- target 点云 = `target_id ± N` 帧
- 所有帧先变换到 map 系再直接拼接
- 环境预设采用 `N=40`；`TgtVoxel` 室内为 `0.2 m`、室外为 `0.4 m`
- 只有渲染视图会限点，配准仍使用完整点云

### 2. RS Spatial Submap

- keeps the online RS-style local-submap logic
- selects spatial neighbors near the target node, with a source-to-target minimum time gap

- 保留在线 RS 风格的局部子图逻辑
- 按 target 附近的空间邻居选帧，并对 source 和 target 施加最小时间间隔过滤

## Point Cloud Editing | 点云编辑

The tool is intentionally optimized for the common ground-robot workflow.

本工具有意针对地面机器人最常见的手工闭环流程进行了收敛设计。

### Default edit modes | 默认编辑模式

- `XY+Yaw`: the main mode for horizontal alignment
- `Z`: used occasionally from a side view to move the whole source cloud up or down

- `XY+Yaw`：主要模式，用于平面内对齐
- `Z`：偶尔从侧视图整体上移或下移 source 点云

### Mouse behavior | 鼠标操作

#### View mode

- Left drag: orbit camera
- Right drag: pan camera
- Wheel: zoom

- 左键拖动：旋转视角
- 右键拖动：平移视角
- 滚轮：缩放

#### Edit mode

- The editable object is always the `source` cloud.
- `target` remains fixed.
- In `XY+Yaw` + `Drag=XY`: left drag changes source `x/y`.
- In `XY+Yaw` + `Drag=Yaw`: left drag changes source `yaw`.
- In `Z`: left vertical drag changes source `z`.
- Right drag still pans the camera.
- `Alt + Left drag` or middle drag still rotates the camera.
- Wheel always zooms the camera.

- 可编辑对象始终是 `source` 点云。
- `target` 始终固定。
- `XY+Yaw` + `Drag=XY`：左键拖动修改 source 的 `x/y`。
- `XY+Yaw` + `Drag=Yaw`：左键拖动修改 source 的 `yaw`。
- `Z`：左键上下拖动修改 source 的 `z`。
- 右键仍用于平移视角。
- `Alt + 左键拖动` 或中键仍用于旋转视角。
- 滚轮始终用于缩放视角。

Manual dragging and the right-side `Manual Delta` controls are synchronized both ways.

点云拖拽和右侧 `Manual Delta` 控件是双向同步的。

## Auto Yaw Sweep | 自动 Yaw 遍历

This feature targets ground robots whose main unknown is usually yaw.

这个功能主要面向地面机器人，因为这类场景下不确定量通常主要是 yaw。

Behavior:

功能逻辑：

- the current manual delta is used as the base pose
- only yaw seeds are swept
- each seed first updates the point-cloud preview
- GICP is then run from that seed
- the best result is chosen using overlap quality and RMSE

- 以当前手工 delta 作为基准位姿
- 仅遍历 yaw 初值
- 每个初值都会先刷新点云预览
- 然后从该初值运行 GICP
- 最终按重叠质量和 RMSE 选出最佳结果

## Graph Changes | 图改动列表

The bottom table stores accepted changes in the current session.

底部表格用于保存当前工作会话中已经接受的图改动。

Types:

类型包括：

- `Manual Add`
- `Replace Existing Loop`
- `Disable Existing Loop`

Common statuses:

常见状态包括：

- `Accepted`
- `Applied`
- `Disabled`

Notes:

说明：

- `Replace Existing Loop` is shown as a single logical row.
- Replacement operations are treated as one atomic user action for undo / redo semantics.

- `Replace Existing Loop` 在表格中会聚合为单行业务动作。
- 替换操作在撤销语义上被视为一次原子动作。

## Exported Files | 导出文件

When you optimize and export, the tool generates:

执行优化和导出后，工具会生成：

- edited input `g2o`
- manual constraint CSV
- optimized `pose_graph.g2o`
- optimized `optimized_poses_tum.txt`
- `scans.pcd`
- `trajectory.pcd`
- `pose_graph.png`
- `manual_loop_report.json`

The heavyweight `scans.pcd`/Scan Context rebuild happens in a background
worker. Keep the window open until the Export progress bar reaches 100%; the
interface remains usable while files are generated.

耗时较长的 `scans.pcd`/Scan Context 重建会在后台执行。请在 Export 进度达到 100% 前保持
窗口打开；文件生成期间界面仍可正常响应。

## Practical Tips | 实用建议

- Start with `XY+Yaw` and only use `Z` when the source cloud is obviously too high or too low.
- Use `Temporal Window` for most manual work.
- Run optimization after a small number of accepted changes, instead of accumulating too many edits first.
- If you are revising an existing loop edge, prefer `Replace Edge` over adding a duplicate manual edge.

- 优先使用 `XY+Yaw`，只有当 source 整体明显过高或过低时再切到 `Z`。
- 大多数手工配准场景建议使用 `Temporal Window`。
- 建议每累计少量已接受改动后就优化一次，而不是一次性堆积过多编辑。
- 若你是在修订已有闭环边，优先使用 `Replace Edge`，避免叠加重复约束。
