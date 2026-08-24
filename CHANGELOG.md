# Changelog

> **2026-08-24：产品正式命名为 LiDAR Map Refiner。** GUI、双语 README、工具说明与
> 命令行描述统一采用新名称。现有 `manual_loop_closure` Python 包、会话格式、运行目录
> 保持兼容，不做破坏性重命名。

> **2026-08-24：删除 Ghost Loop Legacy。** GUI 移除 `Ghost Sug.` 与
> `Ghost Loop (Legacy)` 控件、诊断驱动的回环提议代码和自动 PGO/BALM 循环；headless
> 与正式重建入口仅接受 `production`。历史运行目录和报告仍可读取，但旧
> `ghost/radius/scloop/ba-only` 命令会由参数解析器明确拒绝。地图不自洽扫描继续作为
> 可选只读诊断，不会生成约束。

> **2026-08-24：GUI/headless 自动修复合并。** 无头 `production` 入口成为唯一自动
> 修复实现；GUI 的 `Repair Map` 直接运行该入口，不再维护“批量先加入、事后再撤回”
> 的另一套准入顺序。每条候选先通过 trial-PGO，再进入批量 PGO，并继续接受求解后
> NIS/leave-one-out 审计。无头入口最终原子发布带 run ID 的结果协议；GUI 只有在原始
> TUM/G2O、终态 TUM/G2O/约束 CSV 和 proposal ledger 哈希全部匹配后才采用结果。
> `Stop Repair` 会终止隔离进程组（包括 PGO/BALM 子进程），取消、崩溃与半成品均保持
> 当前已加载结果不变。手工编辑继续独立保留。

> **2026-08-14：生产流程重命名与 BALM 停止语义修正。** 用户可见流程统一为
> `Initial Loop Search → Audited PGO → Final Map Refinement`；`stage0`、
> `seed`、`auto_seed` 仅作历史 schema 兼容别名。地图不自洽扫描默认关闭，
> 且明确为只读诊断。BALM 报告分开平移/旋转更新量；只有两者均收敛
> 才记为 `pose_converged`，RMS 平台改记为 `objective_plateau`，不再冒充
> 数值收敛。GUI 可中止最终精修并保留 PGO，也可一键导出橙/蓝两色
> PGO–BALM CloudCompare 对比。

> **2026-08-13：GUI/无头生产 PGO 统一。** Auto Seed 现在保留通过 GICP 的
> 候选，只用 10° 重力灾难门；生产 PGO 固定为 GICP 对角因子、
> `sigma_t=1.265 m`、`sigma_r=0.5°`、信息尺度 `0.1` 与 GNC-TLS。
> 修复 GUI 将 radian 标准差误写进 degree 列的单位错误，并把求解后审计统一为
> 平移/旋转联合 NIS。参数在四条开发序列选择后只在 hall02/hall04 验证一次，
> 未按留出结果回调。空闲 GUI 会释放 Stage-0/GICP 缓存，默认共享缓存上限由
> 约 3.6 GB 降为约 0.72 GB；target 子图缓存改成 2 项 LRU，自动候选不再
> 保留仅供渲染的点数组，Auto Seed 批量操作只写一个有界 undo 快照。PGO/BALM
> 输出现在按位姿数、时间戳、有限值以及平移/旋转运动 fail-closed 验证；失败时恢复
> 上一个有效运行目录，不允许半成品成为导出源。

> **2026-08-13：生产流程收缩。** 默认 `Repair Map` 与 headless 路径现在都是
> `Stage-0 → audited PGO → 一次双面 BALM → 结束`。重影只作为 BALM 后的诊断输出，
> 不再提议新回环，也不参与 Stage-0 或 BALM 的接受/回滚。历史循环保留为显式
> `ghost` / `Ghost Loop (Legacy)` 模式。

> **2026-08-13：双面 BALM 已恢复。** 恢复范围是 BALM 的视线符号分面与鬼影检测的
> 同侧判定；法向门 ICP 仍留在 `OneDrive/icra2027_tro_base_20260806/`，不进入当前
> ICRA 方法。

- Publish a PID-owned busy marker only while GUI QThread/QProcess work is
  active, allowing formal 1x replay gates to detect in-process GICP without
  treating an idle open window as competing compute; stale markers require a
  live matching GUI PID before they are accepted.
- Build production Auto Seed descriptors in one sequential PCD pass without
  filling the multi-gigabyte raw-frame GICP cache. Legacy ghost experiments
  still combine descriptor and diagnosis-cloud preparation in that pass.
- Recover the one-click Repair Map state after an exception while applying a
  completed background result, so controls do not remain stuck in a busy state.
- Bound GUI-only point-cloud uploads while retaining the complete cloud for
  GICP, and use responsive target-map presets (±40 frames, 0.2 m indoor / 0.4 m
  outdoor) instead of rendering a 201-frame target at 0.1 m in every scene.
- Let the plain system-Python launcher re-exec the documented validated venv,
  with `MANUAL_LOOP_GUI_PYTHON` available as an explicit override; use the same
  validated environment as an optimizer-backend candidate.
- The one-click Repair Map button now becomes `Stop Repair` throughout Auto
  Seed, Optimize, and final BALM. A stop request finishes the current
  stage and preserves its artifacts instead of force-killing an active writer.
- Closing the GUI no longer silently kills an Optimize/BALM writer. Active
  Repair stages require safe stop-after-stage; standalone optimization asks
  explicitly before a force stop and defaults to waiting.
- Headless Repair now matches the GUI environment presets end-to-end: indoor
  GICP/target voxels are 0.2 m, outdoor values are 0.4 m, and outdoor
  BALM uses its 4 m root. Each run writes a structured preset summary for
  pipeline validation.
- Production Repair always stops after its single final BALM. The old iterative
  diagnosis proposal path and fixed remeasurement tail are excluded from the
  default after truth/runtime ablations showed no dependable added value.
- Dense fine-voxel preview maps use Open3D's tensor CPU reduction with the
  legacy voxel origin; small/coarse maps keep the faster legacy path. On the
  2464-keyframe GUI benchmark this reduced indoor preview median wall time by
  27.9% and maximum event-loop gap by 37.0% without changing GICP results.
- Auto Seed and suggested-loop verification now evaluate independent GICP
  initial guesses with a bounded worker pool in their existing background
  stage. Complete escalator/simulation A/B runs reduced Repair wall time
  785→335 s (2.34x) and 348→204 s (1.71x); terminal constraint CSVs and
  trajectories were byte-identical in both comparisons.

All notable changes to this project will be documented in this file.

本项目的重要变更会记录在本文件中。

## [Unreleased]

### Changed

- Distinguish a successful BALM run from numerical convergence in
  `balm_report.json`: top-level `run_status`, `numerically_converged`, the
  detailed termination reason, and the independent production-adoption gate
  are now recorded separately. The legacy top-level `converged` field keeps
  its strict final-stage numerical meaning for compatibility.

- Make production BALM convergence-driven rather than truth-tuned: the
  absolute PGO translation/rotation trust weights default to zero, no IMU prior
  is used, and the coarse/fine stages now stop on pose-update convergence or an
  RMS plateau with a 20-update total cap. The former one-update-per-stage
  setting remains reproducible with `--max-iterations 2`, but is no longer a
  production default because its stopping point was selected using truth that
  is unavailable in deployment. A pinned
  official BALM2 cross-check reproduced its dense residual/gradient exactly
  and Hessian to 2.91e-11. Post-hoc truth still shows that deeper pure-plane
  descent can worsen ATE; that is recorded as an objective-model limitation,
  not hidden by a truth-selected runtime iteration count.
  The outdoor preprocessing is frozen with the evaluated setting (4 m root,
  0.4 m leaf, 80 m range).
- Reset the LM trust-region floor whenever BALM rebuilds plane associations.
  Re-association changes the local objective; carrying damping decayed on the
  previous model made Spires fine-stage pose updates grow from 2 cm to 36 cm
  while the plane RMS appeared flat. Raised damping after a rejected step is
  preserved, so this adds no new tuning parameter.
- Restore double-sided BALM as the default: opposite observed faces of a thin
  wall are split by sensor-to-surface view sign before coarse plane fitting;
  `--no-double-sided` remains the explicit single-surface ablation. This does
  not restore the archived normal-gated ICP path.
- BALM 生产配置改为无先验、按收敛停止：PGO 平移/旋转绝对信任项默认为零，不使用
  IMU 先验，粗/细阶段按位姿更新量或 RMS 平台停止，总更新上限为 20。原两次更新仅作
  可复现实验，不再用部署时不可获得的真值选择停止轮数。

### Added

- GUI responsiveness/correctness pass: GICP, yaw sweep, seed/suggestion
  verification, session parsing/validation, and final map Export run off the Qt event loop with live
  progress; Preview/Final review labels now map to the correct clouds; and
  first-run `Repair Map` creates a valid zero-change optimizer baseline when no
  seed is found instead of stalling before BALM. Manual point-cloud dragging
  now replaces only the movable source geometry (the large target submap stays
  resident), while resize/orbit/pan renders are coalesced to display cadence.

- BALM plane bundle adjustment refinement (hku-mars/BALM style) after factor-graph
  optimization: a new `BALM` button in the Commit section refines the last optimized
  trajectory by minimizing the smallest covariance eigenvalue of adaptively voxelized
  planar patches. Pure numpy/scipy backend (`python_optimizer/balm.py` + `balm_cli.py`)
  run as a background QProcess; results are written to a new
  `manual_loop_runs/<timestamp>_balm/` directory (refined `optimized_poses_tum.txt`,
  updated `pose_graph.g2o` vertices, `balm_report.json`) so Undo and Export work
  unchanged. Root voxel size and iteration count are tunable in the Advanced tab.
- 因子图优化之后新增 BALM 平面束调整精化：Commit 区新增 `BALM` 按钮，对最近一次优化
  的轨迹做自适应体素平面提取 + 特征值最小化精化，结果写入新的 run 目录，Undo/Export
  流程保持不变。
- BALM defaults to a coarse-to-fine schedule: stage 1 uses a
  12 cm plane-thickness threshold so 10-20 cm ghost wall layers fall into one plane and get
  merged, stage 2 tightens at 5 cm; default downsample leaf 0.2 m and
  convergence/plateau stopping. `balm_cli.py --single-stage` restores the
  one-stage behavior.
- BALM 默认改为粗到细两段式：第一段 12 cm 平面厚度阈值合并 10–20 cm 的重影墙层，
  第二段 5 cm 收紧；默认降采样 0.2 m，并按收敛或 RMS 平台停止。
- BALM handles thin double-sided walls with II-NVM-style view-sign separation
  (chengwei0427/II-NVM): within a candidate plane, points are grouped by the sign of
  dot(normal, sensor − point); opposite faces are never merged across the normal, while
  same-side ghost layers still merge. Synthetic check: a 12 cm wall stays 12.3 cm with the
  feature on vs. squashed to 4.6 cm with it off under the 12 cm coarse stage.
  Disable with `balm_cli.py --no-double-sided`.
  **2026-08-06: removed from this version** (see the banner at the top of this file);
  the flag no longer exists. The mechanism and its evaluation are frozen in
  `OneDrive/icra2027_tro_base_20260806/`.
- BALM 引入 II-NVM 式法向-视线符号分离处理薄双面墙：候选平面内按
  dot(法向, 传感器位置−点) 的符号分组，异侧面永不跨法向合并，同侧重影层照常合并。
- BALM robustness/UX pass: Huber robust kernel down-weights clutter and dynamic points
  (delta = 1.5 x plane thickness; `--robust` off via BalmParams.robust_kernel="none");
  a secondary RMS plateau stop protects explicit long-run ablations;
  post-run ghost scan reports residual same-side double layers with suggested
  manual-loop keyframe pairs (report `ghost_regions` + log lines); GUI adds a
  "BALM Merge [m]" coarse-threshold spin, live iteration/rms progress in the State card,
  and an "Auto BALM after Optimize" checkbox.
- BALM 鲁棒性/体验：Huber 鲁棒核降权动态物与杂物点；RMS 平台期自动提前停止（迭代数变为上限）；
  运行后自动扫描残留重影并给出建议手动回环关键帧对；GUI 新增合并阈值旋钮、实时进度与
  "Optim 后自动 BALM" 选项。
- BALM performance pass (~2x end-to-end on the 2579-kf IH session: 146 s -> 76 s, identical
  result): plane extraction rewritten as level-wise BFS with packed-integer radix sorts and
  batched per-voxel eigendecompositions (no Python recursion); per-pose Hessians/gradients
  now use one small GEMM per pose instead of 27 global reductions; re-association is skipped
  while the median pose motion since the last association stays below
  `reassociate_min_motion` (default 1 cm).
- BALM 性能优化（IH 会话端到端 ~2 倍：146 秒 → 76 秒，结果不变）：平面提取改为逐层 BFS +
  整数键基数排序 + 批量特征分解；每位姿 H/b 改为单次小矩阵 GEMM；位姿中位运动低于阈值时
  跳过重关联。
- One-click ghost repair: "Add Sug." registers every BALM-suggested keyframe pair with
  GICP and accepts those passing the quality gate (fitness >= 0.6, inlier RMSE <= 0.15 m);
  "Auto Loop" iterates Add Suggested -> Optimize -> BALM until no ghost regions remain,
  nothing new can be added, or the round limit (5) is reached — click again to stop.
- 一键重影修复："Add Sug." 自动对建议点对做 GICP 并按质量门槛入图；"Auto Loop" 自动循环
  建议→优化→BALM 直至收敛（无残留/无可添加/达轮数上限），再点一次可中途停止。
- Automatic seed loops ("Auto Seed", Match section): Scan-Context retrieval over the
  session (gravity-canonicalized, FFT yaw alignment, ring-key prefilter) proposes
  large-drift revisit candidates; each is verified by a gated GICP cascade
  (identity -> yaw sweep) and accepted seeds enter the graph on probation — if the
  next BALM diagnosis shows new ghost regions attributed to a seed's segments, that
  seed is disabled automatically. Manual Pick-Nodes seeding is unchanged. Ghost
  regions now also report their surface normal (usable as a registration prior).
- 自动种子回环（Auto Seed 按钮）：SC 检索 + 门控 GICP 级联 + 归因式试用期回退；
  手动选点流程保持不变。

## [0.1.0] - 2026-04-18

### Added

- Standalone bilingual open-source project layout for the manual loop closure tool.
- PyQt GUI entrypoint and extracted backend catkin workspace.
- Detailed English / Chinese installation and tool documentation.
- Version-pinned `requirements.txt` and optional conda environment.
- One-command helper scripts for venv creation, environment checking, backend building, and Ubuntu dependency installation.
- README assets, screenshots, and workflow illustrations.
- GitHub Actions smoke-check workflow.

### Changed

- Adapted GUI path discovery and optimizer lookup for the standalone repository.
- Simplified backend build dependencies so the offline optimizer no longer requires Open3D CMake integration.

### Validated

- `python3 -m py_compile`
- `python launch_gui.py --help`
- `python scripts/check_env.py`
- `bash scripts/build_backend_catkin.sh`

## [Unreleased]

### Changed

- Kept the one-click `Repair Map` progress monotonic by mapping Auto Seed and
  each Auto Loop verification batch into their reserved pipeline stages.
- Moved node-pair, existing-edge, and manual-edge point-cloud Preview builds
  off the Qt event thread; repeated parameter edits are coalesced into one
  refresh, preventing 200--500-frame target submaps from freezing the window.
- Added an optional `Fast ISAM2` optimize mode in `Advanced`, made it the default Python-side working-graph update path, and kept `Accurate LM` as the parity/reference solve.
- Updated GUI/runtime documentation to explain the new optimize-mode selector, the default `TgtVoxel=0.1 m`, the default `MapVoxel=0.1 m`, and the fact that `Add` / `Replace` consume the current GICP candidate.
- Compressed the homepage hero card again by renaming `Graph Changes` to `Graph Ops` and shortening the export call-to-action.
- Hid the legacy C++ backend selector from the normal GUI path and kept it only for explicit developer mode while preserving automatic fallback behavior.
- Added a one-command `make gtsam-python` / `scripts/install_gtsam_python.sh` path for installing the GTSAM 4.3 Python wrapper into the active virtual environment.
- Added first-party Docker support with a repository `Dockerfile`, `.dockerignore`, and `docs/DOCKER.md`.
- Updated installation docs and both README variants to emphasize Docker and Python-first setup while keeping the C++ backend as a developer-only reference path.
- Made the standalone GUI Python-first by default while keeping the legacy C++ optimizer as an optional fallback path.
- Added validated Python/C++ parity reporting for multiple sessions and documented the observed pose, graph, map-point, and runtime differences.
- Refined the trajectory panel to emphasize `Nodes` / `Edges` selection, reduce status-badge prominence, and compact the toolbar and control rows.
- Reworked the right-side control area into persistent `Summary` / `Advanced` tabs with per-tab scrolling so the tab header remains visible while browsing long panels.
- Compressed the `Summary`, `Delta`, and `Registration` layouts, moved `MapVoxel` into `Advanced`, and simplified action labels for a denser but clearer control column.
- Improved the point-cloud viewer defaults with a top-down camera preset, active preset highlighting, and synchronized local/repo UI behavior.
- Improved PALoc-style session loading by ignoring stale explicit g2o paths outside the selected session root and trimming simple trailing unmatched g2o vertices.
- Fixed Python backend discovery so repository-local virtual environments are found correctly, and added unbuffered optimizer progress logging with elapsed time, LM residuals, and map rebuild progress.
- Added animated README demos for adding, replacing, and disabling loop edges.
- Added a repository-hosted wiki content set under `wiki/` with a GitHub-Wiki-ready page structure.
- Enlarged the main README screenshot and switched feature demos from a three-column layout to full-width single-column sections for clearer viewing.
- Updated the wiki content so it can be published directly to GitHub Wiki with sidebar navigation and GitHub-safe asset links.
- Added a prominent author/contact block to the README header.
- Linked the author names in the README header to their GitHub profiles and corrected the `Xieyuanli Chen` spelling.
- Switched the repository front page to English-first README files with a dedicated `README.zh.md` language counterpart.
- Reworked the GitHub Wiki content into English-first pages with matching Chinese switch pages instead of line-by-line bilingual mixing.
- Simplified the hero banner and tutorial card visuals to avoid text overflow on the homepage assets.
- Updated the contact email to `xhubd@connect.ust.hk`.
- Added a beginner-friendly Docker FAQ and linked it directly from both README quick-start sections.
- Added project-aware edit persistence with `manual_loop_projects/`, resumable `project_state.json`, structured `operations.jsonl`, and `execution.log`.
- Added an `Open Project` session-input entry so historical edit projects can be restored explicitly instead of only resuming the latest project.
- Reworked export behavior so `manual_loop_exports/` stores lightweight manifests pointing to `manual_loop_runs/` instead of duplicating the full optimization output.
- Updated README and Wiki pages to explain the new project/run/export directory model and historical-project restore workflow.
- Reworked the optimize/export flow so `Optimize` updates the working graph and optimized TUM first, while `Export` performs the heavyweight final map and trajectory rebuild on demand.
- Fixed node-pair cloud preview after accepting a manual edge by reusing unapplied manual-edge poses as preview seeds, and cleared stale manual delta after optimization to avoid double-applying offsets.

### Validated

- `python3 -m py_compile gui/manual_loop_closure_tool.py gui/manual_loop_closure/open3d_viewer.py`
- `QT_QPA_PLATFORM=offscreen python3 launch_gui.py --help`
- Offscreen widget instantiation for the updated trajectory, tabbed control panel, and camera preset controls
- Offscreen session load and project-resume checks against a real PALoc-style session
