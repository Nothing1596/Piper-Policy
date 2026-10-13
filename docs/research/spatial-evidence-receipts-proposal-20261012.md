# 空间证据回执改进方案与事实核对记录

核对日期：2026-10-12。本文档修正此前改进草案中与当前仓库源码及文献原文不符的前提假设，记录 `recovery-softplan-01` 已集成的五项底层修复，并给出下一阶段聚焦切片——**改善空间证据回执（Spatial Evidence Receipts）**——的工程设计规范。当前正在运行或已冻结的实验目录、账本与历史成绩保持只读，不予改写。

---

## 一、 代码事实核对与前序假设修正

经逐行回溯 [`piperx-cli`](../../piperx-cli/src/piperx_middleware/) 与 [`tools/run_longvil_online.py`](../../tools/run_longvil_online.py)，此前草案中的七项“现有 Bug”前提与真实源码不符，以如下核对结果为准：

| 原草案说法 | 当前源码真实实现与位置 | 工程结论 |
| :--- | :--- | :--- |
| `move_linear` 每个采样点都必须静止 10 步，形成几十次急刹 | `steps < 10` 是后台 1kHz 物理线程追赶墙钟时间的单轮步进上限（[`mujoco_backend.py:255`](../../piperx-cli/src/piperx_middleware/mujoco_backend.py#L255)）。执行循环在中间路点仅检查目标匹配与关节跟踪误差（`<= 0.12°` 或 `<= 0.3°`）即推进 `waypoint_index += 1`，仅在最终目标点检查连续新鲜的 `stable` 与 `settled >= 3`（[`service.py:582-601, 654-668`](../../piperx-cli/src/piperx_middleware/service.py#L582-L668)）。 | 不重写轨迹执行器。若后续怀疑轨迹推进过慢或振荡，须先采集实测关节位置、速度、TCP 轨迹与接触力曲线予以证实。 |
| 夹爪必须移动至少 `0.8 mm`，否则报 `gripper_motion_not_observed` | 当前代码无此阈值或错误码。执行器区分 `completion="width"`（宽度误差 `< 0.001 m`）与 `completion="bilateral_contact"`（双指接触力判定 `supported is True`）（[`service.py:604-609`](../../piperx-cli/src/piperx_middleware/service.py#L604-L609)、[`run_longvil_online.py:163-164`](../../tools/run_longvil_online.py#L163-L164)）。 | 保留现有两种完成模式并清晰解释回执语义，无需修复不存在的零位移报错。 |
| `simulation_observe` 被 `0.5°/s` 阈值阻止采帧 | 采帧入口无速度限制（[`simulation_camera.py:8-24, 102-136`](../../piperx-cli/src/piperx_middleware/simulation_camera.py#L102-L136)）。`max(abs(v)) > 0.5°/s` 位于 [`scene_bridge.py:156-163`](../../piperx-cli/src/piperx_middleware/scene_bridge.py#L156-L163)，用于校验已保存 RGB-D 在做几何解算时是否具备静止对齐的机械臂状态。 | 不放宽几何对齐校验，也不在观察工具内额外推进物理步。 |
| 只读 `scene_query` 总会修改 `plan` 版本 | 仅当显式传入 `question_id` 时才调用 `record_result` 写回答案并递增版本（[`scene_bridge.py:251-260`](../../piperx-cli/src/piperx_middleware/scene_bridge.py#L251-L260)）。机器人状态或标定变化会使旧证据置为 `stale`（[`scene_bridge.py:193-205`](../../piperx-cli/src/piperx_middleware/scene_bridge.py#L193-L205)）。 | 通过每轮请求前同步 `task_plan` 最新版本（[`run_longvil_online.py:579-599`](../../tools/run_longvil_online.py#L579-L599)）解决版本对齐，无需引入 CQRS 框架。 |
| 实验没有清理历史图片 | `append_live` 已限制保留当前与上一组实时图（`cap=2`）及最新一组补看示范（`cap=1`），原地剥离更早消息中的 `image_url`，仅单独保留 `messages[1]` 的初始示范概览（[`run_longvil_online.py:549-554`](../../tools/run_longvil_online.py#L549-L554)）。 | 不得将现状描述为图片无限累积，也不得将初始示范帧当作过期实时图删除。 |
| 模型没有关节工具或显式完成工具 | 已暴露 `robot_move_joints` 与 `finish_task`；无工具调用直接停止记为 `model_stopped_without_finish`（[`run_longvil_online.py:35, 38-41, 134, 628`](../../tools/run_longvil_online.py#L35-L41)）。 | 不增加重复的 `done()` 或未经路径碰撞验证的 `retract_to_home`。 |
| 八组实验限制为 24 轮 | 在线实验上限为 80 次 API、120 次工具、65,536 completion tokens、1,800 秒（[`run_longvil_online_pilot.py:18`](../../tools/run_longvil_online_pilot.py#L18)）。12 轮上限属于原生交互控制台 [`model_agent.py:628`](../../piperx-cli/src/piperx_middleware/model_agent.py#L628)。 | 区分原生 CLI 交互入口与外部在线评测入口，不混为一谈。 |

---

## 二、 文献依据与实验因果边界修正

1. **GPT-Policy 的动作分块不跨越夹爪状态变化**：
   - 原文（[`arXiv:2609.19138v1`](https://arxiv.org/html/2609.19138v1)，Section 3.3 / Figure 3）明确指出：*“Gripper commands remain fixed during Cartesian motion”*，分块仅用于无需中途观察的无接触笛卡尔路径，而在接触、夹爪开合、遮挡或跟踪异常处必须停止并获取新观测。
   - 因此，**不得将“悬停 $\to$ 下降 $\to$ 闭合 $\to$ 抬起”打包为单次开环提交**；下降到位与双指接触闭合恰恰需要闭环反馈确认。
2. **Robo-Harness K1 的上下文窗口参数与消融结果**：
   - 原文附录 E.2 表 14（[`arXiv:2609.29389`](https://arxiv.org/pdf/2609.29389)）中，图像历史 $K$ 指先前实时决策的图组数（非单张示范图），且所有条件均保留完整历史档案与检索工具。表 14 中 $K=0, N=8$ 为 $15/18$（$83.3\%$），$K=1, N=8$ 为 $14/18$（$77.8\%$），每配置仅 18 轮，作者将其作为设计线索而非普适最优超参。
3. **跨模型与跨接口的总差距不能单归因于视觉叠加（Overlay）**：
   - K1 论文中 $16.7$ 个百分点的差距来自 `Gemini 3.7 Flash + K1 全套工具接口` 对比 `GPT-6 Astra + RGB 基线接口`，模型基座与工具集同时不同，不能单独归因于某一项视觉叠加或内联 `note`。
4. **总预算与控制变量的实验边界**：
   - 当研究问题是“在给定总预算（API 轮数、token、墙钟时间）下，启用 `plan` 是否有助于完成任务”时，`plan` 的读写开销与实时仿真在模型思考期间的物理推进本身就是被测系统端到端效果的一部分，不构成“错误控制变量”。不将当前实验静默替换为 `max_motion_calls=16` 的动作预算实验。

---

## 三、 已收尾集成的五项修复（`recovery-softplan-01`）

在进入新切片开发前，以下五项修复已在当前源码中完成集成并通过 139 项针对性测试与 Linux 原生仿真验收：

1. **软计划（Soft Plan）**：在线实验中 `task_plan` 作为可选持久化辅助，不再作为阻断动作或完成的硬门控（[`run_longvil_online.py:99-115, 333`](../../tools/run_longvil_online.py#L99-L115)）。
2. **确认停稳后的受阻恢复（Confirmed-Stopped Blocked Recovery）**：MuJoCo `direct` 模式下运动受阻且经连续新鲜反馈确认停稳后，返回 `error_code="motion_blocked"`, `recoverable=True` 及停稳后实测状态，允许模型基于新观测帧与新 `request_id` 发起向上退避或重抓（[`service.py:670-715`](../../piperx-cli/src/piperx_middleware/service.py#L670-L715)、[`run_longvil_online.py:205-218, 756-771`](../../tools/run_longvil_online.py#L205-L218)）。
3. **重抓与目标恢复评分（Goal-Recovery Grading v2）**：保留所有抓取尝试，严格区分开爪受控释放（`controlled_release`）与闭合滑脱（`grip_lost`），分别报告目标覆盖率（`goal_coverage_success`）与示范顺序达成率（`ordered_success`）（[`run_longvil_online.py:353-511, 772-804`](../../tools/run_longvil_online.py#L353-L511)）。
4. **采集对齐的机器人子树深度自滤除（Capture-Aligned Robot Depth Filter）**：MuJoCo `observe()` 在同一次加锁渲染中利用 geom 分割掩码将 `base_link` 子树像素的深度置为 `NaN`，不篡改 RGB、不脑补被遮挡表面，并在 `depth_filter` 中如实报告过滤计数（[`mujoco_backend.py:559-676`](../../piperx-cli/src/piperx_middleware/mujoco_backend.py#L559-L676)）。无需再叠加第二套 FK 包络裁剪。
5. **计划版本自动同步**：每轮模型调用前自动拉取最新 `task_plan` 并注入 `CURRENT execution_progress.current_plan`（[`run_longvil_online.py:579-599`](../../tools/run_longvil_online.py#L579-L599)）。

---

## 四、 下一阶段聚焦切片：改善空间证据回执（Spatial Evidence Receipts）

在暂缓新轨迹执行器、跨接触动作块、通用 `retract_to_home`、必填运动 `note`、恢复期强制写 `plan` 及 `max_motion_calls=16` 的前提下，下一阶段仅承接一个边界清晰、可独立验证的切片：**改善空间证据回执**。

### 1. 明确定义相对坐标与双轴几何语义字段

1. **采集时刻对齐的相对位移 `tcp_to_target_delta_m`**：
   - 在几何候选与 IK 预览回执中，基于触发该计算的 `observation_id` 所绑定的采集时刻实测 TCP 位姿（`capture_states[observation_id]['robot']['measured_tcp_pose']`），显式计算并返回基座坐标系（`frame="base"`）下的相对位移差：
     $$\Delta \mathbf{p}_{\text{tcp}\to\text{target}} = \mathbf{p}_{\text{target}}^{\text{base}} - \mathbf{p}_{\text{measured\_tcp}}^{\text{base}}(t_{\text{capture}})$$
   - 字段命名与元数据约束：必须同时携带 `reference_frame: "base"`、`reference_observation_id` 与 `reference_capture_sim_time_s`（或采集时间戳），明确该增量仅对应 `observation_id` 采集静止时刻的实测 TCP，运动后自动失效。
2. **严格区分 PCA 物体长轴、短轴与夹指闭合方向**：
   - 复用 [`scene_geometry.py:789-805`](../../piperx-cli/src/piperx_middleware/scene_geometry.py#L789-L805) (`_surface_summary`) 已有的 2D PCA 特征分解，补充并明确区分：
     - `minor_axis_yaw_deg` / `observed_minor_width_m`：可见表面点云在水平面上的**短轴偏航角**及 **2%~98% 分位观测短轴跨度**；
     - `major_axis_yaw_deg` / `observed_major_span_m`：可见表面点云在水平面上的**长轴偏航角**（与短轴正交）及 **2%~98% 分位观测长轴跨度**；
     - `closing_axis_yaw_deg`：PiperX 夹爪在俯视姿态 `rpy = [180, 0, yaw]` 下，双指沿工具坐标系 $+X$ 轴闭合（[`run_longvil_online.py:281-285, 301-320`](../../tools/run_longvil_online.py#L281-L320)），因此夹指闭合方向对齐的是物体的**短轴**（`minor_axis_yaw_deg`），绝不能将物体长轴（`major_axis_yaw_deg`）与夹指闭合方向混用为同一个 `yaw`；
     - `orientation_ambiguous`：当水平面长短特征值接近（$\lambda_1 \le 1.2 \lambda_0$，如近似正方形方块或球体）时为 `True`，此时回执明确声明方向近似各向同性，默认采用 `0°` 闭合方向。
3. **诚实标注“可见表面观测跨度”**：
   - 所有跨度字段统一命名为 `observed_surface_*`（如 `observed_minor_width_m`、`observed_major_span_m`、`observed_height_above_plane_m`），并在 `scope` 中明确声明：该数值仅为当前视角下突破支撑平面阈值且未被机械臂掩码滤除的**可见表面观测跨度**，不代表被遮挡背面或完整物体的真实三维尺寸。

### 2. 复用现有几何与标定，生成可核验的视觉叠加图（Visual Overlay）

1. **设计目的**：
   - 让模型在调用 `grasp_target`、`placement_target` 或 `preview_move_to` 后，不仅收到 JSON 数值回执，还能在同一观察帧的拷贝图像上直观核验候选点与有效表面点是否落在预期目标上，同时**严格保留原图（Raw RGB）、`observation_id`、采集时间与 `calibration_id`**。
2. **视觉叠加内容与诚实性边界**：
   - 利用 [`scene_geometry.py`](../../piperx-cli/src/piperx_middleware/scene_geometry.py) 中已验证的相机内参 `intrinsics` 与外参 `base_from_camera` 逆变换（$\mathbf{T}_{\text{cam}\leftarrow\text{base}} = \mathbf{T}_{\text{base}\leftarrow\text{cam}}^{-1}$），将基座系 3D 点重投影回像素平面 $(u, v)$：
     - **可见有效表面点/ROI 框**：绘制请求的 `table_roi`、`object_roi` / `reference_roi` 矩形框，以及 ROI 内位于支撑平面之上且通过深度一致性与机械臂掩码过滤的有效采样点投影分布（或有效表面中位点 `median_point_m`）；
     - **候选位姿与夹指闭合轴**：将候选 `grasp` / `place` / `preview` 的基座系 3D 坐标投影为像素标记，并按 `closing_axis_yaw_deg` 与 `opening_width_m` 绘制两条对称的夹指闭合方向短线段；
     - **实测 TCP 相对位置**：将同帧采集时刻的 `measured_tcp_pose.xyz_m` 投影到图像上（若在视野内），并用虚线连接至目标候选投影点；
     - **强制防误导图例（Mandatory Watermark / Caption）**：叠加图顶部或配文固定标注 `CANDIDATE OVERLAY ONLY (obs=<id>, cal=<cal_id>, t=<sim_time>s) — NOT collision-checked, NOT grasp/task success`，杜绝模型把“IK 预检通过的候选图”误判为“动作已执行”或“安全无碰撞”。

### 3. 整理共享纯函数与 Skill 渲染，避免复制评测脚本

1. **提取共享空间回执模块**：
   - 将当前内联在 [`tools/run_longvil_online.py`](../../tools/run_longvil_online.py) 中的通用几何合成与回执格式化纯函数（如 `grasp_estimate`、`placement_geometry`、相对增量计算、3D$\to$2D 候选重投影叠加渲染）提取至 [`piperx-cli/src/piperx_middleware/`](../../piperx-cli/src/piperx_middleware/) 下的独立轻量模块（例如 `spatial_receipts.py` 或并入 `scene_geometry.py` / `scene_bridge.py` 的前端辅助层）。
   - **严格隔离实验私有状态**：提取的公共函数仅接收标定矩阵、点云表面摘要（`surface_summary`）、平面参数（`plane`）、运动学实例（`PiperKinematics`）和实测机械臂快照（`robot_state`），**绝不把在线评测脚本私有的 `OnlineGate` 状态机、历史抓取账本 `transfers` 或私有评分逻辑搬入 `piperx-cli` 公共执行器**。
2. **整理 Skill 章节按需渲染**：
   - [`task_plan.py:97-109`](../../piperx-cli/src/piperx_middleware/task_plan.py#L97-L109) 的 `guide()` 已按 Markdown 二级标题（`## `）解析 [`SKILL.md`](../../piperx-cli/src/piperx_middleware/skills/observe-and-plan/SKILL.md)，而 [`run_longvil_online.py:81-89`](../../tools/run_longvil_online.py#L81-L89) 的 `prompt_recipe()` 仍使用硬编码字符串切片（`skill.find('For a numeric question,')`）。可将其统一为基于章节/段落参数的 `guide()` 渲染选项，无需重写新的 skill 框架。
