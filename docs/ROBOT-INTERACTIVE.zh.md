# Piper Robot 单终端操作（0.7.1）

想先完整跑一遍？见 [单次任务演示：用户命令与模型工具调用](ONE-TASK-DEMO.zh.md)。

本次重构将仿真和真机配置分开。ROS `piper-lab/config/hardware.yaml` 的 commissioning 不属于这个终端，不会自动修改为 `true`。

## 安装发行包

需要 Python 3.11+ 和网络。解压到新目录后，Windows 运行 `py -3 install.py`（或 `Setup.cmd`），macOS/Linux 运行 `python3 install.py`。随后用 `.\piper-robot.cmd` / `./piper-robot` 启动；激活包内 `.venv` 后也可使用裸命令 `piper-robot`。安装器不安装 Python、不注册 PATH、不连接硬件；这不是离线包。Windows CANDO 需 x64 Python 及另外准备的厂商驱动与 SDK。升级时先退出旧程序，再从新目录安装，不覆盖已有 `.venv`。

## 启动与连接

安装后运行 `piper-robot`，先选择仿真或真机，再选择本机或已保存的 SSH 主机。真机模式不会沿用上一次选择。所有操作在当前终端完成：

```text
/connect
/status
/tools
/model
/manual robot_move_joints(joints_deg=[1,0,0,0,0,0], speed_percent=5)
/jobs
/quit
```

上例是仿真测试输入，不是现场运动建议。`/manual` 的参数与现有 MCP 工具一致；位置为米，关节角为度，夹爪力度是未经物理标定的协议字段。

`/connect` 枚举执行器所在主机的设备，多设备时在当前终端选择。USB 枚举、CAN 有反馈、控制器 ready 分别显示。没有 CAN 设备时不会退回仿真。操作员不需要输入 HTTP 端口；程序选择本地端口并验证执行器实例和配置身份。

无交互启动需要显式模式：

```sh
piper-robot --mode simulation --simulation-backend sim
```

`sim` 是确定性软件测试替身，默认 `mujoco` 是物理仿真。两者都不能作为真机验收证据。原有 `sim start`、`mcp`、`observe` 和脚本子命令保留。

## 审批与限位

- `/approval risk`：仅风险动作确认，真机默认。缺少相关自动审批阈值也会要求确认。
- `/approval always`：每个动作确认。
- `/approval auto`：不确认动作，但仍拒绝硬限位越界、反馈过期、故障和无有效控制会话。

新建仿真配置默认 `auto`，可直接连接并调用工具；不需要填写四个自动审批阈值。阈值仅在 `risk` 模式使用。已有配置中的审批模式和限制会保留，不会因升级被放宽。

修改审批模式或关键配置时，先显示具体修改，再输入 `/confirm` 确认或 `/cancel` 取消。只有同时存在多个待确认修改时才需要编号（`/confirm CODE`）；无其他待确认修改且本次未改值时，不产生确认请求。有旧提案时会保留明确选择，避免误确认旧修改。动作进入 `awaiting_approval` 后用 `/approve JOB_ID` 或 `/deny JOB_ID` 处理；等待期间 `/status`、`/jobs`、`/stop` 仍可使用。不存在需要另开终端的 OPEN 时间窗口。

`/limits` 显示有效执行限制，并提供设置向导。设备能力、用户执行限制和自动审批阈值是三件不同的事。执行范围取设备与用户限制交集；越界直接报错，不静默截断目标。直接输入 `/limits`，按提示填写即可，回车保留原值；无待确认提案且全部保留时直接结束。auto/always 只问速度、夹爪最大和最小开度三项；risk 才继续询问四个自动审批阈值。未展示的既有阈值保持不变。无需理解内部会话、版本或凭据机制。

下面的 JSON 是脚本用户的可选写法，设置自动审批阈值示例：

```text
/limits {"automatic":{"max_joint_step_deg":5,"max_tcp_step_m":0.02,"max_speed_percent":5,"max_effort_protocol":0.5}}
```

这只是配置语法示例；阈值应根据现场装置确定。关键配置确认绑定本次修改；动作批准绑定参数、当前姿态、设备连接和配置版本，状态变化后需重新提交。审批不代表已做碰撞规划。

`/config` 查看执行器参数；传 JSON 修改已支持的 TCP、连接、执行限制及控制器参数。主机配置经确认后持久保存，先断开 CAN，随后用 `/connect` 重新连接；不会因修改配置自动发送动作。固件参数仍单独返回 ACK/读回与持久性说明。模型凭据只能调用模型工具，不能修改审批模式、执行限制或清除操作员故障锁存。

## 远程主机

```text
/remote add lab robot-windows
/mode real lab
/connect
```

`robot-windows` 是已有 SSH 配置中的主机名/别名。远端必须预先安装兼容版本的 `piper-robot` 和 SSH 服务。程序使用现有 SSH 配置、密钥或 agent，保留主机密钥验证。首次 SSH 信任或登录失败时先按错误提示完成 SSH 设置，不会自动关闭验证或传输私钥。

CAN 和运动求解/执行都在机械臂侧。网络仅传完整请求、状态及结果。一个执行器只接受一个活动控制会话：前端每秒心跳，连续 5 秒无心跳后拒绝后续动作，取消待审批请求，已接受动作按原有反馈和故障规则完成。网络恢复后按原 `request_id` 查结果，再显式 `/connect` 恢复会话；不自动重发未知结果动作。

## 退出、共享服务与迁移

`/quit` 停止接收新任务，取消未批准请求，等待当前已接受动作完成，再释放本前端专用执行器。附着已有共享服务时只释放本控制会话；不会关闭其他操作者的服务。停止仍需使用 `/stop`，退出本身不是急停。

配置、凭据、模型配置和任务日志按模式与远程主机分开保存。旧配置先备份再迁移，保留只读设置、执行限制和故障锁存。旧 `/operator/control-window` 标记弃用但仍兼容；新前端不使用它。老客户端缺少控制会话时收到 `missing_control_session` 等明确错误，不默认为免审。

故障或失联后的动作结果为 `unknown`/`outcome_unknown` 时，只能查询原请求；不能用新编号冒充重试。每次动作不会另起持有 CAN 的进程，执行器持续拥有连接，最终关闭时统一释放。

## 验证范围

软件测试、进程/网络测试、Windows/Linux 部署和真实机械臂验收分别记录在 `implementation/validation.md`。本轮不执行真机动作，不宣称已完成相机标定、碰撞规划或抽屉任务。

## Force 越界恢复（源码新增，尚未包含在 0.7.1 发行包）

适用于机械臂已有轻微越界、普通控制模式切换无法预载当前姿态的情况。默认关闭，按仿真/真机及主机分别保存；这是恢复开关，不是第四种审批模式。

```text
/config force
/config force on
/confirm
/manual robot_set_control_mode(speed_percent=5)
/approve <本次工具返回的 job_id>
/status
```

`/confirm` 只确认开启配置。**每次**动作请求仍会进入 `awaiting_approval`，必须对本次 job 再执行 `/approve`，包括 `auto` 审批模式；`/deny <job_id>` 拒绝。模型不能开启 force 或批准自己的请求。现有工具名、单位及 request_id 重试规则不变。

随后根据 `/status` 的六个实测角度构造 `robot_move_joints(joints_deg=[...], speed_percent=5)`：越界关节可以保持或向正常范围回退，目标不得产生新的越界、加重已有越界或越过另一侧限位。不自动裁剪目标、不自动回零。当前角度超出正常有效范围超过 5° 时直接拒绝，恰好 5° 可申请确认。恢复后关闭：

```text
/config force off
/confirm
```

正常有效范围仍由机型范围、用户限位与当前连接已查询到的控制器限位共同决定。force **不覆盖控制器回报的限位**，也不写固件限位参数。若实测角度已经违反回报的控制器限位，仍拒绝并需核对零点/标定。只读、反馈过期、驱动故障、急停锁存、速度限制、会话与审批失效规则保持有效。SDK 限位检查始终开启；仅在已批准动作的发送期间使用恢复范围，并在异常时恢复原配置，防止 SDK 静默裁剪保持目标。

force 开启时，恢复使用显式关节目标与控制模式工具；笛卡尔运动、home 等运动原语会被拒绝，关闭 force 后可继续使用。夹爪不扩大范围，也需要逐次确认。5° 是软件恢复容差，不代表机械硬限位或碰撞安全保证。本功能的离线/仿真测试不能作为真机恢复验收。

### 本机真实相机观察

在 `real/local` 中使用 `/observe [相机序列号]`，保存当前 RGB、对齐深度和内参，返回图像路径；多相机且未选择时明确列出设备，不任意选取。`/pixel U V` 查询最近一次图像指定像素的深度与光学相机坐标，不生成基座坐标或动作。空深度直接报错，不用背景深度替代。图片是带时间戳的快照，执行前需重新观察。

相机依赖 `pyrealsense2`、`numpy`、`opencv-python`。可在当前 profile 的 `camera.json` 设置 `python`（已具备这些依赖的解释器）、`serial`、`output_root`；默认使用当前解释器与 profile/observations。此配置仅影响本机相机采集。远程目标暂不支持这两个命令，避免把操作者电脑的相机误当机器人侧相机。原独立 `piper-robot observe` 子命令仍用于仿真。
