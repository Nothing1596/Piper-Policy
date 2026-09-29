# 单次任务演示：转动关节、张开夹爪、恢复姿态

适用：当前源码 0.7.1 的 `piper-robot`。本例只用软件仿真，不需要机械臂、相机或 API key；模型代做时才需要配置模型。

**目标：J1 从 0° 转到 5°，夹爪张开到 40 mm，再将关节恢复到起始位置，夹爪保持张开。**

完成标准：三个动作均返回 `succeeded`，最后六关节角为 `[0,0,0,0,0,0]`、夹爪开度为 `0.04 m`、没有正在执行的动作。

## 1. 启动：只开一个终端

以下命令输入操作系统终端。假设 `piper-robot` 已安装并加入 PATH：

```sh
piper-robot --root ./work/demo-one-task --simulation-backend sim
```

使用压缩包且尚未注册 PATH 时，在包目录将命令开头换成 Windows 的 `.\piper-robot.cmd` 或 macOS/Linux 的 `./piper-robot`，其余参数不变。

第一次演示使用一个未用过的目录，例如上面的 `demo-one-task`。`sim` 是没有三维画面的确定性仿真，便于重复同一流程；平时直接运行 `piper-robot` 则默认使用 MuJoCo。

按启动提示输入：

```text
选择模式：1 仿真 / 2 真机 [1]: 1
选择目标 [local]: local
```

出现 `piper-robot>` 后，后面的命令都输入这里。无需另开执行器终端，不需要设置端口或 OPEN 时间窗口。

如果使用管道或脚本，没有交互菜单，则显式指定模式：

```sh
piper-robot --root ./work/demo-one-task --mode simulation --simulation-backend sim
```

## 2. 路径 A：用户手动完成

**逐条输入；三个 `/manual` 命令每条都等到 `succeeded` 再继续，不能整段一次粘贴。**

| 步骤 | 用户输入 | 看什么结果 |
|---|---|---|
| 1. 连接 | `/connect` | `Backend: sim`、`Ready: yes`。程序自动选择仿真设备。 |
| 2. 读初始状态 | `/status` | 初始关节 `[0,0,0,0,0,0]`，夹爪 `0.02 m`。以下固定角度以这个起点为前提。 |
| 3. 转动 J1 | `/manual robot_move_joints(joints_deg=[5,0,0,0,0,0], speed_percent=5)` | 等任务结果 `status: succeeded`。 |
| 4. 张开夹爪 | `/manual robot_gripper(width_m=0.04, effort_protocol=0.5)` | 等任务结果 `status: succeeded`。 |
| 5. 恢复关节 | `/manual robot_move_joints(joints_deg=[0,0,0,0,0,0], speed_percent=5)` | 等任务结果 `status: succeeded`；夹爪开度不变。 |
| 6. 验收 | `/status` | 关节全 0，夹爪 `0.04 m`，`Active job: none`。 |
| 7. 退出 | `/quit` | 退出前端，释放本次专用执行器。 |

最终状态摘录：

```text
Backend: sim (simulation; no hardware)
Ready: yes
Joints (deg): [0, 0, 0, 0, 0, 0]
Gripper: width=0.04 m, ...
Active job: none
```

`joints_deg` 是六个**绝对角度**，不是增量；`width_m` 单位为米；`effort_protocol` 是协议力度字段，不是牛顿。手动调用的请求编号由前端补齐，用户不用生成。

新仿真默认 `auto`，这条路径不要求 `/limits`、`/approval` 或任何确认。若复用了已有配置，则保留其原有审批规则，不会自动放宽。

## 3. 路径 B：把同一任务交给模型

这一节替代路径 A，不需要把手动流程再做一遍。

### 首次配置模型：一次即可

先输入 `/model` 查看 endpoint、模型名、凭据是否存在以及 MCP 是否连通。已经配置好就跳过下面的设置。

内置模型客户端使用 **OpenAI 兼容的 `/chat/completions` 接口**。供应商只支持 Anthropic Messages 时不能直接填入；Claude CLI 的 key 配置也不会自动同步到这里。

下面是一份需要替换占位值的配置示例。key 文件应已存在，文件中仅存放 key；把路径换成本机的绝对路径：

```text
/model set endpoint="https://YOUR-GATEWAY/v1" model="YOUR-MODEL" api_key_file="C:/Users/YourName/.config/piper/model.key"
/model check
```

macOS/Linux 的 key 文件路径例如 `/home/yourname/.config/piper/model.key`。endpoint 填供应商给出的 API 基础地址，不包含 `/chat/completions`；并非所有供应商都带 `/v1`。

等待 `Model check: status=ok`。该检查验证基本模型调用，实际工具调用能力由后面的任务验证。配置会保存到当前模式、当前目标的配置目录。

### 每次任务：连接、说目标、退出

先输入：

```text
/connect
```

然后输入这段**普通文本，不加 `/`**：

> 仅在当前 sim 仿真中执行。先读取状态，确认 ready 且六关节均为 0°，记录起始关节角。以 5% 速度把 J1 转到绝对角度 5°，其他关节保持原值；完成后将夹爪张开到 0.04 米，协议力度 0.5；完成后以 5% 速度恢复起始关节角，夹爪保持张开。每步确认 succeeded 后再做下一步。最后读取状态，报告是否达到目标。如果起点不符合、动作失败或结果未知，就停止后续步骤并说明原因，不要重新发送动作。

前端会显示工具调用与结果。最终应报告三个动作成功、关节已恢复、夹爪为 `0.04 m`。结束时用户输入：

```text
/quit
```

已完成模型配置时，每次任务用户只需 **`/connect` → 一段目标描述 → `/quit`**。模型负责选择工具、填参数、检查结果。

### 模型实际需要调用什么

下面是该任务的工具调用示意，**不是让用户输入终端的命令**。对应当前前端管理的 MCP 连接；模型不能调用 `/confirm` 等操作员命令。

| 顺序 | 工具 | JSON 参数 |
|---|---|---|
| 1 | `robot_status` | `{}` |
| 2 | `robot_move_joints` | `{"joints_deg":[5,0,0,0,0,0],"speed_percent":5,"request_id":"demo-run-001-turn"}` |
| 3 | `robot_gripper` | `{"width_m":0.04,"effort_protocol":0.5,"request_id":"demo-run-001-open"}` |
| 4 | `robot_move_joints` | `{"joints_deg":[0,0,0,0,0,0],"speed_percent":5,"request_id":"demo-run-001-return"}` |
| 5 | `robot_status` | `{}` |

上表编号只是演示。每次新的任务生成新的编号；同一动作查询或恢复时保留原编号。恢复角度应取第一步读到的状态，不应在其他场景中机械套用全 0。

前端 `invoke_tool` 会等待任务结束。直接调用 MCP 时，动作可能只返回 `job_id` 和 `accepted`；这时须查询到 `succeeded` 才执行下一步，例如：

```text
工具：robot_status
参数：{"job_id":"实际返回的 job_id"}
```

若动作响应丢失，按原 `request_id` 查询结果，而不是换编号再发一次：

```text
工具：robot_status
参数：{"request_id":"demo-run-001-turn"}
```

外部独立 MCP 客户端还需要已有执行器连接和有效控制会话；本节不假定复制工具参数就能完成外部客户端接入。

## 4. 可选：演示同一个终端里的动作确认

若想看审批交互，在动作开始前输入：

```text
/approval always
/confirm
```

之后再提交路径 A 的一个动作。出现 `awaiting_approval` 时，复制屏幕上返回的实际编号：

```text
/approve 实际的JOB_ID
```

不同意则输入 `/deny 实际的JOB_ID`。等待时仍可以 `/status`、`/jobs` 或 `/stop`。这是任务审批；`/confirm` 只确认配置修改，不能代替 `/approve`。

演示完如需恢复仿真默认行为：

```text
/approval auto
/confirm
```

本例切换时仅有一项待确认配置；多项待确认时需按提示指定确认码。

## 5. 遇到问题时输入什么

| 现象 | 下一步 |
|---|---|
| 想看某个工具参数 | `/tools robot_move_joints` 或 `/tools robot_gripper` |
| 想查看执行记录 | `/jobs` |
| 想中断当前动作 | `/stop`；再 `/status` 检查是否停下。 |
| `Ready: no` | `/status` 查看具体原因，先不继续动作。 |
| 出现 `awaiting_approval` | 用实际任务编号 `/approve JOB_ID` 或 `/deny JOB_ID`。 |
| 模型不调用工具或连接失败 | `/model`、`/model check`；也可用路径 A 验证机器人链路。 |
| 超时或断网，动作结果未知 | 按原请求编号查询；不要重复提交动作。 |

## 验证记录

2026-09-29，在 macOS 的隔离临时配置中，通过当前交互控制器 → MCP → HTTP → `SimBackend` 实际执行了路径 A 的三个动作，均为 `succeeded`。最终关节全 0、夹爪 `0.04 m`；清理后无残留 `runtime.json`。

本次没有调用外部模型供应商，也没有操作真实机器人；路径 B 展示当前接口支持的调用流程，具体模型是否能完成任务仍需实际运行验证。本例没有抓物、碰撞规划、相机标定或开抽屉验收含义。
