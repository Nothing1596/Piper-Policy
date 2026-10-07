# piper-robot 命令与功能 / Command reference

适用 **0.7.1**。[中文安装](ROBOT-GUIDE.zh.md) · [English installation](ROBOT-GUIDE.en.md) · [单次任务演示](ONE-TASK-DEMO.zh.md)

## 推荐：单终端 / Recommended console

安装后运行 `piper-robot`，选择仿真/真机和本机/远端。包内入口为 Windows 的 `.\piper-robot.cmd` 或 macOS/Linux 的 `./piper-robot`；安装不自动注册 PATH。Start the console, choose mode and target, and operate in one terminal.

```text
piper-robot
# 选择仿真、本机后逐条输入 / Select simulation and local, then enter:
/connect
/status
/tools
/quit
```

完整命令包括 `/manual`、`/model`、`/approval`、`/limits`、`/config`、`/remote`、`/mode`、`/approve`、`/deny`、`/confirm`、`/cancel`、`/stop`、`/shutdown`。见 [交互指南](ROBOT-INTERACTIVE.zh.md)。No backend ports or second-terminal OPEN windows are needed.

没有 TTY 的控制台启动须显式指定 `--mode simulation` 或 `--mode real`。`--simulation-backend sim` 选择确定性测试替身，默认 `mujoco` 是物理仿真。`--root DIR` 选择前端数据根目录，模式与目标配置分开保存。`/quit` 等待已接受动作结束并清理专用执行器，附着共享服务时只释放本会话；退出不是急停。

## 源码新增：force recovery（未发布）

`/config force` 查看；`/config force on|off` 提议修改，再 `/confirm` 应用。仅允许已有 ≤5° 越界的保持/向内恢复，每次动作请求必须 `/approve <job_id>`，即使审批为 `auto`。不覆盖控制器回报限位、不改固件、不自动裁剪目标。仅显式关节动作及控制模式支持越界恢复；HTTP/MCP 共用相同准入和确认规则。详见 [恢复操作](ROBOT-INTERACTIVE.zh.md#force-越界恢复源码新增尚未包含在-071-发行包)。

Operator-only force recovery is off by default. Enable/disable with `/config force on|off`, then `/confirm`. Every action request requires separate operator approval, including auto mode. The 5-degree allowance covers an existing model/site overrun only: hold or move inward, never create or worsen an overrun. Queried controller limits remain enforced. This is an unreleased source feature, not a physical acceptance claim.

## 兼容脚本与独立 MCP / Script and MCP compatibility

以下参数表用于已有脚本和独立托管服务，不是普通交互用户的连接步骤。包装命令参数在命令后，执行器全局参数在命令前：

```text
piper-robot sim start --root DIR [--port 8798] [--seed 200]
piper-robot observe --root DIR [--url URL] --workspace DIR --output NEW_PATH
piper-robot mcp [--root DIR] [--url URL] [--token-file FILE] [--workspace DIR]
piper-robot [--root DIR] [--url URL] [--token-file FILE] COMMAND [ARGS]
```

`sim start`、`observe`、`mcp` 使用独立解析器；不要写成 `piper-robot --root DIR sim start`。脚本/MCP 的 root 是执行器配置目录，不是前端 profiles 的父目录。These entries do not provide the console's managed lifecycle by themselves.

执行器默认 root 为 Windows `%LOCALAPPDATA%/PiperXMiddleware`、其他平台 `~/.local/state/piperx-middleware`。脚本 URL 顺序为显式 `--url` → `PIPERX_URL` → 本地配置 → `http://127.0.0.1:8765`；独立 MCP/observe 不自动读取 sim 的 8798 端口。`init`/`serve` 使用本地配置，不接受全局 URL/token；`init` 后端 sim/mujoco/agx 分别为软件替身、物理仿真、真机。

**保留入口不等于免审执行。** 独立客户端仍需有效控制会话，MCP 桥可通过 `PIPERX_CONTROL_SESSION` 接收会话；创建和心跳由其控制端管理。单独运行旧 `arm` 时间窗口不会代替新会话，缺少会话时会明确拒绝动作。新交互前端自动维护这些状态。HTTP/MCP cannot bypass approval, limits or session checks.

## 单位、动作与反馈 / Units and execution

| 命令 / Command | 单位与行为 / Meaning |
|---|---|
| `move-joints J1 ... J6` | 六个绝对关节角，度 / Absolute joint degrees |
| `gripper WIDTH` | 绝对开度，米；effort 是未标定协议值 / Metres; effort is uncalibrated |
| `move-to X Y Z` | 基座系绝对 TCP 位置，米；可选 `--rpy` 是绝对姿态，度 / Absolute base-frame TCP target |
| `move-by DX DY DZ` | 相对平移，米，`--frame base` 默认或 `tcp` / Relative translation |
| `rotate R P Y` | 基座系绝对 RPY，度，保持位置 / Absolute orientation, fixed position |
| `move-linear X Y Z` | 定姿态直线目标，`--step` 默认 0.002 米；`--native` 请求真机 MOVE_L / Linear target |

`move-to` / `move-by` / `rotate` 是 IK 求解后的关节空间端点运动，不保证 TCP 走直线；没有通用碰撞规划。`move-linear` 的采样检查也不等于连续路径无碰撞证明。RPY 使用 extrinsic xyz，即 Rz@Ry@Rx。

动作命令通常有 `--request-id`、`--wait`、`--wait-timeout`（默认 130 秒）。未给 ID 时生成一次并写到 stderr；结果未知时按原 ID 查询，不自动重发动作。`--speed` 默认 5%，动作 `--timeout` 默认 30 秒。`gripper` 没有 speed；`execute` 使用已有计划。

提交成功返回 job_id，不代表动作完成。`job ID --wait` / `request ID --wait` 查询结果。等待超时返回 outcome_unknown，不自动停机、撤销或重放。`--output FILE` 独占创建 JSON 文件；多数脚本命令 stdout 是 JSON。失败/取消/未知等结果返回非零。Status summaries default to readable text; use `--json` for scripts.

`gripper --completion bilateral_contact` 仅适用于 MuJoCo 闭合时的稳定双指接触，不证明抬起成功。`stop` 请求软件停止；`estop` 锁存软件停止；`shutdown` 只在空闲时释放连接并退出；都不替代物理急停。

`arm`、`estop`、`clear-estop`、`configure-runtime`、`sim-fault`、`limits --refresh` 使用操作员权限，默认读取 root/operator.token；普通命令默认 model.token。`--token-file` / `PIPERX_TOKEN_FILE` 可覆盖。MCP 模型工具不暴露这些管理命令。

## 内置模型与交互终端 / Built-in model and console

`model`（单数）是下位机自己的 OpenAI-compatible 文本/工具调用入口，不是视频侧的 `models`，也不是多供应商视频配置。Configuration lives in root/model.json; it stores credential references, not key values.

在交互前端使用 `/model`、`/model set endpoint=... model=... api_key_file=...` 和 `/model check`。`check` 发起最小真实补全请求；配置存在不证明模型或工具调用可用。随后普通文本交给模型；手动 `/manual 工具名(参数)` 不依赖模型。详细例子见 [单次任务演示](ONE-TASK-DEMO.zh.md)。

脚本 `model run --allow-motion` 仅允许该回合提出动作，仍须满足后端会话和审批规则。内置文本/工具回合不等同于视频理解或完整视觉策略。`control-mode` 固定请求 CAN/MOVE_J，不接受任意模式、使能或复位字段。

## MCP 工具 / MCP tools

`mcp` 不启动执行器、不另开 CAN。加 `--workspace` 才暴露 `simulation_observe`；目录须存在，输出不能覆盖既有路径。该取图工具当前只支持 MuJoCo，返回图像并保存 RGB/深度/标定，不返回隐藏物体状态或评分。

| 工具 / Tool | 主要参数 / Main arguments |
|---|---|
| `robot_status` | 可选 `job_id` 或 `request_id` / Optional job/request lookup |
| `robot_connect` | `reconnect=false`, `device_id` 可选 |
| `robot_disconnect`, `robot_stop`, `robot_diagnostics` | 无 / none |
| `robot_set_control_mode` | `request_id`, `speed_percent=5`, `timeout_s=3` |
| `robot_move_joints` | `joints_deg[6]`, `request_id`, speed / timeout |
| `robot_gripper` | `width_m`, `request_id`, `effort_protocol=0.5`, `timeout_s=10` |
| `move_to` | `xyz_m[3]`, `request_id`, 可选 `rpy_deg`, speed / timeout |
| `move_by` | `delta_m[3]`, `request_id`, `frame=base`, speed / timeout |
| `rotate` | `rpy_deg[3]`, `request_id`, speed / timeout |
| `set_gripper` | `width_m`, `request_id`, effort / timeout, `completion=width` 或 `bilateral_contact` |
| `move_linear` | `xyz_m[3]`, `request_id`, speed / timeout；MCP 不提供 CLI 的 step/native 参数 |
| `simulation_observe` | `output`；仅 workspace 已配置时 / Requires configured workspace |

常规 MCP 运动 speed 默认 5%、timeout 默认 30 秒；夹爪默认 10 秒。以客户端 `tools/list` 的完整 schema 为准，CLI 与 MCP 并非逐参数等价。Tool calls return jobs; agents must check completion and fresh images.

支持命令不等于已完成真机验收。SDK、CAN 驱动、标定需现场部署；本版验证范围见 [验收记录](implementation/validation.md)。Supported commands do not establish physical hardware acceptance.

## 命令索引 / Command index

| Command | 功能 / Function |
|---|---|
| `sim start` | 初始化并启动 MuJoCo / Initialize and serve MuJoCo |
| `observe` | 保存当前 RGB-D / Capture current RGB-D |
| `mcp` | 独立 stdio MCP 桥 / Independent stdio MCP bridge |
| `init` | 创建配置与凭据 / Initialize configuration and credentials |
| `serve` | 启动共享执行器 / Start shared executor |
| `shell` | 交互终端 / Interactive console |
| `connect` | 连接设备 / Connect device |
| `disconnect` | 释放连接 / Disconnect device |
| `state` | 原始状态 / Raw state |
| `status` | 状态概览 / Status summary |
| `monitor` | 持续状态采样 / Repeated status reads |
| `tools` | 执行器工具清单 / Executor tool inventory |
| `params` | 配置与实测参数 / Configured and measured parameters |
| `calls` | 调用记录 / Call history |
| `jobs` | 作业清单 / Job inventory |
| `job` | 按 job_id 查询或等待 / Query or await a job |
| `request` | 按 request_id 恢复查询 / Query by request ID |
| `events` | 增量事件 / Incremental events |
| `doctor` | 反馈、新鲜度与诊断 / Feedback freshness and diagnostics |
| `devices` | 枚举执行器端设备 / Enumerate executor-side devices |
| `limits` | 限位及来源，可主动刷新 / Limits and provenance, optional refresh |
| `control-mode` | 设置控制模式 / Set control mode |
| `move-joints` | 六关节绝对角度 / Six absolute joint angles |
| `gripper` | 夹爪绝对开度 / Absolute gripper opening |
| `move-to` | TCP 绝对位置 / Absolute TCP position |
| `move-by` | TCP 相对平移 / Relative TCP translation |
| `rotate` | TCP 绝对姿态 / Absolute TCP orientation |
| `move-linear` | 定姿态直线采样 / Fixed-orientation linear sampling |
| `preview` | 预览 JSON 命令 / Preview a JSON command |
| `execute` | 执行 plan_id / Execute a previewed plan |
| `stop` | 请求停止 / Request stop |
| `shutdown` | 空闲时关闭服务 / Shut down idle executor |
| `arm` | 操作员限时控制窗口 / Operator control window |
| `estop` | 锁存软件停止 / Latch software stop |
| `clear-estop` | 清除软件停止锁存 / Clear software stop latch |
| `configure-runtime` | 操作员运行参数 / Operator runtime configuration |
| `sim-fault` | 注入仿真故障 / Inject simulation faults |
| `manual` | 调用具名 MCP 工具 / Invoke a named MCP tool |
| `model show` | 读取模型配置 / Read model configuration |
| `model set` | 保存端点与模型名 / Save endpoint and model configuration |
| `model list` | 查询服务模型清单 / List server models |
| `model check` | 检查模型服务 / Check model service |
| `model run` | 有界工具调用回合 / Bounded model tool-calling turn |

## 完整参数 / Exact help snapshots

以下内容来自 0.7.1 的实际 `--help`，未启动后端或调用模型。Generated from the current CLI help without starting a backend or calling a model.

<details>
<summary>piper-robot --help</summary>

```text
usage: cli.py [-h] [--root ROOT] [--url URL] [--token-file TOKEN_FILE]
              [--mode {simulation,real}] [--target TARGET]
              [--simulation-backend {sim,mujoco}]
              {shell,init,serve,state,connect,disconnect,stop,shutdown,arm,status,monitor,tools,params,calls,jobs,clear-estop,configure-runtime,control-mode,devices,doctor,estop,events,execute,gripper,job,limits,manual,move-by,move-joints,move-linear,move-to,preview,request,rotate,sim-fault,model}
              ...

PiperX shared robot executor and operator controls

positional arguments:
  {shell,init,serve,state,connect,disconnect,stop,shutdown,arm,status,monitor,tools,params,calls,jobs,clear-estop,configure-runtime,control-mode,devices,doctor,estop,events,execute,gripper,job,limits,manual,move-by,move-joints,move-linear,move-to,preview,request,rotate,sim-fault,model}
    shell               Interactive robot console (also the default with no
                        command)
    disconnect          Release CAN without exiting the server
    shutdown            Release CAN and exit an idle executor gracefully; not
                        a robot stop
    arm                 Operator-only bounded control window; does not itself
                        move
    status              Read-only summary: connection, robot, joints, gripper,
                        TCP, diagnostics
    monitor             Poll status repeatedly; Ctrl-C stops the watch only,
                        never the robot
    tools               List MCP tools from executor capabilities
    params              Show configured parameters and reported state,
                        distinctly labeled
    calls               Recorded executor calls, newest first; not proof of
                        physical completion
    jobs                Known jobs, newest first; job status is the execution
                        result
    clear-estop         JSON CLI: clear-estop
    configure-runtime   JSON CLI: configure-runtime
    control-mode        JSON CLI: control-mode
    devices             JSON CLI: devices
    doctor              JSON CLI: doctor
    estop               JSON CLI: estop
    events              JSON CLI: events
    execute             JSON CLI: execute
    gripper             JSON CLI: gripper
    job                 JSON CLI: job
    limits              JSON CLI: limits
    manual              JSON CLI: manual
    move-by             JSON CLI: move-by
    move-joints         JSON CLI: move-joints
    move-linear         JSON CLI: move-linear
    move-to             JSON CLI: move-to
    preview             JSON CLI: preview
    request             JSON CLI: request
    rotate              JSON CLI: rotate
    sim-fault           JSON CLI: sim-fault
    model               Local/OpenAI-compatible model configuration and
                        bounded tool calling

options:
  -h, --help            show this help message and exit
  --root ROOT
  --url URL             Executor origin for the console and inspection
                        commands (env PIPERX_URL; default: local config, else
                        http://127.0.0.1:8765)
  --token-file TOKEN_FILE
                        Bearer token file for the console and inspection
                        commands (env PIPERX_TOKEN_FILE; default:
                        <root>/model.token)
  --mode {simulation,real}
                        Explicit startup mode; required without a TTY
  --target TARGET       Local executor or saved SSH target
  --simulation-backend {sim,mujoco}
piper-robot: sim start | observe | mcp | [--root DIR] <executor command>
Use sim --help, observe --help, mcp --help, or the executor help below.
```

</details>

<details>
<summary>piper-robot sim start --help</summary>

```text
usage: piper-robot sim [-h] --root ROOT [--port PORT] [--seed SEED] {start}

positional arguments:
  {start}

options:
  -h, --help   show this help message and exit
  --root ROOT
  --port PORT
  --seed SEED
```

</details>

<details>
<summary>piper-robot observe --help</summary>

```text
usage: piper-robot observe [-h] --root ROOT [--url URL] --workspace WORKSPACE
                           --output OUTPUT

options:
  -h, --help            show this help message and exit
  --root ROOT
  --url URL
  --workspace WORKSPACE
  --output OUTPUT
```

</details>

<details>
<summary>piper-robot mcp --help</summary>

```text
usage: standalone_cli.py [-h] [--url URL] [--root ROOT]
                         [--token-file TOKEN_FILE] [--workspace WORKSPACE]

PiperX MCP stdio client for a shared executor; never opens CAN

options:
  -h, --help            show this help message and exit
  --url URL             Executor URL; defaults to PIPERX_URL or
                        http://127.0.0.1:8765
  --root ROOT           Local executor data directory (for its token file)
  --token-file TOKEN_FILE
                        Model token file; value is never printed
  --workspace WORKSPACE
                        Enable RGB-D capture inside this existing directory
```

</details>

<details>
<summary>piper-robot init --help</summary>

```text
usage: cli.py init [-h] [--backend {sim,agx,mujoco}] [--port PORT]
                   [--can-interface {socketcan,agx_cando}]
                   [--can-channel CAN_CHANNEL] [--sdk-root SDK_ROOT]
                   [--cando-source CANDO_SOURCE] [--tcp-offset-m X Y Z]
                   [--tcp-offset-rpy-deg ROLL PITCH YAW]
                   [--profile {direct,calibration}] [--read-only]

options:
  -h, --help            show this help message and exit
  --backend {sim,agx,mujoco}
  --port PORT
  --can-interface {socketcan,agx_cando}
  --can-channel CAN_CHANNEL
  --sdk-root SDK_ROOT
  --cando-source CANDO_SOURCE
  --tcp-offset-m X Y Z
  --tcp-offset-rpy-deg ROLL PITCH YAW
  --profile {direct,calibration}
  --read-only
```

</details>

<details>
<summary>piper-robot serve --help</summary>

```text
usage: cli.py serve [-h] [--config CONFIG] [--managed]
                    [--allow-motion | --read-only]

options:
  -h, --help       show this help message and exit
  --config CONFIG
  --managed        Private lifecycle-managed executor with automatic port
  --allow-motion   Enable configured action tools
  --read-only      Run observations only
```

</details>

<details>
<summary>piper-robot shell --help</summary>

```text
usage: cli.py shell [-h]

options:
  -h, --help  show this help message and exit
```

</details>

<details>
<summary>piper-robot connect --help</summary>

```text
usage: cli.py connect [-h] [--reconnect] [--device-id DEVICE_ID]
                      [--output OUTPUT] [--json]

options:
  -h, --help            show this help message and exit
  --reconnect           Release and reopen an idle CAN connection
  --device-id DEVICE_ID
                        Explicit enumerated CAN device ID
  --output OUTPUT
  --json                JSON is the default
```

</details>

<details>
<summary>piper-robot disconnect --help</summary>

```text
usage: cli.py disconnect [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT
  --json           JSON is the default
```

</details>

<details>
<summary>piper-robot state --help</summary>

```text
usage: cli.py state [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT
  --json           JSON is the default
```

</details>

<details>
<summary>piper-robot status --help</summary>

```text
usage: cli.py status [-h] [--json]

options:
  -h, --help  show this help message and exit
  --json      Print the raw server response as JSON
```

</details>

<details>
<summary>piper-robot monitor --help</summary>

```text
usage: cli.py monitor [-h] [--interval INTERVAL] [--count COUNT] [--json]

options:
  -h, --help           show this help message and exit
  --interval INTERVAL  Seconds between polls; positive and finite (default 2)
  --count COUNT        Stop after this many polls (default: until Ctrl-C)
  --json               Print one JSON state object per line
```

</details>

<details>
<summary>piper-robot tools --help</summary>

```text
usage: cli.py tools [-h] [--json]

options:
  -h, --help  show this help message and exit
  --json      Print the raw server response as JSON
```

</details>

<details>
<summary>piper-robot params --help</summary>

```text
usage: cli.py params [-h] [--json]

options:
  -h, --help  show this help message and exit
  --json      Print the raw server response as JSON
```

</details>

<details>
<summary>piper-robot calls --help</summary>

```text
usage: cli.py calls [-h] [--limit 1..500] [--json]

options:
  -h, --help      show this help message and exit
  --limit 1..500
  --json          Print the raw server response as JSON
```

</details>

<details>
<summary>piper-robot jobs --help</summary>

```text
usage: cli.py jobs [-h] [--limit 1..500] [--json]

options:
  -h, --help      show this help message and exit
  --limit 1..500
  --json          Print the raw server response as JSON
```

</details>

<details>
<summary>piper-robot job --help</summary>

```text
usage: cli.py job [-h] [--output OUTPUT] [--json] [--wait]
                  [--wait-timeout WAIT_TIMEOUT]
                  id

positional arguments:
  id

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --wait
  --wait-timeout WAIT_TIMEOUT
```

</details>

<details>
<summary>piper-robot request --help</summary>

```text
usage: cli.py request [-h] [--output OUTPUT] [--json] [--wait]
                      [--wait-timeout WAIT_TIMEOUT]
                      id

positional arguments:
  id

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --wait
  --wait-timeout WAIT_TIMEOUT
```

</details>

<details>
<summary>piper-robot events --help</summary>

```text
usage: cli.py events [-h] [--output OUTPUT] [--json] [--after AFTER]
                     [--limit LIMIT]

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
  --after AFTER
  --limit LIMIT
```

</details>

<details>
<summary>piper-robot doctor --help</summary>

```text
usage: cli.py doctor [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
```

</details>

<details>
<summary>piper-robot devices --help</summary>

```text
usage: cli.py devices [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
```

</details>

<details>
<summary>piper-robot limits --help</summary>

```text
usage: cli.py limits [-h] [--output OUTPUT] [--json] [--refresh]

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
  --refresh        Operator: actively query six controller limits
```

</details>

<details>
<summary>piper-robot control-mode --help</summary>

```text
usage: cli.py control-mode [-h] [--output OUTPUT] [--json]
                           [--request-id REQUEST_ID] [--wait]
                           [--wait-timeout WAIT_TIMEOUT] [--speed SPEED]
                           [--timeout TIMEOUT]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
  --speed SPEED
  --timeout TIMEOUT
```

</details>

<details>
<summary>piper-robot move-joints --help</summary>

```text
usage: cli.py move-joints [-h] [--output OUTPUT] [--json]
                          [--request-id REQUEST_ID] [--wait]
                          [--wait-timeout WAIT_TIMEOUT] [--speed SPEED]
                          [--timeout TIMEOUT]
                          joints joints joints joints joints joints

positional arguments:
  joints

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
  --speed SPEED
  --timeout TIMEOUT
```

</details>

<details>
<summary>piper-robot gripper --help</summary>

```text
usage: cli.py gripper [-h] [--output OUTPUT] [--json]
                      [--request-id REQUEST_ID] [--wait]
                      [--wait-timeout WAIT_TIMEOUT] [--timeout TIMEOUT]
                      [--effort EFFORT]
                      [--completion {width,bilateral_contact}]
                      width

positional arguments:
  width

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
  --timeout TIMEOUT
  --effort EFFORT
  --completion {width,bilateral_contact}
                        MuJoCo closing grasp only: require fresh stable
                        bilateral finger contact
```

</details>

<details>
<summary>piper-robot move-to --help</summary>

```text
usage: cli.py move-to [-h] [--output OUTPUT] [--json]
                      [--request-id REQUEST_ID] [--wait]
                      [--wait-timeout WAIT_TIMEOUT] [--speed SPEED]
                      [--timeout TIMEOUT] [--rpy RPY RPY RPY]
                      vector vector vector

positional arguments:
  vector

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
  --speed SPEED
  --timeout TIMEOUT
  --rpy RPY RPY RPY
```

</details>

<details>
<summary>piper-robot move-by --help</summary>

```text
usage: cli.py move-by [-h] [--output OUTPUT] [--json]
                      [--request-id REQUEST_ID] [--wait]
                      [--wait-timeout WAIT_TIMEOUT] [--speed SPEED]
                      [--timeout TIMEOUT] [--frame {base,tcp}]
                      vector vector vector

positional arguments:
  vector

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
  --speed SPEED
  --timeout TIMEOUT
  --frame {base,tcp}
```

</details>

<details>
<summary>piper-robot rotate --help</summary>

```text
usage: cli.py rotate [-h] [--output OUTPUT] [--json] [--request-id REQUEST_ID]
                     [--wait] [--wait-timeout WAIT_TIMEOUT] [--speed SPEED]
                     [--timeout TIMEOUT]
                     vector vector vector

positional arguments:
  vector

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
  --speed SPEED
  --timeout TIMEOUT
```

</details>

<details>
<summary>piper-robot move-linear --help</summary>

```text
usage: cli.py move-linear [-h] [--output OUTPUT] [--json]
                          [--request-id REQUEST_ID] [--wait]
                          [--wait-timeout WAIT_TIMEOUT] [--speed SPEED]
                          [--timeout TIMEOUT] [--step STEP] [--native]
                          vector vector vector

positional arguments:
  vector

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
  --speed SPEED
  --timeout TIMEOUT
  --step STEP
  --native              Use controller MOVE_L on agx; simulator still uses
                        explicit synthetic samples
```

</details>

<details>
<summary>piper-robot preview --help</summary>

```text
usage: cli.py preview [-h] [--output OUTPUT] [--json] file

positional arguments:
  file             JSON command object, not a request envelope

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
```

</details>

<details>
<summary>piper-robot execute --help</summary>

```text
usage: cli.py execute [-h] [--output OUTPUT] [--json]
                      [--request-id REQUEST_ID] [--wait]
                      [--wait-timeout WAIT_TIMEOUT]
                      plan_id

positional arguments:
  plan_id

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --request-id REQUEST_ID
                        Stable ID for retries; generated once if omitted
  --wait                Wait for measured completion
  --wait-timeout WAIT_TIMEOUT
```

</details>

<details>
<summary>piper-robot stop --help</summary>

```text
usage: cli.py stop [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT
  --json           JSON is the default
```

</details>

<details>
<summary>piper-robot shutdown --help</summary>

```text
usage: cli.py shutdown [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT
  --json           JSON is the default
```

</details>

<details>
<summary>piper-robot arm --help</summary>

```text
usage: cli.py arm [-h] [--seconds SECONDS] [--radius-deg RADIUS_DEG]
                  [--gripper] [--output OUTPUT] [--json]

options:
  -h, --help            show this help message and exit
  --seconds SECONDS
  --radius-deg RADIUS_DEG
  --gripper
  --output OUTPUT
  --json                JSON is the default
```

</details>

<details>
<summary>piper-robot estop --help</summary>

```text
usage: cli.py estop [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
```

</details>

<details>
<summary>piper-robot clear-estop --help</summary>

```text
usage: cli.py clear-estop [-h] [--output OUTPUT] [--json]

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
```

</details>

<details>
<summary>piper-robot configure-runtime --help</summary>

```text
usage: cli.py configure-runtime [-h] [--output OUTPUT] [--json]
                                [--tcp-m TCP_M TCP_M TCP_M]
                                [--tcp-rpy TCP_RPY TCP_RPY TCP_RPY]
                                [--payload {empty,half,full}]
                                [--collision-rating COLLISION_RATING]
                                [--joint-acc JOINT_ACC]

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
  --tcp-m TCP_M TCP_M TCP_M
  --tcp-rpy TCP_RPY TCP_RPY TCP_RPY
  --payload {empty,half,full}
  --collision-rating COLLISION_RATING
  --joint-acc JOINT_ACC
```

</details>

<details>
<summary>piper-robot sim-fault --help</summary>

```text
usage: cli.py sim-fault [-h] [--output OUTPUT] [--json]
                        {none,stale,collision,tracking,driver,teaching}

positional arguments:
  {none,stale,collision,tracking,driver,teaching}

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Save the JSON result (exclusive create)
  --json                JSON is the default for this command
```

</details>

<details>
<summary>piper-robot manual --help</summary>

```text
usage: cli.py manual [-h] [--output OUTPUT] [--json] expression

positional arguments:
  expression       Named literal MCP call, e.g. robot_status()

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Save the JSON result (exclusive create)
  --json           JSON is the default for this command
```

</details>

<details>
<summary>piper-robot model show --help</summary>

```text
usage: cli.py model show [-h] [--output OUTPUT] [--endpoint ENDPOINT]
                         [--name NAME]

options:
  -h, --help           show this help message and exit
  --output OUTPUT
  --endpoint ENDPOINT
  --name NAME
```

</details>

<details>
<summary>piper-robot model set --help</summary>

```text
usage: cli.py model set [-h] [--output OUTPUT] [--endpoint ENDPOINT]
                        [--name NAME] [--api-key-env API_KEY_ENV]

options:
  -h, --help            show this help message and exit
  --output OUTPUT
  --endpoint ENDPOINT
  --name NAME
  --api-key-env API_KEY_ENV
```

</details>

<details>
<summary>piper-robot model list --help</summary>

```text
usage: cli.py model list [-h] [--output OUTPUT] [--endpoint ENDPOINT]
                         [--name NAME]

options:
  -h, --help           show this help message and exit
  --output OUTPUT
  --endpoint ENDPOINT
  --name NAME
```

</details>

<details>
<summary>piper-robot model check --help</summary>

```text
usage: cli.py model check [-h] [--output OUTPUT] [--endpoint ENDPOINT]
                          [--name NAME]

options:
  -h, --help           show this help message and exit
  --output OUTPUT
  --endpoint ENDPOINT
  --name NAME
```

</details>

<details>
<summary>piper-robot model run --help</summary>

```text
usage: cli.py model run [-h] [--output OUTPUT] [--endpoint ENDPOINT]
                        [--name NAME] [--allow-motion]
                        prompt

positional arguments:
  prompt

options:
  -h, --help           show this help message and exit
  --output OUTPUT
  --endpoint ENDPOINT
  --name NAME
  --allow-motion       Allow motion proposals in this bounded turn
```

</details>
