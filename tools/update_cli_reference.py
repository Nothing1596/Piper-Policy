"""Generate command references from actual --help output; never start a backend."""
import argparse
import os
from pathlib import Path
import subprocess

VIDEO = [
    ('candidates', '本地 CV 候选帧 / Local candidate extraction'),
    ('compile', '模型解析视频并生成证据包 / Compile a demonstration'),
    ('inspect', '读取示范包或示范 ID / Inspect a bundle or ID'),
    ('find', '检索示范库 / Search the demonstration store'),
    ('evaluate', '与给定人工标注比较 / Compare supplied annotations'),
    ('models show', '显示配置，不进行推理 / Show configuration only'),
    ('models probe', '单图真实模型请求 / Real single-image API probe'),
    ('mcp', '视频工具 stdio 服务 / Video MCP stdio server'),
]
ROBOT = [
    ('sim start', '初始化并启动 MuJoCo / Initialize and serve MuJoCo'),
    ('observe', '保存当前 RGB-D / Capture current RGB-D'),
    ('mcp', '独立 stdio MCP 桥 / Independent stdio MCP bridge'),
    ('init', '创建配置与凭据 / Initialize configuration and credentials'),
    ('serve', '启动共享执行器 / Start shared executor'),
    ('shell', '交互终端 / Interactive console'),
    ('connect', '连接设备 / Connect device'),
    ('disconnect', '释放连接 / Disconnect device'),
    ('state', '原始状态 / Raw state'),
    ('status', '状态概览 / Status summary'),
    ('monitor', '持续状态采样 / Repeated status reads'),
    ('tools', '执行器工具清单 / Executor tool inventory'),
    ('params', '配置与实测参数 / Configured and measured parameters'),
    ('calls', '调用记录 / Call history'),
    ('jobs', '作业清单 / Job inventory'),
    ('job', '按 job_id 查询或等待 / Query or await a job'),
    ('request', '按 request_id 恢复查询 / Query by request ID'),
    ('events', '增量事件 / Incremental events'),
    ('doctor', '反馈、新鲜度与诊断 / Feedback freshness and diagnostics'),
    ('devices', '枚举执行器端设备 / Enumerate executor-side devices'),
    ('limits', '限位及来源，可主动刷新 / Limits and provenance, optional refresh'),
    ('control-mode', '设置控制模式 / Set control mode'),
    ('move-joints', '六关节绝对角度 / Six absolute joint angles'),
    ('gripper', '夹爪绝对开度 / Absolute gripper opening'),
    ('move-to', 'TCP 绝对位置 / Absolute TCP position'),
    ('move-by', 'TCP 相对平移 / Relative TCP translation'),
    ('rotate', 'TCP 绝对姿态 / Absolute TCP orientation'),
    ('move-linear', '定姿态直线采样 / Fixed-orientation linear sampling'),
    ('preview', '预览 JSON 命令 / Preview a JSON command'),
    ('execute', '执行 plan_id / Execute a previewed plan'),
    ('stop', '请求停止 / Request stop'),
    ('shutdown', '空闲时关闭服务 / Shut down idle executor'),
    ('arm', '操作员限时控制窗口 / Operator control window'),
    ('estop', '锁存软件停止 / Latch software stop'),
    ('clear-estop', '清除软件停止锁存 / Clear software stop latch'),
    ('configure-runtime', '操作员运行参数 / Operator runtime configuration'),
    ('sim-fault', '注入仿真故障 / Inject simulation faults'),
    ('manual', '调用具名 MCP 工具 / Invoke a named MCP tool'),
    ('model show', '读取模型配置 / Read model configuration'),
    ('model set', '保存端点与模型名 / Save endpoint and model configuration'),
    ('model list', '查询服务模型清单 / List server models'),
    ('model check', '检查模型服务 / Check model service'),
    ('model run', '有界工具调用回合 / Bounded model tool-calling turn'),
]

VIDEO_INTRO = '''# piper-video 命令与功能 / Command reference

适用 **0.3.0**。[返回中文 README](../README.md) · [English README](../README.en.md) · [安装向导](VIDEO-GUIDE.zh.md) · [Installation guide](VIDEO-GUIDE.en.md)

前端为 `piper-video`；Windows 离线包使用 `.\\piper-video.cmd`。只负责视频证据，不提供机器人动作或仿真启动。The video-only CLI selects frames, interprets demonstrations and manages evidence; it does not execute robot actions.

## 调用流程 / Workflow

```powershell
.\\piper-video.cmd candidates --video examples\\transfer.mp4 --output work\\candidates-01
.\\piper-video.cmd models show --model-config profiles\\openai.json
.\\piper-video.cmd models probe --model-config profiles\\openai.json --image examples\\probe.jpg --output work\\probe-01
.\\piper-video.cmd compile --video examples\\transfer.mp4 --task "描述动作与最终位置，不确定则记录未知" --model-config profiles\\openai.json --output work\\demo-01 --store work\\demos.sqlite
.\\piper-video.cmd inspect --bundle work\\demo-01
.\\piper-video.cmd find --store work\\demos.sqlite --task "搬运"
.\\piper-video.cmd mcp --workspace . --model-config profiles\\openai.json
```

配置自己的视觉模型与密钥变量后再运行 `probe` / `compile`。选帧不调用大模型；`show` 不验证推理。Use your own vision model and environment-variable credentials before API calls. Candidate extraction is local; configuration display is not an inference test.

## 参数与结果 / Semantics

- `compile`：`--max-keyframes` 默认 24；`--detector-onnx` 可选；`--cache` 指定缓存；`--store` 注册结果。使用 `--model-config` 时不能同时使用旧的 `--model` / `--model-url` / `--model-manifest`。
- `inspect`：`--bundle` 与 `--id` 二选一；ID 查询需要 `--store`。`evaluate` 需要示范包、标注 JSON 和输出路径；它比较给定标注，不生成独立人工真值。
- 模型配置 JSON 的主要字段：`provider`、`model`、`base_url`、`api_key_env`、`timeout_s`（默认 180）、`max_images`（默认 6）、`max_output_tokens`（默认 4096）、`structured_output`、`trust_env`、`artifact_manifest`。Provider adapters: LM Studio, OpenAI Responses, OpenAI-compatible, Anthropic, Gemini. Endpoint/model compatibility must be verified with `probe`.
- 输出通常为 JSON。示范保存到 `demo.json`，带阶段、图像引用和 unknown。候选帧保存到 `manifest.json`。模型日志保留在输出旁的 model-calls 目录。使用新输出目录，避免覆盖证据。
- CLI 正常返回 0；示范 unresolved 等已识别失败结果返回 2；参数错误通常也是 2，其他异常返回非零。Read both the exit code and JSON verdict. A model's `supported` is not human acceptance or robot task success.
- `--max-keyframes` 是选帧预算，`max_images` 是模型请求配置，二者不等于上下文 token 上限。它不负责自动训练权重或跨场景技能迁移。

## MCP 工具 / MCP tools

启动参数：`mcp --workspace DIR [--model-config JSON] [--detector-onnx FILE]`。workspace 必须已存在；工具路径限制在其中。stdio 服务等待客户端协议输入，不是交互聊天窗口。Without a model profile, local CV and evidence tools remain available; compilation reports a missing profile.

| 工具 / Tool | 参数 / Arguments | 功能 / Behavior |
|---|---|---|
| `pipeline_capabilities` | 无 / none | 显示配置；不会验证推理 / Configuration only |
| `video_candidates` | `video`, `output` | 本地选帧 / Local candidate extraction |
| `video_candidate_page` | `manifest`, `offset=0`, `limit=24` | 分页帧信息，limit 1..100 / Page frame metadata |
| `video_frame` | `image` | 返回实际图像 / Return pixels |
| `video_compile` | `video`, `task`, `output`, `max_keyframes=24` | 固定配置模型解析；发送选中图片 / Compile via configured provider |
| `video_inspect` | `bundle` | 读取结论与未知项 / Read claims and unknowns |
| `video_evaluate` | `bundle`, `annotations`, `output` | 对照标注 / Compare annotations |

MCP 与 CLI 的参数并非完全一一对应，例如 MCP 编译不接受任意模型地址，也没有 store 参数。外部代理可以自己看 `video_frame` 返回的图片；这不等于把代理登录态当作模型 API。The MCP profile is fixed by the operator; CLI and MCP argument sets differ.
'''

ROBOT_INTRO = '''# piper-robot 命令与功能 / Command reference

适用 **0.6.0**。[返回中文 README](../README.md) · [English README](../README.en.md) · [安装向导](ROBOT-GUIDE.zh.md) · [Installation guide](ROBOT-GUIDE.en.md)

Windows 离线包入口 `.\\piper-robot.cmd`；安装环境中也可用 `piper-robot`。保留 `piperx` 与 `piperx-mcp` 旧入口。The lower controller is host middleware, not arm firmware.

## 两类语法 / Invocation forms

包装命令的参数放在命令后；执行器命令的全局参数放在命令前：

```text
piper-robot sim start --root DIR [--port 8798] [--seed 200]
piper-robot observe --root DIR [--url URL] --workspace DIR --output NEW_PATH
piper-robot mcp [--root DIR] [--url URL] [--token-file FILE] [--workspace DIR]
piper-robot [--root DIR] [--url URL] [--token-file FILE] COMMAND [ARGS]
```

`sim start`、`observe`、`mcp` 使用各自的参数解析器。不要写成 `piper-robot --root DIR sim start`。The wrapper commands have their own parsers; executor global flags precede its command.

执行器默认 root：Windows `%LOCALAPPDATA%/PiperXMiddleware`；其他平台 `~/.local/state/piperx-middleware`。常规控制命令的 URL 顺序：显式 `--url` → `PIPERX_URL` → root 配置 → `http://127.0.0.1:8765`。独立 `mcp` / `observe` 使用 RobotClient 默认地址，不自动套用 sim 的 8798 端口；建议始终显式传 `--url`。

`init` / `serve` 使用本地 root/config，不接受全局 `--url` 或 `--token-file`。`init --backend sim` 是轻量模拟；`mujoco` 才是物理仿真；`agx` 是真实后端。默认 init 端口 8765，与 `sim start` 默认 8798 不同。

## 仿真例子 / Simulation example

终端 A / Terminal A:

```powershell
.\\piper-robot.cmd sim start --root work\\sim --port 8798 --seed 200
```

终端 B / Terminal B:

```powershell
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 connect
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 status --json
.\\piper-robot.cmd observe --root work\\sim --url http://127.0.0.1:8798 --workspace . --output work\\frame-01
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 jobs --json
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 stop
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 shutdown
```

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

动作命令通常有 `--request-id`、`--wait`、`--wait-timeout`（默认 130 秒）。未给 ID 时生成一次并写到 stderr；同一动作重试沿用原 ID。`--speed` 默认 5%，动作 `--timeout` 默认 30 秒。`gripper` 没有 speed；`execute` 使用已有计划。

提交成功返回 job_id，不代表动作完成。`job ID --wait` / `request ID --wait` 查询结果。等待超时返回 outcome_unknown，不自动停机、撤销或重放。`--output FILE` 独占创建 JSON 文件；多数脚本命令 stdout 是 JSON。失败/取消/未知等结果返回非零。Status summaries default to readable text; use `--json` for scripts.

`gripper --completion bilateral_contact` 仅适用于 MuJoCo 闭合时的稳定双指接触，不证明抬起成功。`stop` 请求软件停止；`estop` 锁存软件停止；`shutdown` 只在空闲时释放连接并退出；都不替代物理急停。

`arm`、`estop`、`clear-estop`、`configure-runtime`、`sim-fault`、`limits --refresh` 使用操作员权限，默认读取 root/operator.token；普通命令默认 model.token。`--token-file` / `PIPERX_TOKEN_FILE` 可覆盖。MCP 模型工具不暴露这些管理命令。

## 内置模型与交互终端 / Built-in model and console

`model`（单数）是下位机自己的 OpenAI-compatible 文本/工具调用入口，不是视频侧的 `models`，也不是多供应商视频配置。Configuration lives in root/model.json; it stores credential references, not key values.

```powershell
.\\piper-robot.cmd --root work\\sim model set --endpoint http://127.0.0.1:1234/v1 --name YOUR_MODEL --api-key-env PIPERX_MODEL_API_KEY
.\\piper-robot.cmd --root work\\sim model show
.\\piper-robot.cmd --root work\\sim model list
.\\piper-robot.cmd --root work\\sim model check
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 model run "只读取并总结当前状态"
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 manual "robot_status()"
.\\piper-robot.cmd --root work\\sim --url http://127.0.0.1:8798 shell
```

`model check` 会发起一次不带工具的最小真实补全请求，不只是检查配置。`control-mode` 固定请求 CAN/MOVE_J 控制切换，不接受任意模式或使能/复位字段。

`model run --allow-motion` 显式允许该回合提出动作，仍受后端权限和校验限制。`show/set` 不证明模型可用。内置回合不等同于视频理解或完整视觉策略。要让外部视觉代理看图，请使用下述带 workspace 的独立 MCP。

交互命令 / Console commands: `/help`, `/status`, `/connect`, `/disconnect`, `/model`, `/manual`, `/tools`, `/calls`, `/jobs`, `/params`, `/stop`, `/quit`。`/quit` 不会停止机器人或断开 CAN。直接 `piper-robot` 显示帮助；使用 `shell` 明确进入交互终端。

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

支持命令不等于已完成真机验收。SDK、CAN 驱动、标定需现场部署；独立包已验证的范围见 [验收记录](SPLIT-VALIDATION.md)。Supported commands do not establish physical hardware acceptance.
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video-python', required=True)
    p.add_argument('--robot-python', required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    env.pop('PYTHONPATH', None)
    for kind, interpreter, module, commands, intro in [
        ('VIDEO',a.video_python,'piperlab.video_entry',VIDEO,VIDEO_INTRO),
        ('ROBOT',a.robot_python,'piperx_middleware.standalone_cli',ROBOT,ROBOT_INTRO),
    ]:
        text = intro+'\n## 命令索引 / Command index\n\n| Command | 功能 / Function |\n|---|---|\n'
        text += ''.join(f'| `{name}` | {purpose} |\n' for name,purpose in commands)
        text += '\n## 完整参数 / Exact help snapshots\n\n以下内容来自对应发行版的实际 `--help`，未启动后端或调用模型。Generated from installed release help, without starting a backend or calling a model. Video subcommand help retains the legacy `piper-lab demo` program label; invoke it as `piper-video` followed by the listed command.\n'
        for name, _ in [('', ''), *commands]:
            result = subprocess.run([interpreter,'-m',module,*name.split(),'--help'],env=env,
                                    capture_output=True,text=True,encoding='utf-8',check=True,timeout=20)
            display = 'piper-'+kind.lower()+(' '+name if name else '')
            text += f'\n<details>\n<summary>{display} --help</summary>\n\n```text\n{result.stdout.rstrip()}\n```\n\n</details>\n'
        path = repo/'docs'/f'{kind}-CLI-REFERENCE.md'
        path.write_text(text,encoding='utf-8')
        print(f'{path.name}: {len(commands)} command entries verified')


if __name__ == '__main__':
    main()
