# 新机器部署向导：从解压到看图执行

适用版本：piper-lab **0.2.1** + piperx-middleware **0.5.0**，Windows x64、Python 3.12。本文中的命令在 PowerShell 执行。没有自动配置向导窗口；这份文档就是逐步操作向导。

## 先理解三个部分

CLI 是你下命令的入口；执行后端是常驻进程，负责机械臂状态、队列和动作；模型或外部代理负责决定下一步。安装程序不会自动启动这三者。

| 使用方式 | 你给谁下任务 | 谁组织循环 | 管线是否要配置模型 |
|---|---|---|---|
| 管线自主执行 | `piper policy run --task ...` | 管线取图→模型决策→动作校验→执行→新图复核 | 要 |
| 代理使用工具 | Codex、Claude Code 等 | 代理读图并调用 MCP，后端返回执行反馈 | 仅看图和操作仿真时不用；`video_compile` 仍需要 |
| 直接操作下位 CLI | `piper device ...` | 你逐条控制 | 不用 |

上位 `--model-config`、下位 `device model` 和 Codex 自身的模型设置是不同配置。本向导主要使用上位 profile 或 MCP，不要求同时配置三个模型。

## 1. 解压和准备 Python

把 ZIP 解压到固定目录，例如 `D:\PiperCLI`，找到其中同时包含 `install.ps1`、`piper.cmd`、`wheels` 的文件夹。后续称它为“安装目录”。不要在 ZIP 预览窗口里直接运行。

本包只提供 Windows x64 / CPython 3.12 的依赖 wheel，不能直接用于 Linux、macOS、ARM 或 Python 3.13。检查：

```powershell
py -3.12 -c "import sys,struct; print(sys.version); print(struct.calcsize('P')*8)"
```

应看到 3.12.x 和 64。如果找不到 `py`，安装 Python 3.12 x64，或在下一步使用已有解释器的绝对路径。安装器会再次检查版本和位数。

## 2. 安装并确认完整性

在安装目录打开 PowerShell：

```powershell
Set-Location 'D:\PiperCLI\piper-cli' # 改为你实际的安装目录
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper.cmd doctor
.\piper.cmd --help
```

若解释器不由 `py` 管理：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1 -PythonExe 'C:\Path\Python312\python.exe'
```

安装器先逐文件核对 `SHA256.json`，再创建 `.venv`，从包内 101 个 wheel 离线安装并做 `pip check`。无需激活环境，`piper.cmd` 会使用本包自己的 Python。不要在安装前修改包内 profiles 或文档，否则完整性检查会拒绝；安装后复制 profile 到工作目录再修改。

成功标准：最后出现 `Installed`；`doctor` 的 `ok` 为 true，piper-lab 为 0.2.1、piperx-middleware 为 0.5.0。这里的 `model_inference_tested:false`、`hardware_connected:false` 是正常结果。

如果失败，不要在损坏的 `.venv` 上反复覆盖。保存报错，重新解压到另一个新目录处理。安装后的 `.venv` 包含绝对路径，不能直接搬到另一台电脑；每台机器从 ZIP 重新安装。

## 3. 不用模型，先把仿真连起来

先建立工作目录。以下各终端都从同一个安装目录执行：

```powershell
New-Item -ItemType Directory -Force .\work | Out-Null
```

**终端 A**：启动 MuJoCo，保持窗口运行。

```powershell
.\piper.cmd sim start --root work\sim --port 8808 --seed 200
```

看到监听 `127.0.0.1:8808` 说明进程启动；并不表示已经连接仿真机器人。它是无独立场景窗口的后端，相机画面通过 MCP 获取。它会在 `work\sim` 创建配置和本机 token；不需要拷贝旧机器的 token。

**终端 B**：连接并检查。

```powershell
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 connect
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 status
```

应看到后端 MuJoCo、连接成功、新鲜反馈、没有活动作业。第一次安装或服务重启后，应明确执行 connect，不能省略。端口被占用时，选其他空闲端口，并同步修改后面所有 URL。已有 root 的 seed/port/backend 不一致时，使用新的 root，不要覆盖旧试验。

## 4A. 让 Codex / Claude Code 看图操作

这一路径可以先不配置本地模型或云 API。前提是你的代理客户端已安装、登录，并支持 MCP 图像内容；该代理可能把图片发送到它自己的模型服务。

仍在终端 B 的安装目录：

```powershell
$piperPython = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$piperWork = (Resolve-Path '.\work').Path
$piperSim = (Resolve-Path '.\work\sim').Path
```

Codex CLI 注册：

```powershell
codex mcp add piper -- $piperPython -m piperlab.harness --workspace $piperWork --robot-root $piperSim --robot-url http://127.0.0.1:8808
codex mcp get piper --json
```

Claude Code 注册：

```powershell
claude mcp add --transport stdio piper -- $piperPython -m piperlab.harness --workspace $piperWork --robot-root $piperSim --robot-url http://127.0.0.1:8808
```

这两个命令选择自己使用的客户端即可，不用都执行。若客户端已有同名配置，先核对它的用途，再选择新的名称；不要覆盖别人的服务。重新连接 MCP 或打开新的代理会话，确认工具列表包含 `robot_status`、`simulation_observe`、`move_to` 等。具体刷新操作取决于客户端，单纯注册成功不代表当前会话已加载。

可把下面这段任务发给代理：

> 当前只操作 MuJoCo 仿真。先调用 robot_status 确认后端和连接；调用 simulation_observe，output 使用 observations/first-01，查看返回的实际图像。描述红块、蓝块和绿色托盘的位置。确认目标后，把红块放进绿色托盘；每个动作等待作业完成，不重发未知结果。抓起后和释放退开后重新取图，报告可见结果及不确定性。不要连接实体机械臂。

`simulation_observe` 保存 `rgb.jpg`、`depth.npy`、`observation.json`，并直接向代理返回图片；没有隐藏物体坐标或评分。output 相对 workspace，必须使用新名字。重试可用 `observations/first-02`。工作目录只暴露任务素材，不要设为整个用户目录。

如果代理说“没有看图工具”，先确认注册的是 0.2.1 的解释器，且参数带 `--robot-root`。只运行 MCP 服务不会自动启动仿真，终端 A 必须还在运行。单独在终端运行 `piper mcp ...` 后没有提示符也属正常：它正等待 MCP 客户端的协议输入，不是聊天终端。

## 4B. 配置模型，让管线自行循环执行

选择一个 profile，复制后再修改。以下以 OpenAI API 为例：

```powershell
Copy-Item .\profiles\openai.json .\work\model.json
notepad .\work\model.json
```

把 `YOUR_VISION_MODEL_ID` 改为你账户实际可用、支持图片与结构化输出的模型 ID。OpenAI 兼容服务还要修改 base_url；路径通常应包括 `/v1`，不要重复添加 `/chat/completions`。各 provider 与变量如下：

| profile | 要设置的变量 / 条件 |
|---|---|
| lmstudio.json | 本机已有视觉模型已加载，服务在 127.0.0.1:1234；model 与服务实际标识一致 |
| openai.json | OPENAI_API_KEY |
| openai-compatible.json | PIPER_MODEL_API_KEY，实际 HTTPS 地址和视觉模型 ID |
| anthropic.json | ANTHROPIC_API_KEY |
| gemini.json | GEMINI_API_KEY |

PowerShell 不回显输入的密钥设置方法（示例变量用于 OpenAI）：

```powershell
$keyInput = Read-Host 'API key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $keyInput).Password
Remove-Variable keyInput
.\piper.cmd models show --model-config work\model.json
```

密钥仍会以正常 API 客户端所需的形式存在于进程环境中；上述方式避免把字面值写入命令历史。这个变量仅对当前终端和它之后启动的子进程生效。不要把密钥写进配置 JSON、报告或截图。已经打开的代理应用不会自动继承新终端的环境变量。

先做一次单图调用：

```powershell
.\piper.cmd models probe --model-config work\model.json --image examples\probe.jpg --output work\probe-01
```

成功标准是生成 result.json 和成功调用记录，而不只是 show 显示 key 存在。包内 probe.jpg 来源于合成示范，不能证明模型识别人类动作的能力。云端调用可能收费；本安装程序没有执行任何云端调用。

本地 LM Studio 同样需要先 probe。包不安装、下载或自动加载模型，也不会根据旧机器的显卡序号分配设备。根据新机器显存选择模型和上下文，并发先设 1；不要把本机曾验证的 T10+CPU 配置当作所有机器通用设置。

选用兼容端点而它只支持 JSON object 时，可明确配置 `structured_output: "json_object"`；本地仍验证业务 schema。不同供应商对 schema 的支持范围可能不同，单图 probe 通过后仍要测试真实任务。不要把 HTTP 200 当作任务成功。

## 5. 编译参考视频（可选）

先用合成样例熟悉目录结构，再替换成自己的视频：

```powershell
.\piper.cmd demo compile --model-config work\model.json --video examples\transfer.mp4 --task "Move the red block into the green tray as demonstrated." --output work\demo-01 --store work\demo-store --detector-onnx models\yolo11n.onnx
.\piper.cmd demo inspect --bundle work\demo-01
.\piper.cmd demo find --store work\demo-store --task "red block tray"
```

管线先解码和选帧，再让模型筛选关键帧、解析阶段和证据。查看 demo.json 中的 stages、unknowns、outcome_verdict 和引用图片。`unknown` 或 `unresolved` 不能作为已完整理解视频的证明。它形成可引用的示范记忆，不会更新模型权重。

使用 API 时，选出的图像及提示会发送给所选供应商。示范记录和日志可能包含任务内容；请先确认素材适合发送。云端模型别名不能证明远端权重版本，因此不要为 API 配置需要权重身份证明的持久 `--cache`。

如果使用代理调用 `video_compile`，MCP 启动参数还要追加 `--model-config` 和配置文件的绝对路径。只让代理看 `video_frame` 或 `simulation_observe` 时，无需该配置。

## 6. 自主执行仿真任务

保持终端 A 运行，确认终端 B 的模型环境变量仍存在，且第 3 步 connect/status 正常：

```powershell
.\piper.cmd policy run --model-config work\model.json --task "Move the red block into the green tray." --backend http://127.0.0.1:8808 --token-file work\sim\model.token --output work\policy-01 --max-decisions 12
```

如需参考第 5 步的示范，在命令末尾追加 `--demo work\demo-01`。模型提出结构化动作；程序用当前图像、深度、候选区域及校验规则处理，再交执行器运行，并用新画面复核。不是把任意模型文本当作 shell 命令执行。

当前视觉策略针对 30 mm 彩块与绿色托盘、有限抓放/推移动作，并要求 MuJoCo。不能把该命令的 backend 改成实体机械臂地址就宣称完成真机部署。它也不是任意物体、任意自然语言任务的通用机器人。

完成后看 `work\policy-01\report.json`、报告 HTML、逐步记录和前后图片。模型说 done 只是声明，继续做独立评分：

```powershell
.\piper.cmd sim evaluate --backend http://127.0.0.1:8808 --token-file work\sim\model.token --operator-token-file work\sim\operator.token
```

红块任务重点核对 cube_red.in_tray=true、grasped=false 和最终图像。评分需要 operator token，应在动作结束后使用，不要把它交给规划模型作为作弊答案。一次成功不等于多场景成功率。

## 7. 停止、退出和下一次运行

```powershell
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 stop
.\piper.cmd device --root work\sim --url http://127.0.0.1:8808 shutdown
```

stop 停当前动作；shutdown 在空闲时退出执行器；二者含义不同。Ctrl-C 停一个监视窗口不等于停止机器人。请求超时先按原 request_id/job_id 查询，结果未知时不要换一个新 ID 重发动作。

下一次从第 3 步启动并连接，用 `policy-02`、`demo-02` 等新目录。重启 MuJoCo 会重建场景，不保留上次物理终态。迁移到新机器时复制素材/必要结果，不复制已安装 .venv；重新生成连接配置和 token。

## 8. 连接实体机械臂之前

本包的安装与自检不连接 CAN、不使能、不回零、不校准。下位机提供独立的设备枚举、状态、控制和动作接口；用 `piper device --help`、`piper device doctor --help`、`piper device init --help` 查看实际参数。

真机部署还取决于 CAN 适配器/驱动、固件及协议、限位、工具 TCP、相机标定和操作员授权。当前真实视觉闭环输入来自 MuJoCo；RealSense/真机接入和安全验收不能仅凭本包的仿真成绩视为完成。尚未准备这些条件时停留在仿真步骤即可。

## 9. 常见问题

| 现象 | 检查和处理 |
|---|---|
| py 或 Python 3.12 找不到 | 指定 -PythonExe；确认是 x64 和 3.12 |
| Existing .venv / Hash mismatch | 新目录重新解压；不要在安装前修改原包 |
| Could not find distribution | 本包限定 Windows x64 Python 3.12；不要改成联网安装掩盖版本问题 |
| 连接拒绝、401 | 确认终端 A、端口、root、该实例生成的 token；不要打印 token 内容 |
| not_connected | 用同一 root 和 URL 执行 device connect |
| 相机工具不存在 | 0.2.1 解释器、robot-root 参数、重新加载 MCP；只有视频模式时没有相机工具 |
| OpenGL/渲染失败 | 模型推理与渲染是不同依赖；检查显示驱动和当前会话图形支持，不把仅资产加载通过当相机通过 |
| missing_api_key_env | 在启动管线或 MCP 的那个进程环境设置对应变量 |
| 401/403/404/429 | 核对账户权限、模型 ID、地址、额度；不自动切供应商 |
| schema 不支持 | 先确认模型支持视觉/结构化输出；兼容服务可显式 json_object，仍需任务验证 |
| 截断、上下文溢出、超时 | 查看错误记录；调整模型/上下文/预算或用较短视频重新运行，不把不完整 JSON 当成功 |
| output already exists | 新运行使用新目录；保留失败证据 |
| no_free_reachable_placement / 执行失败 | 查看实时图片、区域和作业错误，勿放宽限制硬执行 |
| 模型说成功，但实际没完成 | 以新画面和独立评分为准，保留报告供排查 |

## 10. 本包到底验证了什么

查看包根目录 `PACKAGE-VALIDATION.json`，它记录该包实际的安装、CLI、解码、检测器接口、MuJoCo 资产和 MCP 通信检查。`docs/RELEASE-NOTES.md` 区分历史实际任务与本次打包测试。API 协议模拟测试不等于你的账户已接通，仿真不等于真机验收。

包内模型权重清单 `model-files.json` 仅用于对应既有文件的校验，不下载模型，也不证明当前服务加载了哪个文件。第三方许可见 `THIRD-PARTY.md` 和 `licenses/`。
