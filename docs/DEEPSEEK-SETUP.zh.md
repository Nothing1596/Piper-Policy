# 无需 Codex：下位机接入 DeepSeek API

流程：用户文字 → 下位机内置模型循环 → DeepSeek 返回工具调用 → 本地执行器执行 → 回传工具结果。远端电脑不需要 Codex，也不需要本地大模型/GPU。

当前官方配置为 `https://api.deepseek.com`、`deepseek-flash`，见 [首次调用](https://api-docs.deepseek.com/zh-cn/)。DeepSeek 默认开启思考模式，工具多轮要求回传 reasoning_content；0.6.1 对官方端点显式设置 `thinking: {type: disabled}`，使用非思考工具循环。提示词中的 `/no_think` 不能替代 API 参数。见 [思考模式要求](https://api-docs.deepseek.com/guides/thinking_mode/)。

本次兼容测试使用模拟 HTTP 响应，验证探测、工具调用、工具结果回传，以及非 DeepSeek 端点不受影响；尚未使用用户密钥做真实 DeepSeek 请求。

## 1. 升级已安装的下位机包

先完成原完整包安装，打开**完整包根目录**的 PowerShell。从 [robot-v0.6.1](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.6.1) 下载 wheel 到当前目录。

```powershell
$wheel = '.\piperx_middleware-0.6.1-py3-none-any.whl'
if ((Get-FileHash $wheel -Algorithm SHA256).Hash -ne 'ee6b259e8ed22c0d24159209a61638185b053a96761c774054e52c42ac787ec4') { throw 'SHA256 mismatch' }
& .\.venv\Scripts\python.exe -m pip install --no-index --no-deps --upgrade $wheel
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -c 'import piperx_middleware;print(piperx_middleware.__version__)'
```

应显示 `0.6.1`。升级已有环境即可，不复制源码、不安装 Codex。旧 0.6.0 ZIP 的锁文件仍固定旧版本，**升级后不要再运行旧包安装器**，否则可能降回旧版；此 wheel 是现有安装的更新，不是完整离线依赖包。

## 2. 配置密钥与模型

先在 [DeepSeek 平台](https://platform.deepseek.com/) 创建有可用额度的 API key。在当前 PowerShell 中输入密钥，输入不回显，也不进入命令历史：

```powershell
$env:DEEPSEEK_API_KEY = [Net.NetworkCredential]::new('', (Read-Host 'DeepSeek API key' -AsSecureString)).Password
$Root = Join-Path $PWD 'work\ds-check'
.\piper-robot.cmd --root $Root model set --endpoint https://api.deepseek.com --name deepseek-flash --api-key-env DEEPSEEK_API_KEY
.\piper-robot.cmd --root $Root model show
.\piper-robot.cmd --root $Root model list
.\piper-robot.cmd --root $Root model check
```

环境变量只对当前窗口及其子进程有效；新开窗口需重新设置。配置文件只记录变量名，不记录密钥。`list` 检查模型清单；`check` 才发送一条真实小请求。401 通常先查密钥，402 查账户额度，超时查远端网络；只提供错误文字，不发送密钥。

## 3. 先在仿真完成一次只读工具调用

终端 A，在同一个完整包根目录：

```powershell
.\piper-robot.cmd sim start --root .\work\ds-check --port 8798 --seed 200
```

终端 B，继续使用上一步已经设置密钥的窗口：

```powershell
.\piper-robot.cmd --root $Root --url http://127.0.0.1:8798 connect
.\piper-robot.cmd --root $Root --url http://127.0.0.1:8798 model run '调用 robot_status 读取当前状态，用中文说明连接状态和关节反馈，不执行动作。'
```

检查报告的 `tool_results` 是否有 `robot_status` 及实际返回值，不能只看模型说“成功”。不加 `--allow-motion` 时，该入口只提供 `robot_status` / `robot_diagnostics` 两个只读工具。`model check` 通过不等于工具闭环通过。

## 4. 改用真机已有执行器

真机需要已有正确的 `agx` 配置、厂商 SDK、CAN 接口和运行中的执行器；USB/CAN 已插上不代表这些已就绪。将 `--root` 换成**已有真机执行器的数据目录**，`--url` 换成其实际地址，并对这个 root 重新执行 `model set`。先用 `status --json` 核对 backend 是 `agx` 和反馈有效，再运行上面的只读 `model run`。不要用 `sim start` 或新初始化目录替代已有真机配置。

正式动作入口是 `model run '具体任务' --allow-motion`；它只允许模型提出动作，不绕过控制器的权限、反馈、限位和完成判定。本指南不提供未经标定的真机移动目标。确认型号、坐标、急停与执行器配置后再做单步验收。

当前内置入口传递文字和工具结果，不自动上传相机帧；接通 DeepSeek 不等于完成“看真人视频并让真机复现”。视频管线、视觉输入与真机技能迁移是另外的验收项。
