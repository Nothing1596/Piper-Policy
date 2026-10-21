# Robot 0.6.1：DeepSeek 兼容更新与使用说明

[返回版本导航](README.zh.md) · [正式下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.6.1) · [该 tag 的配置指南](https://github.com/Nothing1596/Piper-Policy/blob/robot-v0.6.1/docs/DEEPSEEK-SETUP.zh.md)

## 版本与改动

2026-09-27 发布的更新 wheel `piperx_middleware-0.6.1-py3-none-any.whl`，用于已经装好的 Robot 0.6.0 环境。没有完整离线依赖，不是新机器安装包。共享执行器、运动与 MCP 的使用方式继承 0.6.0。

主要变化：内置文本工具循环在代码识别的官方 DeepSeek 端点上显式发送关闭 thinking 的参数，避免旧多轮工具协议不匹配；其他兼容端点保持原行为。这是该版本的历史协议策略，不是对今天所有 DeepSeek 产品能力的声明，也不是通过 `/no_think` 提示词实现。

## 升级已安装环境

退出旧客户端和旧执行器，在完整包目录核对发布页 SHA256SUMS.txt 后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --no-deps --upgrade .\piperx_middleware-0.6.1-py3-none-any.whl
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import piperx_middleware; print(piperx_middleware.__version__)"
```

最后应显示 0.6.1。升级后不要重新执行旧 0.6.0 固定依赖安装器，否则可能降级。新机器先安装完整 0.6.0 再更新，或直接选择较新的 0.7.1；不要把本 wheel 当完整包。

## 模型配置与只读验收

操作者先在当前窗口设置自己的 DEEPSEEK_API_KEY 环境变量，不把字面密钥写进命令历史或 JSON。模型名和服务端点以自己的供应商配置为准，下面的占位值需要替换：

```powershell
$RobotDataRoot = Join-Path $PWD 'work\ds-check'
.\piper-robot.cmd --root $RobotDataRoot model set --endpoint https://YOUR-ENDPOINT/v1 --name YOUR-MODEL --api-key-env DEEPSEEK_API_KEY
.\piper-robot.cmd --root $RobotDataRoot model show
.\piper-robot.cmd --root $RobotDataRoot model list
.\piper-robot.cmd --root $RobotDataRoot model check
```

配置只保存变量引用。show 不调用推理，list 不证明工具调用，check 才发小型真实请求。只有该版代码识别的官方 DeepSeek 地址会获得本次协议修正，自建网关不能仅靠模型名获得相同行为。

终端 A 用这个 root 启动仿真：

```powershell
.\piper-robot.cmd sim start --root work\ds-check --port 8798 --seed 200
```

终端 B 保留先前的环境变量：

```powershell
.\piper-robot.cmd --root $RobotDataRoot --url http://127.0.0.1:8798 connect
.\piper-robot.cmd --root $RobotDataRoot --url http://127.0.0.1:8798 model run '调用 robot_status 说明当前连接和关节反馈，不执行动作。'
```

检查实际 tool_results 中有 robot_status。未加 `--allow-motion` 的这条入口只提供只读工具；模型返回自然语言“成功”不等于调用已经发生。需要动作时仍须执行器当前权限、反馈、限位和旧控制机制，不能靠模型开关绕过。

## 能力边界和故障

本版内置入口传文字及工具结果，不自动上传相机帧；接通 DeepSeek 不意味着具备真人视频到真机复现。视觉视频应另外安装 video 管线，外部代理可独立看图调用机器人 MCP。

401/403 查凭据及权限，超时查网络，工具参数错误查实际 schema；不要通过盲换模型或重发动作处理未知结果。关闭与停止沿用 0.6.0。公开保存验证为 52 通过/5 外部 SDK 跳过、已有环境升级和 pip check；模型协议用模拟 HTTP，未在该发布验证中实测用户云账户或真机。本次没有重新请求供应商。
