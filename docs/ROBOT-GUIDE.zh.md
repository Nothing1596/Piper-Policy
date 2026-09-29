# Piper Robot 0.7.1 部署指南

[English](ROBOT-GUIDE.en.md) · [下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.7.1) · [版本说明](ROBOT-RELEASE-0.7.1.md)

一个终端完成选择模式、连接、工具调用、审批和退出。此包包含机器人执行器、HTTP/MCP、仿真资源和源码，不包含视频解析管线或模型服务，也不会刷写机械臂固件。

## 安装

准备 **Python 3.11+ 和可下载依赖的网络**，推荐使用已验证的 Python 3.12。Windows CANDO 后端要求 x64 Python（WOA 使用 x64 模拟），厂商驱动和 SDK 另外准备。

下载 `piper-robot-0.7.1-interactive.zip` 及 `.zip.sha256`。解压到新目录，在包含 `install.py` 的目录运行：

Windows（也可运行 `Setup.cmd`）：

```powershell
py -3 install.py
.\piper-robot.cmd
```

macOS/Linux：

```sh
python3 install.py
./piper-robot
```

安装器创建包内 `.venv`，联网安装依赖并检查 CLI 帮助。不自动安装 Python、不注册 PATH、不连接机器人；已存在 `.venv` 时拒绝覆盖。如果没有 Windows `py` 启动器，用已安装的 Python 绝对路径执行 `install.py`。

若需裸命令 `piper-robot`，可激活该虚拟环境后使用。下文的裸命令在未激活时替换为 `.\piper-robot.cmd` 或 `./piper-robot`。不要将旧 `installer-v1` 补充包覆盖到本包。

## 跑通一次任务

启动后选择 **仿真 → 本机**，然后在同一终端逐条输入：

```text
/connect
/status
/tools
```

等 `Ready: yes` 后再发送动作。默认 MuJoCo 提供物理仿真；确定性软件演示可用 `piper-robot --root ./work/demo-one-task --simulation-backend sim` 启动新配置，完整操作见 [单次任务演示](ONE-TASK-DEMO.zh.md)。

`/manual 工具名(参数)` 手动调用不需要模型。`/model` 查看模型配置与 MCP 连接，`/model set ...` 保存自己的 OpenAI 兼容 Chat Completions 端点，`/model check` 实际测试模型 API；之后普通文本交给模型处理。手动操作、模型输入、每步预期结果都在演示文档中。

`/connect` 自动管理执行器和端口，并在多设备时询问选择；没有 CAN 不会悄悄切回仿真。连接成功、反馈存在和 Ready 是不同状态；未 Ready 时查看 `/status`、`/params` 与错误提示。

## 审批、远程与退出

新仿真默认 `auto`，真机默认 `risk`；已有配置保留原审批和限位。`/approval`、`/limits`、`/config` 都在当前终端操作，配置修改通过 `/confirm` 确认。动作等待审批时使用 `/approve JOB_ID` 或 `/deny JOB_ID`，仍可 `/status` 或 `/stop`。不再需要另开终端输入 OPEN。

`/remote` 管理已安装 SSH 和同版本 CLI 的远端，`/mode` 切换模式和目标。SSH 信任及登录需要先配置好；程序不跳过主机密钥验证。详细语法见 [交互指南](ROBOT-INTERACTIVE.zh.md)。

`/quit` 等待已接受动作结束，清理本次专用执行器；共享服务只释放当前控制会话。需要停止动作时用 `/stop`；退出不是急停。失联或结果未知时查询原 request_id，不能换新编号重发。

模型经 MCP/HTTP 调用同一执行器，不能修改操作员审批与限制。独立 MCP 和脚本入口保留，但新会话规则同样生效：旧客户端缺少有效控制会话会被拒绝。普通用户优先使用单终端入口；高级参数见 [CLI 参考](ROBOT-CLI-REFERENCE.md)。

## 升级与验证范围

退出旧前端，确认本次专用执行器已清理，将新版解压到新目录重新安装；不要复制旧 `.venv`。配置按模式和目标隔离；旧配置先备份再迁移，保留只读、限位和故障锁存。ROS commissioning 不会自动改为已通过。

0.7.1 在 macOS 上通过 **533 项测试，6 项跳过**，新 wheel 安装、MCP/HTTP 仿真三步动作和退出清理通过。首次安装冒烟有一次未捕获状态的 Ready 断言失败，后续四次未复现，原因尚未确定。详见 [验证记录](implementation/validation.md)。本版未重新完成 Windows/Linux 或真实机械臂验收；相机标定、碰撞规划、真机动作成功率不在本版证据内。
