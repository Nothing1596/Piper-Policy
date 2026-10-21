# Robot 0.7.1：单终端功能与使用说明

[返回版本导航](README.zh.md) · [正式下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.7.1) · [完整单次任务演示](../ONE-TASK-DEMO.zh.md)

## 适用对象及功能

这是 2026-09-29 发布、截至 2026-10-21 仍标为 Latest 的机器人交互包。新机器需要单终端控制时以它为稳定入口。包含源码、wheel、测试、MuJoCo 资源、安装脚本与指南；视频独立包仍为 0.3.0，0.7.2 开发功能未进入此 ZIP。

- 启动先选择 simulation/real 和 local/已保存远端，不自动延续上次真机模式。
- `/connect` 自动启动或复用执行器、识别实例、发现设备；多设备在本终端选择，无需手填后端端口或另开 OPEN 窗口。
- 连接、状态、工具调用、模型配置、审批、限位、远程目标、模式切换和退出在同一前端完成。
- 单一执行器持有 CAN；请求有身份和结果查询，失联不重发未知动作，退出清理专用进程。
- 手动工具不需要模型 key；模型只能使用已有工具，不能修改操作员策略。

## 安装与版本检查

下载 `piper-robot-0.7.1-interactive.zip` 及其 `.sha256`，核对后解压到新目录。先安装 Python 3.11+，历史验证使用 3.12。Windows：

```powershell
py -3 install.py
.\piper-robot.cmd
```

macOS/Linux：

```sh
python3 install.py
./piper-robot
```

安装会联网下载依赖，不自动安装 Python 或注册 PATH；不要套旧 installer-v1。单独 wheel 面向已有环境。升级时退出旧客户端，不覆盖运行目录、不复制 `.venv`；不同模式和远程目标有独立 profile。

## 完成一个不需模型的任务

先用确定性 sim 测试替身熟悉交互，它没有三维图像：

```powershell
.\piper-robot.cmd --root .\work\first-task --mode simulation --simulation-backend sim
```

进入提示符后逐条输入，每个动作等 succeeded 再继续：

```text
/connect
/status
/tools robot_move_joints
/manual robot_move_joints(joints_deg=[5,0,0,0,0,0], speed_percent=5)
/manual robot_gripper(width_m=0.04, effort_protocol=0.5)
/manual robot_move_joints(joints_deg=[0,0,0,0,0,0], speed_percent=5)
/status
/quit
```

这个例子要求新 sim 的初始关节全 0。终态应为六关节 0°、夹爪 0.04 m、无活动作业。正常启动不指定 simulation-backend 时默认 MuJoCo。六关节为绝对角、夹爪为米，力度字段未标定为牛顿。不要把此固定数值套到已有真机姿态。

## 给模型下目标

前端输入 `/model` 查看 endpoint、模型、凭据引用与 MCP 状态。只在需要模型代做时配置：

```text
/model set endpoint="https://YOUR-ENDPOINT/v1" model="YOUR-MODEL" api_key_file="C:/Users/YourName/.config/piper/model.key"
/model check
/connect
```

随后输入普通目标文本，不带 `/`，要求模型先读状态、每步查实际结果、最后复核终态。内置入口使用兼容 chat/completions 的服务；客户端自己的登录和 key 不会自动同步。check 只证明最小请求，真实工具循环需单独观察。0.7.1 的内置模型链以文字/工具为主，不包含后续实时视觉工作流。

## 确认、远端与退出

| 命令 | 用途 |
|---|---|
| `/approval`、`/limits`、`/config` | 查看/提议修改策略和范围 |
| `/confirm`、`/cancel` | 应用/取消配置提案；单提案可省编号 |
| `/approve JOB_ID`、`/deny JOB_ID` | 对具体待审批动作作决定 |
| `/remote`、`/mode` | 管理保存目标、切换模式 |
| `/status`、`/jobs`、`/stop` | 等待时仍能检查或停止 |
| `/quit`、`/shutdown` | 退出会话或按条件关闭服务 |

新仿真默认 auto，真机默认 risk；always 每动作确认。auto 仅取消确认，不解除限位、反馈、故障或身份检查。配置确认和动作批准是两件事。SSH 使用已有密钥/agent 和主机密钥核验，远端需预装 SSH 与匹配 CLI；CAN 与动作执行仍在机器人侧。

`/quit` 不接收新任务，等待已接受动作完成，释放自己的会话和专用执行器；附着共享服务时不杀别人的服务。它不是急停。超时/断线先按原 request_id 查结果，不能通过新编号重放。

## 排障与验证范围

Ready:no 先读状态和具体原因；CAN 不通检查设备分配、驱动和是否有其他上位机占用。`require can position control` 应按该版工具表申请 `robot_set_control_mode` 并核验结果，不直接猜任意使能/复位命令。

0.7.1 不含 force 恢复、`/video`、`/demo watch`、`/plan`、`scene_query` 或行为复核；这些请读开发/源码页。独立旧 MCP 桥不会自动承担新前端生命周期，旧客户端不能默认免审动作。

保存的发布验证为 macOS 533 通过/6 跳过，干净 wheel 安装、MCP/HTTP 仿真三动作和冷启动通过；有一次动作前 Ready 断言未复现且原因未定。该发布未重新验收 Windows/Linux、跨主机真实 SSH 或真实机械臂。本次文档工作没有重跑上述测试。
