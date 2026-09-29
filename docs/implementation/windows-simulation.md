# Windows 主机仿真实测（2026-09-28）

本轮在用户指定的 Windows 主机 上运行，未启动 Parallels 虚拟机。系统 `Windows-11-10.0.26200-SP0`，Python 3.12.10 AMD64，MuJoCo 3.14.0。测试只选择 `simulation / mujoco`，没有访问真实 CAN 或相机。

## 实际结果

- 前端 `/connect` 成功，MCP 发现 13 个工具。
- `/approval always` 后在同一前端确认配置；每个 `/manual` 动作先进入 `awaiting_approval`，再通过 `/approve` 执行，等待期间可查 `/status`。
- 关节目标 `[5,0,0,0,0,0]`：J1 反馈 4.999951°，任务 succeeded。
- 夹爪目标 30 mm：反馈 30.001901 mm，任务 succeeded。
- 六关节目标全零：J1 在完成判定时为 0.054742°，其余轴接近零，任务 succeeded；这是软件容差内完成，不是数学意义上的精确零。
- 三次动作分别按原 request_id 查询结果，模拟后端 tx 计数依次为 1、2、3。
- 动作前后均从 Windows MuJoCo 捕获 640×480 RGB 和深度，图片已拉回本地。
- `/quit` 等待完成并退出，runtime.json 被移除，执行器日志确认正常 shutdown。

[结果 JSON](evidence/windows-simulation-20260928/result.json) · [前图](evidence/windows-simulation-20260928/before.jpg) · [后图](evidence/windows-simulation-20260928/after.jpg)

## 实测中处理的问题

1. **进程探测缺陷**：原 `_pid_alive()` 在所有平台使用 `os.kill(pid, 0)`；[Python 的 Windows 语义](https://docs.python.org/3/library/os.html#os.kill) 会将零信号映射为终止操作。dsh CLI 将 Windows 分支改为 OpenProcess/GetExitCodeProcess/CloseHandle 只读查询，显式声明 64 位句柄类型；未知错误保守视为存活。macOS 模块回归 146 passed、1 Windows 专用跳过；Windows 探测测试 13 passed、1 POSIX 专用跳过，包含反复探测子进程仍存活的真实测试。
2. **依赖获取**：主机 PyPI 首次安装因网络超时中断。独立测试虚拟环境复用了已有依赖目录，新版 piperx 包安装在自己的 site-packages 中；没有覆盖旧执行器。补装 Pillow 后 RGB-D 接口通过。此结果不是完全离线安装或全新 Windows 环境安装验收。
3. **测试插件**：补装 pytest-asyncio 后再运行 Windows 进程回归；结果见附录。

## 可复现入口与范围

测试脚本：`tools/verify_windows_simulation.py`，不提供 real 模式选项。Windows 工作目录：

```text
%USERPROFILE%\Piper-070-Simulation-20260928
```

当前测试安装的 wheel SHA-256：`eac898dec96a0125a14b21c3873e819a757ddbbcf5d17380a43df7217c461761`。

本次安装来自包含 Windows 进程修复的工作区 wheel；**上一轮压缩包未包含该修复，不能用上一轮包代替本次测试版本**。本轮未重新封包或对外发布。代码及修复留在工作区。

只验证关节/夹爪、审批、通信、仿真 RGB-D 与退出。没有执行抓取彩块、抽屉任务、真实 CAN、实际摄像头、自动标定或真实机器人验收。远程控制是 SSH 在 Windows 本地运行测试；不等同于已验证前端自管 SSH 隧道全流程。

## Windows 进程回归附录

`test_managed_console_process.py`：**7 passed，26.74 秒**。覆盖真实前端/执行器进程、审批、两个管理器共享同一执行器、非交互模式检查、前端异常退出后的回收、stdin 未关闭时退出、慢 MCP 握手和已接受动作响应丢失后的查单。

进程测试命令及结果记录如上，原始日志保留在本地测试产物目录。加上进程存活探测，Windows 共 20 项通过、1 项 POSIX 专用跳过；另有 MuJoCo 三动作与 RGB-D 场景运行成功。
