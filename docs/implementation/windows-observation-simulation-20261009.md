# Windows 独立仿真验收 — 2026-10-09

本轮按用户要求部署到远程原生 Windows x64 主机 100.77.34.109（NOTHING），不是 Parallels。
仅仿真；没有打开真实 CAN/相机，没有真实机械臂动作，没有推送、合并或公开发布。
旧 `Piper-070-Simulation-20260928` 环境及用户既有配置保持原样。

## 部署与复现

目录：`C:\Users\fogfront-agent.nothing\Piper-072dev3-Simulation-20261009`。
基础解释器为已有 Python 3.12.10 AMD64。目录下独立机器人 `.venv` 和视频 `.video-venv`，
不复用旧环境的 site-packages；版本分别为 piperx-middleware 0.7.2.dev3、piper-lab 0.3.1.dev2。
MuJoCo 3.14.0。依赖检查通过，具体版本保存为 `robot-freeze.txt` / `video-freeze.txt`。

在该目录启动 `Start-Simulation.cmd`，进入后：

```text
/connect
/status
/tools
/manual robot_move_joints(joints_deg=[5,0,0,0,0,0], speed_percent=5, request_id="demo-joint-001")
/manual robot_gripper(width_m=0.03, effort_protocol=0.5, request_id="demo-gripper-001")
/manual robot_move_joints(joints_deg=[0,0,0,0,0,0], speed_percent=5, request_id="demo-home-001")
/quit
```

此专用入口显式选择 simulation/mujoco，配置在 `interactive-profile` 下，自动接线独立视频 Python。
同一入口加 `--agent` 提供 JSONL，加 `--agent-mcp` 提供模型 stdio MCP；同一 profile 同时只启动一个控制前端。
不需要开放另一个终端的控制窗口。新仿真 profile 默认 auto；审批验证中另外使用了独立的 always profile。

## 实际通过的链路

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 已安装 wheel 的 JSONL | 计划创建/更新、深度点查询与写回、仿真动作后证据 stale、quit 清理通过 | 合成深度 fixture + sim 后端 |
| 已安装 wheel 的 stdio MCP | initialize、列工具、skill guide、scene_query、EOF 退出通过 | 17 个前端工具；其中 13 个来自机器人执行器 |
| MuJoCo 图像/深度 | 640×480 RGB-D 前后帧实际生成 | 物理仿真传感器渲染，不是实拍相机 |
| 仿真图像到几何 | 使用模拟器已知内外参反投影，25 像素有效邻域，base 点计算与预期投影差 <2 cm；结果写入计划 | 这是数值/接线检查，不是实机外参精度验收 |
| 动作与审批 | J1 0→5°、夹爪 0.03 m、关节回零共 3 个动作；全部 awaiting_approval→approve→succeeded | 使用原 request_id 查结果，未重发动作 |
| 动作后计划 | 已答几何问题自动标为 stale | 接线/生命周期行为 |
| 视频前端实际子进程 | 指定时间点 direct、两个不同区间 pipeline、第二次复用 CV cache、返回评价通过 | 合成视频；无真实视觉 API、无 YOLO 权重 |
| 视频回归 | 95 passed，3.98 秒 | 已安装独立视频 wheel；合成视频与测试模型 |
| 修复后 Windows 前端回归 | 20 passed，1 deselected，15.46 秒 | 被排除的是下述无权限创建 symlink 的测试，不计通过 |
| 相机组子进程测试 | 补 OpenCV 后 8 passed，5.79 秒 | 合成输入与进程生命周期；没有打开物理相机 |
| 修复的 macOS 回归 | 21 passed，5.34 秒 | frontend_mcp、agent_console、observation_workflow |
| 启动入口 | Start-Simulation.cmd 执行 connect/status/tools/quit，返回 0 | 非交互喂入命令，未人工操作完整 TTY |
| 清理 | 仿真 runtime.json 均移除；验收结束检查没有残留 Python 进程 | 当次观测，不代表任意崩溃场景全面验收 |

## 发现与处理

### MCP 冷启动阻塞

初版安装包的 MCP 握手成功，但第一次 tools/list 不返回。线程栈显示在 scene_bridge 首次导入
NumPy 的本地扩展初始化中停住；关闭输入后恢复。预加载 NumPy 的对照烟测立即返回 17 个工具。
因此对 JSONL/MCP 两个入口做最小修复：启动 stdin 后台读取前解析工具定义、加载数值依赖。
不修改工具、审批或连接协议，也不重放请求。更底层的 Windows DLL/输入锁机制没有单独证明。

修复后重新构建并安装 wheel，标准入口、实际工具调用和退出全部复测通过。MCP 集成测试增加了
30 秒请求超时，避免以后回归时无限等待。0.7.2.dev3 尚未公开发布，本次修复 wheel SHA-256：

`fcf6def0aaab9438f9332a485101cab7b39441f57124148a6813605cafc577c6`

初始传输包 SHA-256 为 `365236ef9de0fe223f4d3e068ba817d0a89e24c9fe252f1375567bdfceae3035`；
它保留为 kit.zip 以追踪第一次验收，最终使用修复后的 kit-fixed.zip 与新 wheel，不能混用旧包证据。

### 安装和未覆盖的 Windows 项

原安装调用被 Windows PowerShell 的 NativeCommandError/标准错误重定向打断，机器人依赖已经安装。
在同一新目录内续装视频环境，使用 Python 管理日志后完成；没有删除或覆盖旧用户环境。

第一轮扩大回归为 **245 passed、6 failed、2 skipped**，不是全绿：

- 2 个相机组测试缺机器人环境的 cv2。补装 OpenCV 4.11.0.86 后，所属文件 8 项全部通过。
- 2 个 symlink 测试被当前 Windows 账号的 WinError 1314 阻止，无法执行构造；未提升账号权限。
- 2 个远端会话缓存测试直接断言 POSIX 0600/0700，在 Windows stat 语义下失败。
  本轮不将其改成无条件通过，也没有证明等价 Windows ACL 隔离，留作后续跨平台验证。
- 2 个原有 skipped 不计通过。完整 Windows 测试集未验收。

## 证据与下一步

远端目录保留 `mujoco-run-02`（最终图像、深度、几何、计划、动作报告），
`demo-run-01`（视频 direct/pipeline 结果）、`final-checks.json`、各测试日志与复现脚本。
本机记录在 `/Users/nothing/DEV/piper-win-sim-20261009`。

本次已证明单终端工具、视频子进程、模拟 RGB-D、几何查询和计划存储能够在 Windows 安装环境运行。
尚未证明真实 VLM 观看示范后自主完成任务，也未配置/调用模型 API 或 YOLO 检测权重。
下一实验应固定模拟任务与随机种子，对比“仅示范视频”“视频+观察计划”“视频+计划+几何后端”，
记录成功率、耗时、模型调用/图像数量与几何请求；在此之前不能声称新管线提高了操作成功率。
