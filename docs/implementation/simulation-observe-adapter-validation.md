# 仿真观察适配验证 — 2026-10-09

## 结论和范围

0.7.2.dev4 已在 macOS 和远程原生 Windows 11 x64 上验证模型 CLI 的
`camera_observe → task_plan → camera_observe(question_id) → move_to → camera_observe`。
本次由 Codex 实际查看 CLI 回传图片、选择像素、提出问题、根据几何结果决定仿真动作。
不是只用内部函数或预设脚本补齐观察字段。

没有连接真实 CAN、真实相机或操作机械臂；没有验证抓取、接触、人体示范复现或视频引导的性能收益。
本轮不调用外部视觉 API、不安装 YOLO 权重，也不将像素测量误差当作系统定位精度。

## 最小实现

- MuJoCo 与 real/local 复用 camera_observe、SceneBridge、TaskPlanStore 和原动作执行器。
- 每次请求返回一张新渲染的 RGB-D；可附 ROI。明确标记 on_demand_snapshot，不虚构窗口历史。
- 图像与机器人状态在同一后端锁内取样，保留各自时间戳、执行器实例和连接代次。
- 内外参与深度单位规范化为现有几何后端格式。模拟标定保存在独立文件，不覆盖真实 camera.json。
- JSONL 保留 result，图片附在可选 content；MCP 使用标准 ImageContent，远端图片不再要求模型读取远端磁盘。
- 独立 observe 命令使用同一规范化格式，并从 profile 的 runtime.json 发现并核对执行器身份。
- 未增加服务、控制窗口、动作权限或新的规划框架。

## Windows 阻塞及修复

首次冷启动取图和 Windows 端到端测试超时。诊断线程栈显示，simulation_camera 校验变换矩阵时
导入 scene_geometry，连带加载 scipy.ndimage 及其原生模块，前端停在加载过程中。
将 SciPy 的导入移到真正清理深度图的计算函数后，计算仍由原 geometry worker 承担，
前端矩阵校验不再触发该重型依赖。修复前的会话不能热更新，重启后正式 SSH/CLI 入口复测通过。
没有通过放宽限位、伪造反馈或修改真实配置解决此问题。

## 自动验证

| 环境 / 范围 | 结果 |
|---|---|
| macOS，最终机器人全量测试 | 667 passed，8 skipped，67.09 s；一个现有 Starlette/httpx 弃用警告 |
| macOS，观察链路、模拟相机、几何相关测试 | 31 passed，5.15 s |
| Windows 11 x64，Python 3.12.10，独立 venv、已安装 wheel | 8 passed，12.74 s |

Windows 8 项包括：冷启动 MCP 取图与问题写回、动作后 stale/reobserve、JSONL 内联图片、
旧 observe 自动发现地址及直接几何消费、模拟标定隔离、重连旧证据失效、非法 ROI/身份拒绝。
退出断言检查本前端 runtime.json 被清理。Windows 本轮没有重跑全量平台测试；Linux 未在本轮覆盖。

## 模型通过实际 CLI 操作的记录

入口为独立部署目录内 Start-Simulation.cmd --agent-mcp，通过 SSH 标准输入输出接入。
客户端仅使用 MCP 协议，保存服务器返回的 JSON/ImageContent；没有导入产品内部函数、
绕过前端调用 HTTP、读取模拟器隐藏物体坐标或成功评分器。

| 步骤 | 输入 / 得到的证据 |
|---|---|
| 连接、第一次观察 | robot_connect；camera_observe(max_frames=1)，看到桌面、两个色块及绿色托盘 |
| 建立计划 | task_plan guide/create/update；选择画面空白桌面像素 [230,300]，保存 table_height 问题 |
| 绑定问题观察 | camera_observe(question_id=table_height, roi=[200,270,260,330])，同时返回完整图、裁剪图与几何结果 |
| 几何证据 | point_base_m=[0.447247839,-0.028455360,0.000738067]，25 像素支撑、有效比例 1，明确为 mujoco_sensor |
| 根据测量选动作 | 保持 x/y、姿态，将目标 TCP z 设为测得桌面高度 + 0.37 m |
| 执行 | move_to，request_id=measured-table-clearance-001，速度 5%，succeeded；TCP z 从 0.352861008 到 0.370738067 m |
| 更新证据 | task_plan get 显示 stale；重新 camera_observe 生成不同文件和时间戳，问题变为 answered |
| 核验与退出 | 查看新图片并读取 robot_status；保存解释；关闭 MCP，执行器正常退出 |

正式成功会话共 13 次 MCP 工具调用，其中 3 次 camera_observe、1 次 move_to；
返回 4 张图片（3 张完整图、1 张 ROI），Codex 实际查看了动作前后 2 张完整图。
13 次工具响应耗时合计 3.967 s，**不等于任务总时长**，不包含模型思考、工具之间等待及先前调试。
证据文件为 model-cli-evidence-final-02/01.json 至 13.json，包含逐次请求、响应、图片路径和耗时。

观察的桌面点不是末端正下方的同一 XY；这里只在平桌面仿真场景中用它作为高度参考。
默认 TCP 仍为已配置的 flange，未把它冒称为实测指尖；move_to 是关节空间终点动作，没有路径碰撞证明。
后续图像可见夹爪上移、色块仍在桌面；这只能支持本次小幅移动，不能证明抓取。

## 部署与启动

Windows 新环境：`C:\Users\fogfront-agent.nothing\Piper-072dev4-Simulation-20261009`。
保留之前版本和所有真实 profile。机器人和视频依赖分开安装。
交互启动运行 `Start-Simulation.cmd`，输入 `/connect`，再 `/observe`。
模型入口运行 `Start-Simulation.cmd --agent-mcp` 或 `--agent`。

实验 profile：model-chain-final-02；任务 ID d2c5bfd6-0752-4cad-9c5b-3ec99e333433。
退出后检查：没有 Python 进程、没有 runtime.json，执行器日志记录完整关闭。
此检查仅描述检查时刻，不将先前阻塞的会话称为正常退出。

下一步应采用固定任务与相同初始场景，比较“直接看图”与“提问+几何后端”的成功率、
观测次数和总时长；本次接线通过尚不能支持更好视频引导的实验结论。
