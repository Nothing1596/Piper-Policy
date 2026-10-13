# 模型 CLI 观察闭环实测 — 2026-10-09

结论：**尚未打通观察 → plan → observe/几何 → execution → 再观察闭环**。
本轮直接通过 SSH 使用远程 Windows 的已安装 piper-robot CLI，没有用 Python 控制器内部接口或手动构造观察数据来补齐链路。
只运行 MuJoCo 仿真，没有打开真实 CAN/相机，没有修改产品代码、部署版本或发布。

## 环境与流程

- 主机：100.77.34.109，原生 Windows 11 x64。
- 部署：`C:\Users\fogfront-agent.nothing\Piper-072dev3-Simulation-20261009`，修复 MCP 冷启动后的 0.7.2.dev3。
- 独立 profile：部署目录下 `chain-cli-20261009`，simulation/local，MuJoCo。
- 入口：`Start-Simulation.cmd --root <独立目录> --agent`。
- task_id：`b110fabc-4623-43bc-8c58-6aeac8a31226`，最终 version=7。
- 共发送 17 条模型 JSONL 请求（包含 tools、quit 两条管理请求），另运行 2 次独立 CLI observe 命令。两个仿真运动请求均完成，没有重发未知动作。

| 步骤 | 实际请求/结果 | 判断 |
|---|---|---|
| 连接 | robot_connect；backend=mujoco、ready=true | 通过 |
| 读取工作流 | task_plan(operation=guide) | 返回 observe-and-plan skill |
| 历史视频查询 | demo_watch(pipeline, 0–2 s, max_frames=3) | 返回 3 帧路径、质量/变化评价；没有配置检测权重和快速视觉模型 |
| 建立计划 | create → update；加入 surface 问题与 surface_point 查询 | 持久化成功，问题 pending |
| 动作前观察 | observe(question_id=surface, max_frames=1) | camera_unavailable: Live camera supports real/local only |
| 工具发现 | tools | 17 个工具；仿真工具表没有 camera_observe/camera_video |
| 独立动作探测 | J1 0→5°，其余保持初始 [30,-30,0,0,0]°，速度 5% | succeeded；完成时 J1≈4.943°；这不是视觉引导动作 |
| 动作后观察 | 相同 observe 请求 | 同样 camera_unavailable |
| 读取计划 | task_plan(get) | surface 仍 pending，没有数值证据 |
| 恢复初始姿态 | 目标 [0,30,-30,0,0,0]° | succeeded；完成时 J1≈0.064° |
| 旧独立取图命令 | piper-robot observe，只传 root/workspace/output | 报 requires MuJoCo；未自动使用当前动态执行器地址 |
| 显式使用 CLI 已显示的地址 | 同一命令补 --url http://127.0.0.1:49249 | 成功保存 RGB、深度和 observation.json |
| 原样消费旧取图文件 | scene_query(surface_point, observation_path, question_id=surface, expected_version=6) | observation_path: intrinsics object is required |
| 记录缺口并退出 | plan 更新至 blocked_at_observation_adapter → quit | version=7，问题 pending；draining 后进程退出码 0 |

本次的合成视频用于检查接口数据返回，不能提供抓杯子或开抽屉的动作示范。JSONL 返回远程图像路径；本轮未把这些图片实际送入视觉模型，不计视觉语义理解通过。

## 已确认的接线缺口

1. `console_interaction.py` 中 camera()/camera_observe 仅允许 real/local，available_tools() 也只在该模式暴露相机工具。仿真 observation API 尚未接到模型 observe 入口。
2. `simulation_camera.capture()` 保存内参在 `camera_info.intrinsics`，几何后端要求规范化的顶层 intrinsics 对象及深度来源等字段。单纯开放仿真工具名不能解决格式、标定来源和姿态时间对齐。
3. 旧独立 observe 命令需显式指定动态执行器 URL；仅给 profile root 没有自动复用已连接前端的地址。这不符合单前端自动管理连接的预期。

上一轮 `verify_observation_simulation.py` 手动把模拟 RGB-D 转成 geometry 需要的格式、登记模拟标定、关联反馈后，才调用 scene_query。因此上一轮证明的是后端分段接线，不是本轮要求的模型 CLI 完整闭环。

## 最小后续实现范围

复用当前 camera_observe/observe 和 plan 写回流程，增加 MuJoCo 观察来源适配：从前端已有连接取图，统一证据结构，使用模拟器已知内外参并明确 simulated provenance，保存对应机器人状态与连接身份。模型通过同一工具得到图像和 observation_path，绑定的问题进入现有 scene_query；运动后再取新图更新证据。

不新增独立服务、不复制 plan 存储或计算后端。最终验收须让同一模型 CLI 会话在不改 JSON 文件、不补端口、不直接调用控制器内部接口的条件下完成：看图 → 提问/plan → observe 自动测量写回 → 根据测量选择动作 → 再观察验证。

本次任务状态保存在远程独立 profile 的 tasks 目录。本机导出为 `/Users/nothing/DEV/piper-win-sim-20261009/cli-chain-plan.json`。
