# Piper Robot 0.7.2 — 视频、计划与反馈闭环

本版汇总 0.7.1 之后的主要源码更新，并附带独立 Video 0.3.1 wheel。下载包为 `piper-robot-0.7.2-interactive.zip`；它联网安装依赖，不是 Windows 离线依赖包。

## 重要更新

1. **从示范到当前场景。** `/demo watch` 支持明确视频地址、direct/pipeline、时间点和区间，返回原图、真实时间和引用；`/video` 管理本机连续相机，MuJoCo `/observe` 返回按需 RGB-D 快照。录制示范与实时证据始终分开。
2. **共享 plan、skill 和可复用计算。** `/plan`、`task_plan` 和 observe-and-plan skill 贯通内置模型、JSONL 与前端 MCP；`scene_query` 提供 CPU 表面点、局部点云、平面和距离。在线 plan 保持软状态，模型负责解释、决策和根据反馈更新。
3. **反馈恢复和任务收尾。** 修正可见区域抓取几何、求解后的新反馈、重连身份和旧观察失效；确认停稳的 MuJoCo 受阻可以交给模型新决策。失败/中止可正常收尾，未知动作不自动重放。
4. **公开行为复核。** 对重复测量、重复拒绝、计划步骤消失及缺证据完成给出确定性提醒。只看公开工具结果，不读隐藏思考，不自动改参数、判成功或调用监督模型。
5. **源码中的实验与上下文。** 提供公开 session 缓存、历史图读回、完整视频动作/进度复核、终态关系修订和有限恢复工具。新增 token 预算与结构化回执压缩模块供实验适配器使用，日常 CLI 未自动启用；这些实验工具不保证全部包含在机器人运行 wheel 中。
6. **完整版本文档与打包身份。** 新包包含完整 docs 树、组件源码和 BUILD-INFO.json，版本与 wheel 一致性在构建时检查；六个历史发布项和四个开发阶段分别有安装及功能说明。

## 下载与安装

先准备 Python 3.12+。Windows 运行 `py -3 install.py` 或 Setup.cmd；macOS/Linux 运行 `python3 install.py`。安装器分别建立机器人 `.venv` 和视频 `.video-venv`，启动器自动连接视频解释器；不自动安装 Python、注册 PATH、下载检测权重或配置模型凭据。只装机器人 wheel 时最低 Python 为 3.11。

安装前核对发布页 SHA-256；升级使用新解压目录，保留旧环境和 profiles，不复制 `.venv`。启动后先选仿真、本机，输入 `/connect`、`/status`、`/observe`。详细步骤见 [中文安装](ROBOT-GUIDE.zh.md)、[English](ROBOT-GUIDE.en.md)、[完整本版指南](versions/robot-0.7.2.zh.md)。

MCP 模型入口为 `piper-robot --mode simulation --target local --agent-mcp`；JSONL 为 `--agent`。旧独立 `mcp` 桥仍保留其执行器与会话要求。操作员确认、限位、反馈和故障检查继续有效。

## 验证与限制

本轮实际检查集中列于 [发布验证](implementation/release-0.7.2-validation.zh.md)，避免把历史实验结果移作本包成绩。macOS 软件回归、安装包检查和有限仿真冒烟分别记录；本轮没有付费模型任务、Windows/Linux 新安装或真实 CAN/相机验收。

没有自动标定、通用目标分割、隐藏几何重建、连续路径碰撞规划、插入/接触成功判决或真机任务成功率声明。软件修复与模型性能改善是不同结论。厂商 SDK、驱动、相机依赖和实际外参仍需现场准备。

新 release 不改旧 tags/资产、冻结实验结果或 `experiment/derivation` 历史。独立首帧 seed 研究接口未纳入本版。
