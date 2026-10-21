# Robot 0.7.2：安装与完整工作流

[返回版本导航](README.zh.md) · [下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.7.2) · [主要更新](../ROBOT-RELEASE-0.7.2.md)

## 与旧版本的关系

0.7.2 将 dev1–dev4 及后续主线修订整合成正式机器人版本，附 Video 0.3.1。保留 0.7.1 的单终端、审批和会话机制，增加示范观看、当前图像、plan/几何、公开复核和反馈恢复。旧开发构建仍由 commit/manifest 区分，不能与本版互换。

实验 token_context 已在源码中，供特定适配器调用；安装运行包不会自动给日常内置模型启用 32k 压缩，也没有自动学习经验或监督模型。

## 新机器与升级

下载交互 ZIP 及 SHA-256 校验文件。完整包需要 Python 3.12+，首次安装联网；Windows 在新目录运行 `py -3 install.py`，macOS/Linux 运行 `python3 install.py`。安装器分别创建 `.venv` 与 `.video-venv`，不覆盖已有环境、不自动安装 Python/PATH、检测器权重或模型服务。

Windows 用 `.\piper-robot.cmd`，macOS/Linux 用 `./piper-robot`。要用裸命令可激活机器人虚拟环境。单独 wheel 用户自行安装 simulation extra；相机需要 camera extra 与平台驱动。配置按 simulation/real 和目标隔离，保留旧只读、限位和锁存；不用旧 installer-v1。

## 一次任务的建议顺序

启动时选择仿真、本机，逐条输入：

```text
/connect
/status
/observe {"max_frames":1}
/demo prepare /absolute/demo.mp4
/plan new "按示范的完整动作顺序操作当前物体"
/manual task_plan(operation="guide")
/manual demo_reference(overview=True)
/plan
```

demo 路径由用户提供；没有示范时跳过 demo 操作。MuJoCo 返回当前快照，录制图提供方法和顺序，不提供当前物体坐标。先确认模型真正收到图片，再根据模糊处用 watch 的 direct/pipeline 和时间区间补看。

模型通过 task_plan 更新完整序列、问题、证据与进度；通过 scene_query 复用一次 RGB-D 的表面点、平面和距离。问题答案与 interpretation 分开，使用真实 task_id/version，旧 CAS 更新不能覆盖新计划。测量例子见 [观察计划指南](../ROBOT-OBSERVATION-PLAN.zh.md)。

然后使用 `/manual` 或配置 `/model` 后给普通文本目标，动作仍由原执行器处理。每步读取真实结果并用新图复核；发生受阻、接触变化、掉落或参考物体移动时重新判断并更新 plan。完成需要对应任务的过程、松爪撤离及稳定证据，不能只看 primitive succeeded 或模型说 done。最后 `/quit`；需要中断动作先 `/stop`。

## 模型工具与行为提醒

外部 MCP：`piper-robot --mode simulation --target local --agent-mcp`；JSONL：同参数改为 `--agent`。实际工具以 tools/list 为准。manual 自动补动作请求编号，外部模型动作显式提供 request_id；未知结果查询原编号，不换 ID 重放。

`harness_review` 提醒重复查询/拒绝、稳定步骤 ID 消失或失败引用支撑 completed 等公开问题，不看隐藏 CoT，不拦动作、不自动判任务成功。在线 plan 是软状态。终态评分、实验预算和只在特定 runner 中暴露的工具，不等于日常 CLI 默认功能。

## 真实相机、恢复与限制

本机真实相机支持持续帧缓存和最多两台 fixed/wrist 配置，实际标定来自操作员；没有外参时不能把光学点直接当基座目标。MuJoCo 只按需取图，没有连续窗口。远端不会偷偷回退到本机相机。

force 只覆盖已存在 ≤5° 越界的保持/向内恢复，须配置确认且每动作批准，不允许继续向外。确认停稳后继续新决策的恢复只适用于指定 MuJoCo/direct 失败，不放宽真机反馈或未知执行规则。

SDK、驱动、标定、物体隐藏结构、碰撞/接触/插入及真机任务性能仍各自验收。本版实际软件与包检查见 [验证记录](../implementation/release-0.7.2-validation.zh.md)，不要用历史模型案例替代本轮验收。
