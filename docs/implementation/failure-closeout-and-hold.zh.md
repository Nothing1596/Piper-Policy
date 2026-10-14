# 失败收尾与夹爪固定保持诊断

## 最小接口修正

仅改实验 runner，不修改真实机器人接口、控制器、摩擦参数或审批机制。

默认 `finish_task` 兼容原来的完成路径。未完成任务可明确返回：

```json
{"name":"finish_task","arguments":{"outcome":"failed","summary":"持物滑落，未完成受控放置","uncertainties":["最终支撑关系尚未确认"]}}
```

主动中止用 `outcome: "aborted"`。两种结果都正常结束决策循环并执行评分、清理，无需松爪、撤离或制造一次成功动作；允许没有当前 observation_id。它们不清除 unknown_outcome，也不会成为 task_complete。完成路径仍要求当前观察，严格实验仍核验松爪撤离和稳定性。

评分报告首先区分 grip_lost、incomplete_transfer、uncontrolled_release。真正缺失/损坏的正常阶段快照会指出具体字段，例如 missing state_before_release、invalid state_held，而不是统称 missing objects。不能用失败标记取得有效搬运资格。

## 固定姿态实验

当前 Codex 负责判断和参数选择，不调用 Gemini 或 DeepSeek。准备动作复用 2026-10-13 Codex 运行中基于公开 RGB-D 确定的参数；固定程序只是可复现的测试操作，不冒充模型逐步推理。

同一初始场景、夹取参数、力度和目标上方姿态，独立复验两次，保持 240 秒。记录 0、30、60、120、186、240 秒对应状态，并用原有物理录像日志在结束后量化相对 TCP 的物体滑移、转动、双指接触及机械臂漂移。保持期间不追加闭爪、重抓或自适应运动，也不暂停仿真、修改摩擦或传送物体。

这些时间点属于同一次保持轨迹，不能当作六个独立样本。两次复验只能做诊断；既不证明模型性能差异，也不等于真实机械臂夹持验收。正式结果及环境故障独立记录，不覆盖上一轮。

## 2026-10-14 验证状态

- 实验工具回归测试：252 passed、1 skipped；尚无本次变更的远程物理验收。
- 使用新代码只读复核上一轮搬运记录，原因明确为 `grip_lost_between_decisions`。原始报告、搬运记录和评分文件哈希保持不变，未覆盖旧成绩。
- 固定保持测试的参数、运行脚本及源码快照准备在本机 `/Users/nothing/DEV/piper-hold-stability-20261014/`，两次复验均尚未启动；这些本机运行产物不在本次提交中。
- 阻塞：原 Windows/WSL 仿真入口未恢复。先出现 `Wsl/Service/E_UNEXPECTED`，最后连接检查时 Windows SSH 也超时。环境恢复前不把测试准备描述为物理验证完成。
- 之前的 Codex 单轮实操结果见 [实操报告](codex-top-placement-20261013.zh.md)。
