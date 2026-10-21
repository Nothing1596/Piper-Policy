# Robot 0.7.2.dev4：MuJoCo 模型观察闭环

[返回版本导航](README.zh.md) · [原开发包说明](../ROBOT-RELEASE-0.7.2.dev4.md) · [完整验证记录](../implementation/simulation-observe-adapter-validation.md)

## 状态与关键修复

这是未公开发布的开发构建，没有同名 GitHub tag/release。它打通“模型能在仿真中看图、把结果交给几何后端、写 plan、动作后再观察”的接口缺口。不是新增动作规划器，也没有提高动作权限。

当前 main 仍使用 0.7.2.dev4 这个机器人包版本号，但在原 dev4 后继续合入了多批 harness 修订。因此部署需同时记 commit；[main@09256a5](source-09256a5.zh.md) 和历史 dev4 包不能只凭同版本号当成完全一样。

## 新环境与启动

使用已有历史 dev4 包时保留其 manifest、校验和独立 profile；没有历史包时按源码版建环境，安装 `piperx-cli[simulation]`。首次无模型连接：

```sh
piper-robot --mode simulation --target local --root ./work/dev4-demo
```

进入提示符：

```text
/connect
/status
/observe {"max_frames":1}
/plan new "观察桌面并测量一个可见表面点，暂不执行动作"
/manual task_plan(operation="guide")
/plan
```

默认仿真后端是 MuJoCo。指定 `--simulation-backend sim` 会变成无图像测试替身，不能拿它验证视觉链。无需物理相机或 CAN，也不需要 YOLO/视觉 API 才能获得模拟 RGB-D。

## 让观察、计划和测量连起来

1. 真正查看 observe 返回的全图，选择清楚的目标像素。
2. task_plan.update 创建 geometry 问题，带实际 task_id 与 expected_version，query 使用 surface_point 和该像素。
3. `/observe {"question_id":"实际问题ID","max_frames":1}` 返回一张新图及数值答案。
4. `/plan` 核对证据、版本和坐标系；解释写 interpretation，测量由后端填写。
5. 如需动作，用既有原语及当前测量决定参数；动作后新观察，旧图和旧答案不再冒充当前状态。

完整可替换参数的更新例子见 [dev3](robot-0.7.2.dev3.zh.md)。这里不提供可盲执行的固定抓取坐标。看到一个表面点不证明把柄完整、孔洞可插入或路径无碰撞。

## 外部模型、图像和地址发现

| 入口 | 调用方式 | 图像怎样交付 |
|---|---|---|
| 人工交互 | `/observe`、`/manual`、`/plan` | 回执与证据文件 |
| JSONL | `piper-robot --mode simulation --target local --agent` | result 保留，另外附可选 content 图片 |
| 前端 MCP | `piper-robot --mode simulation --target local --agent-mcp` | 标准 MCP ImageContent |
| 兼容取图命令 | `piper-robot observe --root DIR --workspace DIR --output NEW_PATH` | 规范化 RGB-D 文件与 JSON |

兼容 observe 在未显式传 URL 时可从该 profile 的 runtime.json 发现并核验执行器实例。不要把前端 profiles 父目录当作执行器 root；使用实际 runtime 所在的 profile/执行器配置。远端模型无需自行读取远端磁盘图片路径，仍需客户端真正消费图像内容。

每次仿真观察都是按需新快照，capture_mode=on_demand_snapshot；即使请求 max_frames>1，也不会虚构连续视频窗口。图像、模拟内外参和机器人状态同锁采样并保留身份/代次。模拟标定单独保存，不覆盖真实 camera.json；数据标记 simulated=true。

## 修复原因、排障与验收边界

旧 Windows 取图冷启动曾因矩阵校验连带导入 SciPy 原生模块而阻塞。dev4 把重依赖留在真正几何 worker 的使用路径，前端不会为简单校验提前加载整套计算依赖。修复需要新进程，旧进程不会热更新；不能靠放宽限位或伪造 feedback 处理。

若返回无相机工具，核对 MuJoCo、前端 MCP 握手和实际版本；若反馈或观察过期，重新观察并更新计划；若实例改变，显式重新建立连接身份。`max_frames` 是上限，不是新帧数量承诺。

历史验收包含 macOS 全量 667 通过/8 跳过、Windows 相关 8 项通过，以及 Codex 经真实 CLI 看图后的单次仿真抬升。成功会话 13 次 MCP 工具调用、3 次观察、1 次 move_to；这不是完整抓放、人类视频复现、模型性能对比或真机任务成功。Linux、实机相机/标定精度及 CAN 在该轮未验收。本次没有重跑这些实验。

后续受阻、抓取几何、重连身份、软 plan、持久上下文及评分更新都在 [源码版逐批变化](source-09256a5.zh.md) 中单列；不能假设历史 dev4 ZIP 已包含它们。
