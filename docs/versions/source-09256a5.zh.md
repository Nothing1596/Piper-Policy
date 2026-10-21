# 已推送源码 main@09256a5：安装、功能与使用

[返回版本导航](README.zh.md)。本页固定在 `09256a50935bfc237f7dd737b718dffe22e3d5e0`，不把今后的 main 变更归入本快照。核对日期为 2026-10-21。

## 身份与适用场景

这份源码已在 GitHub main；它不是新的 release。机器人分发元数据为 `piperx-middleware 0.7.2.dev4`，视频侧为 `piper-lab 0.3.1.dev2`。它比公开 Robot 0.7.1 多出视频、plan、几何、上下文和实验反馈能力；版本字符串没有随每次提交更新，部署记录应包含 commit。

普通单终端用户可以按下述方式进入仿真。模型实验工具只在专门 runner 中提供，不能把实验工具表当作日常 `/tools` 的保证。

## 从源码建立独立环境

需要 Git、Python 3.12 与依赖下载网络。下面是新目录安装，不更新已有运行目录。Windows PowerShell：

```powershell
git clone https://github.com/Nothing1596/Piper-Policy.git Piper-Policy-09256a5
Set-Location Piper-Policy-09256a5
git checkout --detach 09256a50935bfc237f7dd737b718dffe22e3d5e0
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".\piperx-cli[simulation]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\piper-robot.exe --mode simulation --target local --root .\work\source-demo
```

macOS/Linux 使用相同的新 clone 与 commit，再执行：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e './piperx-cli[simulation]'
.venv/bin/python -m pip check
.venv/bin/piper-robot --mode simulation --target local --root ./work/source-demo
```

要使用示范视频解析，在同一环境另外安装 `./piper-lab[video,detector,model,mcp]`；真实相机需要机器人 `camera` 可选依赖及匹配平台驱动。安装依赖本身不加载模型、不连接 CAN。本页没有验证一次全新的联网安装。

## 一次视频引导任务

先输入以下命令，按每步返回信息继续，不整段盲贴：

```text
/connect
/status
/tools
/observe
/plan new "参照示范，将目标物体放到指定位置"
/demo watch {"address":"/absolute/demo.mp4","mode":"direct","max_frames":6}
/manual task_plan(operation="guide")
/plan
```

`/demo watch` 的路径换成当前主机可访问的真实视频。watch 返回的是录制证据；`/observe` 返回当前仿真。先核对完整动作序列和顺序，再用当前画面选择区域。没有配置模型时，可以人工读图并 `/manual`；配置内置模型后，普通文本目标触发模型工具循环。配置例子见 [0.7.1](robot-0.7.1.zh.md)，具体视觉/多图行为见 [观察与计划](../ROBOT-OBSERVATION-PLAN.zh.md)。

图像返回后可以调用 `scene_query` 测量，再把解释、未决问题和证据引用写到 plan。执行只使用已存在的运动原语；每个动作后核对 job 和新观察，更新已完成与遗漏步骤。失败请求不是完成依据；松爪后还需要撤离和稳定证据。结束时 `/quit`，异常动作先 `/stop`，未知结果按原 request_id 查询。

## 相对 0.7.1 的逐批变化

| 提交 | 新增或修正 | 为什么改、怎样使用 |
|---|---|---|
| `40d231b` | 有界 force 恢复、本机 RGB-D 观察 | 已有轻微越界可以申请向内恢复；`/config force on` 后 `/confirm`，动作仍逐次批准；不是解除硬限位 |
| `af65e62` | 视频示范、持续相机、plan/几何、模拟观察、持久公开上下文 | 把看图→提问→计算→反馈连起来；使用 `/demo`、`/observe`、`/plan`，历史证据通过实验缓存按需读回 |
| `8b021c2` | failed/aborted 正常收尾、持物丢失归因 | 失败任务可以结束并保存原因，无需凑一个成功动作；`finish_task` 是实验入口工具 |
| `c5582d6` | 可见区域抓取几何、停稳核验、求解后新反馈、重连观察身份 | 处理局部选区矛盾、tracking_error 和 stale_feedback；让新反馈驱动新的决策，不重放未知动作 |
| `8317f85` | 跨前端公开行为复核 | 重复测量、重复拒绝、计划步骤丢失、无依据完成会得到提醒；提醒不拦动作、不读取隐藏思考 |
| `1d37a5f` | 软计划、进度写回、计数边界、终态关系失效 | 用真实版本同步 plan；参考物体移动后旧终态关系不再自动有效；模型调用与内部测量分开统计 |
| `3253c98` | 视频经验研究、有限恢复、远端作业、近似恢复 | 供冻结实验使用；恢复先查账，缺完整物理检查点的近似回合单列，不当作连续执行 |
| `09256a5` | 汇总实现与实验边界文档 | 使使用者能追溯配置和验证范围；文档提交不等于新增模型成功数据 |

plan 在在线链路中是软状态，不需要为了运动先“达成计划指标”。离线 R2 强制协议属于单独实验，不能套成日常动作门槛。几何候选只是当前可见区域的测量及端点建议，不提供对象身份、隐藏形状或无碰撞保证。

## 模型入口和实验入口

外部代理可以启动 `piper-robot --mode simulation --target local --agent-mcp`；JSONL 使用 `--agent`。两者经过同一前端控制器，模型不能修改操作员策略或代替操作员批准。独立旧 `mcp` 桥不负责单终端生命周期，不能互换使用方式。

`tools/run_longvil_online.py` 才提供 LongVIL 在线实验工具、冻结输入、私有独立评分与条件配置。`--plan on|off`、`--skill on|off`、`--effort low|high`、`--decision-policy standard|short-feedback` 是该 runner 的参数，不是 `piper-robot` 斜杠命令。`--behavior-monitor` 开启公开行为提醒，`--no-behavior-monitor` 用于对照。启动实验还需要已审核场景、示范证据、模型配置和账本，不应复制旧私有任务路径硬启动。

## 文件、故障与验证

- profile 保存目标、配置、任务 plan 和 harness 审计；原始图像与几何证据单独存放，历史缓存不会获得当前动作资格。
- 请求被拒绝时先检查反馈、观察身份、模式及批准；只换 request_id 不会修复前提。显式重连后重新观察；实例身份改变或未知执行不能靠重发动作解决。
- 公开行为复核不分析 CoT，不自动调用监督模型，也不替模型写完成状态；有些低效搜索不触发规则，提醒不是语义评分器。
- 既有公开行为复核文档保存了当时相关回归 `839 passed / 8 skipped`，不是本源码快照的新全量验收；本次没有重跑这些测试或模型/真机实验。细分范围见 [实现文档](../implementation/public-behavior-monitor.zh.md)，不同轮次不能相加。
- 对本页固定的 09256a5 而言，token 预算压缩尚未纳入；后续 956f916 已提交该实验模块。首帧 seed 仍在独立目录。安装本页旧 commit 不会获得这些后续工作，见 [工作状态](unmerged-work-20261021.zh.md)。
