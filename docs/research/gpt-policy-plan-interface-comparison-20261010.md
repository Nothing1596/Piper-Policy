# GPT-Policy 与 Piper 的 plan / 执行接口比较

核对日期：2026-10-10。GPT-Policy源码固定在公开main提交`3a74ed4b45d25ff9af4939c9209292e725712f50`，本地只读检出位于`/Users/nothing/DEV/paper-timing-review-20261009/gpt-policy-source-20261010`。未运行其硬件代码，未改动正在运行的LongVIL作业。比较的是公开默认协议和当前Piper代码，不能替代所有私有实验入口的审查。

## 核心判断

两者在“通用模型结合示范、当前观察、历史与计算工具，产生动作并根据反馈继续决策”的范式上高度重合。plan+code、可复用原语、确定性IK、动作后观察均不能独自构成我们的研究贡献。当前更明确的接口差异是显式的问题/证据状态、可主动按时间段补看示范，以及可积累的局部RGB-D几何。接口差异尚未证明方法优越或学术新颖。

## GPT-Policy 的 plan 含义

默认`harness/protocol.py`要求每次返回一个`name/arguments`工具选择。`configs/tools.json`启用move_to、move_eef_chunk、set_gripper、check_path、locate_point、done、give_up；没有单独task_plan工具或计划问题状态接口，也不执行任意生成Python程序。模型可以进行语义规划，短note说明观察证据和下一目的，历史中保留先前决策及反馈。

`motion/planner.py`的EefTrajectoryPlanner是确定性运动规划器：读取当前关节、转换TCP、对路径插值、逐点IK、生成时间化关节参考。它不是恢复任务子目标的第二个语言模型。因此需要区分语义计划、结构化动作请求和数值轨迹规划。

Piper真机入口同样使用工具调用；LongVIL离线实验则让模型提交整段getpos/moveto/pick/place/open/close符号程序。后者与GPT-Policy逐步真机决策属于不同评测任务，不能直接用一致率比物理成功率。

## 已有共同能力与具体差异

| 维度 | GPT-Policy公开默认实现 | 当前Piper |
|---|---|---|
| 示范视频 | 窗口内语义选帧、全局复核、保留初终态和接触/释放/重复事件 | demo_watch允许模型查询时间段/时间点；v2初始概览固定均匀最多12帧 |
| 任务状态 | 对话中的观察、决策、previous_result，Codex刷新时重放文字历史 | task_plan持久化goal/stage/interpretation/questions及后端答案状态 |
| 几何 | locate_point返回标定射线；同一静态特征的两次腕部观察可三角定位，并检查视差/条件数/残差 | scene_query用RGB-D算表面点、积累局部点云、拟合平面和距离 |
| 证据关联 | 工具结果和历史step关联 | 可用question_id将测量写回特定问题，保留来源；动作或配置变化使历史答案过期 |
| 执行 | move_to/chunk，check_path，插值/IK/时间规划，动作后新观察 | 固定Piper后端与运动原语；此类封装是部署选择，非独立创新 |
| 当前效果证据 | 论文中的小规模真机上下文消融 | 27任务324轮离线语义实验；B/C没有启用3D或物理执行 |

GPT-Policy已有几何工具和反馈闭环，不能将其概括为只有RGB猜坐标。两次定位是腕部不同时刻的RGB视线，而非单张RGB恢复深度。其check_path不证明碰撞或接触成功；Piper的平面和距离计算同样不证明插入或抓取成功。RGB-D与三角定位的差别含传感器因素，不能将由深度硬件带来的收益全部归因于harness。

## 应当验证的候选贡献

让示范解释输出操作关系和候选子目标；显式记录会改变下一步动作的未知量；据此选择补看区间、相机视角、ROI或计算操作；将返回证据绑定该问题并据有效期修订计划；执行后再验证结果。研究重点是按任务决策组织和获取证据，而非新增一个plan文件。

当前接口有question_id、证据与过期状态，但主实验尚未稳定体现完整机制。108轮C中88轮只成功写一次plan，只有6轮在问题写入后又在更晚API中补看。多数plan是解释账本；离线没有scene_query，也没有证实测量驱动执行的收益。

下一步应在相同Piper、相机输入、示范、控制原语和模型预算下比较GPT-Policy式观察/历史/工具基线与显式问题协议，再控制3D工具是否开放。基线同样允许主动换视角和使用几何工具，避免用削弱的基线制造区别。测量本身、结构化plan和额外多轮复核分别有控制，记录语义准确、几何误差、调用/费用和物理任务结果。这里是实验建议，未自动启动。

## 一手来源

- [公开默认工具目录](https://github.com/cheng-haha/GPT-Policy/blob/3a74ed4b45d25ff9af4939c9209292e725712f50/configs/tools.json)
- [控制提示与输出协议](https://github.com/cheng-haha/GPT-Policy/blob/3a74ed4b45d25ff9af4939c9209292e725712f50/src/gpt_policy/harness/protocol.py)
- [两视角几何定位](https://github.com/cheng-haha/GPT-Policy/blob/3a74ed4b45d25ff9af4939c9209292e725712f50/src/gpt_policy/vision/perception.py)
- [确定性TCP轨迹规划](https://github.com/cheng-haha/GPT-Policy/blob/3a74ed4b45d25ff9af4939c9209292e725712f50/src/gpt_policy/motion/planner.py)
- [视频选帧和全局复核](https://github.com/cheng-haha/GPT-Policy/blob/3a74ed4b45d25ff9af4939c9209292e725712f50/src/gpt_policy/harness/video_selector.py)
- [Codex历史重放](https://github.com/cheng-haha/GPT-Policy/blob/3a74ed4b45d25ff9af4939c9209292e725712f50/src/gpt_policy/harness/providers/codex.py)
- [论文](https://arxiv.org/html/2609.19138)
