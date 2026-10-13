# DeepSeek V4.1 Flash 小规模配对消融

日期：2026-10-09。范围：原生 Windows MuJoCo 仿真，机器人包 0.7.2.dev4。
这是探索性预实验，不预设 plan 有效，不作为真机或视频模仿学习的验收。

## 因素与固定项

四组：无 plan/无几何、plan/无几何、无 plan/几何、plan/几何。
“plan”指模型可见的 task_plan 工具和显式提问、记录、更新流程，不是模型是否进行内部思考。
“几何”指 scene_query 和绑定问题的自动几何计算可用与否；运动求解器所有组相同。
所有组预先建立相同的空任务存储容器，因为几何结果需要落在 tasks 目录。
没有容器造成的报错不能计为 plan 的效果；发现此问题的预试验不纳入正式数据。

固定模型为 dsh 所用的 DeepSeek-V41-Flash，API ID deepseek-flash，官方 endpoint，
reasoning_effort=low。保留并回传 reasoning_content，不调用第二个视觉模型。
思考模式下 temperature 不生效，不以 temperature=0 声称模型确定性。
每轮 API 最多 8192 输出 token；每任务最多 20 轮 API、40 个有效工具调用、240 秒轮次启动预算。
已开始的 API/动作允许在其自身超时内结束，因此总时间可能略大于 240 秒。
两个最新图片消息组保留在上下文，所有组一致。

每次任务都建立独立对话 messages、session_id、CLI 进程、profile、任务目录和 MuJoCo 实例。
不恢复先前会话，不给模型查看其他任务日志，不让模型访问 shell、文件系统或评分器。
API 请求没有跨任务对话历史；供应商前缀缓存不等于共享会话，记录 cached tokens 并随机化组顺序。

## 任务与分组

正式任务：

1. hover_red：末端指尖 TCP 悬停到红块顶面中心上方 120 mm。
2. blue_then_red：先到蓝块顶面中心上方 120 mm，再到红块上方同样高度。

两块都不得移动；完成后重新观察。所有组得到相同初始 RGB-D 图像、状态与工具说明，
相同向下姿态建议 [180,0,0]、10% 速度和指尖 TCP 配置。没有提供目标真实坐标。
两个场景种子 401、402，各运行四组，总计 16 个任务。每个“任务×种子”为一个配对区组，
区组内以固定随机种子打乱四组顺序，串行运行。独立 seed=400 用于入口检查，不纳入分析。
正式启动前 protocol.json 固定排程、提示、预算、评分阈值及 runner 哈希；正式期间不调提示或阈值。

本轮不输入示范视频、YOLO 检测或更换模型，否则无法分离 plan 与几何因素。

## 判定与度量

独立操作员评分器在动作后读取模拟器实际位置，结果不回传模型：

- XY 到方块中心误差 ≤15 mm；TCP 高度相对顶面+120 mm 的误差 ≤10 mm。
- 顺序任务须在实际动作后的状态中先达到蓝块目标，再达到红块目标，最终保持红块目标。
- 两个方块相对初态最大位移 ≤5 mm。
- 记录末次动作后是否成功取图，单独于位置成功报告；模型宣称完成不等于成功。

报告成功数、配对胜负、最终误差、API 次数、工具调用、观察数、图片数、输出/输入 token、
API 时间和任务墙钟时间；效率同时给全体任务与成功任务，避免将提前失败当作“更快”。
每组仅 4 次，任务和种子有限；即便出现优势，也不能推断通用收益。
基础设施错误单列，不冒充任务失败；截断、预算耗尽和模型主动放弃保留为实际结果。

## 可复用入口

- tools/run_plan_ablation.py：模型/API → 标准 MCP → 已部署 CLI；API key 从环境变量读取。
- tools/ablation_sim_host.py：只允许新仿真目录，独立评分、退出检查；从未暴露给模型。
- tools/tests/test_plan_ablation.py：因素隔离、顺序评分和位移约束测试。

实验沿用项目工具，不修改生产动作权限或真实配置。数据目录位于仓库外，密钥不写入配置、报告或包。

方法参考：用任务和场景种子作为区组，减少无关变异，参见
[NIST randomized block designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)。
模型图片、reasoning_effort 与消息协议参照 [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)；
工具回合保留模型返回的推理字段，参照 [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)。
