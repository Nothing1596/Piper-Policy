# LongVIL：高推理预算、有无显式 plan 的后台对照

日期：2026-10-09。用户确认先做 LongVIL 官方离线指标，并要求提交后台任务后停止监视。
本轮不运行真实机械臂，也不把符号执行器的 FSA 当作 Piper 的物理成功率。

## 对上一轮输入的核查

上一轮 run_plan_ablation.py 在第一次模型请求中附带 camera_observe 返回的实际 RGB 图片、
内外参和机器人反馈，因此 plan 前有当前仿真画面。它没有示范视频，没有自动视觉模型语义预解析；
几何条件组可调用 scene_query。旧的实时图像上下文仅保留最近两组，不能称为始终保存初始图。

新实验是离线 LongVIL：初始参考图为该视频 t=0 的录制图，不是 live camera。
每次请求固定保留完整初始示范概览和这张单独标注的初态参考；另保留最近两组主动补看图片。
模型输入只给 object_list，绝不传入 code、action_sequences、positions、task_horizon 或预动作标签。

## 固定协议

数据来自 cq838/LongVILBench，版本 `59f002209575ae1b3e619a2ffd5a831a15194c2a`：

- level1_4_blockmanipulation_1
- level2_6_blockmanipulation_1
- level3_12_blockmanipulation_1

每个任务用 clean/complex 两种视频，共 6 段；每段做 no_plan 与 plan 两组，总计 12 次。
这是预先指定的三级小样本，不是全量基准。两种变体属于相同任务，不视作六个独立语义任务。
每段内的条件顺序用固定随机种子打乱，每次新会话、新 CLI/profile、新 plan 目录。

两组均使用 `deepseek-flash`（dsh 配置对应 DeepSeek V4.1 Flash），`reasoning_effort=high`，
每轮最多 65,536 输出 token、每任务最多 30 轮 API/60 次工具调用，900 秒新轮次启动预算。
单次 API 超时 300 秒。预算不足仍记录截断/耗尽，不为某组重试或增额。
这些预算均高于上一轮，但任务同时变化，不能拿两轮差值解释“提高预算的因果效果”。

模型经真实 Piper CLI 的标准 MCP 调用 demo_watch，plan 组另有原有 task_plan。
两组都可重新查看任意时间点或区间；submit_program 仅提交离线程序，不会驱动机器人。
No plan 指没有显式持久化工作流，模型内部思考仍为 high。

## 视频处理与附图

两组使用相同管线：真实 PTS 解码、OpenCV 清晰度/变化分析、固定 YOLO11n ONNX 检测、原始图像。
初始概览 12 帧覆盖全视频；模型可主动补看。YOLO 的 COCO 标签不等于积木形状真值，
没有检测到物体须保留 unknown。未配置额外语义 VLM，不预先把示范编译成动作答案；
同一个 DS4.1 high 执行模型自行解释图像及中间件评价。
本轮不同时比较 direct/pipeline，以免增加混杂因素。

提交前发现并修复：demo_watch 可返回 12 帧元数据，DemoBridge.image_inputs 却只附前 6 帧。
修复后 MCP 发送至多 12 张相应图片；demo_reference 分页仍最多 6 张。
内置 model_agent 的每轮 6 图预算仍独立存在，本实验走外部模型 API→MCP，不经其裁剪。
增加末帧传递回归测试，并经实际 CLI 验证 12 张图、YOLO 评价及正常释放执行器。

## 评分及证据边界

采用 LongVIL 上游 `aa8fbb18b293b3e2de4a86530b92756a35463f76` 的 evaluate.py：

- EMA：规范化代码逐行完全一致，变量名等差异可能导致严格指标失败。
- SMS：从开头连续匹配的代码行比例，不是无序步骤准确率。
- FSA：官方 CodeExecutor 的符号终态是否一致，不是物理模拟接触或真实抓取。

评分器只在模型结束后运行，不向模型反馈标准答案或分数。
无程序、输出截断、预算耗尽按任务失败保留；程序非法单列且计失败；
连接/管线/评分基础设施错误单列，停止余下排程，不自动重试。
每次结束保存原始响应、工具调用、附图计数审计、指标及 CLI 释放状态；自动更新汇总。

## 后台运行及产物

实验目录：`/Users/nothing/DEV/piper-longvil-ablation-20261009`。
源码及参考评分器冻结后启动，密钥由现有 dsh 配置在调用进程内读取，不写入产物。
离线模型和 CLI 运行在本机 Mac；未冒充此次在 Windows 上执行。

- sources.json / dataset-info.json：固定数据版本、下载地址及视频哈希。
- frozen/：本轮执行代码快照及来源哈希。
- formal/protocol.json：固定排程、提示、预算及组件哈希。
- formal/<session>/：Result.json、events.jsonl、initial.json、图片和逐轮 audit。
- formal/episodes.jsonl / episodes.csv / summary.json / RESULTS.md：自动生成的逐任务及汇总结果。
- job.json / job-exit.json / background.log：提交身份、退出状态和日志。

用户要求停止监视，所以提交后不等待指标、不建立自动监视、不推送发布；
“已提交”与“已完成并验证”分开报告。后台只有这 12 次排程，不循环扩展实验。

参考：[LongVIL 代码及评测说明](https://github.com/chenquan2002/LongVIL/tree/aa8fbb18b293b3e2de4a86530b92756a35463f76)，
[LongVILBench 数据集](https://huggingface.co/datasets/cq838/LongVILBench)，
[DeepSeek 思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)。
