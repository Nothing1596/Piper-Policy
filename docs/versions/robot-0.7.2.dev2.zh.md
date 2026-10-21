# Robot 0.7.2.dev2：示范视频与分层观看

[返回版本导航](README.zh.md) · [示范命令详解](../ROBOT-DEMONSTRATION.zh.md) · [示范/实时边界](../DEMO-AND-LIVE-CONTEXT.zh.md)

## 状态及新增功能

这是未公开发布的开发包阶段，没有同名 release/tag。继承 dev1 的持续相机和单终端执行，增加 `/demo watch`、示范引用与前端 stdio MCP，让模型可以按时间反复看原图、读取 CV/检测评价，再结合当前场景决策。安装使用历史开发包自己的入口，或按 [当前源码](source-09256a5.zh.md) 新建环境；后者包含后续修订，不是原 dev2 二进制复刻。

机器人与视频可使用独立依赖环境，profile/video-pipeline.json 只保存视频解释器、检测器和模型配置文件的引用。模型 key 不随安装包分发，检测器也不等同于视觉 API。

## 直接看、按时间看、经过管线看

在 piper-robot 提示符输入：

```text
/demo watch {"address":"/absolute/demo.mp4","mode":"direct","at_s":1.5}
/demo watch {"address":"/absolute/demo.mp4","mode":"direct","start_s":1,"end_s":3,"max_frames":6}
/demo watch {"address":"/absolute/demo.mp4","mode":"pipeline","start_s":0,"end_s":3.8,"max_frames":6}
```

address 必填，可以是可访问本地文件、file URL 或可直接下载的视频 HTTP(S) 地址；网页播放器 URL 不保证能用。时间使用源视频实际 PTS，返回实际帧时间和 seek 偏差，不造出视频里不存在的帧。

| 模式 | 给模型的内容 | 是否需要语言模型 API |
|---|---|---|
| direct | 原图、时间及引用 | 不需要管线视觉评价 API |
| pipeline | 图像及 evaluation：质量、变化、检测、事件和未知项 | CV/ONNX 不需要；配置快速视觉评价后才调用其 API |

快速视觉评价只对实际看过的帧负责，不能拿一张终态图证明整个动作过程。评价与检测标签是可被反驳的假设；缺权重/未配置 API 应明确标出，已配置评价调用失败不静默当成功。后续 dev3 保留给执行模型的原图范围，避免低层筛选掩盖证据。

## 载入示范，让任务实际使用它

```text
/demo prepare "/absolute/demo.mp4"
/demo show
/manual demo_reference(offset=0, limit=3)
/demo clear
```

prepare 使用本地 candidates，不必配置 API；得到候选帧并选为当前示范。也可 `/demo use "/absolute/demo.json"` 载入已有编译包，或 use manifest.json 载入候选。`/demo compile {"video":"/absolute/demo.mp4","task":"描述完整动作顺序与目标"}` 才调用配置的语义编译器。只有编译过且具有语义阶段的包才能按 stage_id 查询，候选帧不能冒充阶段答案。

watch 是补看，不替换已经选择的任务示范。开始执行时，应确认当前模型真的获得示范图片和引用；还需 `/observe` 获取当前场景。示范表达“怎样做”，当前画面决定“在哪里做”；视频像素不能直接当基座坐标。

## 外部模型与配置

`piper-robot --mode real --target local --agent-mcp` 提供一个托管前端 MCP，包含机器人、camera_observe/camera_video、demo_reference/demo_watch。使用客户端 tools/list 查看实际 schema；MCP 返回标准图片与结构化评价。JSONL 仍用 `--agent`，模型不可配置策略或替操作员批准。

pipeline 配置包含 `python`、`detector_onnx`、`model_config` 的文件引用。视觉配置明确 provider/model/base_url、密钥环境变量或文件引用及图片/输出预算；服务是否支持这些字段需 probe。历史部署使用过低思考 Flash 做快速视觉评价，不是强制绑定某个供应商的设计。

Mac 开发联合包带 Setup.command 与 Piper-Robot.command，并分别建立 .venv/.video-venv；它与公开 0.7.1 安装包不是同一个制品。缺视频解释器先修配置，勿把机器人解释器缺视频依赖当相机故障。

## 完成标准与限制

最小链路是：真实视频成功解码→返回选定时间的图片和引用→pipeline 有明确评价/不可用说明→模型获得实际图像→用当前状态决定后续动作。不会因为 evaluation 有 supported 就自动宣称机器人成功。

dev2 尚无 dev3 的持久 plan/几何工具，也不证明真人示范迁移效果。历史软件/视频/仿真记录不等于新模型账户、原生 Mac CAN/D435 或真机抓取验收。本次未调用 API、相机或机器人。
