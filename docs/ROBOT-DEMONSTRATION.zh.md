# 示范视频接入与分层观看（0.7.2.dev2）

```text
视频地址 → 原 piper-video 解码/PTS校验 → CV/YOLO证据筛选
                                      → DeepSeek V4.1 Flash（low）评价
                                      → 执行模型结合实时画面和状态选择动作
```

## 按时间观看

```text
/demo watch {"address":"/完整路径/demo.mp4","mode":"direct","at_s":1.5}
/demo watch {"address":"/完整路径/demo.mp4","mode":"direct","start_s":1,"end_s":3,"max_frames":6}
/demo watch {"address":"/完整路径/demo.mp4","mode":"pipeline","start_s":0,"end_s":3.8,"max_frames":6}
```

address 必填，支持本地路径、file URL 和可直接下载的 HTTP(S) 视频地址；网页播放器地址不是视频文件。时间以源视频真实 PTS 为准，时间点取第一张 PTS 不早于请求的真实帧，返回实际时间及 seek_offset_s。区间保留端点；不外推不存在的帧。

- direct：返回实际图片、原视频时间和引用编号，不调用评价模型。
- pipeline：返回图片和 `evaluation`，包含清晰度、变化量、检测框与置信度、事件、全视频上下文的相机补偿信息及未知项。有配置的快速视觉模型时还返回 `evaluation.vision`：摘要、阶段、目标、未知项、历史结果判断和帧引用。

CV 全视频上下文与请求区间的评价分别标注。API 只审阅端点和变化/清晰度突出的中间帧，最多三张；执行模型收到该层筛出的图片，仍可用 direct 重看某个时间点。单帧不能证明运动过程或接触。pipeline 的评价是历史证据，`task_success=unverified` 不代表本次真机成功。

目标检测是成熟感知中间件，不需要语言模型 API；视觉评价是另一层，用 DeepSeek V4.1 Flash，而执行决策由当前工具模型完成。缺权重、未配置视觉 API 会明确标注；已配置评价模型但调用/引用校验失败时返回错误，不静默降级。

## 载入示范作为任务参考

```text
/demo prepare "/完整路径/demo.mp4"
/demo show
/demo use "/完整路径/manifest.json"
/demo use "/完整路径/demo.json"
/demo clear
```

prepare 调用原 piper-video candidates，生成候选帧并选择为当前示范，无 API 要求。use 可读取候选或已编译的阶段包。任务开始时把当前示范的阶段/目标/未知项和首批实际图片加入模型上下文；`demo_reference(offset=0,limit=3)` 可继续分页。watch 不覆盖已选示范。

原完整语义编译功能仍可用：`/demo compile {"video":"/完整路径/demo.mp4","task":"描述示范目标"}`，调用原编译器并保留引用/时间顺序验证。它使用明确配置的模型，不能把候选帧伪装成已编译阶段。

示范传递方法与目标，例如“抓住把柄后提起”；模型必须依据当前画面重新定位，不能把视频像素当基座坐标。

## 配置与模型入口

profile/video-pipeline.json 只引用独立视频解释器、检测权重和模型配置：

```json
{"python":"视频解释器绝对路径","detector_onnx":"YOLO11n ONNX绝对路径","model_config":"视觉模型配置绝对路径"}
```

视觉配置示例（不携带密钥值）：

```json
{"provider":"openai-compatible","model":"deepseek-flash","base_url":"https://api.deepseek.com/v1","api_key_env":"PIPER_DEEPSEEK_API_KEY","reasoning_effort":"low","structured_output":"json_object","max_images":3,"max_output_tokens":8192}
```

支持配置 api_key_file 引用私有文件，环境变量优先。本机部署复用用户授权的 dsh 凭据引用；安装包不包含凭据、现场配置或视频。`deepseek-flash` 已经服务 metadata 核验为 DeepSeek-V4.1-Flash，支持图像与 low。dsh 自己的默认设置不修改。

外部模型：`piper-robot --mode real --agent-mcp`，一个 stdio MCP 前端提供原机器人工具、camera_video、camera_observe、demo_reference、demo_watch。视频工具返回标准 MCP 图片和结构化评价；启动不连接 CAN 或发送动作。模型不能改策略或批准动作。

`demo_watch` 的 MCP 图片输出与请求一致，最多 12 帧；`demo_reference` 每页最多 6 帧。内置模型代理仍有自己的每轮图片预算，不能把结构化结果里的帧数当成最终模型请求的附图数量。

## Mac 安装

Python 3.12+，双击 Setup.command 安装，再双击 Piper-Robot.command；也可运行 `./piper-robot`。机器人和视频依赖分别在 .venv/.video-venv，启动器自动连接它们，无需另开上位机终端。

Mac 本轮验证原生 CLI、MCP、仿真和视频处理；原生 CAN/D435 不在该验收范围内。当前实机可继续运行在 Parallels Windows，由 Mac 模型通过 stdio 连接。旧机器配置、审批与限位保持独立。

示范与实时流为何共用基础设施、但采用不同策略，见 [上下文边界](DEMO-AND-LIVE-CONTEXT.zh.md)。
