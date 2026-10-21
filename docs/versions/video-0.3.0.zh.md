# Video 0.3.0：功能与使用说明

[返回版本导航](README.zh.md) · [正式下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/video-v0.3.0) · [完整命令参考](../VIDEO-CLI-REFERENCE.md)

## 定位与包含内容

独立的视频证据管线，2026-09-27 发布。下载 `piper-video-0.3.0-win64-py312.zip`，用于 Windows x64 / Python 3.12 离线安装。入口是 `piper-video`；Python 分发名仍为 piper-lab。无需机器人包，也不启动 MuJoCo、CAN 或运动。

它把视频解码为带时间的候选帧，结合可选 ONNX 检测和视觉 API 生成阶段、结论、帧引用及未知项。它不是自动机器人控制器，不更新模型权重，不因检测到标签就证明动作发生。

## 安装与无 API 自检

从发布页核对 `video-SHA256SUMS.txt`，解压到新目录：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper-video.cmd --help
.\piper-video.cmd candidates --video examples\transfer.mp4 --output work\candidates-01
```

检查输出 `manifest.json`、实际帧文件与时间。此步骤只验证解码和本地选帧，不调用大模型。旧包需要更便捷安装或 PATH 注册时，使用 [installer-v1](installer-v1.zh.md)，不要用机器人 0.7.1 的安装脚本。

## 模型配置、解析和结果

复制包内 profiles 模板，填写实际 provider、model 和 base_url；api_key_env 指向操作者设置的环境变量。支持的适配器包括 LM Studio、OpenAI Responses、兼容端点、Anthropic 与 Gemini；服务是否支持图像和结构化输出必须实际 probe。

```powershell
.\piper-video.cmd models show --model-config work\model.json
.\piper-video.cmd models probe --model-config work\model.json --image examples\probe.jpg --output work\probe-01
.\piper-video.cmd compile --video examples\transfer.mp4 --task "描述完整抓取、移动、释放顺序，无法确认则保留未知" --model-config work\model.json --output work\demo-01 --store work\demos.sqlite
.\piper-video.cmd inspect --bundle work\demo-01
.\piper-video.cmd find --store work\demos.sqlite --task "搬运"
```

检测器是可选项：确认实际 ONNX 路径后为 compile 加 `--detector-onnx FILE`。`--max-keyframes` 默认 24，与模型 max_images、token 上限是不同限制。compile 产物通常有 `demo.json`、引用图片和 model-calls；supported 是模型假设，unknown 不应静默删除。人工标注准备好后才运行 evaluate，它不会生成独立真值。

## 给外部模型使用

stdio MCP 启动命令是 `piper-video mcp --workspace DIR --model-config JSON`；workspace 必须存在。主要工具是：

| 工具 | 用途 |
|---|---|
| `pipeline_capabilities` | 查看当前配置，不发送推理请求 |
| `video_candidates` / `video_candidate_page` | 本地选帧与分页索引 |
| `video_frame` | 真正返回图片，供代理自行解释 |
| `video_compile` | 使用操作者固定的模型配置编译视频 |
| `video_inspect` / `video_evaluate` | 读取证据与对照已有标注 |

未传模型配置时，选帧和取图仍能用，compile 明确提示缺配置。MCP 服务等待协议输入，没有普通聊天提示符属正常。想执行动作需另接 Robot MCP，并查看当前机器人场景；录制示范不授予动作许可。

## 文件管理与排障

输出目录必须新建命名，不能覆盖旧证据；同视频重试使用 demo-02 等新目录。API 401/429/超时检查 provider 日志，配置存在或 HTTP 200 都不等于语义解析成功。上下文超限可缩小待解析帧预算，但不将缺帧后的结论冒充完整视频判断。

此独立版没有 Robot dev2 的 `/demo watch` 斜杠接口，也没有实时腕部相机、计划或几何计算。安装包带选定检测器和合成样例，不含私人视频、云账户或 VLM 权重。许可与检测器源代码见发布资产和 THIRD-PARTY 文档。

历史独立包验证覆盖全新安装、CLI/stdio MCP、选帧、ONNX 初始化和图像返回；发布记录的上位源码回归为 290 通过/3 跳过。未声称本次模型 API 推理、真人语义准确率或真机控制验收。本次仅重新核对版本资料。
