# 视频管线独立部署指南

[English](VIDEO-GUIDE.en.md)

[完整命令与功能参考](VIDEO-CLI-REFERENCE.md)

版本 0.3.0，入口 `piper-video`。此包独立安装视频处理、模型 API 和 MCP；不安装下位机或 MuJoCo。Python 分发名仍为 `piper-lab`，保留旧上位机模块以兼容源码，但本指南使用视频专用入口。

## 1. 安装

准备 Windows x64 和 Python 3.12 x64。解压到一个新目录，在该目录打开 PowerShell：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper-video.cmd --help
```

安装器核对 SHA256，再从包内 wheels 离线安装到 `.venv`。不改全局 PATH，不启动模型，不保存密钥。若 Python 没有 `py` 启动器，给安装器传入 `-PythonExe C:\Python312\python.exe`。换机器时重新解压安装，不复制 `.venv`。

## 2. 不接模型，先验证视频入口

```powershell
.\piper-video.cmd candidates --video examples\transfer.mp4 --output work\candidates-01
```

查看生成的 `manifest.json` 和帧图。这里完成解码、运动分析与候选帧筛选，没有理解动作语义。输出目录必须是新目录；重跑使用不同名字。

## 3. 接入模型 API

`profiles` 包含 LM Studio、OpenAI、OpenAI-compatible、Anthropic 和 Gemini 示例。复制合适的 JSON，填写实际模型名、端点，保留密钥为环境变量引用。不要把密钥写进 JSON 或发给模型。

```powershell
$env:OPENAI_API_KEY = '填写你自己的密钥'
.\piper-video.cmd models show --model-config profiles\openai.json
.\piper-video.cmd models probe --model-config profiles\openai.json --image examples\probe.jpg --output work\probe-01
.\piper-video.cmd compile --video examples\transfer.mp4 --task "描述抓取、搬运、释放以及最终位置；看不清则记录未知" --model-config profiles\openai.json --detector-onnx models\yolo11n.onnx --output work\demo-01
.\piper-video.cmd inspect --bundle work\demo-01
```

先核对 `models` 目录中的 ONNX 实际文件名；上例以 `yolo11n.onnx` 为例。检测器可选，删除 `--detector-onnx` 即不使用它。`show` 只检查配置，`probe` 才真实发送图片。云服务会收到选中图像并可能产生费用；本地模型服务及其大模型权重不包含在包中。

结果 `demo.json` 记录阶段、帧引用和未知项。`supported` 是模型判断，不是人工验收。`find --store ... --task ...` 搜索已注册示范；编译时用 `--store work\demos.sqlite` 注册。`evaluate --bundle ... --annotations ... --output ...` 比较人工标注，包中注释模板不是现成真值。

## 4. 模型通过 MCP 调用

在支持 stdio MCP 的代理中配置以下服务器，所有路径替换为解压后的绝对路径：

```json
{
  "mcpServers": {
    "piper-video": {
      "command": "D:/PiperVideo/.venv/Scripts/python.exe",
      "args": ["-m", "piperlab.video_entry", "mcp", "--workspace", "D:/PiperVideo", "--model-config", "D:/PiperVideo/profiles/openai.json"]
    }
  }
}
```

不同代理的配置文件格式可能不同；上面给出标准进程命令和参数，按其 MCP 设置填写。密钥变量需由客户端进程继承。省略 `--model-config` 时仍可选帧、翻页、取图及读取证据；`video_compile` 会提示尚未配置模型。不要往 MCP 的标准输入输出混入普通终端文字。

工具包括 `video_candidates`、`video_candidate_page`、`video_frame`、`video_compile`、`video_inspect`、`video_evaluate`。也可让代理直接执行 `piper-video.cmd`，读取 JSON 和图片。外部代理自己的视觉能力可以用于看图；不等于把代理登录态作为模型 API 凭证。

## 5. 结构与边界

```text
视频 → PyAV 解码 → OpenCV 运动分析/选帧
                     ↓ 可选 ONNX Runtime 检测和跟踪
              选中帧 + 时间信息 → 模型 API
                     ↓
           结构校验 → 阶段/证据/unknown → demo.json
```

这是一条示范解析和证据管理管线，不是更新模型权重的训练器。代理若要执行任务，需另外接入 `piper-robot` MCP，结合机器人当前图像重新判断；历史视频不授权机械臂动作。真人六棱柱/玻璃容器视频的解析结果不等于当前彩块/绿色托盘技能能够执行。

## 6. 排障与许可

- API 失败：核对环境变量、端点、实际视觉模型与输出格式支持，查看输出旁的 model-calls 目录；未完成的步骤保留为未知。
- 上下文超限或超时：减少关键帧，确认服务上下文容量，不把重试超时当作成功。
- MCP 无响应：先用 `--help` 检查解释器路径，再检查客户端日志；不要启动多个竞争本地模型的编译任务。
- 依赖或哈希失败：保留日志，重新解压到新目录，不复用半安装环境。

查看 `THIRD-PARTY.md`、`licenses` 和 `DETECTOR-SOURCE.md`。ONNX 检测器有独立的上游许可条件。发布页提供对应 Ultralytics 源码；本项目尚未给原创代码指定统一开源许可证。
