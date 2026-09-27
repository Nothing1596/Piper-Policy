# piper-video 命令与功能 / Command reference

适用 **0.3.0**。[返回中文 README](../README.md) · [English README](../README.en.md) · [安装向导](VIDEO-GUIDE.zh.md) · [Installation guide](VIDEO-GUIDE.en.md)

前端为 `piper-video`；Windows 离线包使用 `.\piper-video.cmd`。只负责视频证据，不提供机器人动作或仿真启动。The video-only CLI selects frames, interprets demonstrations and manages evidence; it does not execute robot actions.

## 调用流程 / Workflow

```powershell
.\piper-video.cmd candidates --video examples\transfer.mp4 --output work\candidates-01
.\piper-video.cmd models show --model-config profiles\openai.json
.\piper-video.cmd models probe --model-config profiles\openai.json --image examples\probe.jpg --output work\probe-01
.\piper-video.cmd compile --video examples\transfer.mp4 --task "描述动作与最终位置，不确定则记录未知" --model-config profiles\openai.json --output work\demo-01 --store work\demos.sqlite
.\piper-video.cmd inspect --bundle work\demo-01
.\piper-video.cmd find --store work\demos.sqlite --task "搬运"
.\piper-video.cmd mcp --workspace . --model-config profiles\openai.json
```

配置自己的视觉模型与密钥变量后再运行 `probe` / `compile`。选帧不调用大模型；`show` 不验证推理。Use your own vision model and environment-variable credentials before API calls. Candidate extraction is local; configuration display is not an inference test.

## 参数与结果 / Semantics

- `compile`：`--max-keyframes` 默认 24；`--detector-onnx` 可选；`--cache` 指定缓存；`--store` 注册结果。使用 `--model-config` 时不能同时使用旧的 `--model` / `--model-url` / `--model-manifest`。
- `inspect`：`--bundle` 与 `--id` 二选一；ID 查询需要 `--store`。`evaluate` 需要示范包、标注 JSON 和输出路径；它比较给定标注，不生成独立人工真值。
- 模型配置 JSON 的主要字段：`provider`、`model`、`base_url`、`api_key_env`、`timeout_s`（默认 180）、`max_images`（默认 6）、`max_output_tokens`（默认 4096）、`structured_output`、`trust_env`、`artifact_manifest`。Provider adapters: LM Studio, OpenAI Responses, OpenAI-compatible, Anthropic, Gemini. Endpoint/model compatibility must be verified with `probe`.
- 输出通常为 JSON。示范保存到 `demo.json`，带阶段、图像引用和 unknown。候选帧保存到 `manifest.json`。模型日志保留在输出旁的 model-calls 目录。使用新输出目录，避免覆盖证据。
- CLI 正常返回 0；示范 unresolved 等已识别失败结果返回 2；参数错误通常也是 2，其他异常返回非零。Read both the exit code and JSON verdict. A model's `supported` is not human acceptance or robot task success.
- `--max-keyframes` 是选帧预算，`max_images` 是模型请求配置，二者不等于上下文 token 上限。它不负责自动训练权重或跨场景技能迁移。

## MCP 工具 / MCP tools

启动参数：`mcp --workspace DIR [--model-config JSON] [--detector-onnx FILE]`。workspace 必须已存在；工具路径限制在其中。stdio 服务等待客户端协议输入，不是交互聊天窗口。Without a model profile, local CV and evidence tools remain available; compilation reports a missing profile.

| 工具 / Tool | 参数 / Arguments | 功能 / Behavior |
|---|---|---|
| `pipeline_capabilities` | 无 / none | 显示配置；不会验证推理 / Configuration only |
| `video_candidates` | `video`, `output` | 本地选帧 / Local candidate extraction |
| `video_candidate_page` | `manifest`, `offset=0`, `limit=24` | 分页帧信息，limit 1..100 / Page frame metadata |
| `video_frame` | `image` | 返回实际图像 / Return pixels |
| `video_compile` | `video`, `task`, `output`, `max_keyframes=24` | 固定配置模型解析；发送选中图片 / Compile via configured provider |
| `video_inspect` | `bundle` | 读取结论与未知项 / Read claims and unknowns |
| `video_evaluate` | `bundle`, `annotations`, `output` | 对照标注 / Compare annotations |

MCP 与 CLI 的参数并非完全一一对应，例如 MCP 编译不接受任意模型地址，也没有 store 参数。外部代理可以自己看 `video_frame` 返回的图片；这不等于把代理登录态当作模型 API。The MCP profile is fixed by the operator; CLI and MCP argument sets differ.

## 命令索引 / Command index

| Command | 功能 / Function |
|---|---|
| `candidates` | 本地 CV 候选帧 / Local candidate extraction |
| `compile` | 模型解析视频并生成证据包 / Compile a demonstration |
| `inspect` | 读取示范包或示范 ID / Inspect a bundle or ID |
| `find` | 检索示范库 / Search the demonstration store |
| `evaluate` | 与给定人工标注比较 / Compare supplied annotations |
| `models show` | 显示配置，不进行推理 / Show configuration only |
| `models probe` | 单图真实模型请求 / Real single-image API probe |
| `mcp` | 视频工具 stdio 服务 / Video MCP stdio server |

## 完整参数 / Exact help snapshots

以下内容来自对应发行版的实际 `--help`，未启动后端或调用模型。Generated from installed release help, without starting a backend or calling a model. Video subcommand help retains the legacy `piper-lab demo` program label; invoke it as `piper-video` followed by the listed command.

<details>
<summary>piper-video --help</summary>

```text
usage: piper-video [-h] {candidates,compile,inspect,find,evaluate,models,mcp}

Independent video evidence pipeline

positional arguments:
  {candidates,compile,inspect,find,evaluate,models,mcp}

options:
  -h, --help            show this help message and exit
```

</details>

<details>
<summary>piper-video candidates --help</summary>

```text
usage: piper-video candidates [-h] --video VIDEO --output OUTPUT

options:
  -h, --help       show this help message and exit
  --video VIDEO
  --output OUTPUT
```

</details>

<details>
<summary>piper-video compile --help</summary>

```text
usage: piper-lab demo compile [-h] --video VIDEO --task TASK --output OUTPUT
                              [--model-config MODEL_CONFIG] [--model MODEL]
                              [--model-url MODEL_URL]
                              [--model-manifest MODEL_MANIFEST]
                              [--max-keyframes MAX_KEYFRAMES] [--store STORE]
                              [--cache CACHE] [--detector-onnx DETECTOR_ONNX]

options:
  -h, --help            show this help message and exit
  --video VIDEO
  --task TASK
  --output OUTPUT
  --model-config MODEL_CONFIG
                        Explicit JSON provider profile; keys are read from
                        named environment variables
  --model MODEL
  --model-url MODEL_URL
  --model-manifest MODEL_MANIFEST
  --max-keyframes MAX_KEYFRAMES
  --store STORE
  --cache CACHE
  --detector-onnx DETECTOR_ONNX
```

</details>

<details>
<summary>piper-video inspect --help</summary>

```text
usage: piper-lab demo inspect [-h] (--bundle BUNDLE | --id ID) [--store STORE]

options:
  -h, --help       show this help message and exit
  --bundle BUNDLE
  --id ID
  --store STORE
```

</details>

<details>
<summary>piper-video find --help</summary>

```text
usage: piper-lab demo find [-h] --store STORE --task TASK

options:
  -h, --help     show this help message and exit
  --store STORE
  --task TASK
```

</details>

<details>
<summary>piper-video evaluate --help</summary>

```text
usage: piper-lab demo evaluate [-h] --bundle BUNDLE [--store STORE]
                               --annotations ANNOTATIONS --output OUTPUT

options:
  -h, --help            show this help message and exit
  --bundle BUNDLE
  --store STORE
  --annotations ANNOTATIONS
  --output OUTPUT
```

</details>

<details>
<summary>piper-video models show --help</summary>

```text
usage: piper-lab models show [-h] --model-config MODEL_CONFIG

options:
  -h, --help            show this help message and exit
  --model-config MODEL_CONFIG
```

</details>

<details>
<summary>piper-video models probe --help</summary>

```text
usage: piper-lab models probe [-h] --model-config MODEL_CONFIG --image IMAGE
                              --output OUTPUT

options:
  -h, --help            show this help message and exit
  --model-config MODEL_CONFIG
  --image IMAGE
  --output OUTPUT
```

</details>

<details>
<summary>piper-video mcp --help</summary>

```text
usage: piper-video mcp [-h] --workspace WORKSPACE
                       [--model-config MODEL_CONFIG]
                       [--detector-onnx DETECTOR_ONNX]

options:
  -h, --help            show this help message and exit
  --workspace WORKSPACE
  --model-config MODEL_CONFIG
  --detector-onnx DETECTOR_ONNX
```

</details>
