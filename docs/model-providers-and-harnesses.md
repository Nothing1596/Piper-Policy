# 模型 API 与 MCP/CLI 接入（0.2.0）

前端仍是 CLI。现在可以把“谁负责看图”和“谁安排整个任务”分别配置。例如，你在 Codex 中提出“分析这段示范”，Codex 通过 MCP 调用管线；管线可以把选出的帧交给你配置的 API，也可以继续使用 LM Studio。换供应商不需要重写视频选帧、示范存储或机器人动作校验。

另一个路径不需要为管线配置 API：代理调用 `video_candidates`，通过 `video_candidate_page` 取得帧列表，再用 `video_frame` 读取实际图片，由代理自身模型解释。这个解释属于代理会话，不会自动变成已经通过管线 schema 和引用复核的示范包。需要规范示范包时，用配置了模型的 `video_compile`。

| provider | 接口 | 鉴权环境变量 |
|---|---|---|
| lmstudio | 现有本地接口，默认保留 | 无 |
| openai | Responses + 图片 + JSON Schema | OPENAI_API_KEY |
| openai-compatible | Chat Completions + 图片 | PIPER_MODEL_API_KEY |
| anthropic | Messages + base64 图片 + structured outputs | ANTHROPIC_API_KEY |
| gemini | generateContent + inlineData + JSON Schema | GEMINI_API_KEY |

支持这些协议不等于供应商的每个模型都能看图。选择的模型必须支持图片以及相应结构化输出。示例配置中的 `YOUR_VISION_MODEL_ID` 必须替换成账户实际可用的模型标识；兼容接口还需要修改地址。代码不会自动选择付费模型、切换供应商、下载权重或使用 Codex/Claude Code 的登录凭据。

仓库示例位于 `config/model-profiles/`，交付 ZIP 中是 `profiles/`。以 ZIP 为例，编辑一份自己的配置，然后在设置好相应环境变量的终端执行：

```powershell
.\piper.cmd models show --model-config profiles\openai.json
.\piper.cmd models probe --model-config profiles\openai.json --image my-test.png --output runs\api-probe-01
.\piper.cmd demo compile --model-config profiles\openai.json --video examples\transfer.mp4 --task "Describe the demonstrated transfer" --output runs\api-demo-01
.\piper.cmd demo inspect --bundle runs\api-demo-01
```

`show` 只检查配置和环境变量是否存在；`probe` 才会实际发送图片并产生费用。API key 只从环境变量读取，不能写进 JSON。不要把 key 直接写入 MCP 配置或提交到版本库。需要代理环境时显式设置 `trust_env: true`，默认不读取系统代理。远程 endpoint 要求 HTTPS；本地兼容服务可以用 loopback HTTP。

`demo compile`、`policy run`、`eval run` 都接受 `--model-config`。旧的 `--model`、`--model-url`、`--model-manifest` 继续用于本地服务，不能与 profile 混用。兼容供应商若只支持 JSON object，可显式配置 `structured_output: "json_object"`；仍在本地验证完整 schema，不会静默退化到自由文本。OpenAI 等严格 schema 的可选字段用 nullable 表达，收到 null 后恢复为省略字段，再按原业务 schema 验证。

请求日志记录模型配置、图片哈希、提示、结构化结果、耗时和供应商返回的用量；不保存鉴权头和错误响应正文。日志包含任务内容，应与视频一样妥善保存。超时、截断、拒答、错误 JSON 和错误 schema 都不能当成功。API 层不自动重试，编译器原有的有界重询仍可能发出额外请求。模型别名不能证明远端权重，因此云 API 不启用需要权重身份证明的持久编译缓存，也不能混入旧本地模型回归统计。

## MCP 连接

合并包已带齐依赖；从源码安装视频 MCP 服务可用 `pip install ".[mcp]"`。要注册机器人 MCP 工具，还需安装下位机包 `piperx-middleware`。

使用绝对路径配置客户端。先建立一个只存放本任务视频和结果的 workspace；不要把整个用户目录暴露给代理。MCP 输出目录必须是新路径，访问会检查 workspace 边界。下列示例假设安装包位于 `D:\PiperCLI`，工作目录为 `D:\PiperWork`。

Codex CLI：

```powershell
codex mcp add piper -- D:\PiperCLI\.venv\Scripts\python.exe -m piperlab.harness --workspace D:\PiperWork --model-config D:\PiperCLI\profiles\openai.json
```

Claude Code：

```powershell
claude mcp add --transport stdio piper -- D:\PiperCLI\.venv\Scripts\python.exe -m piperlab.harness --workspace D:\PiperWork --model-config D:\PiperCLI\profiles\anthropic.json
```

如果只让代理读取帧并自行解释，去掉 `--model-config`。支持 stdio MCP 的其他 harness 使用相同 command/args；下面是常见 `mcpServers` 格式，具体配置文件位置和外层字段以客户端为准，不声称所有客户端完全相同：

```json
{
  "mcpServers": {
    "piper": {
      "command": "D:\\PiperCLI\\.venv\\Scripts\\python.exe",
      "args": ["-m", "piperlab.harness", "--workspace", "D:\\PiperWork"]
    }
  }
}
```

代理可调用 `pipeline_capabilities`、`video_candidates`、`video_candidate_page`、`video_frame`、`video_compile`、`video_inspect`、`video_evaluate`。`video_frame` 返回实际像素，客户端可能把它交给自己的云模型；`video_compile` 则把选定帧交给启动时指定的供应商。工具不会从视频文字或模型输出中取得新 endpoint、key 或命令。

0.2.1 补充 `simulation_observe(output)`：启用 robot-root 且后端确认为 MuJoCo 时，返回当前相机图像，并在 workspace 的新目录保存 RGB、depth.npy 和标定/时间元数据。不会调用隐藏物体坐标或操作员评分。2026-09-27 已用真实 MCP 调用完成一次“看图选红块→抓取→抬起回看→移到托盘→释放→退开回看”，终态物理评分在动作结束后独立确认红块入盘。不是供应商 API 基准，也不构成人类视频技能迁移验收。

需要同时开放下位机能力，在 MCP 启动参数追加 `--robot-root D:\PiperWork\sim --robot-url http://127.0.0.1:8808`。仿真服务需要先通过 `piper sim start` 启动。此时同一 MCP 服务注册已有下位机工具，沿用原 token、会话、连接和动作检查。MCP 启动自身不启动后端、不连接 CAN、不自动授权运动。没有 `--robot-root` 就没有机器人工具。

## 验证边界

本轮验证 API 请求/响应契约、模拟 HTTP 下的完整视频编译链、真实 stdio MCP 通信、工作目录限制和合并包安装。没有供应商账户/模型的真实调用结果，没有改动用户 Codex/Claude Code 全局配置，也没有通过每个 harness 的交互 UI 验收。供应商实测先执行单图 `probe`，再用匹配任务的合成示范和相同仿真种子评测。旧本地合成 6/6、冻结 64/90、真人 LongVIL 解析都保留原身份，不能当这些新供应商的成绩。

最终独立安装环境运行上位全套测试：285 通过、3 跳过；54 个已安装 Python 模块与当前源码哈希一致。MCP 经过真实客户端/服务端 stdio 初始化、工具枚举、像素传输、合成视频抽帧及路径越界拒绝。下位包仍为既有 0.5.0，本轮验证组合工具注册及安装资产，没有重新运行真机或模型仿真回归。首次 MCP 视频调用曾停在 NumPy 原生库初始化；将 CPU 视频依赖提前到 stdio 读线程之前加载后，上述测试正常通过。

实现对照的官方资料：[OpenAI 结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)、[图片输入](https://developers.openai.com/api/docs/guides/images-vision)、[Codex MCP](https://developers.openai.com/codex/mcp)、[Anthropic 结构化输出](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[Claude Code MCP](https://code.claude.com/docs/en/mcp)、[Gemini generateContent](https://ai.google.dev/api/generate-content)。
