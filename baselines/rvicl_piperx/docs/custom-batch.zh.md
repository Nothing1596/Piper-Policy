# 服务器上配置自己的视频、模型和自动化

三个 WatchAct 视频只是已配好场景/评分的示例。自定义作业用 `run --manifest`，不调用示例 prepare、不读取默认三题，也不要求视频来自 WatchAct。

## 配置三个文件

在已部署的 `/home/featurize/work/rpent2piperx` 工作；下载包用户在解压目录工作。

```sh
mkdir -p jobs
cp examples/custom-manifest.json jobs/my-tasks.json
cp examples/model-config.json jobs/my-run.json
cp examples/model.env.example .env.model
```

编辑以下内容：

| 文件 | 配置内容 |
|---|---|
| `jobs/my-tasks.json` | 任意数量的任务行：视频本地路径、时间区间、公共指令、对应单臂场景、独立评分、seed、唯一 job_id |
| `jobs/my-run.json` | 模型名、推理强度、自动化预算、示范取帧和 SAM3 服务参数；可复用已有兼容配置 JSON |
| `.env.model` | `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，私有保存，不提交到 Git |

视频必须已下载为本地文件。`end_s: null` 表示使用真实视频结尾；剪辑的开始时间必须与配置的场景初态对应，程序不会自动从视频重建世界。路径相对于任务清单文件解析，建议服务器上直接写绝对路径。

当前模型入口是 OpenAI-compatible Chat Completions。后端需支持图片和原生工具调用，模型名、reasoning_effort 与 schema 功能也需供应商支持；不能只将文本专用服务填入 endpoint 就声称等价。本轮不加视觉代理、不切换原推理方法。视频层级构建与动作决策使用同一配置。**已有 shell 中的 OPENAI/CODEX 环境变量优先于 env 文件**；换后端前应核对它们，避免旧变量覆盖新文件，不要把密钥打印到日志。

同一批次共用一份模型配置。多模型对照分别使用对应 config/env 和不同 output 运行，不在一个任务执行中途切换后端。

最小任务行（完整可编辑模板见 `examples/custom-manifest.json`）：

```json
[{
  "job_id": "my-task-seed-0",
  "kind": "model", "arm_count": 1, "seed": 0,
  "video": {"path": "/absolute/my-demo.mp4", "start_s": 0, "end_s": null},
  "language_instruction": "Perform the demonstrated sequence in the current robot scene.",
  "camera_perspective": "front", "spatial_reference": "camera",
  "simulation_asset": "/absolute/matched-piper-scene.xml",
  "evaluation_file": "/absolute/operator/evaluation.json"
}]
```

一个视频对应一条或多条独立任务行；重复测试显式添加不同 job_id 和 seed。自定义 manifest 不使用 IG 数据集的 episodes_per_env 自动扩展，避免将一个未生效的选项当作重复次数。

## 运行与覆盖参数

```sh
python3 quickstart.py run \
  --manifest jobs/my-tasks.json --config jobs/my-run.json \
  --env-file .env.model --output runs/my-batch --dry-run

python3 quickstart.py run \
  --manifest jobs/my-tasks.json --config jobs/my-run.json \
  --env-file .env.model --output runs/my-batch
```

Featurize 的 ffmpeg 不在 PATH 时，追加 `--ffmpeg-dir /home/featurize/work/gpt-policy9/.venv/bin`。dry-run 输出必须检查 `blocked: []`；它列出文件/评分绑定问题，不代替正式启动前的场景编译与 SAM3 服务预检。预检失败不会调用模型。

可直接覆盖已有选项，命令行优先于 JSON：

```sh
python3 quickstart.py run --manifest jobs/my-tasks.json \
  --config jobs/my-run.json --env-file .env.model --output runs/another-batch \
  --model my-vision-model --reasoning-effort high \
  --max-turns 100 --max-episode-steps 500 \
  --sam3-endpoint http://127.0.0.1:8114 --primitive-image-history latest
```

| JSON 字段 | 实际含义 |
|---|---|
| `model`、`reasoning_effort` | 请求的模型别名及推理强度 |
| `max_turns` | 每回合原 RPent 决策轮数上限 |
| `max_episode_steps` | 适配器执行计数上限，不是物理仿真 tick |
| `trial_timeout`、`api_timeout` | worker 墙钟秒数上限、单次 API 超时 |
| `candidate_fps`、`max_candidates`、`max_keyframes`、`clip_budget` | 原视频候选采样、概览和局部补看预算 |
| `sam3_enabled`、`sam3_endpoint`、`sam3_protocol`、`sam3_timeout` | 原感知接口；改变配置必须单独记录，不能冒充同配置对照 |

当前串行运行；没有新增并发调度或模型策略重试。重复相同命令保留已完成失败，未知动作不重放；基础设施异常在确认清理后显式继续未启动任务。独立重跑用新的 output 路径。

## 场景与评分边界

自定义视频不等于自动得到匹配仿真。`simulation_asset` 必须满足现有单臂 PiperX/相机/物体遥测约定；`evaluation_file` 使用现有有限布尔指标或 WatchAct 抓取搬放几何/过程评分。倒水、折盒等新物理和目标并未自动实现。不要让任意视频在默认三题场景里运行后宣称复现成功。

目标和 oracle 只供独立评分，不进入模型；原 native planner、视频层级、检索和动作原语不变。结果仍看 `run-*/summary.json`、逐题 score/receipt/worker.log，而非只看模型 finish。
