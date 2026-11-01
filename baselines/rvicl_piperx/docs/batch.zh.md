# 单臂 PiperX 批处理（2026-11-01）

入口与原项目布局一致：`scripts/eval_imitator_game.py`，但调用独立的 `piper_evaluate`，不修改原 rvicl2。批处理仅支持 Linux/macOS；真实机器人不在本入口范围内。

## 已部署目录与运行环境

Featurize 项目目录为 `/home/featurize/work/rpent2piperx`。`.venv` 指向独立的 `/home/featurize/.local/share/rpent2piperx/runtime`：工作目录所在 NFS 安装大量小文件很慢，因此 Python 环境放在本地盘，源码、依赖锁、安装轮子及日志仍在项目目录。更换实例后按锁文件和轮子重建环境，不依赖这个链接可移植。

共享 RPent 为 `/home/featurize/work/RPent`，启动核验全部锁定的框架/机器人 Python 源码。认证只运行时引用原私有 env 文件，不复制到新目录。不要把原 `.env.example` 当成可以公开的示例，它实际含密钥。

## 先验收流程

```sh
cd /home/featurize/work/rpent2piperx
export MUJOCO_GL=egl
.venv/bin/python -u scripts/eval_imitator_game.py \
  --smoke --ffmpeg-dir /home/featurize/work/gpt-policy9/.venv/bin \
  --output-root var/acceptance
```

运行两轮独立的脚本化响应＋真实 MuJoCo，模型付费调用为零。每轮自动启动与清理执行器。原命令再次执行只核对和跳过已结束回合，不重放动作；它不是模型完成任务的分数。

## 原五任务清单检查

```sh
.venv/bin/python -u scripts/eval_imitator_game.py \
  --env-file /home/featurize/work/rvicl2/.env.example \
  --ffmpeg-dir /home/featurize/work/gpt-policy9/.venv/bin \
  --eval-config /home/featurize/work/yuan/The-Imitator-Game/examples/baselines/lerobot_dataset/eval/exp_list/unseen_5tasks_env_list.txt \
  --model gpt-6.1-sol --sam3-endpoint http://127.0.0.1:8114 \
  --sam3-protocol rest --primitive-image-history latest --levels L0 \
  --episode-start 0 --episodes-per-env 1 \
  --output-root var/unseen5 --dry-run
```

**这五题目前没有匹配的单臂资产与评分绑定，不可直接开跑。** 去掉 `--dry-run` 仍在 API 之前拒绝，不会把双 Panda 任务放进默认两方块世界。`configs/piper_scenes.json` 按环境 ID 绑定 `arm_count: 1`、`simulation_asset`、`evaluation_file`。不会自动把双臂轨迹翻译成单臂答案。

## 单臂公开任务清单

`--manifest` 替代原 IG 的任务抽样，其他推理参数不变。最小 JSON 列表：

```json
[{
  "job_id": "single-arm-001",
  "kind": "model",
  "arm_count": 1,
  "video": {"path": "/absolute/human.mp4", "start_s": 0, "end_s": 10},
  "language_instruction": "将视频中的完整操作应用于当前场景。",
  "simulation_asset": "/absolute/matched-piper-scene.xml",
  "evaluation_file": "/absolute/operator-only-goals.json",
  "seed": 0
}]
```

模型只得到视频、公共语言与视角信息、实际相机和原工具；不接收目标谓词或 oracle plan。清单和评分路径属于作业端数据，不能把整份作业目录交给拥有自由文件读取权限的模型代理。

通用布尔评分接口示例（已有 WatchAct 独立几何/过程评分另见下文）：

```json
{"scope":"piper_scene_metrics","predicates":[
  {"path":["objects","cube_blue","in_tray"],"equals":true},
  {"path":["objects","cube_blue","grasped"],"equals":false}
]}
```

这是既有两方块/托盘场景的有限评分接口，不是 WatchAct BDDL、完整场景匹配或独立长时稳定性验证。启动前实际编译场景并检查指标存在、初态未已达成目标；结束后只看实测末态，不信任模型 finish 声明。缺字段算评分故障，不能当作模型失败。目前另有三题 WatchAct 专用场景与独立评分；不支持任意 BDDL 自动生成场景或评分。详见 [三题批次说明](watchact-batch-20261101.zh.md)。

## WatchAct 导入边界

```sh
.venv/bin/python scripts/import_watchact.py \
  --data-root /absolute/WatchAct-dataset --task Imitation --view front \
  --limit 3 --output configs/watchact-front.json
.venv/bin/python scripts/eval_imitator_game.py \
  --manifest configs/watchact-front.json --output-root var/watchact --dry-run
```

导入只读公开 `data/*.jsonl`，不读含 oracle 的 `meta_data/`。不会下载整套数据、修改 BDDL 或生成假场景。未下载视频或未绑定场景/评分会在清单中明确报告。WatchAct 同时有语言与视频，非 Imitation 子任务的目标、初态可能不同，不能统一理解为照抄视频终态；语言指令必须保留到模型输入。

## 续跑与统计

输入、源代码与 RPent 哈希决定 `run-<hash>` 目录。相同输入重复运行进入同一目录。改变代码、模型、视频、资产或评分形成新目录，不冒充续跑。

- 每个作业先持久化 running 回执，再启动子进程。批次 OS 锁避免相同作业双启动。
- 已完成成功、失败、异常均不自动重跑。异常暂停后，显式再次运行可继续后续任务；未确认清理时继续暂停。
- 中断只有原结果完整且 `simulation/cleanup.json` 确认停止，才按原记录认定结束。未知结果、损坏回执、无回执的残留目录不启动新动作。没有物理检查点时不恢复场景状态。
- 每轮保存 worker 日志、配置、回执、模型请求/图片/usage、工具与执行器请求、独立评分和清理证据。层级与 planner API 一并统计；未知 usage 为 null，并单列缺失数量。脚本化验收与真实模型成绩分开。
- `summary.json`、`results.json` 在每轮更新。`completed` 是流程结束，`task_success` 才是当前适配评分；不输出原双臂 TSS 或官方 WatchAct/LIBERO 成绩。

SAM 使用原服务 `/segment` 与 `/segment_point`。预检检查 `/openapi.json` 的实际接口；该服务没有 `/health`。接口存在只证明服务可达，真实分割另做诊断。原 `CUDA_VISIBLE_DEVICES=0 ... serve_sam3_local.py --checkpoint .../sam3.pt --device cuda --port 8114 --host 127.0.0.1` 命令保持不变；无需每批重复启动或改权重。
