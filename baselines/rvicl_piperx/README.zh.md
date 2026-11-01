# RPent → PiperX：下载后运行单臂仿真

保留原 RPent 的视频层级、递归检索、planner 和动作原语，连接现有 PiperX MuJoCo 执行器。没有我们的 skill、plan 或行为复核；本入口只做仿真。

**下载发布页的 `rpent-piperx-1.6.zip`，不是 GitHub 自动生成的 Source code.zip。** 运行包附带固定版本 RPent 源码与 Piper 执行器 wheel；首次安装依赖和准备视频仍需联网。

## 1. 准备环境

- Linux / macOS；Windows 请在 WSL2 内运行，原生 Windows 不支持本批处理入口。
- 安装 `uv`、`ffmpeg`（含 ffprobe），能运行 `python3`。macOS 可用 `brew install uv ffmpeg`；Ubuntu 安装 ffmpeg，uv 按[官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装。
- Linux 无头仿真默认 EGL，需要可用的 OpenGL/EGL 环境；不能渲染时先检查显卡驱动。macOS 使用原生渲染。
- 真实模型任务另需自己的模型 API 和可访问的 SAM3 REST 服务；自测不需要二者。

解压后进入目录，所有命令在**同一个终端**完成：

```sh
cd rpent-piperx-1.6
python3 quickstart.py setup
python3 quickstart.py smoke
```

`setup` 自动建立独立 Python 3.12 环境、安装锁定依赖并核验 RPent 源码。重复执行不会删除实验记录。
`smoke` 串行运行两题脚本化响应＋真实 MuJoCo，自动启动、清理执行器，**没有模型 API 费用，也不代表模型完成了任务**。

## 2. 准备三题 WatchAct 示例

```sh
python3 quickstart.py prepare
```

按固定版本下载原始资产和三个完整视频，逐个核对 SHA-256，生成本机路径的场景、评分和清单并编译检查。下载失败会报错；重复准备只复用哈希相同的资源，不覆盖已修改文件。第三方资产不随 ZIP 再分发。

| 作业 | 完整视频 | 操作目标 |
|---|---|---|
| imitation_data_315 | GX010391.MP4 | 汤罐入篮，番茄酱保持 |
| imitation_data_320 | GX010392.MP4 | 拿起并放回番茄酱，再将汤罐入篮 |
| imitation_data_355 | GX010399.MP4 | 汤罐入篮，橙汁盒保持 |

## 3. 配置模型，然后启动自动化

```sh
cp examples/model.env.example .env.model
# 编辑 .env.model：填自己的 API key、兼容接口地址
# 编辑 examples/model-config.json：核对模型名和 sam3_endpoint
python3 quickstart.py run --dry-run
python3 quickstart.py run
```

`run` **会发起付费模型请求**，依次运行三个独立场景；安装、自测、准备和 dry-run 不发起模型请求。默认模型/推理为 `gpt-6.1-sol / low`，SAM3 为 `http://127.0.0.1:8114`。示例文件不含可用密钥。SAM3 服务和权重需预先部署，或通过 SSH 转发已有服务；预检失败会在模型调用前退出，不会静默关闭 SAM 或换模型。

想用自己的配置：

```sh
python3 quickstart.py run --config /absolute/model-config.json --env-file /absolute/private.env --output runs/my-batch
```

再次运行相同命令会核对原结果，**不重跑已结束成功、失败或异常回合，不重放未知动作**。基础设施异常后先检查回执；确认清理后显式再次运行才会继续未启动任务。要启动一组独立新实验，明确换 `--output runs/new-batch`。

## 4. 看结果

打开 `runs/watchact/run-*/summary.json`，逐题看 `score.json`、`receipt.json` 和 `worker.log`。运行正常结束不等于任务成功。

- 独立评分检查罐是否完整入篮、实际接触提起、松手撤离和终态稳定；第二题还检查番茄酱先拿起归位。
- `model_calls/` 和 `hierarchy/` 的真实 usage 一并计入；缺失用量明确保留，工具调用与内部路点分别统计。
- 每题自动清理自己的执行器，退出证据在 `rollout/simulation/cleanup.json`。
- 密钥、运行认证、原始日志和下载缓存留在本机，分享时不要直接打包整个 runs/var 目录。

## 已知限制与详细说明

最新三题是 **2 题正常结束但独立评分失败，1 题适配器异常**，不是成功基准。地面高度和滚动问题已通过动力学验收，稳定抓持尚未通过；180° 附近数值舍入可能触发关节限位拒绝并暂停。此版本保留问题记录，没有为模型失败更改策略。评分是 Piper 适配指标，不是官方 WatchAct/LIBERO 成绩。

- [原理、坐标、原语和手动命令](docs/adapter-reference.zh.md)
- [批处理、续跑和自定义任务](docs/batch.zh.md)
- [三题视频、物理修复、实际调用与评分结果](docs/watchact-batch-20261101.zh.md)
- [软件与仿真验收记录](VALIDATION.zh.md)
