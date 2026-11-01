# RV-ICL 单臂 PiperX 仿真适配

这是本地 `rvicl2` 的 **human-RGB / IG 移植版**接入 PiperX 的独立入口。原目录、旧实验和 Piper 日常 CLI 不变。本入口只运行 MuJoCo，不提供真机开关。

视频层级构建、L0 页面、L1/L2 索引、按需 L3 原视频检索和 RPent 原生 `ApiAgentLoop` 保留。Piper 的 `InteractiveConsoleController`、skill 注入、`task_plan`、视频经验库和行为复核不进入模型请求。

## 冻结了什么

| 部件 | 冻结来源与边界 |
|---|---|
| 本地 rvicl2 | Git `4052a54caac9acb4ce42c050c9a96d11b3e251a4`，加本轮开始时未提交的 IG 源码；逐文件哈希见 `BASELINE.lock.json` |
| RPent | 2026-11-01 核实用户指定 Featurize 的实际版本 `c9026c2bf9c1342b88cdc682e60d578770f871d3`；当前适配器使用此干净版本，关键 planner/原语/感知/会话/工具分发源码与远端哈希一致。旧 v1.1 的 `d6daf341…` 冻结副本与实验保持原样 |
| Piper 下层 | 当前仓库 `1e8c5f4f3d7d1a4aefc042f526ed88cbe56eac87` 的 0.7.2 执行器；执行器、机器人资产与控制参数保持；WatchAct 任务支撑面物理修订单列于 [三题批次](docs/watchact-batch-20261101.zh.md) |
| 认证与历史产物 | `.env*`、运行日志、缓存不作为公开源码复制；运行时可引用已有私有 env 文件 |

启动前核验未改的视频/提示/录制源码与锁定 RPent 的全部 `robots/`、`rpent/` Python 文件哈希，包括实际 API 工具分发实现。每轮先保存适配代码、提示、配置和示范视频哈希，再开始模型请求及执行器启动。输出目录必须是新目录，不隐式恢复动作。

本地共享库中的原 Panda 场景与接口保留供兼容检查；Piper 入口不会实例化它们，也不会将 IG 的私有轨迹或评分传给模型。本变体不使用原 LIBERO 的 pi0.5，不作为原论文完整复现或官方分数。

## 机器人适配边界

| 原接口 | Piper 绑定 |
|---|---|
| `arm` | 保留字段，只允许 `left`，代表唯一 PiperX；不伪造右臂，不暴露 `move_both` |
| `move_to` / `rotate_wrist` | 继承冻结 RPent 的实际方法体。底层调用已有 Piper `resolve_primitive(MoveTo)` 与 IK，输出六关节弧度路径，再转成执行器的角度请求。没有新增笛卡尔规划器 |
| 路径 | Piper 既有的关节空间终点语义，关节线性采样后由原语按 `substeps` 选择；不保证 TCP 直线或避障。`substeps=0` 保留路径，默认 25 限制路点数，1 仅选终点 |
| 姿态 / TCP | 基座坐标米制 xyz、wxyz 四元数，调用下层时转 extrinsic xyz RPY 角度。仿真 TCP 使用资产既定的法兰偏移 `[0,0,0.1425]` m。工具 +Z 朝向指端，夹缝沿工具 X 开合；世界 Z 朝上。`rotate_wrist` 是世界 Z yaw，绝不是 J6 增量 |
| 客观硬件信息 / IK 反馈 | `state.robot_parameters` 提供运行时限位、MDH、TCP 变换、开度配置和 IK 能力边界；实测 TCP 基向量用于明确轴向。失败回显本次请求、实测种子和原错误，不推断碰撞或全局不可达，不提供替代姿态或私有预检结果 |
| 夹爪 | 原 `0..1` 开度映射到执行器配置宽度；反馈按实际最大开度归一化。闭合在已有执行器上以双侧接触结束；空闭合达到目标宽度时，三帧反馈核验后请求保持。原始 cancelled 回执保留，并单列宽度完成证据；不会将 cancelled 改写成 succeeded，也不声称抓住物体 |
| 同时关节/夹爪更新 | 复用下层串行动作，按关节、夹爪顺序发送，不声称原双臂模拟器的原子同步。预算计已接收执行器动作/等待次数，不等价于 IG 500 个物理步 |
| 相机 / 反投影 | `head` 固定相机、`left_wrist` 腕部相机；顺序取帧，各自保留时间、深度与标定，不冒充同步。复用 RPent 的 `back_project` 和 SAM 逻辑；新捕获不会因机械臂未动而复用旧深度 |

默认推理配置对齐用户指定的远端五任务入口：`gpt-6.1-sol / low / latest`，SAM3 开启、REST、地址 `http://127.0.0.1:8114`。在 Featurize 上可复用原服务；在 Mac 上运行时，需使已有服务通过 SSH 转发可达，或用独立 `--config` 显式设置实际地址。不会下载模型、启动额外视觉模型或在 SAM 失败时静默关闭它。`smoke` 不调用 `segment`，因此 smoke 通过不代表 SAM 服务已验收。不同感知配置的结果不能直接充当同配置方法对照。

原视频协议与原 planner 的每响应首个工具执行规则均保留。模型可以在其原生推理中做计划，但没有我们自己的持久化计划工具。

## 安装与启动

需要 Python 3.12、ffmpeg/ffprobe。以下命令在仓库根目录的同一终端运行；不需要另开执行器终端或 OPEN 窗口。

```sh
git clone https://github.com/RLinf/RPent.git baselines/rvicl_piperx/deps/RPent
git -C baselines/rvicl_piperx/deps/RPent checkout --detach c9026c2bf9c1342b88cdc682e60d578770f871d3
uv venv baselines/rvicl_piperx/.venv --python 3.12
uv pip install --python baselines/rvicl_piperx/.venv/bin/python -r baselines/rvicl_piperx/requirements.lock.txt -e './piperx-cli[simulation]'
export RPENT_ROOT="$PWD/baselines/rvicl_piperx/deps/RPent"
export PYTHONPATH="$PWD/baselines/rvicl_piperx:$RPENT_ROOT"
```

先运行不收费的接口测试：

```sh
baselines/rvicl_piperx/.venv/bin/python -m rvicl_ig.piper_cli smoke \
  --rpent-root "$RPENT_ROOT" --output baselines/rvicl_piperx/runs/smoke-001
```

它使用**脚本化 API 响应与真实 MuJoCo 物理**，核对视频检索、实际 TCP 位移/旋转、空闭合/张开、反馈、调用数量及退出。它不证明模型理解或抓取成功。

真实模型运行：准备私有 env 文件，包含 `OPENAI_API_KEY` 与可选 `OPENAI_BASE_URL`（兼容接口的根地址）。模型及推理配置来自 `piper_simulation.json`；默认为指定远端实验所用的 `gpt-6.1-sol / low`，保留其 `latest` 图片历史和 SAM 感知。远端原默认 JSON 仍写旧模型；对齐依据是用户命令覆盖后保存的任务清单。

```sh
baselines/rvicl_piperx/.venv/bin/python -m rvicl_ig.piper_cli run \
  --rpent-root "$RPENT_ROOT" \
  --env-file /absolute/private/model.env \
  --video /absolute/path/human-demo.mp4 \
  --output baselines/rvicl_piperx/runs/model-001
```

`--start-s` / `--end-s` 选择完整视频中的区间；不传终点则读取实际时长。`--config` 指向另一个冻结配置；`--max-turns` 可用于明确标记的连通诊断，默认使用配置的 100 轮。诊断轮数不作为完整 benchmark 预算。

入口自动启动一个隔离的 MuJoCo 执行器并保持控制会话心跳。反馈或身份异常、未知动作结果会暂停；只按原 request_id 查回执，不重新提交动作。退出先释放会话，再经已有 RuntimeManager 关闭本轮拥有的服务。旧输出不能覆盖，新试验使用新目录。

## 日志与验收

- `source/`、`source_manifest.json`、`BASELINE.lock.json`、`config.json`、`agent_prompt.txt`：源码与输入身份。
- `hierarchy/`、`demo_source.json`：示范原图、层级、真实时间与示范哈希。
- `model_calls/`、`planner_result.json`：原 RPent 请求、图片、模型回包和 usage；层级构建调用另存于 `hierarchy/`，统计总调用时必须一起计入。
- `transcript.json`：已结构化返回的工具；`simulation/tools.json` 还记录抛错的工具尝试。工具总量以 native planner stats 与这些日志核对，不把内部关节路点当模型调用。
- `simulation/requests/`、`receipts/`：实际执行器请求和原始结果；`observations/` 与 `execution_frames/` 留存当前与历史证据。
- `result.json` / `error.json`：流程正常结束或异常暂停。`completed` 仅表示运行结束；任务是否成功需看模型 finish 声明与独立匹配的任务评分，不能读它推断抓取成功。
- `simulation/cleanup.json`：本轮执行器释放结果。`simulation/executor/` 含本地认证文件，**不纳入共享实验包**。

测试：

```sh
baselines/rvicl_piperx/.venv/bin/python -m pytest baselines/rvicl_piperx/tests -q
```

本轮具体软件/仿真证据与未通过项见 `VALIDATION.zh.md`。本机 macOS 已验证的内容不扩展为 Windows/Linux、真机、完整 LIBERO/IG 或方法性能验收。

## 求解器与比较边界

RPent 的 `move_to` 本身不绑定 Panda，依赖环境提供 `plan_arm_path` 和 qpos 执行。原 IG 环境使用 Panda 模型配置下的 MPlib `plan_screw`；Piper 当前使用自身六轴运动学的有限多初值数值 IK，再生成关节插值路径。不能把 Panda 的七轴路径或 `panda_hand_tcp` 直接交给六轴 Piper。MPlib 框架可以配置其他机器人的 URDF/SRDF、关节顺序、限位与 TCP，但本轮没有新增这一后端。

原语方法体、`xyz + wxyz`、`substeps` 和 `val/steps` 不变，并不保证轨迹、可达解、接触时序或控制步预算相同。关节插值的中间 TCP 可能偏离任务空间路线；需要沿边插入、拉动或保持接触的任务对此尤其敏感。现有 Piper `move_linear` 仅支持固定姿态的离散直线，不能静默替代任意姿态的 `plan_screw`。

比较我们的方法与此基线时，双方必须共用同一 Piper 求解器、执行器、相机、物理配置及评分。保留模型工具调用、内部执行器请求和墙钟/物理时间各自的统计，不将两边的 500 个计数视作相同物理预算。此入口是 RV-ICL/RPent 的 PiperX 移植基线，不是原 IG 官方成绩。

## 单终端批处理与 WatchAct

新批处理入口保持原项目的脚本名称与主要参数，串行启动独立 PiperX 仿真并清理，保存逐轮回执、实际调用与用量，重启不重放已结束或未知动作。先用 `scripts/eval_imitator_game.py --smoke` 验证两轮流程；原五个 IG 环境缺单臂匹配资产时会在模型调用前阻止执行。

WatchAct 可作为下一步单臂视频任务来源：导入公共语言和视频，保留任务语义，不把私有 oracle/目标交给模型。当前只完成清单接线，**尚未完成 WatchAct 场景与 BDDL 到 PiperX 的物理适配**。详见 [批处理部署与操作说明](docs/batch.zh.md)。

单题真实物理验证现已完成：WatchAct `imitation_data_315` 的完整视频、单臂场景、私有评分与 native RPent 已接通，最终模型回合正常结束但没有抓起汤罐，独立评分失败。格式兼容、夹爪完成条件和篮底几何修订的开发回合全部保留，不能冒称任务成功。配置、代码边界、实际调用/用量与重现入口见 [WatchAct 单题接入和结果](docs/watchact-imitation63.zh.md)。
