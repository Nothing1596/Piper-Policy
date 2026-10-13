# 实时视频与模型 CLI（0.7.2.dev1）

开发版新增持续相机采集与 JSONL 模型入口，机器人继续使用原执行器。软件、相机和实机动作的验证范围分别记录。

本轮验证见 [验证记录](implementation/live-video-validation.md)。

## 人工操作

```text
piper-robot
# 选择 real、local
/video start
/observe
/observe {"window_s":3,"max_frames":3,"roi":[200,120,900,700]}
/video status
/video stop
/quit
```

多相机用 `/video start SERIAL`。`/observe SERIAL` 保留兼容，首次开始采集，之后复用。切换序列号、异常或拔插后显式 stop/start，不静默重连。`/pixel U V` 查询最后一张选中帧，仍是未标定的光学相机坐标。

默认取最近 2 秒，最多 3 帧：起点、变化明显的中间帧、最新帧。静止画面只返回起终帧；`max_frames=1` 只取最新帧，上限为 6。ROI 是原始 RGB 的 `[左,上,右,下]` 像素区域，影响变化评分，并保存局部图。完整 RGB 和深度仍保留。腕部相机运动也能造成画面变化，选帧不自动判断接触或抓取成功。

视频开始后，动作终态结果附 `visual_evidence`：请求前单帧和结束时选出的帧。`before_request` 可能早于操作员审批，不声称是精确运动开始帧。长动作超过缓存时报告覆盖截断；请求前帧已单独保存。重复请求返回历史结果时，不把新画面归到旧动作上。

## 模型操作

```text
piper-robot --mode real --target local --agent
```

此入口读 JSONL，stdout 每行一个响应，提示发到 stderr；必须显式指定模式。仿真测试用 `--mode simulation --simulation-backend sim`。它供外部模型进程使用，无须配置 `/model`。

按行发送：

```json
{"id":"tools-1","command":"tools"}
{"id":"camera-1","command":"video","arguments":{"operation":"start"}}
{"id":"scene-1","command":"observe","arguments":{"window_s":2,"max_frames":3}}
{"id":"connect-1","command":"call","tool":"robot_connect","arguments":{}}
{"id":"state-1","command":"status"}
```

动作参数遵循 `tools` 返回的原工具 schema。以下角度仅用于隔离仿真示例：

```json
{"id":"step-1","command":"call","tool":"robot_move_joints","arguments":{"joints_deg":[1,0,0,0,0,0],"request_id":"example-step-01","speed_percent":3}}
{"id":"job-1","command":"status","job_id":"实际返回的 job_id"}
{"id":"stop-1","command":"call","tool":"robot_stop","arguments":{}}
{"id":"end-1","command":"quit"}
```

`id` 是输入行编号，`request_id` 是动作幂等编号，动作必须提供后者。响应可能按完成顺序返回，按 id 匹配。未知结果须查询原 job/request_id，不能换编号重发。运行中的动作允许另发 status/stop。

模型入口不提供配置或审批工具。待审批立即返回 `awaiting_approval`，由操作员通过原审批渠道处理；auto 仍受执行约束限制。quit 或输入 EOF 后释放会话、等待执行器收尾、结束相机进程。新前端不使用外部时间窗口。

## 证据与依赖

`piper.observation.v1` 包含时间顺序的 frames、RGB/深度/ROI 路径、storyboard 拼图、bundle JSON 和覆盖范围。超过缓存返回 `history_truncated=true`；最新帧超过 2 秒拒绝返回。

每帧附最近的机器人 HTTP 状态和接收时间差，差距超过 0.5 秒或缺失时状态为 null。这是同主机接收时间的尽力对齐，不是硬件曝光同步；设备时钟单独保留，不与机械臂时钟相减。未验证标定时 `calibration_applied=false`，不生成基座抓取坐标或任务成功结论。

外部模型需读取实际图片。内置 `/model` 循环新增 `camera_video`、`camera_observe`，可把本前端实际生成的图片送给配置的支持视觉的 Chat Completions endpoint；每次最多 3 张，仅保留最近两组实时图像，历史文本和文件引用保留。ROI 图优先。文本模型收到路径并不等于看到图像。

安装相机 extra：`pip install 'piperx-middleware[camera]'`，需要平台可用的 RealSense 库。沿用 profile/camera.json 可指定已有解释器：

```json
{"python":"已有相机解释器绝对路径","serial":"相机序列号","sample_hz":3,"buffer_s":20}
```

RGB-D 流为 15 fps，默认 3 Hz 缓存、20 秒上限，并设 128 MiB 内存上限。sample_hz 上限 15，buffer_s 上限 120，内存不随配置扩大。只把选中帧保存到磁盘，不产生完整 MP4，已保存证据不自动删除。

连续视频当前支持 real/local D435，不替换成操作者电脑的相机。0.7.2.dev4 的 MuJoCo 支持同名 observe 的按需新快照，尚无连续视频历史；返回值明确区分。相机子进程不导入机械臂 SDK 或占用 CAN。EOF/退出优先请求 SDK 正常释放，超时后回收本前端创建的相机进程。
