# Robot 0.7.2.dev1：连续观察与模型 JSONL

[返回版本导航](README.zh.md) · [原功能说明](../ROBOT-LIVE-VIDEO.zh.md) · [保存的验证记录](../implementation/live-video-validation.md)

## 状态与解决的问题

dev1 是未公开发布的开发阶段，没有对应 GitHub release/tag。它在 0.7.1 单终端执行器上增加 camera-only 连续采集、变化选帧和外部模型 JSONL；避免每次看图都重新打开相机。公开 0.7.1 ZIP 不包含这些功能。

需要安装时按 [源码版说明](source-09256a5.zh.md) 建立独立环境；当前源码包含后续改进，并不等于能够精确还原历史 dev1 构建。已有历史开发包的读者使用包内启动器，先核对版本与构建记录，不假定 pip 仓库有可安装的同名版本。

## 人工观察一段实时画面

此阶段连续视频支持 real/local 相机。先保证 SDK 和相机解释器可用，进入前端选择真实模式、本机；只启动相机不等于连接机器人 CAN：

```text
/video start
/observe
/observe {"window_s":3,"max_frames":3,"roi":[200,120,900,700]}
/video status
/pixel 320 240
/video stop
/quit
```

ROI 和像素是语法例子，应从实际 RGB 尺寸与画面选择。指定设备用 `/video start SERIAL` 或 `/observe SERIAL`；切换、拔插或异常后显式 stop/start，不能默认旧流已恢复。`/pixel` 使用最后选中图像，未标定时仅是光学坐标，不是机械臂目标。

默认观察最近 2 秒、最多 3 帧：起点、明显变化帧和最新帧；max_frames=1 只取最新帧，上限为 6。相机移动也会制造变化，因此返回变化帧不等于检测到抓取或插入。

## 返回文件与时间

观察 bundle 保存顺序帧、RGB/深度/ROI 路径、拼图与覆盖范围。完整图仍保存，ROI 不替代场景上下文。默认 3 Hz 缓存、20 秒、总内存上限 128 MiB；底层 RGB-D 流 15 fps，只将选中证据落盘，不输出完整 MP4。

缓存不足返回 history_truncated；最新帧过期会拒绝。机器人状态按同机 HTTP 接收时间近邻对齐，差距超过 0.5 秒或缺失则为 null。这不是曝光硬同步，也不代表验证过外参。动作前后的 visual_evidence 保留自己的采样时刻；审批前帧不能冒充精确动作开始帧。

## 外部模型怎样调用

启动 `piper-robot --mode real --target local --agent`，逐行发送 JSON。只读例子：

```json
{"id":"tools-1","command":"tools"}
{"id":"camera-1","command":"video","arguments":{"operation":"start"}}
{"id":"scene-1","command":"observe","arguments":{"max_frames":1}}
{"id":"state-1","command":"status"}
{"id":"end-1","command":"quit"}
```

stdout 每行一个响应，stderr 放提示；响应按 id 对账，不能假定并发请求按发送顺序返回。动作仍走 `command=call` 和 tools 中的原 schema，并显式提供 request_id；行 id 不能替代动作编号。模型无操作员配置/审批权限，EOF/quit 回收当前会话和相机进程。

内置 `/model` 入口也可把本前端取得的实际图片发送给支持视觉的兼容服务；该阶段每次最多 3 张，保留最近两组实时图像。模型收到一个图片文件名不代表真正看到了像素，需核对交付内容。

## 配置、故障与边界

camera extra 需要平台可用的 RealSense 依赖；可在 camera.json 指定相机解释器、serial、sample_hz 和 buffer_s。不要同时启动竞争同一相机的上位机；相机进程不导入机械臂 SDK、不另占 CAN。退出优先正常释放，超时才回收前端拥有的子进程。

该历史阶段没有 dev3 的 plan/几何或 dev4 的 MuJoCo 同接口观察。sim 测试替身没有真实图像；远端目标不能静默改用操作者本机相机。软件回归、真实相机可取帧和物理任务成功分别验收；本次仅核对文档/代码，没有采集或发送硬件动作。
