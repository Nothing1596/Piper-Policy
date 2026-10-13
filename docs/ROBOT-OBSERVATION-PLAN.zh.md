# 示范视频、观察计划与几何证据

本版把「看示范—提出会改变动作的问题—补看/测量—执行」接进同一个
`piper-robot`。这是观察工作流，不是新的动作审批门槛。机械臂动作仍走既有执行器。

## 一次任务怎样操作

进入 `piper-robot` 选择模式和主机，再 `/connect`。实时相机目前只支持
`real/local`；MuJoCo 仿真通过同一 `/observe` / `camera_observe` 返回一张新 RGB-D 快照。
`max_frames` 是预算上限；仿真尚无连续历史，返回 `capture_mode=on_demand_snapshot`。
模拟内外参与采样时机器人状态由执行器提供，不使用前端与远端主机之间的时钟差。
普通 `sim` 后端没有图像，会明确报错；远端不会偷偷改用本机物理相机。

```text
/demo prepare /absolute/demo.mp4
/plan new "按示范方式抓住杯子把柄"
/manual demo_reference(overview=True)
```

先看首尾与阶段概览，再按需要细看：

```text
/manual demo_reference(stage_id="grasp", offset=0, limit=3)
/demo watch {"address":"/absolute/demo.mp4","mode":"pipeline","start_s":1.0,"end_s":2.0,"max_frames":6}
```

`stage_id` 必须来自已编译示范返回的阶段，纯 CV 候选文件没有语义阶段。
`pipeline` 返回选中帧的质量、检测、变化和已配置的快速视觉评价；评价只代表模型实际看过的帧。
执行模型仍能读取选中的原始图片。相同源文件/检测器配置的全视频 CV 分析会缓存；
当前区间的取帧和视觉评价仍重新执行，不把旧评价当作新结论。

模型先调用 `task_plan(operation="guide")` 读取随包 skill，再记录少量关键问题。
内置对话模型自动得到同一份 skill、当前计划和示范概览。自然语言首条任务会建立计划；
换目标请显式 `/plan new`，不要把多个任务混在旧计划里。

下面 `TASK_ID`、`VERSION` 换成 `/plan` 返回值，像素只是语法示例，必须从当前图像选择：

```text
/manual task_plan(operation="update", task_id="TASK_ID", expected_version=VERSION, changes={"questions":[{"id":"surface","question":"选中区域的可见表面在哪里？","affects_action":True,"source":"geometry","query":{"operation":"surface_point","pixel":[320,240]}}]})
/observe {"serial":"WRIST_SERIAL","question_id":"surface","max_frames":1}
/plan
```

`/observe` 会采集图像、执行绑定的查询并写回数值证据。双相机时要明确选择产生这些像素的相机。
图像语义判断写入 `task_plan.update` 的 `changes.interpretation`，例如“把柄疑似位于杯子右侧”；
它和后端测量值分开保存。模型不能直接伪造问题的 `result` 或 `state`。

也可以反复计算同一份证据，不必重新取图：

```text
/manual scene_query(operation="surface_point", observation_path="FRAME_OBSERVATION_PATH", pixel=[320,240])
/manual scene_query(operation="accumulate_region", target_id="cup", batch_id="scan1", observations=[{"observation_path":"VIEW_A","roi":[100,100,220,240]},{"observation_path":"VIEW_B","roi":[110,90,230,240]}])
/manual scene_query(operation="fit_plane", target_id="cup", batch_id="scan1", reference_normal=[0,0,1])
/manual scene_query(operation="point_distance", point_id="RETURNED_POINT_ID_A", other_point_id="RETURNED_POINT_ID_B")
/manual scene_query(operation="point_plane_distance", point_id="RETURNED_POINT_ID", plane_id="RETURNED_PLANE_ID")
```

`observation_path` 选每帧返回的路径，不选多帧 bundle。ROI 是 `[x1,y1,x2,y2]`；也支持像素多边形。
复用一个 batch_id 只允许相同输入；改变视角集合、ROI 或场景时使用新 batch_id。
测距输入必须是后端返回的结果 ID，不接受模型臆测的三维坐标。直接调用 scene_query 绑定 question_id 时必须同时传 expected_version；旧版本请求会被拒绝。显式访问旧任务前需先 resume，不能把后台结果写到另一个当前任务。

视角不足时，模型通过既有运动工具请求一个更有信息量的观察位置，然后再次 `/observe`。
这些观察动作使用原有审批和限位；`observe`、`scene_query` 自己不移动机器人。
中断后 `/plan resume TASK_ID` 恢复目标，把实时/几何答案标为过期，重新观察。

## 相机与标定配置

```text
/config cameras
/config cameras [{"serial":"FIXED_SERIAL","role":"fixed","calibration_file":"/absolute/fixed.json"},{"serial":"WRIST_SERIAL","role":"wrist","calibration_file":"/absolute/wrist.json"}]
/video start
/observe {"max_frames":2}
/video status
/video stop
```

交互输入 `/config cameras` 会逐项询问相机数量、序列号、角色和标定文件；直接 JSON 写法适用于脚本。只用一台相机就只填一项。旧 `camera.json` 的 `serial`、`python` 等设置保留兼容。
最多两台；共享总计 128 MiB 帧缓存预算。图像输出保留全图和可选 ROI，不用裁剪替代上下文。
相机重新配置会关闭旧采集进程，并使实时证据过期。

标定 JSON 的固定格式如下。矩阵必须来自实际标定；示例用字母表示，不能照填成单位矩阵。

```json
{
  "camera_serial": "FIXED_SERIAL",
  "role": "fixed",
  "units": "m",
  "base_from_camera": "实际标定得到的 4×4 数值矩阵"
}
```

腕部相机将 `role` 改为 `wrist`，矩阵字段改为 `flange_from_camera`。
坐标变换为 `base_from_flange(实际关节角) × flange_from_camera`。
当前运动学默认末端是法兰，不能当作指尖。固定相机直接使用 `base_from_camera`。
配置文件内容哈希作为标定身份；修改后旧采集数据不能按新标定直接重新解释。
读取当前计划时会重验已保存测量的来源；即使直接替换原路径的标定文件，受影响答案也会标为过期。

基座坐标计算要求采集时有新鲜、静止且已对齐的机器人反馈。HTTP 接收时间对齐不是硬件同步。
尚无标定时 surface_point 可以返回相机光学坐标，但不能据此直接下发基座运动。
当前仅支持零畸变的对齐深度数据；不支持的畸变、无效深度、过期/移动中的姿态会明确拒绝或返回 unavailable。

## 后端计算与证据边界

- 计算在 CLI 所在主机的独立 CPU 子进程完成，超时/取消会回收；不打开 CAN，不连接 GPU 服务。
- 像素通过深度和内参反投影，经已知外参变换，局部点云默认按 2 mm 体素合并。2 mm 是采样尺度，不是测量精度。
- 平面使用确定性 RANSAC 和 SVD；返回内点比例、残差和法向约定。残差不包含标定误差和深度系统误差。
- 距离是保存的实测点或平面间距离，不能据此声称已插入、可达或无碰撞。
- 遮挡区域没有测量，不会被补成空闲空间。不做自动把柄分割、孔洞重建、SLAM、ICP 或完整仿真器。
- `plan.json` 保存在各 profile 的 `tasks/TASK_ID/`。保存目标、问题、版本和证据引用，不保存 API 密钥、原始图像或模型隐藏推理。图像与几何文件单独保存。
- 过期版本的计算结果不会覆盖新计划；改变问题会清除原答案。软件结果与真实双相机、标定精度、抓取成功率验收分别记录。

## MuJoCo 模型闭环（0.7.2.dev4）

仿真也使用“先 camera_observe 看图 → task_plan 记录问题 → camera_observe(question_id=...) → 执行 → 再观察”。
JSONL 回复保留 result 并附可选 content（文字和 image_url），无需客户端读取远程磁盘上的图片路径；
MCP 使用标准 ImageContent。仿真工具表提供 camera_observe，不提供连续 camera_video。
模拟标定保存到 simulation-camera.json / simulation-calibration.json，不覆盖操作员 camera.json。
几何来源包含 simulated=true；模拟数据不证明实际标定、碰撞安全或抓取成功。
