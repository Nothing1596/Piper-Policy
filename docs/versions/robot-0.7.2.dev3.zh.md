# Robot 0.7.2.dev3：plan、观察 skill 与几何计算

[返回版本导航](README.zh.md) · [完整观察计划指南](../ROBOT-OBSERVATION-PLAN.zh.md) · [保存的软件验收](../implementation/observation-plan-validation.md)

## 状态与作用

未公开发布的开发阶段，无对应 GitHub release/tag。首次把 `/plan`、`task_plan`、`scene_query`、双相机配置和随包 observe-and-plan skill 接进同一前端。功能目标是让模型提出会改变动作的问题，再用图片及可引用计算结果回答，避免只凭单张图猜深度。

安装见 [源码版](source-09256a5.zh.md)；当前源码还包含 dev4 及后续反馈修订。正式 0.7.1 包没有这些命令。历史 dev3 的实时相机主要为 real/local；MuJoCo 同接口观察由 dev4 接通。

## 创建任务与获得正确版本

进入对应模式和主机后：

```text
/connect
/demo prepare /absolute/demo.mp4
/plan new "参照示范，抓住杯子把柄后提起"
/manual task_plan(operation="guide")
/plan
```

guide 返回共享 skill；内置模型也注入同源说明。任务文件保存目标、问题、解释、版本及证据引用。新目标用 plan new，不把不同任务塞到同一计划。task_plan 的 create/get/update/resume 使用真实 ID 和版本；不能让模型猜 UUID。

下面 TASK_ID/VERSION 用 `/plan` 返回值替换，pixel 用当前图像真实选择值：

```text
/manual task_plan(operation="update", task_id="TASK_ID", expected_version=VERSION, changes={"questions":[{"id":"surface","question":"选中可见表面的位置在哪里？","affects_action":True,"source":"geometry","query":{"operation":"surface_point","pixel":[320,240]}}]})
/observe {"question_id":"surface","max_frames":1}
/plan
```

实时观察会对问题执行对应查询并写数值证据。视觉语义解释写 interpretation，不能伪造后端 state/result；问题未解决时保留未知。动作或场景变化会使旧测量过期，应重新观察。在线 plan 是可修订指南，不是必须满足所有问题才能运动的新门槛；后续 main 对软计划和反馈有更完整修订。

## 复用一次观察，减少重复看图

```text
/manual scene_query(operation="surface_point", observation_path="ACTUAL_OBSERVATION_PATH", pixel=[320,240])
/manual scene_query(operation="point_distance", point_id="ACTUAL_POINT_A", other_point_id="ACTUAL_POINT_B")
```

observation_path 来自单帧实际回执，不是整个 bundle；距离输入使用后端返回的点 ID。其他 operation 包括 accumulate_region、fit_plane、point_plane_distance。点云积累需要每个视角有已知外参和静止对齐反馈；不是自动 SLAM 或未知位姿重建。

同一 batch_id 只允许完全相同的输入集合；改图、ROI 或标定就新建 batch。ROI 可以包含多表面，计算不会自动识别目标。直接 scene_query 绑定计划问题时必须带当前 expected_version；过期结果不覆盖新计划。恢复旧任务用 `/plan resume TASK_ID`，实时答案会过期，不能把旧图当当前姿态。

## 一台或两台相机

```text
/config cameras
/video start
/observe {"max_frames":2}
/video status
/video stop
```

向导配置最多两台，注明 serial、fixed/wrist 角色和真实标定文件。固定相机使用 base_from_camera，腕部使用 flange_from_camera，再由实时关节姿态求基座变换。不要拿示例单位矩阵冒充标定，也不要把默认法兰 TCP 当指尖。只有一台就配一项；两台共享总缓存预算，不是每台各自无限增长。

没有外参时 surface_point 可以返回相机光学坐标，但不能直接下发基座动作。非法畸变、无效深度、移动中或过期姿态会返回拒绝/不可用；标定文件变化也使旧结果失效。看不到的区域不会被补成无障碍空间。

## 计算原理、模型作用和输出

CPU worker 将深度像素通过内参反投影，利用已知外参变换；局部点云可按 2 mm 体素合并，平面用确定性 RANSAC/SVD，距离用保存的实测点/平面。2 mm 是采样尺度，不是系统精度；残差不包含全部标定与深度误差。

这些结果让 VLM 不必心算三维距离，也能追溯哪个视角支持判断；目标选区、语义解释、是否需要新视角和动作仍由模型决定。没有自动把柄分割、插入判决、完整对象网格、SLAM、碰撞规划或抓取成功证明。worker 不连接 CAN/GPU，超时会回收。

## 故障和验证

版本冲突先 get 最新 plan，重新判断后更新；不要盲重发旧 update。跨相机像素必须指定来源，跨未知姿态不能直接测距。单项 primitive succeeded 不证明语义步骤完成；解释、测量和物理完成是不同证据。

保存的 macOS 软件验证为机器人 663 通过/8 跳过、视频相关 95 通过，并做隔离安装及 JSONL/MCP 接线检查；不是当天完整 Windows/Linux 或实机验收，本次未重跑。后续 Windows 和 MuJoCo 修复单列在 dev4，不反写成 dev3 当时已验收。
