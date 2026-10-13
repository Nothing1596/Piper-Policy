# 0.7.2.dev3 — 观察计划与可复用几何计算（开发包）

增加 `/plan`、`task_plan`、`scene_query` 和 `/config cameras`，内置模型、JSONL、前端 MCP 共用控制器。
新增随包 `observe-and-plan` skill。实时观察可以绑定问题并写回测量证据；模型不能写入测量结果字段。

示范视频支持首尾/阶段概览和按阶段分页，快速视觉模型的帧子集不再过滤执行模型的原始图像。
prepare/compile 传递配置的检测器；watch 按视频内容和检测器配置缓存全视频 CV 分析。

支持最多两个本机相机的固定/腕部配置，CPU 计算表面点、局部已知位姿点云、平面和距离。
没有自动标定、接触/插入判定、碰撞规划或真实任务成功率声明。详见 `ROBOT-OBSERVATION-PLAN.zh.md`。

本包从保留现有工作区改动的源码构建，用于本地验证；不是公开 release，不自动部署或连接真机。
验证记录见 `implementation/observation-plan-validation.md`。
