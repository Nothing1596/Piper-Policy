# 0.7.2.dev4 — MuJoCo 模型观察闭环（开发包）

修复模型 CLI 在仿真中无法 observe、旧仿真观察文件无法送入 scene_query 的接线缺口。
MuJoCo 与 real/local 共用 camera_observe、任务问题绑定、几何计算和 plan 写回流程。
仿真返回按需新快照，包含模拟内外参及同次锁定采样的机器人状态，不宣称连续视频历史。

JSONL 保留原 result，并在有图片时附加 content；MCP 返回标准 ImageContent。
旧独立 observe 命令复用规范化观察格式，并支持从指定 profile 的 runtime.json 自动发现和核验执行器地址。
模拟标定单独保存，不覆盖操作员真实相机配置；观察和几何结果明确标记模拟来源。

不增加运动权限、服务或控制窗口，不连接真实硬件，不包含视觉模型凭据或 YOLO 权重。
本包为未公开发布的开发构建；验证结果见 implementation/simulation-observe-adapter-validation.md。

Windows 冷启动取图的 SciPy 原生模块加载阻塞已修复；重型计算依赖留在几何 worker 使用路径。
验证：macOS 667 passed / 8 skipped；Windows 相关 8 项通过，并完成实际 CLI 看图驱动的一次仿真抬升。
