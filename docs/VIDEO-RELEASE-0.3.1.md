# Piper Video 0.3.1 — 可重复观看与可引用评价

独立视频组件更新，对应 `piper_lab-0.3.1-py3-none-any.whl`。这是 wheel 和源码发布，不是新的 Windows 全离线套件；原 Video 0.3.0 离线包继续保留。Robot 0.7.2 交互包已附带此 wheel，并独立安装视频环境。

## 重要更新

- `piper-video watch` 支持本地/file/可直接下载视频地址、时间点/区间、direct 原图和 pipeline 评价。返回实际 PTS、帧 ID、seek 偏差及评价所见范围。
- 复用同内容、同检测配置的全视频 CV 分析；所选图片和评价分别有明确引用，不把检测标签或快速视觉假设当作真值。
- 示范概览、阶段引用、图像与元数据传递等后续修订进入正式组件版本；机器人侧通过 demo_watch/demo_reference 接入，历史视频不成为当前观察。
- 额外提供轻量 `onnx` 依赖组，便于已有检测器用户安装 ONNX Runtime；模型权重、API key 和现场素材不随 wheel 分发。

## 安装与最小验证

Python 3.12+，在新的虚拟环境中联网安装：

```sh
python -m pip install './piper_lab-0.3.1-py3-none-any.whl[video,model,mcp,onnx]'
python -m pip check
piper-video --help
piper-video watch --address /absolute/demo.mp4 --mode direct --at-s 1.5 --output work/watch-01
```

最后一行使用自己实际存在的视频和新 output 目录。pipeline 不配置视觉 API 时仍可返回本地 CV 及明确的未配置说明；完整视觉评价需自己的兼容模型配置。旧环境升级前备份，升级后不要重新运行锁定 0.3.0 的旧安装器。

详见 [中文指南](VIDEO-GUIDE.zh.md)、[English](VIDEO-GUIDE.en.md)、[本版功能说明](versions/video-0.3.1.zh.md)。独立视频 CLI/MCP 不安装机器人、不打开 CAN、不训练权重。

发布验证与 Robot 0.7.2 使用同一源码提交并单列组件检查，见 [验证记录](implementation/release-0.7.2-validation.zh.md)。本轮不声称新视觉供应商实测、示范理解准确率或机器人任务性能收益。
