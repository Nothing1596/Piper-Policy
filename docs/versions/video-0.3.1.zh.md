# Video 0.3.1：按需观看、评价与独立使用

[返回版本导航](README.zh.md) · [下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/video-v0.3.1) · [更新摘要](../VIDEO-RELEASE-0.3.1.md)

## 安装形态

发布 `piper_lab-0.3.1-py3-none-any.whl` 和校验文件，不包含 Windows 全离线依赖、检测器权重、示范素材、模型服务或凭据。Python 3.12+；新环境联网安装 wheel 的 video/model/mcp/onnx extra。Robot 0.7.2 ZIP 已将该 wheel 放入独立视频环境，不需要再重复装一个视频目录。

旧 Video 0.3.0 离线包仍有自己的历史指南；升级后不要重新运行旧固定版本安装器。新建环境安装步骤见 [视频指南](../VIDEO-GUIDE.zh.md)。

## 无模型先看原始证据

```sh
piper-video candidates --video /absolute/demo.mp4 --output work/candidates-01
piper-video watch --address /absolute/demo.mp4 --mode direct --at-s 1.5 --output work/watch-point-01
piper-video watch --address /absolute/demo.mp4 --mode direct --start-s 1 --end-s 3 --max-frames 6 --output work/watch-range-01
```

address 必填，支持文件、file URL 或可直接下载的视频 URL。返回真实解码帧、源 PTS 和引用，时间点使用首张不早于目标时刻的真实帧并报告偏差；输出目录不得覆盖。candidates 是选帧，direct 是看原图，都不自动调用视觉 API。

## 经过管线时返回什么

```sh
piper-video watch --address /absolute/demo.mp4 --mode pipeline --start-s 0 --end-s 3 --max-frames 6 --analysis-cache work/cv-cache --output work/evaluation-01
```

返回选中图像、质量/变化/可选检测和 evaluation；同视频/检测配置的全局 CV 分析可缓存。配置 `--detector-onnx FILE` 才启用自己的 ONNX；配置 `--model-config FILE` 才调用视觉评价。已配置 API 失败不能假装只是未配置；评价仅覆盖实际所见帧，不推断完整隐藏过程或当前机器人成功。

完整语义阶段解析仍用 compile；随后 inspect/find/evaluate 管理证据、检索和与已有标注比较。模型需要真实图像支持、正确端点和结构输出；先 probe，再编译，不把 models show 当推理成功。

## 模型接入与产物

独立 MCP 入口仍为 `piper-video mcp --workspace DIR --model-config FILE`，以工具目录的实际 schema 为准；watch 的 CLI 命令不自动意味着有同名 MCP 工具。机器人前端另提供 demo_watch/demo_reference，保留录制/实时标签和证据引用。

模型可读选中图片、manifest、demo.json 和 model-calls。检测标签与快速评价都是假设，原图仍需可核验；图片/元数据不一致、未知引用、上下文超限或截断都保留失败，不静默生成成功计划。pipeline 不训练权重、不执行动作、不建立自动三维仿真器。

本轮组件验证见 [发布验证](../implementation/release-0.7.2-validation.zh.md)。视频读取、评价 API、完整模型任务和硬件成功是不同结论；没有用本轮打包验证冒称任务收益。
