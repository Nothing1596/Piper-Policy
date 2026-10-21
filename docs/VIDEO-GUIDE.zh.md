# Video 0.3.1 独立使用指南

[English](VIDEO-GUIDE.en.md) · [下载](https://github.com/Nothing1596/Piper-Policy/releases/tag/video-v0.3.1) · [版本说明](VIDEO-RELEASE-0.3.1.md) · [功能与命令](VIDEO-CLI-REFERENCE.md)

本版发布 Python 3.12+ wheel，不是新的 Windows 全离线包。旧 Video 0.3.0 ZIP 及 installer-v1 的说明见 [历史指南](versions/video-0.3.0.zh.md)，不要将旧安装脚本与本版 wheel 混用。Robot 0.7.2 交互包已附带此视频组件并单独安装，无需重复安装下面的独立环境。

## 安装

下载 piper_lab-0.3.1-py3-none-any.whl 与 SHA256SUMS.txt 并核对。Windows PowerShell，在新的工作目录：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install './piper_lab-0.3.1-py3-none-any.whl[video,model,mcp,onnx]'
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\piper-video.exe --help
```

macOS/Linux：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install './piper_lab-0.3.1-py3-none-any.whl[video,model,mcp,onnx]'
.venv/bin/python -m pip check
.venv/bin/piper-video --help
```

首次安装下载平台依赖。没有安装机器人、模型服务或大模型权重，也没有自动注册 PATH。后面的 piper-video 命令在未激活环境时替换为以上绝对/相对启动器。wheel 不包含示例视频、ONNX 权重和自己的云端配置。

## 先验证视频，再接模型

使用自己存在的视频，且每次使用新输出目录：

```sh
piper-video candidates --video /absolute/demo.mp4 --output work/candidates-01
piper-video watch --address /absolute/demo.mp4 --mode direct --at-s 1.5 --output work/watch-point-01
piper-video watch --address /absolute/demo.mp4 --mode pipeline --start-s 0 --end-s 3 --max-frames 6 --output work/watch-range-01
```

candidates/direct 不调用视觉 API。pipeline 返回本地 CV 评价和可选检测/视觉评价；图像、真实时间、引用和实际所见范围分别保留。address 支持本地路径、file URL 或可直接下载的视频 URL，不保证支持网页播放器。

要接视觉模型，建立自己的 provider JSON（provider/model/base_url、api_key_env 或 api_key_file 引用、图片与输出预算）。源码 profiles 模板可作参考，密钥不要写入 JSON 或发布包。先执行 `models show`，再 `models probe --model-config FILE --image FILE --output NEW_DIR` 验证真正图片请求；show 不是推理验收。

`compile --video FILE --task TEXT --model-config FILE --output NEW_DIR` 生成阶段/证据/unknown 的 demo.json，`inspect --bundle DIR` 读取。为 watch 添加 `--model-config FILE` 才启用快速视觉 API，添加 `--detector-onnx FILE` 才使用实际权重；模型评价可能收费，检测器不需要语言模型服务。

## MCP 与模型工具

`piper-video mcp --workspace DIR --model-config FILE` 启动 stdio 服务，目录必须存在，外部代理通过准确解释器和参数接入。工具包括 capabilities、候选帧、分页、取图、编译、检查和标注评估，以 tools/list 为准。watch 是本版 CLI 子命令，不假定独立视频 MCP 有同名工具；机器人前端另外提供 demo_watch/demo_reference。

省略模型配置时仍可用本地证据工具。代理自行看图不等于把其登录态转成管线 API key。完整使用场景见 [本版功能](versions/video-0.3.1.zh.md)。

## 结果与边界

保存 manifest、图片、demo.json 和 model-calls。图片来源、时间和评价对应，选帧或检测标签不证明动作发生；单帧不证明接触或完整时间过程。缓存 CV 分析不等于缓存当前 API 的语义判断。输出已有、引用未知、图像不一致、截断或 API 故障应保留错误，不静默宣称成功。

这是示范解析与证据管线，不训练权重、不提供机器人动作或自动三维建模。历史视频要结合当前机器人观察才可用于决策。当前发布检查见 [验证记录](implementation/release-0.7.2-validation.zh.md)；它不等于模型任务准确率或真机成功率。
