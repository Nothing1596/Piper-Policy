# 源码开发 / Source development

独立离线包从 Releases 下载，分别按 VIDEO-GUIDE.zh.md / ROBOT-GUIDE.zh.md 操作。NEW-MACHINE-GUIDE.md 对应旧联合版。以下是联网安装源码的开发步骤，不下载视觉模型权重，也不启动机器人。

For source development, install Python 3.12 and run from the repository root. This path downloads dependencies; use the release ZIP for offline Windows deployment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".\piperx-cli[simulation,hardware,test]" -e ".\piper-lab[video,detector,model,mcp,simulation,test]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest piper-lab\tests -q
.\.venv\Scripts\python.exe -m pytest piperx-cli\tests -q
```

缺少外部 SDK 时，SDK 合约测试会明确跳过；测试不连接实体 CAN。SDK contract tests skip if a compatible external SDK is absent. Do not describe these tests as hardware validation.

Use independent entry points `piper-video` / `piper-robot`; legacy `piper-lab` / `piperx` remain available. To install only video, install `piper-lab[video,detector,model,mcp]` in its own environment; only robot needs `piperx-cli[simulation,hardware]`. The commands below demonstrate the retained combined dispatcher:

```powershell
.\.venv\Scripts\python.exe piper-lab\packaging\joint-cli\piper.py --help
.\.venv\Scripts\python.exe -m piperlab.cli sim start --root .runtime-sim --port 8808 --seed 200
```

Open another terminal in the same directory:

```powershell
.\.venv\Scripts\python.exe -m piperx_middleware.cli --root .runtime-sim --url http://127.0.0.1:8808 connect
.\.venv\Scripts\python.exe -m piperx_middleware.cli --root .runtime-sim --url http://127.0.0.1:8808 status
```

`piper-lab/config/hardware.yaml` 为未验收仿真模板，GPU UUID 已置空。训练前明确选择部署机器的设备。The hardware configuration is an uncommissioned simulation template; select the target GPU explicitly before training.

Build wheels after installing `build`:

```powershell
.\.venv\Scripts\python.exe -m pip install build
.\.venv\Scripts\python.exe -m build --wheel piper-lab
.\.venv\Scripts\python.exe -m build --wheel piperx-cli
```

The full release also includes a fixed Windows dependency wheelhouse, detector, synthetic examples and notices. Project wheels alone are not the complete offline bundle. Never commit credentials or private datasets.
