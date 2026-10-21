# Video 0.3.1 installation and workflow

[中文](VIDEO-GUIDE.zh.md) · [Download](https://github.com/Nothing1596/Piper-Policy/releases/tag/video-v0.3.1) · [Release notes](VIDEO-RELEASE-0.3.1.md)

This release provides a Python 3.12+ wheel and source, not a new offline Windows bundle. The older 0.3.0 kit keeps its [historical guide](versions/video-0.3.0.zh.md). Robot 0.7.2 already includes this wheel and installs a separate video environment.

## Install independently

Verify the downloaded wheel against SHA256SUMS.txt. Create a fresh Python 3.12 virtual environment, then use its interpreter:

```sh
python -m pip install './piper_lab-0.3.1-py3-none-any.whl[video,model,mcp,onnx]'
python -m pip check
piper-video --help
```

Dependencies download during installation. No robot distribution, model weights, detector weights, demonstration files or credentials are bundled. Do not copy an old .venv or rerun an installer locked to 0.3.0 after upgrading. Use the environment's Scripts/piper-video.exe on Windows or bin/piper-video on macOS/Linux if it is not activated.

## View and evaluate a demonstration

```sh
piper-video candidates --video /absolute/demo.mp4 --output work/candidates-01
piper-video watch --address /absolute/demo.mp4 --mode direct --at-s 1.5 --output work/point-01
piper-video watch --address /absolute/demo.mp4 --mode pipeline --start-s 0 --end-s 3 --max-frames 6 --output work/range-01
```

Use an existing video and fresh output paths. direct returns actual decoded frames and timestamps without a model API. pipeline adds CV evaluation, optional ONNX detections and optional vision-model evaluation. Provide --detector-onnx FILE or --model-config FILE explicitly. Evaluation references and the frames actually reviewed remain auditable; a historical image is never a current robot capture.

Configure your own provider/model/base URL and credential reference, then use models probe with a real image before compile. models show does not test inference. compile produces stages, evidence and unknowns; inspect/find/evaluate retain their prior functions. Cloud requests may incur cost.

## MCP and boundaries

piper-video mcp --workspace DIR --model-config FILE exposes the configured evidence tools through stdio. Use tools/list for exact schemas. The watch CLI subcommand does not imply a same-named standalone video MCP tool; Robot's frontend separately offers demo_watch/demo_reference.

This is an evidence pipeline, not a weight trainer, robot controller or automatic 3D reconstruction system. Missing or inconsistent frames, unsupported references, truncation and API errors remain explicit failures. See the [current validation record](implementation/release-0.7.2-validation.zh.md); package checks do not establish semantic accuracy or hardware task success.
