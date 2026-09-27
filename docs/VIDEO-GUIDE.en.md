# Independent video pipeline

[中文](VIDEO-GUIDE.zh.md)

Version 0.3.0, CLI `piper-video`. The Python distribution remains `piper-lab` for compatibility; this installation contains no robot distribution or MuJoCo. Legacy upper-level modules remain in the wheel, while this guide uses the video-only entry point.

## Install

Install Python 3.12 x64 on Windows x64. Extract into a fresh directory and run PowerShell there:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper-video.cmd --help
.\piper-video.cmd candidates --video examples\transfer.mp4 --output work\candidates-01
```

The installer verifies SHA256 and installs bundled wheels offline into `.venv`. It does not change global PATH, start models, or configure credentials. Without the `py` launcher, pass `-PythonExe C:\Python312\python.exe`. Reinstall on each new machine instead of copying a virtual environment. Always use a new output directory.

The last command decodes video and selects candidate frames locally. Inspect its `manifest.json` and images. Candidate extraction alone does not interpret actions.

## Configure a model API

The `profiles` directory includes LM Studio, OpenAI, OpenAI-compatible, Anthropic and Gemini examples. Copy a profile, set the actual endpoint and vision model, and keep credentials in the named environment variable. Local model servers and large model weights are not bundled.

```powershell
$env:OPENAI_API_KEY = 'your-own-key'
.\piper-video.cmd models show --model-config profiles\openai.json
.\piper-video.cmd models probe --model-config profiles\openai.json --image examples\probe.jpg --output work\probe-01
.\piper-video.cmd compile --video examples\transfer.mp4 --task "Describe grasp, transport, release and final position; record uncertainty" --model-config profiles\openai.json --output work\demo-01
.\piper-video.cmd inspect --bundle work\demo-01
```

`show` checks configuration; `probe` sends a real image. Cloud providers receive selected images and may charge for calls. Optionally pass `--detector-onnx` with the actual ONNX filename under `models`. This detector is separate from the vision-language model.

Use `compile --store work\demos.sqlite` to register a bundle, `find --store ... --task ...` to retrieve it, and `evaluate --bundle ... --annotations ... --output ...` to compare supplied human annotations. The bundled annotation example is a template, not ground truth. `demo.json` records stages, citations and unknowns. A model's `supported` verdict is not independent human acceptance.

## MCP and terminal agents

Configure a stdio MCP server in your agent, replacing all paths:

```json
{
  "mcpServers": {
    "piper-video": {
      "command": "D:/PiperVideo/.venv/Scripts/python.exe",
      "args": ["-m", "piperlab.video_entry", "mcp", "--workspace", "D:/PiperVideo", "--model-config", "D:/PiperVideo/profiles/openai.json"]
    }
  }
}
```

Client configuration formats vary; use the command and arguments in your client's MCP settings. The client process must inherit API key variables. Without `--model-config`, local selection, paging, image viewing and evidence inspection remain available; compilation reports that no model is configured. Keep normal terminal text out of the MCP stdio stream.

Tools: `video_candidates`, `video_candidate_page`, `video_frame`, `video_compile`, `video_inspect`, `video_evaluate`. Agents can alternatively invoke `piper-video.cmd` and consume JSON/images. An agent can use its own vision model to inspect returned frames; its login is not an inference API credential for this pipeline.

## Architecture and limits

```text
Video → PyAV decode → OpenCV motion analysis/frame selection
                            ↓ optional ONNX detection/tracking
                 selected frames + timestamps → model API
                            ↓
                  schema checks → stages/evidence/unknowns → demo.json
```

This pipeline interprets demonstrations and stores evidence; it does not train model weights. To act, separately connect the `piper-robot` MCP server and examine the current robot scene. Historical video neither authorizes motion nor proves that the robot's skills can reproduce the task. Hexagonal-object/glass-container demonstrations exceed the current colored-cube/green-tray skill scope.

For API failures, inspect the sibling model-calls directory and check credentials, endpoint, vision support and output format. For context overflow, reduce selected frames and check server capacity. Do not treat timeouts as success or run competing local inference jobs. For failed installation, preserve logs and extract afresh.

See `THIRD-PARTY.md`, `licenses` and `DETECTOR-SOURCE.md`; the release includes matching Ultralytics source. Detector licensing is independent. No repository-wide license has yet been assigned to original project code.
