# Independent robot pipeline

[中文](ROBOT-GUIDE.zh.md)

[Complete command and feature reference](ROBOT-CLI-REFERENCE.md)

For automatic Python setup and user PATH registration, see [one-click installation](ONE-CLICK-INSTALL.md). Double-click `Setup.cmd` in new kits; original ZIPs use the small add-on. The original manual entry point remains documented below.

Version 0.6.0, CLI `piper-robot`; legacy `piperx` and `piperx-mcp` remain available. This package contains the executor, CLI, HTTP/MCP, MuJoCo and robot assets. It installs no video pipeline, detector or vision-language model. The lower controller is host middleware, not arm firmware.

## Install and simulate

Use Windows x64 and Python 3.12 x64. Extract to a fresh directory and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\piper-robot.cmd --help
.\piper-robot.cmd sim start --root work\sim --port 8798 --seed 200
```

The installer verifies SHA256 and installs offline into `.venv`, without changing global PATH or enabling hardware. If needed, pass `-PythonExe C:\Python312\python.exe`. Reinstall rather than copying `.venv` to another machine.

Leave the simulator running in terminal A. In terminal B:

```powershell
.\piper-robot.cmd --root work\sim --url http://127.0.0.1:8798 connect
.\piper-robot.cmd --root work\sim --url http://127.0.0.1:8798 status
.\piper-robot.cmd observe --root work\sim --url http://127.0.0.1:8798 --workspace . --output work\frame-01
```

The evidence directory contains RGB, depth and calibration metadata. Use new output names; evidence cannot be overwritten. Camera capture currently supports MuJoCo only. Existing roots must match backend, port and seed; use a new root when changing configuration. If a port is occupied, change it consistently in all commands.

## Connect an agent

No model API configuration is required in this package. An external agent interprets instructions and images, then invokes tools. Example stdio MCP process configuration:

```json
{
  "mcpServers": {
    "piper-robot": {
      "command": "D:/PiperRobot/.venv/Scripts/python.exe",
      "args": ["-m", "piperx_middleware.standalone_cli", "mcp", "--root", "D:/PiperRobot/work/sim", "--url", "http://127.0.0.1:8798", "--workspace", "D:/PiperRobot"]
    }
  }
}
```

Replace paths and adapt to your client's MCP settings. Start the executor first. MCP connects to this shared owner and does not open a separate CAN connection. Tokens are read from local root files; never include them in prompts or commits. Omitting `--workspace` disables the camera tool.

Agent loop: `robot_status` → `robot_connect` if needed → `simulation_observe` → `move_to` / `move_by` / `move_linear` / `set_gripper` → query the returned job_id → observe again. Reuse request_id for retries of the same action; use a new ID for a new action. A job submission is not completion; check final status and fresh evidence.

Terminal agents can use `piper-robot --root ... --url ... <command>`. Consult each command's `--help`. Use `stop` to request a stop and `shutdown` to close the service. Process termination is not a physical emergency stop.

## Architecture

```text
External model/agent → CLI or MCP → HTTP client
                                     ↓
              shared executor: authentication/deduplication/queue/feedback/limits
                                     ↓
                        MuJoCo or configured CAN backend
                                     ↓
                             status/jobs/RGB-D evidence
```

One executor owns device state. Models do not own CAN directly. Contact, successful lifting and completed placement are distinct outcomes.

Install `piper-video` separately for demonstrations. An agent can connect both MCP servers, inspect a demonstration and then execute using current robot observations. Video output does not automatically become hardware commands. Combined release v0.2.1 retains the previous integrated policy entry point.

## Hardware and verification boundaries

Validate in simulation first. Real hardware additionally requires the correct platform, CAN driver, vendor SDK, robot model, calibration and on-site commissioning. `python-can` is bundled; the vendor SDK is not. Real Windows CAN execution has not been validated. See `piper-robot init --help` for initialization options and verify emergency stop/workspace arrangements before enabling hardware.

Standalone-package checks cover offline installation, MCP handshake, simulation connection and actual RGB-D return. Earlier colored-cube evaluations do not establish hardware acceptance. Preserve the third-party notices in `licenses`; original project code has no repository-wide license yet.
