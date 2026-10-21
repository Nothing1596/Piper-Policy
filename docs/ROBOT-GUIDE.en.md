# Piper Robot 0.7.2 installation

[中文](ROBOT-GUIDE.zh.md) · [Download](https://github.com/Nothing1596/Piper-Policy/releases/tag/robot-v0.7.2) · [Release notes](ROBOT-RELEASE-0.7.2.md)

Select a mode, connect, invoke tools, approve actions and exit in one terminal. The kit contains the robot executor, HTTP/MCP, simulation assets and source; this kit also includes the independent Video 0.3.1 wheel and source. No model service, model weights or vendor SDK are bundled, and no firmware is flashed.

## Install

Prepare **Python 3.12+ and internet access** for this kit with video. The standalone robot wheel alone requires Python 3.11+. Windows CANDO requires x64 Python, including x64 emulation on Windows on ARM; vendor drivers and SDK are separate prerequisites.

Download `piper-robot-0.7.2-interactive.zip` and its `.zip.sha256`. Extract into a fresh directory. Beside `install.py`, run:

Windows (or run `Setup.cmd`):

```powershell
py -3 install.py
.\piper-robot.cmd
```

macOS/Linux:

```sh
python3 install.py
./piper-robot
```

The installer creates the robot `.venv` and video `.video-venv`, downloads dependencies and checks CLI help. Launchers pass the independent video interpreter automatically. It does not install Python, register PATH or connect hardware. An existing `.venv` is preserved and installation stops; use a fresh extraction. Without the Windows `py` launcher, invoke `install.py` with the absolute path of your Python interpreter. Do not apply the old `installer-v1` add-on to this kit.

Activate the kit's virtual environment to use the bare `piper-robot` command. Otherwise replace it with `.\piper-robot.cmd` or `./piper-robot` in examples.

## First task

Run the launcher, select **simulation → local**, then enter these commands separately in the same terminal:

```text
/connect
/status
/tools
```

Wait for `Ready: yes`. MuJoCo is the default physical simulator. For the deterministic software demo, launch `piper-robot --root ./work/demo-one-task --simulation-backend sim`, then follow the [task walkthrough](ONE-TASK-DEMO.zh.md). It moves J1 to 5°, opens the gripper to 0.04 m and returns the joints to zero. Wait for each job to succeed before submitting the next.

`/manual tool(arguments)` needs no model account. `/model` displays the endpoint and MCP connection; `/model set ...` configures your OpenAI-compatible Chat Completions endpoint, and `/model check` makes an actual API request. Plain text then goes to that model. The walkthrough includes credentials-by-file examples and expected results.

`/connect` manages the executor and ports, prompting if several devices are found. Missing CAN never silently selects simulation. Connected, feedback available and Ready are separate states; inspect `/status`, `/params` and the error before proceeding.

## Video, plan and metric evidence

Use `/observe` in MuJoCo for a fresh RGB-D snapshot, then `/plan new "task goal"`. `/demo prepare /absolute/demo.mp4` selects a demonstration; `/demo watch` supports direct images or pipeline evaluation at a timestamp or interval. Recorded evidence never becomes a current robot capture.

`task_plan` stores revisable interpretations, questions and evidence versions. `scene_query` computes visible surface points, local point clouds, planes and distances on CPU. Advisory behavior review highlights repeated queries, ignored rejections and unsupported completion; it neither changes motion parameters nor judges semantic success. Online plans remain soft state.

See the [observation workflow](ROBOT-OBSERVATION-PLAN.zh.md) and [0.7.2 guide](versions/robot-0.7.2.zh.md). Physical cameras require the camera extra, drivers and real calibration separately. MuJoCo provides on-demand snapshots, not continuous history. Detector weights and vision API credentials are not bundled.

## Approval, remote hosts and exit

New simulations default to `auto`; real hardware defaults to `risk`. Existing approval settings and limits are preserved. `/approval`, `/limits` and `/config` operate in the same terminal; confirm proposed configuration changes with `/confirm`. Use `/approve JOB_ID` or `/deny JOB_ID` for pending actions. `/status` and `/stop` remain available while waiting. No second-terminal OPEN window is required.

`/remote` manages hosts with SSH and the matching CLI already installed; `/mode` selects mode and target. Configure SSH login and host trust first. Host-key verification stays enabled. See the [interaction guide](ROBOT-INTERACTIVE.zh.md) for details.

`/quit` waits for accepted actions and cleans up a dedicated executor, or releases only this session when attached to a shared service. `/stop` requests an action stop; quitting is not an emergency stop. Query the original request_id after connection loss or an unknown result; do not resend under a new ID.

MCP and HTTP use the same executor rules. Model credentials cannot change operator policy or limits. Standalone MCP/script entries remain available, but old clients without a valid control session are rejected. Use the console for the managed workflow; advanced options are in the [CLI reference](ROBOT-CLI-REFERENCE.md).

## Upgrade and evidence

Exit the old console, verify its dedicated executor has ended, and install a fresh extraction without copying `.venv`. Mode/host profiles are separate. Migration backs up old configuration and preserves read-only settings, limits and fault latches; it does not approve ROS commissioning.

Current software, package and isolated-install checks are recorded in the [0.7.2 validation report](implementation/release-0.7.2-validation.zh.md). Unit fixtures, MuJoCo, model tasks and hardware are separate evidence tiers. This release cycle does not re-qualify physical hardware, camera calibration, collision planning or task success rates. Historical 0.7.1 results remain in its own release documentation.
