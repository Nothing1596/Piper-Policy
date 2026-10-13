# Live video / model CLI validation — 2026-10-07

## Follow-up: 0.7.2.dev2 + video 0.3.1.dev1

- Existing demonstration pipeline is now connected to the robot frontend. Direct `demo_watch` keeps exact decoded PTS; pipeline mode returns CV/YOLO evidence and a bounded DeepSeek V4.1 Flash evaluation at reasoning effort low. Actual API calls and frame-reference validation passed.
- Windows + Mac-client stdio MCP: 17 tools discovered, demonstration images and actual D435 pixels returned. Video workers now use DEVNULL stdin rather than inheriting the model protocol pipe. After this change the combined call returned; earlier standalone components had already passed. No CAN connection or robot motion was issued.
- Parallels injects empty stdout lines at exit; the narrowly scoped relay drops only whitespace lines. The relay command initialized successfully with 17 tools and CAN disconnected. The user's corrected Codex TOML was parsed and preserved; a subsequent second-client test was rejected because another control session was already active, rather than taking over that session.
- Windows recorded source/index/seek/detection ~8.06 s, Flash ~19.44 s for a selected demonstration interval. These are observations, not latency guarantees.
- Mac arm64 fresh two-environment install: native MuJoCo action, direct point watch, CV/YOLO + Flash watch and clean exit passed. One full call took ~39.72 s, API inference ~12.60 s. Native Mac CAN and D435 remain untested.
- API truncation produced an explicit error, not a valid evaluation. The low thinking setting was retained, evaluation wording shortened, and output ceiling increased to 8192; subsequent integration calls passed. Completion reason/usage and phase timings are retained for diagnosis.
- Detector overlay check: wooden-box region boxes are spatially plausible, but COCO class predictions book/cake are inappropriate and low confidence (first frame 0.145/0.121). This is no proof of complete detector correctness or object identity. Detector results remain proposals.
- Final macOS robot regression: 583 passed / 8 skipped, including both relay lifecycle checks. Focused compiler/video/provider regression with actual PyAV/OpenCV: 97 passed. No public release or push in this work.
- Evidence: `relay-validation/result.json`, `vm-mcp-validation/pipeline-result.json`, `mac-kit-validation/result.json` under the local deployment directory described below.

The earlier dev1 record below is historical and is superseded for real camera availability by the successful dev2 tests.

Source development version: **0.7.2.dev1**, based on `40d231b`. This record describes the current local changes, not a published release or CI run.

## Software

- macOS: `python -m pytest piperx-cli/tests -q` — **576 passed, 8 skipped**, one existing Starlette/httpx deprecation warning, 44.54 seconds. Skipped tests are outside this acceptance claim.
- Actual isolated JSONL frontend/executor processes: connect to the kinematic simulator, execute a joint action, retain its request ID, return structured errors, and drain/release on quit. No CAN opened.
- Actual synthetic camera subprocess: continuous frames, one reused process, endpoint/change selection, ROI crop/storyboard, recorded-pixel compatibility, HTTP state alignment, and EOF cleanup.
- Regression coverage: memory/frame/time budgets, stale frames, missing depth, invalid ROI, subprocess timeout cleanup, no arbitrary image-file ingestion, model image placement after tool replies, bounded image history, stable action IDs, operator command rejection, and stop cancelling local action preparation.

## Windows on ARM / Parallels

An isolated `0.7.2-live-20261007` environment was installed using the already available x64 Python dependencies. The original release remains installed; real profile configuration, policy and camera file hashes were checked unchanged. Launcher files were backed up before selecting the development version.

- JSONL + kinematic simulation: passed; duplicate request returned the same job; frontend/executor exited cleanly.
- Synthetic continuous video through the configured camera interpreter: passed; process reused and reaped.
- Windows embedded camera Python originally did not include the worker script directory. Explicit sibling-module resolution fixed that failure; stderr diagnostics now expose startup errors.
- Real D435: **blocked**, SDK enumeration returned `available=[]`. Failure returned a clear error and the worker was reaped. No real video latency, frame quality, calibration, grasp or hardware-stop acceptance is claimed.
- Physical robot commands: **0**. Real CAN connections opened by this validation: **0**.

Local deployment evidence and rerunnable Windows scripts are in `/Users/nothing/DEV/piper-live-deploy-20261007/`, including `windows-validation.json`, `real-camera-validation.json`, `verify_windows.py` and `probe_real_video.py`.

## Independent review

Direct terminal calls to agy Gemini 3.8 Flash High and Claude `opus5.5` did not produce completed reviews within their bounded runs. These are **not review approvals**. Codex reviewed the implementation and added the above regression tests. No subagents were used.

## Remaining scope

Linux was not validated. Real camera capture requires the D435 to be assigned to Windows and another camera-only run. Current selection uses small grayscale change/clarity and optional ROI; it does not compensate wrist-camera motion or detect contact/slip. Frame/state alignment is bounded host receipt alignment, not exposure synchronization. The robot HTTP/MCP server still owns only robot execution; local camera tools belong to this frontend. No separate video UI, MP4 recorder, new planner, or automatic calibration was added.

## Subsequent read-only hardware attempt

The user authorized a read-only retry. Windows enumerated the candleLight adapter and D435 as OK. An isolated real profile copied the existing limits and set allow_motion=false. The normal CLI attempted robot_connect once; CANDO reported cleanup_incomplete / cando_stop failed, received_frames=0 and tx_frames=0. No joints or gripper feedback was obtained, and no motion/mode/enable command was sent. Only the failed trial's own retained process was terminated after checking its identity. USB replug was requested before another attempt; the main real profile was not modified. This is a failed hardware initialization attempt, not real-robot acceptance.
