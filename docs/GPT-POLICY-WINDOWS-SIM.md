# GPT-Policy on the existing Piper MuJoCo platform

This branch adds original adapter/glue code. It runs the separately supplied
GPT-Policy checkout at `ab970d88bc5570d80a7b4f3e8d3ad97ebe65a007` on the existing
single-arm Piper MuJoCo simulator. It does **not** run the piperlab policy,
connect CAN, change scene assets, enable a physical robot, or ship GPT-Policy
sources. Windows is the target; the implementation's verification environment
is Linux. Native Windows validation remains required.

## Upstream licensing boundary

[GPT-Policy LICENSE](https://github.com/cheng-haha/GPT-Policy/blob/ab970d88bc5570d80a7b4f3e8d3ad97ebe65a007/LICENSE)
allows review/evaluation and has no selected redistribution/commercial license.
Obtain and retain that checkout separately for authorized evaluation. The
preparation command makes narrow edits **in that separate local checkout**.
Do not copy or publish its modified sources in Piper-Policy, release ZIPs or
wheels without the copyright holder's permission. `git diff` in the upstream
checkout shows every adaptation. Its harness, mixed inputs, tool catalog,
provider profiles, runtime loop, recording and motion planner stay upstream.

## What is preserved

- Upstream `EefTrajectoryPlanner`, straight-line/SLERP sampling, per-sample IK,
  continuous DLS refinement, residual audit and scalar-path Ruckig timing
- Model target poses, ordering and every joint sample/timestamp; no endpoint
  replacement, motion-sample dropping or downstream retiming
- Native upstream radian/quaternion model contract; radians/degrees conversion
  is confined to the existing middleware boundary
- Existing durable request IDs, one action owner, session lifecycle, fresh
  feedback, cancel/hold semantics and restart-without-replay behavior
- Existing camera geometry, scene, simulation seed and calibrated Piper TCP
- Upstream provider settings and model selection; no new model defaults

## Deliberate platform differences

- One fixed `top` RGB overview camera. No wrist views, depth or hidden object
  poses enter `CapturedImage` or model observations. Camera calibration comes
  from the existing sensor interface. `locate_point` returns a calibrated ray;
  arm movement cannot create parallax for this fixed camera
- Piper MDH/multistart IK is the solver underneath unchanged upstream continuous
  refinement. ARX/YAM solver equivalence is not claimed
- Piper's existing home is `[0, 30, -30, 0, 0, 0]` degrees. It replaces ARX-only
  home geometry, which violates Piper joint limits
- Measured torque and gripper velocity are unavailable in the current interface
  and reported as null, not fabricated SDK measurements
- MuJoCo receives the supplied timed samples directly as servo references, with
  no second speed profile. The simulator's physical dynamics, actuator response
  and fixed-rate physics thread remain unchanged; an SDK-equivalent actuator
  interpolation model is not claimed
- Gripper closure completes on fresh width **or** stable bilateral force
  contacts. It reports the requested/measured opening and contact evidence.
  This replaces ARX encoder-obstruction feedback; contact never means lift or
  task success. No oracle task score is exposed
- Timeline admission currently requires direct-profile simulation and an
  operator-selected `auto` policy. It rejects calibration/risk/always modes
  rather than bypassing approval. Every sample still obeys site joint limits
- Timeline transport hard envelope: velocity <= pi rad/s, numerical acceleration
  <= 20 rad/s² and jerk <= 500 rad/s³. This rejects forged unsafe references but
  is not a Ruckig provenance or collision certificate. Upstream MotionLimits
  remain tighter by default. Maximum 4096 samples, 600 seconds and 1 MiB request
- If scheduling is over 100 ms late, tracking exceeds 0.12 rad, fresh feedback
  disappears or identity changes, execution holds and reports a non-success
  outcome. Unknown writes are queried by the same request ID, never replayed

## Windows setup (PowerShell, Python 3.12 x64)

From the Piper-Policy checkout:

```powershell
git fetch origin
git switch feat/gpt-policy-windows-sim-baseline
py -3.12 -m venv .venv
$py = (Resolve-Path .venv\Scripts\python.exe).Path
& $py -m pip install -e ".\piperx-cli[simulation,gpt-policy,hardware,test]"
git clone https://github.com/cheng-haha/GPT-Policy.git ..\GPT-Policy
$upstream = (Resolve-Path ..\GPT-Policy).Path
git -C $upstream checkout ab970d88bc5570d80a7b4f3e8d3ad97ebe65a007
& $py -m gpt_policy_piper.prepare --upstream $upstream
& $py -m pip install -e $upstream
& $py -m pip check
```

The optional `hardware` dependency only enables existing SDK-contract test
collection. These commands do not install the ARX/YAM hardware SDKs or connect
any physical robot. The preparation also fixes upstream Hatch's direct-reference
metadata requirement; it otherwise leaves dependency versions intact.

### Start the same simulation

Terminal A, in Piper-Policy:

```powershell
.\.venv\Scripts\piper-robot.exe sim start --root .runtime-gpt-policy --port 8808 --seed 200
```

This is the existing simulator launcher and its existing `.1425 m` fingertip
TCP offset. To compare to a different existing runtime, supply its actual root,
port and seed instead; do not replace its scene or calibration.

Terminal B, in Piper-Policy:

```powershell
$py = (Resolve-Path .venv\Scripts\python.exe).Path
$upstream = (Resolve-Path ..\GPT-Policy).Path
.\.venv\Scripts\piper-robot.exe --root .runtime-gpt-policy --url http://127.0.0.1:8808 connect
& $py -m gpt_policy_piper.prepare --upstream $upstream --runtime-root .runtime-gpt-policy --url http://127.0.0.1:8808
& $py -m gpt_policy_piper.cli --upstream $upstream --check
& $py -m pytest piperx-cli\tests -q
Push-Location $upstream
& $py -m pytest tests -q
Pop-Location
```

Preparation writes `configs/piper-mujoco.local.json` only in the separate
upstream checkout. It refuses to overwrite an existing local profile. Token
files are referenced by path; token contents are never copied into the profile.
The offline `--check` opens no camera/robot and calls no provider.

### Run an intended model task

Use your existing authenticated upstream provider setup. Review its agent
profiles first: Codex, Claude and Kimi retain upstream executable and model
settings. Windows must be able to launch the selected provider CLI. Video
inputs also need the upstream-required FFmpeg tools.

```powershell
& $py -m gpt_policy_piper.cli --upstream $upstream --allow-model-calls --input-json C:\path\to\your-existing-task.json
```

`--allow-model-calls` is explicit acknowledgement that the configured upstream
provider may charge for inference, task naming or video selection. It does not
change the provider or price. No paid inference was run for this implementation.
Keep the same task input and scene seed when comparing with the existing policy.
Recordings and traces use the unchanged upstream run layout.

### Stop and inspect

Ctrl+C follows upstream interruption handling. The first interrupt requests a
hold and then a verified Piper home; a second interrupt cancels homing. A motion
fault latches off further adapter actions for that process. Inspect the job's
`request_id`, `status`, `execution_timing` and fresh simulator feedback before
retrying. Do not resubmit an unknown action under a new request ID.

```powershell
.\.venv\Scripts\piper-robot.exe --root .runtime-gpt-policy --url http://127.0.0.1:8808 stop
.\.venv\Scripts\piper-robot.exe --root .runtime-gpt-policy --url http://127.0.0.1:8808 jobs --json
```

## External checkout edits

The preparation checks exact Git HEAD and SHA-256 of eight managed files before
changing any. A repeat is idempotent only if approved prepared contents match;
unknown edits are rejected. Windows adaptations are explicit:

- Optional Linux V4L2 import so the same upstream CapturedImage is usable
- Real Windows byte-range cache locking, no fake fcntl module
- Bounded worker-thread pipe writes instead of POSIX select on anonymous pipes
- Process-tree termination through Windows taskkill; cleanup uncertainty raises
- Piper factory, sensor factory, calibration facts and offline preflight routing

To undo, first review `git diff` in the external checkout. Restore only the eight
listed managed files using Git, then remove `.piper-gpt-adapter.json`. Do not
blindly restore files if you have added your own edits. Keep the local profile
separately or delete it yourself if you no longer need it.

## Verification scope

Offline tests cover exact timeline transport, hard path bounds, malformed/fast
references, cancel, single owner, expired sessions, TCP identity, idempotency,
unknown submission, bounded chunked requests, actual MuJoCo tracking, RGB-only
capture/calibration and the actual upstream runtime loop with a fake agent.
The pinned upstream's own tests currently give **59 passed, 2 failed** when run
from its checkout. Both failures are pre-existing `tests/test_config.py` uses of
`Path(__file__).parents[2]`, which look for configs outside the repository.
The same failures were reproduced on a pristine pinned checkout. They are not
counted as passing or fixed by this adapter.

Windows helpers have focused contract/mock tests; those do not establish native
Windows execution. No commercial/physical use, paid model, ARX/YAM hardware,
CAN, real camera, task-success rate or Windows provider lifecycle is validated.
