# Single-terminal implementation contract (v1)

Source baseline 639ba59. No real hardware access in implementation/tests. No push/publication.

## Ownership
Root: shared models.py, service.py, http_api.py, mcp_server.py, client.py, console_bridge.py, cli.py, standalone_cli.py, store.py; integration/docs/package.
Gemini via agy: console.py; new console_interaction.py; tests/test_console_interaction.py. Do not edit shared files.
DeepSeek via dsh: new profiles.py, managed_runtime.py, ssh_runtime.py, control_sessions.py; tests/test_managed_runtime.py, test_profiles.py, test_control_sessions.py. Do not edit shared files.
Kimi: new approval_policy.py, approval_binding.py; tests/test_approval_policy.py, test_approval_binding.py. Do not edit shared files.
Claude: independent review and new integration tests only, after integration.

## Shared types: interaction_types.py (root owns)
ApprovalMode = always/risk/auto. RunMode = simulation/real.
ExecutionLimits: joint_lower_deg / joint_upper_deg optional six finite values; max_speed_percent default100; gripper_min_m0, gripper_max_m.07; max_effort_protocol32.767.
AutoApproval: max_joint_step_deg, max_tcp_step_m, max_speed_percent, max_effort_protocol all optional; missing relevant threshold means ask.
InteractionPolicy: mode risk; limits ExecutionLimits; automatic AutoApproval; version1.
RuntimeConnection dataclass: url, model_token_file Path, operator_token_file Path, instance_id, owned bool, profile_root Path, profile_id, mode, target.
Settings gains managed_control bool defaultFalse for legacy sim only; real always requires control session and approval policy. Settings gains interaction_policy optional (server reads persisted operator config otherwise defaults), managed_profile_id optional, port allows0 for internal automatic binding. Physical backend never auto-enabled.

## DSH runtime API
RuntimeManager(base:Path, mode:str, target:str='local', simulation_backend:str='mujoco').
async ensure()->RuntimeConnection. Starts/reuses matching idle executor, no CAN connection. Uses profiles isolated by mode/target, fresh credentials. ensure validates actual health instance/backend/profile identity, not mere listening port. Never ignores SSH host key verification.
async release(connection:RuntimeConnection, shutdown_owned:bool=True)->dict. Wait for current accepted action completion before shutdown, never kill a busy robot executor. Shared instance: detach only. Health identity mismatch must refuse shutdown.
Profiles API: profile_root(base,mode,target)->Path; list_remotes(base)->list[dict]; save_remote(base,name,ssh_host)->dict. Host is SSH config alias/hostname, reject option injection/control chars; no password storage. Caller base per-user app data. Profile provenance/migration preserves old restriction, readonly and latch. Returned remote list {name,ssh_host}. No real fallback to simulator.
RuntimeManager supports remote target saved name, uses managed SSH -L tunnel; remote piper-robot host --stdio accepts structured JSON on stdin. host_main() reads request and returns one JSON response. Code paths/params not interpolated into a remote shell. Remote response credentials travel only in SSH pipe, stored0600 temporary frontend cache; never logs secrets. Remote preinstallation of matching package+SSH required. Interactive private-key/host trust setup can give clear setup guidance; do not disable checking or silently install software.
Managed server start: python -m piperx_middleware.cli --root ROOT serve --managed. Root will add --managed, bind127.0.0.1:0, publish runtime.json atomically with url,instance_id,profile_id,mode,pid,backend. Credentials remain in root files. runtime.json is removed only by matching owner on exit. Runtime manager may set initial config.port=0.
ControlSessions: acquire(owner:str, drained:bool)->dict session_id,owner; heartbeat(session_id)->dict; require(session_id)->None raises DomainError; release(session_id)->dict; expire()->bool (sticky one-shot expiry notification, including expiry observed by reads); public()->dict; monotonic injectable, internal lock. 1s heartbeat client,5s expiry; single owner. First acquire after expiry only allowed after root verifies no active action. Never silently resume after expiry. Inactive vs unknown distinguishes missing_control_session/session_expired/session_conflict. require(None) fails, no auto session. No hardware IO inside session object.

## Kimi policy API
PolicyEngine(policy:InteractionPolicy). evaluate(command:dict, state:dict, *, settings:Settings, resolution:dict|None=None, measured_limits:dict|None=None)->dict {decision:'allow'|'ask',reasons:list[str], effective_limits:dict}. Hard-limit failures raise DomainError (never ask). command is resolved JointMove/GripperMove/ControlMode model_dump. state is State.public(), resolution for Cartesian original target/waypoints. Use PiperKinematics configuredTCP to calculate current->targetTCP displacement for any joint move, not just original Cartesian requests. No ROS/world collision claim. Mandatory hard limits in all modes, settings.gripper_max_m included; all measured limits only when available validated current-epoch passed by root. Validate six joints and full linear waypoints (resolution.joint_waypoints_deg if present; inspect existing primitive keys). Fullauto skips prompts only, always policy asks every write, risk asks critical control_mode and threshold exceed/missing. Position/feedback fault checks remain existing service, not duplicated fake safety.
Operation classify helper evaluate_operation(operation:str, changed_fields:list[str])->dict for operator-critical config changes, not expose operator powers to model.
Approval binding API bind_approval(command:dict,state:dict,*,instance_id:str,connection_epoch:int,policy_version:int,parameter_version:int)->dict; validate_approval(binding:dict,command:dict,state:dict,*,same identity kwargs)->None raising DomainError('approval_changed',...). Canonical command bound; six finite pose, gripper pose if relevant; tolerance current .1deg/.001m. No wall-time expiry; root checks fresh feedback at execution. Persistent job ledger remains root Store; restart cancels awaiting approvals and marks attempted jobs unknown.

## Root HTTP contracts
Operator-auth endpoints:
POST /operator/session {owner:str,shutdown_on_loss:bool=False} -> session_id,owner
POST /operator/session/heartbeat {session_id}
POST /operator/session/release {session_id} (drain active action, cancel pending, reject new)
GET /operator/interaction -> {policy:..., pending:list[job], session:..., effective_limits:...}
PUT /operator/interaction {policy:InteractionPolicy} -> persisted policy; idle required, compare submitted current version then increment; cancel pending.
GET /operator/settings -> persistent host settings. PATCH /operator/settings {changes:dict} -> validate, disconnect, persist, increment parameter version and cancel pending. Backend/mode/storage identity changes excluded; model credentials rejected.
POST /operator/approvals/{job_id} {approved:bool} -> job. Requires active current session header.
Model HTTP/MCP preserves tools, adds automatic X-Piper-Control-Session header in client bridge (not model-visible). Frontend owns session and operator token, heartbeats independent of model/dialogs. All real/managed writes require valid session. Old model without session receives explicit error. Read-only/status/connect/stop do not ask. New action creates awaiting_approval if policy asks; response contains job_id/status/reasons. Same request_id returns same job, never duplicate. Frontend polls pending across clients and offers /approve ID or /deny ID, no blocking popup that disables /stop. No short control windows. Completed jobs keep current semantics; awaiting_approval is nonterminal. State includes interaction.policy_mode/session/control_session_required and scene_collision_checked:false.

## Gemini frontend contracts
Root provides run_managed_console(args,base) startup wrapper and ManagedConsoleController(ConsoleController) integration. Gemini implement within console_interaction.py a reusable InteractiveConsoleController(ConsoleController): __init__(bridge,root,emit=print,*,operator_call,managed=None,prompt=None). operator_call async(method,path,body=None)->dict; managed object async switch(mode,target), shutdown(), remotes(), save_remote(name,host), configure(dict) supplied by root. prompt async(text,default='')->str uses current terminal. Constructor backward compatible console core. Controller reads self.bridge.session_id for heartbeat supplied by root.
Implement /approval [always|risk|auto], /approve JOB, /deny JOB, /limits [JSON] (no arg show+interactive wizard using prompt; no prompt show instructions), /config [JSON] (noarg show; updates via managed.configure), /remote [add NAME SSH_HOST] listing/add via managed, /mode [simulation|real] [TARGET] switch via managed, /shutdown via managed, /connect auto discovery /same terminal selection and criticalmode readiness prompt using normal tool/job path; no rawCAN or firmware enable.
Keep read/status/stop responsive while jobs await approval. Add awaiting_approval handling to existing invoke_tool/_poll_job_until_terminal: show once, poll without original action timeout while waiting; reset execution timer once status becomes accepted/running. operator pending background watcher polls GETinteraction and displays one notice perjob/revision. No autopromote approval mode. /quit drains through managed.shutdown instead of cancelling unknown in-flight write prematurely; prevent new model tool calls while draining, wait alreadydispatched job terminal then close client. UI explicitmode, target, approval, feedbackvsUSB status. Model/tool results never receive operator credentials.
Root handles startup mode menu/prompt via module helper choose_startup(prompt)->(mode,target) if provided; otherwise root. Piped startup requires --mode; do not guess real mode. Tests mock managed/operator so no physical access.

## Integration additions verified 2026-09-28

- Critical frontend changes use local `/confirm CODE` proposals so `/status` and `/stop` remain available. Policy proposals use a compare-and-swap version. Local confirmation is an operator UI; the model only receives the separate model credential.
- `public()` control-session information excludes the session secret; only acquire returns it. Jobs record session generation, never credential/session ID.
- A dedicated executor that never acquired a control session exits after a 30-second startup grace. A dedicated executor with an expired owner drains the current action, marks admission closed atomically, then shuts down. Shared attachment expiry does not shut down the shared instance.
- Profile setting changes disconnect before applying new host settings; reconnect is explicit. Firmware configuration remains on the existing ACK/readback endpoint. ROS commissioning files are untouched.
- Migration rejects malformed restrictions rather than normalizing to defaults, preserves known Settings restrictions and operator policy, backs up originals, and creates independent credentials.
