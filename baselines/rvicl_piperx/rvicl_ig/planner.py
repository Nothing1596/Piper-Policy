"""Adapt IG tools and recordings to RPent's unmodified API planner.

Model requests, tool validation, reasoning, and turn limits belong to
rpent.planner.api_loop.ApiAgentLoop. This module binds the IG environment,
optionally selects the latest execution images via its history capability,
and archives the actual HTTP traffic; it contains no agent loop.
"""

import asyncio
import base64
import json
import threading
from functools import partial
from pathlib import Path

from .io import save
from .image_history import describe_tool_images, label_image_history, latest_execution_images
from .rpent_backend import upstream
from .vlm_trace import VLMTrace, write_bytes


class PlannerTrace:
    """HTTP hooks observing native planner requests without changing their bodies."""

    def __init__(self, client, scene, directory):
        self.client, self.scene, self.directory = client, scene, Path(directory)
        self.requests, self.trace, self.message = 0, None, {}
        self.claimed_calls = set()

    async def request(self, request):
        if request.method != "POST" or not request.url.path.endswith("/chat/completions"):
            return
        if self.trace and self.trace.metadata["status"] == "pending":
            self.trace.error(RuntimeError("Native provider retried an unfinished HTTP request"))
        body = await request.aread()
        self.requests += 1
        self.client.calls += 1
        self.message = {}
        self.claimed_calls.clear()
        self.trace = VLMTrace(self.directory / "model_calls" / f"{self.requests:04d}.json",
                              json.loads(body), body)
        self.client.last_trace_directory = str(self.trace.directory.resolve())
        request.extensions["ig_trace"] = self.trace
        print(f"RPent native planner request {self.requests}, simulation step {self.scene.steps}", flush=True)

    async def response(self, response):
        trace = response.request.extensions.get("ig_trace")
        if trace is None:
            return
        raw = await response.aread()
        if response.is_error:
            trace.error(RuntimeError(f"Planner HTTP status {response.status_code}"), raw)
            return
        try:
            if "text/event-stream" in response.headers.get("content-type", ""):
                # Preserve the wire response as well as the SDK's assembled completion.
                from openai.lib.streaming.chat import ChatCompletionStreamState
                from openai.types.chat import ChatCompletionChunk
                write_bytes(trace.directory / "response.sse", raw)
                state = ChatCompletionStreamState()
                for line in raw.decode().splitlines():
                    if line.startswith("data:") and line[5:].strip() != "[DONE]":
                        state.handle_chunk(ChatCompletionChunk.model_validate_json(line[5:].strip()))
                result = state.get_final_completion().model_dump(exclude_none=True)
                trace.metadata["raw_response"] = "response.sse"
                trace.response(json.dumps(result).encode(), result)
            else:
                result = json.loads(raw)
                trace.response(raw, result)
            choices = result.get("choices", [])
            self.message = choices[0].get("message", {}) if choices else {}
        except Exception as error:
            trace.error(error, raw)
            raise

    def finish(self, error=None):
        if self.trace and self.trace.metadata["status"] == "pending":
            self.trace.error(RuntimeError(error or "Planner stopped before the HTTP request completed"))


def create_toolkit(scene, hierarchy, job, directory, trace, events):
    from PIL import Image
    import numpy as np
    from rpent.session import EnvState
    from rpent.tools.toolkit import Toolkit, ToolResult, readonly
    from .agent import AgentTools, tool_specs

    class ImitatorToolkit(Toolkit):
        def _register_common_tools(self):
            # IG exposes only the authorized robot/video/perception interface.
            # The API planner's native read_image tool uses this toolkit's EnvState.
            pass

        def __init__(self):
            super().__init__(dashboard_events=events, state=EnvState(Path(directory) / "planner_state"), memory=None)
            self.controls = AgentTools(scene, hierarchy, job["rpent_root"])
            self.transcript, self.executed_requests = [], set()
            self.dispatch_lock = threading.Lock()
            for item in tool_specs(job["rpent_root"], job.get("sam3_enabled", True),
                                   robot_profile=getattr(scene, "robot_profile", "panda")):
                spec = item["function"]
                self.add_tool(spec["name"], {"name": spec["name"], "description": spec["description"],
                    "input_schema": spec["parameters"]}, partial(self.dispatch, spec["name"]))

        def remember_images(self, result):
            images = result.get("_images", [])
            if not images:
                return result
            with self.state.record_step(state=result.get("state", {"physics_steps": scene.steps})) as step:
                artifacts = []
                for index, item in enumerate(images, 1):
                    path = Path(item["path"])
                    name = f"image_{index:04d}{path.suffix}"
                    with Image.open(path) as image:
                        if self.state.save(name, np.asarray(image.convert("RGB"))) is None:
                            raise RuntimeError(f"Failed to store planner image {name}")
                    artifacts.append({**{key: value for key, value in item.items() if key != "path"}, "name": name})
            return {**result, "planner_image_step": step, "planner_images": artifacts}

        @readonly
        def dispatch(self, name, **arguments):
            self.raise_if_cancelled()
            calls = trace.message.get("tool_calls") or []
            position = next((i for i, call in enumerate(calls)
                if call.get("function", {}).get("name") == name
                and i not in trace.claimed_calls
                and json.loads(call["function"]["arguments"]) == arguments), None)
            if position is not None:
                trace.claimed_calls.add(position)
            call_id = calls[position].get("id") if position is not None else None
            rejected = trace.requests in self.executed_requests or (position is not None and position > 0)
            self.executed_requests.add(trace.requests)
            self.controls.last_record = None
            try:
                if rejected:
                    # Return a result for every native tool call, while retaining the
                    # existing one-action-per-response contract for this simulator.
                    result = {"tool_rejected": "Only the first tool call in a model response is executed"}
                    if name == "finish":
                        result["error"] = "finish refused: only the first tool call is executed"
                    if hasattr(scene, "record_tool"):
                        with scene.record_tool(name, arguments, turn=trace.requests, call_id=call_id) as record:
                            record["result"] = result
                            self.controls.last_record = record
                else:
                    result = self.controls.dispatch(name, arguments, turn=trace.requests, call_id=call_id)
            except (ValueError, TypeError) as error:
                result = {"tool_rejected": str(error)}
                if name == "finish":
                    result["error"] = "finish refused: " + str(error)
            result = describe_tool_images(result, name, turn=trace.requests, call_id=call_id)
            self.transcript.append({"turn": trace.requests, "call_id": call_id, "name": name,
                "arguments": arguments, "result": result, "recording": self.controls.last_record,
                "selection": {"returned_tool_calls": len(calls), "executed": not rejected},
                "vlm_call": self.client_trace_directory()})
            save(Path(directory) / "transcript.json", self.transcript)
            return self.remember_images(result)

        def client_trace_directory(self):
            return str(trace.trace.directory.resolve()) if trace.trace else None

        def execute_tool(self, name, input_dict):
            with self.dispatch_lock:
                return self.execute_serial_tool(name, input_dict)

        def execute_serial_tool(self, name, input_dict):
            result = super().execute_tool(name, input_dict)
            images = result.result.get("_images", [])
            if images:
                text = {k: v for k, v in result.result.items() if k != "_images"}
                result.content_blocks = ToolResult(name=name, result=text).content_blocks
                for item in images:
                    path = Path(item["path"])
                    result.content_blocks.extend([
                        {"type": "text", "text": item.get("caption", "")},
                        {"type": "image", "source": {"type": "base64",
                            "media_type": "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png",
                            "data": base64.b64encode(path.read_bytes()).decode()}}])
            return result

    return ImitatorToolkit()


def solve_native(client, scene, hierarchy, job, directory, *, prompt, http_transport=None):
    image_history = job.get("primitive_image_history", "budget")
    if image_history not in {"budget", "latest"}:
        raise ValueError("primitive_image_history must be budget or latest")
    upstream(job["rpent_root"])
    import httpx2 as httpx
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.planner.api_loop import ApiAgentLoop
    from rpent.utils.logging import init_output_dir
    import rpent.planner.api_loop as native

    source = Path(native.__file__).resolve()
    if source != Path(job["rpent_root"]).resolve() / "rpent/planner/api_loop.py":
        raise RuntimeError("A different RPent planner checkout was already imported")
    from pydantic_ai.capabilities import ProcessHistory

    def image_processor(messages):
        if image_history == "latest":
            return latest_execution_images(messages)
        return label_image_history(native._prune_history_images(messages))

    class ImageHistoryPlanner(ApiAgentLoop):
        def _build_agent(self, system_prompt, toolkit):
            agent = super()._build_agent(system_prompt, toolkit)
            processors = []
            agent.root_capability.apply(lambda capability: processors.append(capability)
                if isinstance(capability, ProcessHistory) else None)
            if len(processors) != 1:
                raise RuntimeError("Expected one native RPent image history processor")
            processors[0].processor = image_processor
            return agent

    planner_class = ImageHistoryPlanner
    init_output_dir(Path(directory) / "planner_logs")
    trace = PlannerTrace(client, scene, directory)
    # No keepalive connections survive ApiAgentLoop.solve's asyncio.run boundary.
    transport = http_transport if http_transport is not None else httpx.AsyncHTTPTransport(
        limits=httpx.Limits(max_keepalive_connections=0))
    http = httpx.AsyncClient(transport=transport, timeout=client.timeout,
        event_hooks={"request": [trace.request], "response": [trace.response]})
    provider = OpenAIProvider(base_url=client.base, api_key=client.key, http_client=http)
    model = OpenAIChatModel(client.model, provider=provider)
    events = NullDashboardEventSink()
    toolkit = create_toolkit(scene, hierarchy, job, directory, trace, events)
    initial = toolkit.remember_images(describe_tool_images(scene.observe(), "view_env_state", turn=0))
    task_context = {key: job[key] for key in ("language_instruction", "camera_perspective", "spatial_reference")
                    if key in job}
    if any(not isinstance(value, str) or not value.strip() for value in task_context.values()):
        raise ValueError("Recorded task instruction and viewpoint context must be nonempty text")
    instruction = ("Learn the demonstrated human behaviors and manipulation skills, "
        "adapt them to the current robot scene, and achieve analogous outcomes. "
        "Inspect live RGB with view_env_state before the first robot action.")
    if "language_instruction" in task_context:
        instruction = ("Perform the supplied language_instruction using the human video as evidence "
            "and the current robot scene for execution. Inspect live RGB with view_env_state "
            "before the first robot action.")
    user_message = json.dumps({"instruction": instruction,
        **task_context,
        "keyframe_pages": hierarchy.page_count, "max_turns": job["max_turns"],
        **{k: v for k, v in initial.items() if k != "_images"}}, ensure_ascii=False)
    if image_history == "latest":
        cameras = "/".join(getattr(scene, "cameras", ("head", "left_wrist", "right_wrist")))
        prompt += ("\n\nIMAGE HISTORY\nOnly the most recent physical primitive's execution process images, "
                   f"the latest live {cameras} camera batch "
                   "and current perception result image are retained in requests. Older robot images "
                   "are omitted; prior tool calls and text remain. Demonstration images are separate "
                   "evidence. Use the latest live views for current geometry, and view_env_state "
                   "when fresh RGB is needed.")
    planner = planner_class(model=model, reasoning_effort=client.effort or "none",
        dashboard_events=events, timeout_s=job.get("planner_timeout", job.get("trial_timeout", 7200)))
    try:
        result = planner.solve(system_prompt=prompt, user_message=user_message,
            toolkit=toolkit, max_turns=job["max_turns"])
        trace.finish(result.error)
        save(Path(directory) / "planner_result.json", {"planner": "rpent-api",
            "planner_class": f"{planner_class.__module__}.{planner_class.__name__}",
            "native_planner_class": f"{ApiAgentLoop.__module__}.{ApiAgentLoop.__name__}",
            "primitive_image_history": image_history,
            "planner_source": str(source), "finish_result": result.finish_result,
            "messages": result.messages, "stats": result.stats, "error": result.error})
        if result.error:
            raise RuntimeError(f"RPent planner failed: {result.error}")
        turns = result.stats.get("turns_used", trace.requests)
        stop = ("horizon" if scene.truncated else "finish" if toolkit.controls.finished
                else "max_turns" if turns >= job["max_turns"] else "model_stop")
        return {"stop_reason": stop, "turns": turns, "planner": "rpent-api",
                "primitive_image_history": image_history,
                "planner_source": str(source), "planner_stats": result.stats}
    finally:
        trace.finish()
        asyncio.run(http.aclose())
