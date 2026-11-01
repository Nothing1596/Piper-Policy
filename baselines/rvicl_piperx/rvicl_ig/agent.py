"""IG tools and entry point for RPent's upstream planner."""

from contextlib import nullcontext
from pathlib import Path

from .hierarchy import object_schema

PROMPT = (Path(__file__).parent / "prompts/agent.txt").read_text()


def tool_specs(rpent_root, sam_enabled=True, *, robot_profile="panda"):
    from .rpent_backend import primitive_specs
    from .perception import perception_specs
    integer = {"type": "integer"}
    text = {"type": "string"}
    pose = {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 7, "maxItems": 7}
    specs = [
        ("view_task_video", "Read one L0 keyframe page before planning; part starts at 1.", object_schema({"part": integer})),
        ("view_task_map", "Read the text L1 phases and L2 moment index before planning.", object_schema({})),
        ("view_demo_clip", "Read L3 ORIGINAL human frames for exactly one phase or moment.", object_schema({
            "moment": {"type": ["string", "null"]}, "phase": {"type": ["integer", "null"]}})),
        ("view_env_state", "Get live head and both wrist RGB, camera calibration, and robot state.", object_schema({})),
        ("move_both", "Synchronized composition of RPent native plans for both arms. WORLD [x,y,z,qw,qx,qy,qz], null holds one arm.",
            object_schema({"left": pose, "right": pose,
                "substeps": {"type": "integer", "minimum": 0}})),
        ("wait", "Hold current joints for up to 50 physics control steps.", object_schema({
            "steps": {"type": "integer", "minimum": 1, "maximum": 50}})),
        ("finish", "End agent control; only the evaluator assigns scores.", object_schema({
            "status": {"type": "string", "enum": ["success", "failure", "stuck"]}, "summary": text})),
    ]
    if robot_profile not in {"panda", "piperx"}:
        raise ValueError("Unknown robot profile")
    cameras = None
    if robot_profile == "piperx":
        from .piper_binding import primitive_specs
        cameras = ("head", "left_wrist")
        specs = [(n, "Get fresh fixed head and PiperX wrist RGBD with per-camera capture times and robot state." if n == "view_env_state" else d, s)
                 for n, d, s in specs if n != "move_both"]
    return [*primitive_specs(rpent_root), *perception_specs(rpent_root, sam_enabled, cameras=cameras), *[{"type": "function", "function": {
        "name": n, "description": d, "parameters": s}} for n,d,s in specs]]


class AgentTools:
    def __init__(self, scene, hierarchy, rpent_root=None):
        self.scene, self.hierarchy, self.finished = scene, hierarchy, False
        self.rpent_root = rpent_root or str(Path(__file__).resolve().parents[2] / "RPent")
        self._primitives = None
        self.last_record = None

    @property
    def primitives(self):
        if self._primitives is None:
            if getattr(self.scene, "robot_profile", "panda") == "piperx":
                from .piper_binding import create_primitives
            else:
                from .rpent_backend import create_primitives
            self._primitives = create_primitives(self.scene, self.rpent_root)
        return self._primitives

    def dispatch(self, name, arguments, *, turn=None, call_id=None):
        scope = self.scene.record_tool(name, arguments, turn=turn, call_id=call_id) if hasattr(self.scene, "record_tool") else nullcontext(None)
        with scope as record:
            self.last_record = record
            result = self._dispatch(name, arguments)
            if record is not None:
                record["result"] = result
            return result

    def _dispatch(self, name, arguments):
        if name in {"view_task_video", "view_task_map", "view_demo_clip"}:
            return getattr(self.hierarchy, name)(**arguments)
        if name == "view_env_state":
            return self.scene.observe()
        if name in {"segment", "back_project"}:
            return getattr(self.scene.perception, name)(**arguments)
        if name == "finish":
            if set(arguments) != {"status", "summary"} or arguments["status"] not in {"success", "failure", "stuck"}:
                raise ValueError("Invalid finish arguments")
            self.finished = True
            return {"_finish": True, "agent_declaration": arguments}
        if name not in {"move_to", "rotate_wrist", "set_gripper", "release", "move_both", "wait"}:
            raise ValueError("Unknown tool")
        if self.finished or self.scene.truncated:
            raise ValueError("Agent control has ended or the physics budget is exhausted; call finish")
        if getattr(self.scene, "robot_profile", "panda") == "piperx":
            self.scene.check_cancelled()
            if name == "move_both" or (name != "wait" and arguments.get("arm") != "left"):
                raise ValueError("Only arm='left' exists on the single PiperX adapter")
        if not self.hierarchy.ready:
            raise ValueError("Read ALL L0 keyframe pages and the L1/L2 map before moving")
        if name == "wait" and (type(arguments.get("steps")) is not int or not 1 <= arguments["steps"] <= 50):
            raise ValueError("wait steps must be 1..50")
        capture = (self.scene.capture_execution(name, arguments) if hasattr(self.scene, "capture_execution")
                   else nullcontext([]))
        with capture as process_images:
            if name == "wait":
                result = self.scene.wait(**arguments)
            else:
                result = getattr(self.primitives, name)(**arguments)
        feedback = getattr(self.scene, "last_ik_feedback", None)
        if (getattr(self.scene, "robot_profile", "panda") == "piperx"
                and result.get("stop_reason") == "plan_failed" and feedback):
            result = {**result, "upstream_hint": result.get("hint"),
                      "hint": "No validated solution was returned for this requested pose. Collision was not checked; this is not proof of global unreachability.",
                      "ik_feedback": dict(feedback)}
        observation = self.scene.observe()
        return {"execution": result, **observation,
                "_images": [*process_images, *observation.get("_images", [])]}


def solve(client, scene, hierarchy, job, directory):
    from .planner import solve_native
    return solve_native(client, scene, hierarchy, job, directory, prompt=PROMPT)
