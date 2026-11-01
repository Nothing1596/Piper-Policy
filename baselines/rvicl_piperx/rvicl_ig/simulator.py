"""IG-native two-Panda controller. Only the evaluator can obtain task metrics."""

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .dataset import environment_name, episode_seed


def array(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def scalar(value):
    return float(array(value).reshape(-1)[0])


class Scene:
    def __init__(self, job, directory):
        self.job, self.directory = job, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, job["game_root"])
        os.environ.setdefault("MS_ASSET_DIR", str(Path(job["game_root"]) / ".maniskill"))
        level, env_name = environment_name(job["env_id"])
        for number in (1, 2, 3):
            os.environ.pop(f"MANI_SKILL_L{number}", None)
        if level != "L0":
            os.environ[f"MANI_SKILL_{level}"] = "1"
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
        from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
        for number in (1, 2, 3):
            getattr(L0_L3_utils, f"set_l{number}_enabled")(level == f"L{number}")
        L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)
        self.env = gym.make(env_name, num_envs=1, sim_backend="physx_cpu", wrist_sensor=True,
            control_mode="pd_joint_pos", reward_mode="dense", obs_mode="rgbd", render_mode="rgb_array",
            max_episode_steps=job["max_episode_steps"], reconfiguration_freq=1,
            sensor_configs={"shader_pack": job["shader"]},
            human_render_camera_configs={"shader_pack": job["shader"]})
        self.planners = None
        try:
            self.env = CPUGymWrapper(self.env, ignore_terminations=True, record_metrics=True)
            for _ in range(2 * job["reset_index"] + 1):
                self.obs, _ = self.env.reset()
            self.base = self.env.unwrapped
            actual_seed = int(array(self.base._episode_seed).reshape(-1)[0])
            if actual_seed != episode_seed(job["reset_index"]) or actual_seed != job["seed"]:
                raise ValueError(f"Author reset seed mismatch: {actual_seed}")
            self.agents = list(self.base.agent.agents)
            if len(self.agents) != 2:
                raise ValueError("Adapter requires two Panda robots")
            self.mapping = {side: next(i for i, agent in enumerate(self.agents)
                if agent is getattr(self.base, side + "_agent")) for side in ("left", "right")}
            if set(self.mapping.values()) != {0, 1}:
                raise ValueError("Invalid physical arm mapping")
            self.keys = list(self.base.agent.agents_dict)
            self.grippers = [float(np.clip(2 * (array(a.robot.qpos)[0, -2:].mean() + .01) / .05 - 1, -1, 1))
                             for a in self.agents]
            self.steps, self.observation_index = 0, 0
            self.execution_frames, self.execution_sequence = None, 0
            self.truncated, self.first_success = False, None
            self.metrics, self.actions = None, []
            self.perception = None
            self.log = (self.directory / "actions.jsonl").open("w")
            from .recording import ExecutionRecorder
            self.recorder = ExecutionRecorder(self.directory.parent, float(self.base.control_freq), job["save_video"])
            self.recorder.capture(self.video_frame() if self.recorder.enabled else None, self.steps)
        except BaseException:
            try:
                if getattr(self, "recorder", None):
                    self.recorder.close()
            finally:
                if hasattr(self, "log"):
                    self.log.close()
                self.env.close()
            raise

    def joints(self):
        return [array(agent.robot.qpos)[0, :7].copy() for agent in self.agents]

    def step(self, joints):
        if self.truncated:
            return
        action = {self.keys[i]: np.r_[joints[i], self.grippers[i]] for i in (0, 1)}
        self.obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1
        self.truncated = bool(array(truncated).any())
        self.actions.append(np.concatenate([action[key] for key in self.keys]))
        if bool(array(info.get("success", False)).any()) and self.first_success is None:
            self.first_success = self.steps
        self.metrics = {key: scalar(value) for key, value in info["episode"].items()}
        self.log.write(json.dumps({"step": self.steps, "action": self.actions[-1].tolist(),
                                  "episode": self.metrics}) + "\n")
        self.log.flush()
        self.recorder.capture(self.video_frame() if self.recorder.enabled else None, self.steps)
        self._capture_execution_frame()

    def _execution_views(self, cameras):
        sources = self._sensor_sources()
        views = {}
        for camera in cameras:
            rgb = array(self.obs["sensor_data"][sources[camera]]["rgb"])[..., :3]
            views[camera] = rgb.reshape(*rgb.shape[-3:])
        return views

    def _capture_execution_frame(self):
        if self.execution_frames is not None:
            self.execution_frames.capture(self._execution_views(self.execution_frames.cameras), self.steps)

    @contextmanager
    def capture_execution(self, name, arguments):
        from .recording import ExecutionFrames
        cameras = ["head"]
        if name == "move_both":
            cameras.extend(f"{side}_wrist" for side in ("left", "right") if arguments.get(side) is not None)
        elif arguments.get("arm") in {"left", "right"}:
            cameras.append(arguments["arm"] + "_wrist")
        self.execution_sequence += 1
        record = self.recorder.active or {}
        sampler = ExecutionFrames(self.directory / "execution_frames" / f"{self.execution_sequence:04d}_{name}",
            name, cameras, turn=record.get("turn"), call_id=record.get("call_id"))
        self.execution_frames = sampler
        images = []
        try:
            self._capture_execution_frame()
            yield images
        finally:
            self.execution_frames = None
            images.extend(sampler.finish())

    def _sensor_sources(self):
        sources = {"head": "zed2i"}
        for side, i in self.mapping.items():
            matches = [name for name in self.obs["sensor_data"] if name.startswith(f"{self.agents[i].uid}_{i}_")]
            if len(matches) != 1:
                raise ValueError(f"Expected one wrist camera for {side}")
            sources[f"{side}_wrist"] = matches[0]
        return sources

    def video_frame(self):
        from .recording import compose_frame
        overview = array(self.env.render())
        if overview.ndim == 4:
            overview = overview[0]
        views = {}
        for name, source in self._sensor_sources().items():
            rgb = array(self.obs["sensor_data"][source]["rgb"])[..., :3]
            views[name] = rgb.reshape(*rgb.shape[-3:]).astype("uint8")
        return compose_frame(overview[..., :3], views, self.steps, float(self.base.control_freq))

    def record_tool(self, name, arguments, **metadata):
        return self.recorder.tool(name, arguments, **metadata)

    def state(self):
        arms = {}
        for side, i in self.mapping.items():
            agent = self.agents[i]
            pose = agent.tcp.pose.sp
            arms[side] = {"joints": array(agent.robot.qpos)[0, :7].tolist(),
                "tcp_pose_world": [*map(float, pose.p), *map(float, pose.q)],
                "gripper_opening_m": float(array(agent.robot.qpos)[0, -2:].mean())}
        return {"arms": arms, "steps": self.steps, "remaining_steps": self.job["max_episode_steps"]-self.steps,
                "pose_convention": "world [x,y,z,qw,qx,qy,qz], metres"}

    def observe(self):
        from PIL import Image
        self.observation_index += 1
        sources = self._sensor_sources()
        self.camera_sources = sources
        images, cameras, views = [], {}, {}
        folder = self.directory / "observations"
        folder.mkdir(exist_ok=True)
        for name, source in sources.items():
            rgb = array(self.obs["sensor_data"][source]["rgb"])[..., :3]
            rgb = rgb.reshape(*rgb.shape[-3:]).astype("uint8")
            path = folder / f"{self.observation_index:04d}-{name}.png"
            Image.fromarray(rgb).save(path)
            images.append({"path": str(path), "image_kind": "current_observation", "camera": name,
                           "simulation_step": self.steps, "caption": f"LIVE {name}, step {self.steps}"})
            params = self.obs["sensor_param"][source]
            cameras[name] = {"width": rgb.shape[1], "height": rgb.shape[0],
                "intrinsic_cv": array(params["intrinsic_cv"]).reshape(-1, 3, 3)[0].tolist(),
                "extrinsic_cv": array(params["extrinsic_cv"]).reshape(
                    -1, *array(params["extrinsic_cv"]).shape[-2:])[0].tolist()}
            views[name] = {"rgb": rgb,
                "depth_m": array(self.obs["sensor_data"][source]["depth"]).reshape(rgb.shape[:2]) / 1000.,
                **{key: cameras[name][key] for key in ("intrinsic_cv", "extrinsic_cv")}}
        if self.perception is None:
            from .perception import Perception
            self.perception = Perception(self.job["rpent_root"], self.directory / "perception", self.job)
        perception_step = self.perception.record(views, self.steps)
        return {"state": self.state(), "cameras": cameras, "perception_step": perception_step, "_images": images}

    def _plan(self, poses):
        import sapien
        if not isinstance(poses, list) or not 1 <= len(poses) <= 16:
            raise ValueError("Supply 1..16 Cartesian waypoints")
        if self.planners is None:
            from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
            self.planners = [PandaArmMotionPlanningSolver(self.env, base_pose=a.robot.pose,
                multi_robot_id=i, vis=False, debug=False, visualize_target_grasp_pose=False,
                print_env_info=False) for i, a in enumerate(self.agents)]
        starts, segments = self.joints(), []
        for waypoint in poses:
            if not isinstance(waypoint, dict) or set(waypoint) - {"left", "right"}:
                raise ValueError("Waypoint keys must be left/right")
            paths = []
            for i in (0, 1):
                side = next(side for side, j in self.mapping.items() if j == i)
                target = waypoint.get(side)
                path = starts[i][None, :]
                if target is not None:
                    target = np.asarray(target, dtype=float)
                    if target.shape != (7,) or not np.isfinite(target).all() or np.linalg.norm(target[3:]) < 1e-8:
                        raise ValueError("Pose must contain finite XYZ and a nonzero wxyz quaternion")
                    pose = sapien.Pose(target[:3], target[3:]/np.linalg.norm(target[3:]))
                    current = self.agents[i].tcp.pose.sp
                    if segments or not (np.allclose(current.p, pose.p, atol=1e-5)
                                        and abs(np.dot(current.q, pose.q)) > 1-1e-6):
                        result = self.planners[i].planner.plan_screw(np.r_[pose.p, pose.q], starts[i],
                            time_step=self.base.control_timestep, use_point_cloud=False)
                        if result["status"] != "Success":
                            raise ValueError(f"Motion rejected before execution: {side}, {result['status']}")
                        path = result["position"]
                        if not len(path):
                            path = starts[i][None, :]
                starts[i] = path[-1].copy()
                paths.append(path)
            segments.append(paths)
        return segments

    def wait(self, steps):
        if type(steps) is not int or steps < 1:
            raise ValueError("wait steps must be positive")
        targets = self.joints()
        for _ in range(min(steps, self.job["max_episode_steps"] - self.steps)):
            self.step(targets)
        return {"state": self.state()}

    def finish(self):
        # Match author baselines: score at truncation, holding the final joint target.
        if not self.truncated:
            remaining = self.job["max_episode_steps"]-self.steps
            if hasattr(self, "record_tool"):
                with self.record_tool("evaluation_hold", {"steps": remaining}, source="evaluator") as record:
                    record["result"] = self.wait(remaining)
            else:
                self.wait(remaining)
        if not self.truncated or self.steps != self.job["max_episode_steps"] or self.metrics is None:
            raise RuntimeError("Official horizon scoring requires a completed rollout")
        tracker = getattr(self.base, "reward_tracker", None)
        return {"scored": True, "metrics": self.metrics, "SR": self.metrics["success_once"],
            "first_success_step": self.first_success, "simulation_steps": self.steps,
            "native_phase_score": scalar(tracker.total()) if tracker is not None else None,
            "phase_peaks": {k: scalar(v) for k,v in tracker.get_peak_dict().items()} if tracker is not None else {},
            "sub_sr": None, "sub_sr_available": False,
            "sub_sr_unavailable_reason": "Author RewardTracker.total() is continuous; no binary threshold is defined.",
            "scoring_endpoint": "official_horizon", "metric_protocol": "ig-native-horizon-v1"}

    def close(self):
        try:
            self.recorder.close()
        finally:
            self.log.close()
            try:
                if self.planners:
                    for planner in self.planners:
                        planner.close()
            finally:
                self.env.close()
