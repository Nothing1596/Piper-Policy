"""Reuse LIBERO's SAM3 segment/back_project primitives on IG sensor artifacts."""

import base64
import copy
import io
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen

import numpy as np
from PIL import Image

from .rpent_backend import upstream

CAMERAS = ("head", "left_wrist", "right_wrist")
ALIASES = {"agentview": "head", "wrist": "left_wrist"}


def world_map(depth_m, intrinsic, extrinsic):
    """OpenCV axial depth and world-to-camera calibration -> measured world XYZ."""
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsic, extrinsic = np.asarray(intrinsic), np.asarray(extrinsic)
    if depth.ndim != 2 or intrinsic.shape != (3,3) or extrinsic.shape not in {(3,4), (4,4)}:
        raise ValueError("Invalid depth or camera calibration shape")
    if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
        raise ValueError("Camera calibration must be finite")
    rows, cols = np.indices(depth.shape)
    rays = np.stack((cols, rows, np.ones_like(cols)), axis=-1) @ np.linalg.inv(intrinsic).T
    points = (rays*depth[...,None] - extrinsic[:3,3]) @ extrinsic[:3,:3]
    points[~np.isfinite(depth) | (depth <= 0) | (depth > 10)] = np.nan
    return points.astype(np.float32)


class RestSamTransport:
    """Translate the existing /segment and /segment_point service into SAM3 RPC.

    No model or policy code is imported from gpt-policy9. Inference stays in the
    existing SAM service; RPent's Sam3Client still validates/decodes its result.
    """

    def __init__(self, endpoint):
        self.endpoint = endpoint.rstrip("/")
        self._opener = build_opener(ProxyHandler({})) if urlsplit(endpoint).hostname in {"localhost", "127.0.0.1"} else None

    def call(self, method, args=(), kwargs=None, *, timeout_s=120):
        if method not in {"segment", "sam3.segment"} or args:
            raise ValueError("REST SAM adapter supports only the segment RPC")
        values = dict(kwargs or {})
        body = {"image_base64": values["image_base64"]}
        point = values.get("point")
        if point is None:
            route = "/segment"
            body["text_prompt"] = values["text_prompt"]
        else:
            route = "/segment_point"
            # RPent point order is [row,col]; this REST service uses [x,y].
            body["point_coords"] = [point[1], point[0]]
        request = Request(self.endpoint + route, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        open_request = self._opener.open if self._opener else urlopen
        with open_request(request, timeout=timeout_s) as response:
            result = json.load(response)
        if point is None:
            candidates = [{"score": float(r["score"]), "box": r.get("box"),
                "mask": np.frombuffer(base64.b64decode(r["mask_base64"], validate=True), dtype=np.uint8).reshape(r["shape"]) > 0}
                for r in result["results"]]
        else:
            masks = np.frombuffer(base64.b64decode(result["masks_base64"], validate=True),
                dtype=np.dtype(result["masks_dtype"])).reshape(result["masks_shape"])
            scores = result["scores"]
            if len(scores) != len(masks):
                raise ValueError("SAM point mask/score count mismatch")
            candidates = [{"score": float(score), "mask": mask > 0} for score, mask in zip(scores, masks)]
        accepted = [r for r in candidates if np.isfinite(r["score"]) and r["score"] >= values.get("min_score", .2)]
        if not accepted:
            return {"found": False, "reason": "SAM3 found no mask above min_score"}
        selected = max(accepted, key=lambda r: r["score"])
        mask = selected["mask"]
        if mask.ndim != 2:
            raise ValueError("SAM must return a 2D mask at the source image resolution")
        if not mask.any():
            return {"found": False, "reason": "SAM3 returned an empty mask"}
        if selected.get("box") is None:
            rows, cols = np.where(mask)
            selected["box"] = [int(cols.min()), int(rows.min()), int(cols.max()+1), int(rows.max()+1)]
        buffer = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8)*255).save(buffer, format="PNG")
        return {"found": True, "score": selected["score"], "box": selected["box"],
                "mask_shape": list(mask.shape), "mask_png_base64": base64.b64encode(buffer.getvalue()).decode()}


def perception_specs(root, sam_enabled=True, *, cameras=None):
    upstream(root)
    cameras = tuple(cameras or CAMERAS)
    from robots.libero.tools import TOOLS_SPEC
    specs = []
    for original in TOOLS_SPEC:
        if original["name"] not in ({"segment", "back_project"} if sam_enabled else {"back_project"}):
            continue
        spec = copy.deepcopy(original)
        spec["input_schema"]["properties"]["camera"]["enum"] = [*cameras, *ALIASES]
        spec["input_schema"]["properties"]["camera"]["description"] = f"Available cameras: {', '.join(cameras)}; agentview=head, wrist=left_wrist."
        spec["input_schema"]["properties"]["step"]["description"] = "perception_step from a live observation; -1 selects latest. Independent of physics control steps."
        if spec["name"] == "back_project":
            spec["description"] = ("RPent LIBERO back_project: use row/col or a row_range/col_range region on a stored IG RGBD snapshot. "
                "World XYZ is in metres. Use the displayed image's actual dimensions, resolution='high', and perception step (-1=current). "
                "No image resizing is performed. Supports z_min/z_max for region estimates.")
            spec["input_schema"]["properties"]["resolution"].update(
                enum=["high"], description="Use high for the displayed native camera resolution; no image resizing.")
        specs.append({"type": "function", "function": {"name": spec["name"],
            "description": spec["description"], "parameters": spec["input_schema"]}})
    return specs


def create_client(root, config):
    upstream(root)
    from rpent.robots.components.sam3_client import Sam3Client
    endpoint = config.get("sam3_endpoint", "http://127.0.0.1:8114")
    protocol = config.get("sam3_protocol", "rest")
    if protocol == "rest":
        transport = RestSamTransport(endpoint)
    elif protocol == "rpent":
        from rpent.utils.rpc.client_utils import make_rpc_client
        transport = make_rpc_client(endpoint)
    else:
        raise ValueError("sam3_protocol must be rest or rpent")
    return Sam3Client(transport, timeout_s=config.get("sam3_timeout", 120))


class Perception:
    def __init__(self, root, directory, config, *, client=None):
        upstream(root)
        from robots.libero.tools import LiberoPrimitives, back_project
        from rpent.session import EnvState
        self.directory = Path(directory)
        cameras = tuple(config.get("camera_names", CAMERAS))
        if not cameras or len(set(cameras)) != len(cameras) or not set(cameras) <= set(CAMERAS):
            raise ValueError("Invalid camera_names")
        self.states = {camera: EnvState(self.directory / camera) for camera in cameras}
        if client is None:
            client = create_client(root, config)
        self.driver = LiberoPrimitives(env=None, model=None, sam3_client=client, check_cancelled=lambda: None)
        self._back_project = back_project
        self.physics_step = None
        self.capture_id = None
        self.enabled = config.get("sam3_enabled", True)
        self.calls = 0

    def record(self, views, physics_step, *, capture_id=None):
        if self.physics_step == physics_step and self.capture_id == capture_id:
            return self.states["head"].latest_step
        if set(views) != set(self.states):
            raise ValueError("Perception requires exactly the configured camera views")
        for camera, data in views.items():
            rgb, depth = np.asarray(data["rgb"]), np.asarray(data["depth_m"])
            if rgb.shape[:2] != depth.shape:
                raise ValueError("RGB/depth source shape mismatch")
            world = world_map(depth, data["intrinsic_cv"], data["extrinsic_cv"])
            state = self.states[camera]
            with state.record_step(state={"camera": camera, "simulation_step": physics_step,
                    "capture_id": capture_id, "source_stamp_s": data.get("source_stamp_s"),
                    "sim_time_s": data.get("sim_time_s"),
                    "image_shape": list(depth.shape), "world_source": "measured RGBD sensor; no object poses"}) as step:
                for name, value in (("agentview_high.png", rgb), ("agentview_world_high.npz", world)):
                    if state.save(name, value, step=step) is None:
                        raise RuntimeError(f"Failed to save perception artifact {name}")
        self.physics_step = physics_step
        self.capture_id = capture_id
        return self.states["head"].latest_step

    def _camera(self, camera):
        camera = ALIASES.get(camera, camera)
        if camera not in self.states:
            raise ValueError("Unknown IG camera")
        return camera

    def segment(self, prompt="", camera="head", step=-1, point=None, min_score=.2):
        if not self.enabled:
            return {"error": "SAM3 explicitly disabled in config"}
        camera = self._camera(camera)
        if not isinstance(prompt, str):
            raise ValueError("prompt must be text")
        if point is not None:
            if not isinstance(point, list) or len(point) != 2 or any(type(p) is not int for p in point):
                raise ValueError("point must be one [row,col] integer pair")
            shape = self.states[camera].get(step).state["image_shape"]
            if not 0 <= point[0] < shape[0] or not 0 <= point[1] < shape[1]:
                raise ValueError("SAM point outside the selected image")
        self.calls += 1
        result = self.driver.segment(prompt=prompt, camera="agentview", step=step, point=point,
                                     min_score=min_score, state=self.states[camera])
        result.pop("_image_bytes", None)
        result.update(camera=camera, pixel_order="row_col", world_frame="world", unit="metre")
        if result.get("overlay_artifact"):
            path = self.states[camera].artifact_path(result["overlay_artifact"], step=result["step"])
            result["_images"] = [{"path": str(path), "caption": f"SAM3 overlay on {camera}, perception step {result['step']}"}]
        result["artifact_root"] = str((self.directory / camera).resolve())
        return result

    def back_project(self, camera="head", **arguments):
        camera = self._camera(camera)
        result = self._back_project(camera="agentview", state=self.states[camera], **arguments)
        return {**result, "camera": camera}
