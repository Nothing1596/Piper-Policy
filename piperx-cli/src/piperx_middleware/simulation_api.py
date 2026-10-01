"""Authenticated simulation sensor interface. Truth scoring is operator-only."""
from concurrent.futures import ThreadPoolExecutor
import base64
import io
import threading

from fastapi import Depends, Request
from .models import DomainError
from .timed_trajectory import TimedTrajectory, TimedTrajectoryRequest


def attach_simulation_api(app, service, auth, operator_auth):
    # The pool owns its rendering context; ASGI's general pool may change threads.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation-camera")
    app.state.simulation_camera_pool = pool

    def backend():
        if service.backend.name != "mujoco":
            raise DomainError("simulation_only", "MuJoCo backend required", 404)
        if not service.backend.snapshot().connected:
            raise DomainError("not_connected", "Connect the simulator first")
        return service.backend

    def capture(rgb_only=False):
        import numpy as np
        from PIL import Image
        observation = backend().observe()
        rgb = observation.pop("rgb")
        depth = observation.pop("depth")
        image = io.BytesIO()
        Image.fromarray(rgb).save(image, format="JPEG", quality=90)
        depth_file = io.BytesIO()
        np.save(depth_file, np.asarray(depth, dtype=np.float32), allow_pickle=False)
        if rgb_only:
            return {**observation, 'rgb_jpeg_b64': base64.b64encode(image.getvalue()).decode(),
                    'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
                    'provenance': 'mujoco_rgb_sensor', 'source_clock': 'host_monotonic',
                    'instance_id': service.instance_id, 'connection_epoch': service.epoch}
        return {**observation, "rgb_jpeg_b64": base64.b64encode(image.getvalue()).decode(),
                "depth_npy_b64": base64.b64encode(depth_file.getvalue()).decode(),
                "provenance": "mujoco_sensor", "source_clock": "host_monotonic",
                'instance_id':service.instance_id,'connection_epoch':service.epoch}

    @app.get("/v1/simulation/observation", dependencies=auth)
    def observation():
        return pool.submit(capture).result(timeout=30)

    @app.get('/v1/simulation/rgb', dependencies=auth)
    def rgb():
        return pool.submit(capture, True).result(timeout=30)

    @app.post('/v1/simulation/trajectory/preview', dependencies=auth)
    def trajectory_preview(req: TimedTrajectory):
        return service.preview_timed_trajectory(req)

    @app.post('/v1/simulation/trajectory', dependencies=auth)
    def trajectory(req: TimedTrajectoryRequest, request: Request):
        request.state.request_id = req.request_id
        request.state.operation = 'simulation_timed_trajectory'
        result = service.timed_trajectory(req.trajectory, req.request_id)
        request.state.job_id = result.get('job_id')
        return result

    @app.get("/v1/simulation/metadata", dependencies=auth)
    def metadata():
        return backend().metadata()

    @app.get("/operator/simulation/evaluate", dependencies=[Depends(operator_auth)])
    def evaluate():
        return backend().evaluate()

    @app.post("/operator/simulation/reset", dependencies=[Depends(operator_auth)])
    def reset(seed: int = 0):
        with service.lock:
            if service.active:
                raise DomainError("busy", "Reset requires an idle executor")
            service.plans.clear()
            service.lease = None
            service.epoch += 1
            return backend().reset(seed=seed)
