"""RGB-only CapturedImage integration for the existing fixed simulation camera."""
from __future__ import annotations
import base64
import io
import time
from types import SimpleNamespace
from PIL import Image
from gpt_policy.hardware.camera import CapturedImage
from .transport import SimulationTransport


class SimulationCameraSet:
    def __init__(self, settings, width=640, height=480, *, transport=None):
        if (width,height) != (640,480):
            raise ValueError('Existing simulation RGB baseline uses 640x480')
        self.transport = transport or SimulationTransport(settings)
        self._owned_transport = transport is None
        self.cameras = [SimpleNamespace(name='top', path='mujoco:overview_camera',
            format_name='RGB JPEG', width=width, height=height, dropped_frames=0)]
        try:
            observation = self.transport.call('GET','/v1/simulation/rgb')
            self._epoch = observation['connection_epoch']
            calibration = observation['camera_info']
            settings.setdefault('vision',{}).setdefault('camera_intrinsics',{})['top'] = calibration['intrinsics']
            settings.setdefault('calibration',{}).setdefault('base_from_camera',{})['top'] = {'left':calibration['base_from_camera']}
            self.calibration_id = calibration['calibration_id']
        except Exception:
            self.close()
            raise

    def capture(self, stop=None):
        if stop is not None and stop.is_set():
            raise InterruptedError('Simulation capture stopped')
        # captured_at is taken before the sensor request. A response to an old
        # request must never satisfy RunVideo.snapshot(after=a newer time).
        requested_at = time.time()
        observation = self.transport.call('GET','/v1/simulation/rgb')
        if observation['connection_epoch'] != self._epoch:
            raise RuntimeError('Simulation connection changed; reopen camera calibration')
        payload = base64.b64decode(observation['rgb_jpeg_b64'],validate=True)
        with Image.open(io.BytesIO(payload)) as source:
            rgb = source.convert('RGB')
            if rgb.size != (640,480):
                raise ValueError('Unexpected simulation image dimensions')
            rgb_data = rgb.tobytes()
        if stop is not None and stop.is_set():
            raise InterruptedError('Simulation capture stopped')
        return {'top':CapturedImage('top',payload,'image/jpeg',640,480,requested_at,
            rgb_data,observation['source_stamp_s'],observation['source_clock'])}

    def describe(self, images=None):
        image = (images or {}).get('top')
        return [{'name':'top','device':'mujoco:overview_camera','format':'RGB JPEG','width':640,'height':480,
            'dropped_frames':0,'calibration_id':self.calibration_id,'fixed':True,
            'depth_available':False, 'wrist_views_available':False,
            **({'captured_at':str(image.captured_at),'source_timestamp_s':image.source_timestamp_s,
                'source_clock':image.source_clock} if image is not None else {})}]

    def close(self):
        if self._owned_transport:
            self.transport.close()
