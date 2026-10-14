"""One camera-only subprocess: continuously capture, answer JSON lines, exit on EOF.

The SDK capture thread never imports robot/CAN modules. Frame receipt timestamps
use the same host monotonic clock as the parent, not device-clock arithmetic.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np

if __package__:
    from .live_frames import Frame, FrameRing, checked_roi, positive, select
else:
    # Windows embeddable Python ignores PYTHONPATH and the script directory.
    # The configured camera interpreter may deliberately run in that mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_frames import Frame, FrameRing, checked_roi, positive, select


class Stream:
    def __init__(self, serial=None, sample_hz=3, buffer_s=20, source=None, max_bytes=128 * 1024 * 1024):
        self.serial = serial
        self.sample_hz = positive(sample_hz, 'sample_hz', 15)
        self.max_bytes = max_bytes
        self.ring = FrameRing(seconds=buffer_s, max_frames=min(360, max(2, int(sample_hz*buffer_s)+1)), max_bytes=max_bytes)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.stream_id = uuid.uuid4().hex
        self.source = source  # Injected only by process tests; no CLI fake-camera fallback.
        self.thread = threading.Thread(target=self._capture, name='rgbd-stream', daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(15):
            raise RuntimeError('camera_start_timeout')
        if self.error:
            raise RuntimeError(self.error)
        return self.status()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            raise RuntimeError('camera_shutdown_timeout')

    def _capture(self):
        try:
            if self.source:
                self.source(self)
            else:
                self._realsense()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.ready.set()

    def add(self, rgb, depth, metadata):
        import cv2
        stamp = metadata['received_mono_s']
        encoded, jpeg = cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not encoded:
            raise RuntimeError('Cannot encode RGB')
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), (160, 90))
        with self.lock:
            seq = self.ring.frames[-1].sequence+1 if self.ring.frames else 0
            frame = Frame(seq, stamp, dict(metadata, stream_id=self.stream_id, sequence=seq),
                          jpeg.tobytes(), depth.copy(), gray, float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            self.ring.add(frame)
        self.ready.set()

    def _realsense(self):
        import pyrealsense2 as rs
        devices = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
        if self.serial is None:
            if len(devices) != 1:
                raise ValueError(f'Choose /video start SERIAL; available={devices}')
            self.serial = devices[0]
        if self.serial not in devices:
            raise ValueError(f'Camera unavailable; available={devices}')
        pipeline, config = rs.pipeline(), rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        started = False
        try:
            profile = pipeline.start(config)
            started = True
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
            next_sample, warmed = 0., 0
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                received = time.monotonic()
                warmed += 1
                if warmed < 15 or received < next_sample:
                    continue
                frames = align.process(frames)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                intr = color.profile.as_video_stream_profile().intrinsics
                self.add(np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), {
                    'camera_serial': self.serial, 'received_mono_s': received,
                    'captured_at': datetime.now(timezone.utc).isoformat(),
                    'frame_timestamp_ms': color.get_timestamp(), 'timestamp_domain': str(color.get_frame_timestamp_domain()),
                    'frame': 'camera_color_optical', 'calibration_applied': False,
                    'depth_scale_m': scale, 'intrinsics': {
                        'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy,
                        'ppx': intr.ppx, 'ppy': intr.ppy, 'coeffs': list(intr.coeffs), 'model': str(intr.model)}})
                next_sample = received + 1/self.sample_hz
        finally:
            if started:
                pipeline.stop()

    def status(self):
        with self.lock:
            frames = list(self.ring.frames)
            return {'stream_id': self.stream_id, 'camera_serial': self.serial,
                    'status': 'failed' if self.error else 'streaming', 'error': self.error,
                    'sample_hz': self.sample_hz, 'buffered_frames': len(frames),
                    'buffer_bytes': self.ring.bytes, 'dropped_frames': self.ring.dropped,
                    'latest_age_s': max(0., time.monotonic()-frames[-1].received_mono_s) if frames else None,
                    'robot_motion': False}

    def observe(self, output, *, window_s=2, max_frames=3, roi=None, since_mono_s=None):
        import cv2
        positive(window_s, 'window_s', 120)
        now = time.monotonic()
        if self.error:
            raise RuntimeError(f'camera_failed: {self.error}; explicitly restart /video')
        with self.lock:
            picked, coverage = select(list(self.ring.frames), since_mono_s=now-window_s if since_mono_s is None else since_mono_s,
                                      now_mono_s=now, max_frames=max_frames, roi=roi)
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        records, tiles = [], []
        for frame, role, score in picked:
            folder = output / str(frame.sequence)
            folder.mkdir()
            rgb_path, depth_path = folder/'rgb.jpg', folder/'depth.npy'
            rgb_path.write_bytes(frame.jpeg)
            np.save(depth_path, frame.depth)
            info = dict(frame.metadata, rgb_path=str(rgb_path), depth_path=str(depth_path),
                        role=role, visual_change_score=score,
                        source=frame.metadata.get('camera_serial') or self.serial)
            (folder/'observation.json').write_text(json.dumps(info), encoding='utf-8')
            rgb = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if roi:
                x1, y1, x2, y2 = checked_roi(roi, rgb.shape[1], rgb.shape[0])
                info['crop_path'] = str(folder/'crop.jpg')
                cv2.imwrite(info['crop_path'], rgb[y1:y2, x1:x2])
            tile = cv2.resize(rgb, (480, 270))
            cv2.putText(tile, f'{role} #{frame.sequence} {frame.received_mono_s:.3f}', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
            tiles.append(tile)
            records.append(info)
        sheet = output/'storyboard.jpg'
        cv2.imwrite(str(sheet), np.concatenate(tiles, axis=1))
        result = {'schema': 'piper.observation.v1', 'stream_id': self.stream_id,
                  'source': self.serial,
                  'frames': records, 'coverage': coverage, 'storyboard_path': str(sheet),
                  'selection': 'endpoints + spaced visual changes; camera motion is not compensated',
                  'contact_verified': False, 'calibration_applied': False, 'robot_motion': False}
        (output/'bundle.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        result['bundle_path'] = str(output/'bundle.json')
        # Existing /pixel reads the same observation format, always latest selected.
        result['observation_path'] = str(output/str(records[-1]['sequence'])/'observation.json')
        return result


def serve(stream, input_stream=sys.stdin, output_stream=sys.stdout):
    def emit(result):
        print(json.dumps(result, allow_nan=False), file=output_stream, flush=True)
    try:
        emit(stream.start())
        for line in input_stream:
            try:
                req = json.loads(line)
                command = req.pop('command')
                if command == 'stop':
                    break
                if command == 'status':
                    result = stream.status()
                elif command == 'observe':
                    result = stream.observe(**req)
                else:
                    raise ValueError('Unknown camera command')
                emit(result)
            except Exception as exc:
                emit({'error': {'code': 'camera_error', 'message': str(exc)}})
    except Exception as exc:
        emit({'error': {'code': 'camera_start_failed', 'message': str(exc)}})
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial')
    parser.add_argument('--sample-hz', type=float, default=3)
    parser.add_argument('--buffer-s', type=float, default=20)
    parser.add_argument('--max-bytes', type=int, default=128 * 1024 * 1024)
    args = parser.parse_args()
    serve(Stream(args.serial, args.sample_hz, args.buffer_s, max_bytes=args.max_bytes))


if __name__ == '__main__':
    main()
