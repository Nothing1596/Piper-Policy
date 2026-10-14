"""Camera-only worker. Never imports the robot SDK or opens CAN."""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone


def capture(output, serial=None):
    import pyrealsense2 as rs
    import numpy as np
    import cv2
    devices = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
    if serial is None:
        if len(devices) != 1:
            raise ValueError(f'Choose a camera with /observe SERIAL; available={devices}')
        serial = devices[0]
    if serial not in devices:
        raise ValueError(f'Camera {serial} unavailable; available={devices}')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        for _ in range(15):
            frames = pipeline.wait_for_frames(3000)
        frames = rs.align(rs.stream.color).process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise ValueError('Missing RGB or depth frame')
        intr = color.profile.as_video_stream_profile().intrinsics
        rgb_path = output / 'rgb.png'
        if not cv2.imwrite(str(rgb_path), np.asanyarray(color.get_data())):
            raise OSError('Cannot save RGB image')
        np.save(output / 'depth.npy', np.asanyarray(depth.get_data()))
        result = {'camera_serial': serial, 'captured_at': datetime.now(timezone.utc).isoformat(),
                  'depth_path': str(output / 'depth.npy'),
                  'depth_scale_m': profile.get_device().first_depth_sensor().get_depth_scale(),
                  'intrinsics': {'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy,
                                 'ppx': intr.ppx, 'ppy': intr.ppy, 'coeffs': list(intr.coeffs),
                                 'model': str(intr.model)},
                  'frame_timestamp_ms': color.get_timestamp(), 'frame': 'camera_color_optical',
                  'calibration_applied': False, 'robot_motion': False, 'rgb_path': str(rgb_path)}
        (output / 'observation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        if started:
            pipeline.stop()


def pixel(observation, u, v):
    import numpy as np
    info = json.loads(Path(observation).read_text(encoding='utf-8'))
    intr = info['intrinsics']
    if not (0 <= u < intr['width'] and 0 <= v < intr['height']):
        raise ValueError('Pixel outside image')
    if any(intr['coeffs']):
        raise ValueError('Nonzero distortion requires model-aware deprojection; not supported here')
    depth = np.load(info['depth_path'], allow_pickle=False)
    z = float(depth[v, u]) * info['depth_scale_m']
    if not 0 < z < 10:
        raise ValueError('No valid depth at selected pixel; do not substitute background depth')
    return {'pixel': [u, v], 'xyz_m': [(u-intr['ppx'])*z/intr['fx'],
                                     (v-intr['ppy'])*z/intr['fy'], z],
            'frame': 'camera_color_optical', 'captured_at': info['captured_at'],
            'camera_serial': info['camera_serial'], 'calibration_applied': False,
            'note': 'Recorded pixel depth only; not a base-frame motion target or verified grasp point.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--serial')
    parser.add_argument('--observation')
    parser.add_argument('--pixel', type=int, nargs=2)
    args = parser.parse_args()
    try:
        result = pixel(args.observation, *args.pixel) if args.pixel else capture(args.output, args.serial)
        print(json.dumps(result))
    except Exception as exc:
        print(json.dumps({'error': str(exc)}))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
