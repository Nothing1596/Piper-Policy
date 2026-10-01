"""Narrow, real Windows OS adaptations used by the pinned external checkout."""
from __future__ import annotations
import os
import errno
import subprocess
import threading
import time


def acquire_cache_lock(fd):
    """Hold an exclusive interprocess lock until this descriptor is closed."""
    if os.name != 'nt':
        import fcntl
        fcntl.flock(fd,fcntl.LOCK_EX)
        return
    import msvcrt
    if os.fstat(fd).st_size == 0:
        os.write(fd,b'0')
    os.lseek(fd,0,os.SEEK_SET)
    deadline = time.monotonic()+300.
    while True:
        try:
            msvcrt.locking(fd,msvcrt.LK_NBLCK,1)
            return
        except OSError as exc:
            native_error = getattr(exc,'winerror',None)
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK) or (native_error is not None and native_error not in (33,36)):
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError('Video cache lock remained held for 300 seconds') from exc
            time.sleep(.05)


def write_windows_pipe(process, data, deadline):
    """Bound a blocking Windows anonymous-pipe write without POSIX select()."""
    result = []
    finished = threading.Event()
    def write():
        try:
            process.stdin.write(data)
            process.stdin.flush()
        except Exception as exc:
            result.append(exc)
        finally:
            finished.set()
    thread = threading.Thread(target=write,name='gpt-policy-pipe-writer',daemon=True)
    thread.start()
    if not finished.wait(max(0.,deadline-time.monotonic())):
        close_windows_process(process)
        thread.join(timeout=2)
        raise TimeoutError('Agent pipe write deadline reached')
    if result:
        raise result[0]


def close_windows_process(process):
    """Terminate the launched process tree while its root identity still exists."""
    if process.poll() is None:
        try:
            result = subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                timeout=10,check=False)
        except (subprocess.TimeoutExpired, OSError) as exc:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError('Windows agent tree cleanup is unconfirmed') from exc
        if result.returncode and process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def calibration_notes(arms, settings):
    return '''Robot and calibration conventions:
- One PiperX simulation arm has six revolute joints. GPT-Policy joints and RPY remain radians; only the middleware transport uses degrees.
- TCP equals the configured Piper flange-to-tool transform, verified against middleware parameters. The host seeds upstream continuous IK with measured simulation joints.
- Absolute TCP pose_xyzquat uses metres and quaternion xyzw in the arm base frame, +x forward, +y left, +z up.
- The unchanged upstream planner samples straight-line/SLERP segments, solves every sample and uses scalar-path Ruckig retiming. The host streams the full immutable timeline.
- top is the existing fixed MuJoCo overview RGB camera. Intrinsics and fixed base extrinsics come from its calibrated sensor interface.
- No wrist camera, depth image or hidden object pose is provided. locate_point gives a calibrated ray from one fixed RGB view; moving the arm does not create camera parallax.
- Simulation feedback, settling and force contacts do not establish grasp, lift or task success. Inspect fresh visible evidence.
'''


def close_windows_json_process(owner):
    try:
        close_windows_process(owner.process)
    finally:
        for thread in owner._readers:
            thread.join(timeout=2)
        if any(thread.is_alive() for thread in owner._readers):
            raise RuntimeError('Agent descendants still hold pipes; process cleanup is unconfirmed')
        for stream in (owner.process.stdin,owner.process.stdout,owner.process.stderr):
            if stream is not None:
                stream.close()
