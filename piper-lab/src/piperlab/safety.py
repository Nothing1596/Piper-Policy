"""Fail-closed command admission. No ROS or hardware imports."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import secrets
import time
from pathlib import Path
from typing import Callable
import numpy as np
import yaml


class Rejected(RuntimeError):
    pass


def load_config(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    names = config['joint_names']
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError('Exactly six unique arm joints required')
    lo, hi = np.array(config['joint_lower']), np.array(config['joint_upper'])
    if lo.shape != (6,) or hi.shape != (6,) or not np.all(np.isfinite([lo, hi])) or np.any(lo >= hi):
        raise ValueError('Invalid joint limits')
    for key in ('joint_speed_rad_s', 'gripper_speed_m_s', 'stale_timeout_s', 'control_hz'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if config['mode'] not in ('simulation', 'real'):
        raise ValueError('mode must be simulation or real')
    for lo, hi in zip(config['workspace_min_m'], config['workspace_max_m'], strict=True):
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError('Invalid workspace bounds')
    if len(config['workspace_min_m']) != 3:
        raise ValueError('Workspace must have three axes')
    grip = config['gripper']
    if not (0 <= grip['min_m'] < grip['max_m'] <= .2):
        raise ValueError('Invalid gripper width limits')
    return config


def commissioning_errors(c):
    if c['mode'] != 'real':
        return []
    missing = []
    if c.get('commissioned') is not True:
        missing.append('commissioned')
    for group, keys in {'can': ['serial', 'firmware'], 'camera': ['serial', 'mount', 'calibration_file'], 'gripper': ['model']}.items():
        for key in keys:
            if not c.get(group, {}).get(key):
                missing.append(f'{group}.{key}')
    if c.get('camera', {}).get('mount') not in ('fixed', 'wrist'):
        missing.append('camera.mount must be fixed or wrist')
    if not c.get('tcp_m_rad') or len(c['tcp_m_rad']) != 6:
        missing.append('tcp_m_rad')
    if not c.get('limits_verified'):
        missing.append('limits_verified')
    for key in ('collision_scene_verified', 'stop_behavior_verified'):
        if c.get(key) is not True:
            missing.append(key)
    for label, path in [('camera.calibration_file', c.get('camera', {}).get('calibration_file')),
                        ('collision_scene_file', c.get('collision_scene_file'))]:
        if not path or not Path(path).expanduser().is_file():
            missing.append(label + ' must exist')
    if c.get('tcp_m_rad') and not np.all(np.isfinite(c['tcp_m_rad'])):
        missing.append('tcp_m_rad must be finite')
    return missing


def verify_can_identity(c, sys_net=Path('/sys/class/net')):
    device=(sys_net/c['can']['interface']/'device').resolve(strict=True)
    for parent in [device,*device.parents]:
        if (parent/'idVendor').is_file() and (parent/'idProduct').is_file():
            identity=(parent/'idVendor').read_text().strip()+':'+(parent/'idProduct').read_text().strip()
            serial=(parent/'serial').read_text().strip() if (parent/'serial').is_file() else ''
            if identity.lower()!=c['can']['usb_vid_pid'].lower() or not serial or serial!=c['can']['serial']:
                raise Rejected('CAN USB identity or serial mismatch')
            return
    raise Rejected('CAN device is not a verified USB adapter')


@dataclass
class SafetyGate:
    config: dict
    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    owner: str | None = None
    session_id: str = ''
    armed: bool = False
    fault: str | None = None
    sequence: int = -1
    measured: np.ndarray | None = None
    state_at: float = -math.inf
    camera_at: float = -math.inf
    last_command_at: float = -math.inf
    applied_at: float = -math.inf
    applied: np.ndarray | None = None
    events: list = field(default_factory=list)

    @property
    def names(self):
        return self.config['joint_names'] + ['gripper']

    def trip(self, reason):
        was_armed = self.armed
        self.armed = False
        self.fault = reason
        self.owner = None
        self.session_id = ''
        self.sequence = -1
        self.applied = None
        self.events.append({'kind': 'stop', 'reason': reason, 'was_armed': was_armed, 'timestamp': self.wall_clock()})

    def _fresh_stamp(self, stamp):
        age = self.wall_clock() - stamp
        return math.isfinite(age) and -self.config.get('max_clock_skew_s', 0.05) <= age <= self.config['stale_timeout_s']

    def observe(self, names, positions, stamp):
        if len(names) != len(set(names)) or len(names) != len(positions) or not set(self.names).issubset(names):
            self.trip('invalid_joint_feedback')
            return False
        vals = np.array([positions[names.index(n)] for n in self.names], dtype=float)
        if not np.all(np.isfinite(vals)) or not self._fresh_stamp(stamp):
            self.trip('invalid_or_stale_feedback')
            return False
        self.measured = vals
        # Preserve source age; receiving an old frame does not make it fresh.
        self.state_at = self.clock() - max(0.0, self.wall_clock() - stamp)
        return True

    def observe_camera(self, stamp):
        if not self._fresh_stamp(stamp):
            if self.armed:
                self.trip('stale_camera')
            return False
        self.camera_at = self.clock() - max(0.0, self.wall_clock() - stamp)
        return True

    def _observations_ready(self):
        return self.measured is not None and self.clock() - min(self.state_at, self.camera_at) <= self.config['stale_timeout_s']

    def control(self, operation, owner, token=''):
        if operation == 'stop':
            self.trip('manual_stop')
            return ''
        if operation == 'acquire':
            if owner not in ('teleop', 'policy', 'moveit'):
                raise Rejected('Unknown control owner')
            if self.fault:
                raise Rejected('Fault must be reset first: ' + self.fault)
            if self.owner:
                raise Rejected('Control already held')
            self.owner, self.session_id = owner, secrets.token_hex(16)
            self.sequence = -1
            return self.session_id
        if operation == 'reset':
            if self.armed or self.owner:
                raise Rejected('Release control before reset')
            if not self._observations_ready():
                raise Rejected('Fresh feedback and camera required')
            self.fault = None
            return ''
        if token != self.session_id or owner != self.owner or not token:
            raise Rejected('Control session mismatch')
        if operation == 'release':
            self.trip('operator_release')
            return ''
        if operation != 'arm':
            raise Rejected('Unknown operation')
        errors = commissioning_errors(self.config)
        if errors:
            raise Rejected('Hardware not commissioned: ' + ', '.join(errors))
        if self.fault or not self._observations_ready():
            raise Rejected('Fresh observations and cleared fault required')
        self.armed = True
        self.last_command_at = self.clock()
        self.applied_at = self.clock()
        self.applied = self.measured.copy()
        return self.session_id

    def tick(self):
        if self.armed and (not self._observations_ready() or self.clock() - self.last_command_at > self.config['stale_timeout_s']):
            self.trip('watchdog_timeout')
            return True
        return False

    def admit(self, owner, token, sequence, names, positions, stamp):
        self.tick()
        if not self.armed or self.fault:
            raise Rejected('Motion is disabled')
        if owner != self.owner or token != self.session_id or not token:
            raise Rejected('Control session mismatch')
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= self.sequence:
            raise Rejected('Repeated or invalid sequence')
        if not self._fresh_stamp(stamp):
            self.trip('stale_command')
            raise Rejected('Stale command')
        if names != self.names or len(positions) != 7:
            self.trip('joint_mapping_mismatch')
            raise Rejected('Joint mapping mismatch')
        target = np.asarray(positions, dtype=float)
        lo = np.r_[self.config['joint_lower'], self.config['gripper']['min_m']]
        hi = np.r_[self.config['joint_upper'], self.config['gripper']['max_m']]
        if not np.all(np.isfinite(target)) or np.any(target < lo) or np.any(target > hi):
            self.trip('invalid_target')
            raise Rejected('Non-finite or out-of-range target')
        dt = min(1.0 / self.config['control_hz'], max(0, self.clock() - self.applied_at))
        speed = np.r_[np.full(6, self.config['joint_speed_rad_s']), self.config['gripper_speed_m_s']]
        limited = self.applied + np.clip(target - self.applied, -speed * dt, speed * dt)
        # Do not permit accumulated target lead if feedback stalls at one pose.
        limited = self.measured + np.clip(limited - self.measured, -speed * 0.2, speed * 0.2)
        self.sequence = sequence
        self.last_command_at = self.clock()
        return limited

    def mark_applied(self, positions):
        if not self.armed:
            raise Rejected('Gate closed before application')
        self.applied = np.asarray(positions, dtype=float).copy()
        self.applied_at = self.clock()
