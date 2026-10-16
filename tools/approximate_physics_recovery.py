"""Experiment-only pose initialization, deliberately NOT a physics checkpoint."""
import hashlib
import json
from pathlib import Path

import numpy as np


def apply_recorded_pose(backend, recovery):
    """Before any worker starts, restore recorded poses and measured robot hold.

    Only world-parented free objects are supported. Unknown velocities, solver
    state and finger asymmetry are not inferred from a few recorded poses.
    """
    import mujoco
    if backend.connected or backend.worker_thread is not None:
        raise ValueError('Pose initialization must precede simulation connection')
    path = Path(recovery['snapshot'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != recovery['snapshot_sha256']:
        raise ValueError('Recovery snapshot changed')
    state = json.loads(raw)
    model, data = backend.model, backend.data
    if set(state['objects']) != set(backend.object_ids):
        raise ValueError('Recovery object identities differ')
    control = None
    if state.get('bilateral_contact_objects'):
        # A measured opening alone removes the closing preload. Only an audited,
        # previously successful gripper target may initialize this approximation;
        # this does not issue a gripper request or certify retained contact.
        descriptor = recovery.get('gripper_controller_record')
        if not descriptor:
            raise ValueError('Held-object recovery needs the recorded successful gripper target')
        control_raw = Path(descriptor['path']).read_bytes()
        if hashlib.sha256(control_raw).hexdigest() != descriptor['sha256']:
            raise ValueError('Gripper controller record changed')
        result = json.loads(control_raw)['result']
        command = result.get('command', {})
        if result.get('status') != 'succeeded' or command.get('kind') != 'gripper':
            raise ValueError('Gripper controller record is not a successful gripper target')
        control = float(command['width_m'])
        if not np.isfinite(control) or not 0 <= control <= .07:
            raise ValueError('Invalid recorded gripper controller target')
    q = np.deg2rad(np.asarray(state['robot']['q_deg'], dtype=float))
    width = float(state['robot']['gripper_width_m'])
    if q.shape != (6,) or not np.isfinite(q).all() or not 0 <= width <= .07:
        raise ValueError('Invalid measured robot pose')
    # Validate every object before changing any state.
    changes = []
    for name, body in backend.object_ids.items():
        row = state['objects'][name]
        pose = np.asarray(row['position'] + row['quaternion'], dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-5:
            raise ValueError('Invalid recorded pose: '+name)
        if row['kind'] == 'dynamic':
            joint = int(model.body_jntadr[body])
            if (model.body_parentid[body] != 0 or model.body_jntnum[body] != 1
                    or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError('Recovery requires a world-parented free object: '+name)
            changes.append((int(model.jnt_qposadr[joint]), pose))
        else:
            if (not np.allclose(data.xpos[body], pose[:3], atol=1e-7, rtol=0)
                    or abs(np.dot(data.xquat[body], pose[3:])) < 1-1e-7):
                raise ValueError('Static object differs; unsupported recovery: '+name)
    for index, pose in changes:
        data.qpos[index:index+7] = pose
    data.qpos[backend.arm_qpos_indices] = q
    data.qpos[backend.gripper_qpos_indices] = width/2
    data.qvel[:] = 0
    data.qacc_warmstart[:] = 0
    data.ctrl[backend.arm_act_indices] = q
    data.ctrl[backend.gripper_act_indices] = (width if control is None else control)/2
    data.time = 0
    backend.arm_goal_rad = q.copy()
    backend.reference_velocity[:] = 0
    backend.active_target = False
    mujoco.mj_forward(model, data)
    return {'scope': 'approximate_pose_recovery_not_continuous_execution',
            'source_sim_time_s': state['sim_time_s'], 'new_sim_time_s': 0,
            'snapshot_sha256': recovery['snapshot_sha256'],
            'restored': ['world object poses', 'measured arm joints', 'symmetric measured gripper width'],
            'initialized': {'qvel': 'zero, not measured', 'arm_ctrl': 'hold measured pose',
                            'gripper_ctrl': 'hold measured width' if control is None else 'last recorded successful target',
                            'gripper_ctrl_width_m': width if control is None else control,
                            'warmstart': 'cold', 'simulation_clock': 'new zero'},
            'not_restored': ['joint and object velocities', 'full original actuator state',
                             'controller history', 'contact solver history', 'individual finger asymmetry'],
            'actions_replayed': 0}
