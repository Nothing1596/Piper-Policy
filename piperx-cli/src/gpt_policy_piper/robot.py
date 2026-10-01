"""Piper adapter retaining upstream Cartesian sampling, continuous IK and timing."""
from __future__ import annotations
import math
import time
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from piperx_middleware.kinematics import PiperKinematics
from piperx_middleware.models import DomainError, JOINT_LIMITS_DEG
from gpt_policy.geometry.frames import FrameCalibration
from gpt_policy.geometry.poses import rpy_to_quaternion
from gpt_policy.hardware.motion_control import MotionControl, MotionFault
from gpt_policy.motion.ik import ContinuousIK
from gpt_policy.motion.planner import EefTrajectoryPlanner
from gpt_policy.motion.trajectory import MotionLimits, retime_path_segment
from gpt_policy.motion.coordination import path_check_result
from .transport import SimulationTransport, BridgeError


class PiperSolver:
    """Expose the existing Piper MDH solver in upstream's native radian contract."""
    def __init__(self):
        self.kinematics = PiperKinematics()

    def forward_kinematics(self, joints_rad):
        pose = self.kinematics.pose(np.rad2deg(joints_rad).tolist())
        return np.r_[pose['xyz_m'], np.deg2rad(pose['rpy_deg'])]

    def multi_trial_ik(self, pose, seed, trials):
        try:
            answer = self.kinematics.solve(pose[:3], np.rad2deg(pose[3:]), np.rad2deg(seed), nearby=True)
            return 0, np.deg2rad(answer['joints_deg'])
        except DomainError:
            return -1, np.asarray(seed).copy()

    @staticmethod
    def get_ik_status_name(status):
        return 'piper_multistart_converged' if status == 0 else 'piper_no_verified_solution'


class PiperIK(ContinuousIK):
    name = 'piper_mdh_multistart_ik + upstream_continuous_dls_refinement'


class PiperRobot:
    """Single simulated arm; physical transports and raw gain/torque are excluded."""
    dof = 6

    def __init__(self, model='PiperX', interface='mujoco', gripper_open_readout=None,
                 trajectory_hz=30., settings=None, *, transport=None):
        settings = settings or {}
        if settings.get('backend') != 'piper_mujoco' or settings.get('runtime', {}).get('right_interface'):
            raise ValueError('Piper baseline requires backend=piper_mujoco and one arm')
        self.motion = MotionControl()
        self.transport = transport or SimulationTransport(settings, control=True)
        self._owned_transport = transport is None
        self.settings = settings
        self.trajectory_hz = float(trajectory_hz)
        self._last_settle_report = {}
        self._active_job = None
        self._gripper_target = None
        self._arm_target = None
        try:
            parameters = self.transport.call('GET', '/v1/parameters')['configured']
            self.width_m = float(parameters['gripper_max_m'])
            self.frames = FrameCalibration(settings)
            expected = PiperKinematics(parameters['tcp_offset_m'], parameters['tcp_offset_rpy_deg'])._tcp_offset
            if not np.allclose(self.frames.sdk_eef_from_tcp, expected, atol=1e-9):
                raise ValueError('Upstream TCP calibration must match the simulator runtime TCP transform')
            m = settings.get('motion', {})
            self.motion_limits = MotionLimits(
                output_hz=self.trajectory_hz,
                cartesian_step_m=float(m.get('cartesian_step_m', .005)),
                cartesian_step_rad=float(m.get('cartesian_step_rad', .035)),
                tcp_velocity_m_s=float(m.get('tcp_velocity_m_s', .08)),
                tcp_angular_velocity_rad_s=float(m.get('tcp_angular_velocity_rad_s', .5)),
                joint_velocity_rad_s=np.full(6, math.pi)*float(m.get('joint_velocity_scale', .25)),
                joint_acceleration_rad_s2=np.broadcast_to(m.get('joint_acceleration_rad_s2', 2.), (6,)).copy(),
                joint_jerk_rad_s3=np.broadcast_to(m.get('joint_jerk_rad_s3', 12.), (6,)).copy())
            self.ik = PiperIK(PiperSolver(), np.deg2rad(np.array(JOINT_LIMITS_DEG)[:,0]),
                np.deg2rad(np.array(JOINT_LIMITS_DEG)[:,1]), int(m.get('ik_refine_iterations',12)),
                float(m.get('ik_translation_tolerance_m',1e-4)), float(m.get('ik_rotation_tolerance_rad',5e-4)),
                float(m.get('ik_execution_translation_tolerance_m',.002)),
                float(m.get('ik_execution_rotation_tolerance_rad',math.radians(1.))))
            self._raw_state()
            self.planner = EefTrajectoryPlanner(self._read_positions, self.frames, self.ik, self.trajectory_hz,
                self.motion_limits, float(m.get('endpoint_hold_s',.12)))
        except Exception:
            self.close()
            raise

    def _raw_state(self):
        self.motion.check()
        state = self.transport.call('GET', '/v1/state')
        if not state.get('ready') and not (state.get('active_job_id') and state.get('not_ready_reason',{}).get('code') == 'velocity_limit'):
            raise MotionFault({'reason':'simulation_not_ready', 'details':state.get('not_ready_reason')})
        robot = state['robot']
        if not robot['connected'] or robot.get('feedback_age_s', 1.) > .2:
            raise MotionFault({'reason':'stale_simulation_feedback'})
        if self._gripper_target is None:
            self._gripper_target = float(robot['gripper_width_m'])/self.width_m
        return state

    def _read_positions(self):
        return np.deg2rad(self._raw_state()['robot']['q_deg'])

    def state(self):
        raw = self._raw_state()
        feedback = raw['robot']
        q = np.deg2rad(feedback['q_deg'])
        flange = self.ik.solver.forward_kinematics(q)
        tcp = self.frames.sdk_to_tcp(flange)
        return {'joint_positions_rad':q.tolist(), 'joint_velocities_rad_s':np.deg2rad(feedback['velocity_deg_s']).tolist(),
                'joint_torques_nm': None, 'tcp_xyzrpy':tcp.tolist(),
                'tcp_xyzquat':[*tcp[:3].tolist(), *rpy_to_quaternion(tcp[3:])],
                'sdk_eef_xyzrpy':flange.tolist(), 'gripper_position_m':feedback['gripper_width_m'],
                'gripper_normalized':feedback['gripper_width_m']/self.width_m,
                'gripper_command_normalized':self._gripper_target, 'gripper_velocity_m_s':None, 'gripper_torque_nm':None,
                'timestamp_s':feedback['feedback_stamp_s'], 'timestamp_source':'executor_host_monotonic',
                'simulation':True, 'physical_validation':False, 'unavailable_feedback':['joint_torques_nm', 'gripper_torque_nm'],
                'observation_identity':raw['observation_identity']}

    def plan_eef_trajectory(self, requested, note):
        raw = self._raw_state()
        positions = np.deg2rad(raw['robot']['q_deg'])
        # One immutable measured seed for both the planner and admission check.
        planner = EefTrajectoryPlanner(lambda: positions.copy(), self.frames, self.ik, self.trajectory_hz,
            self.motion_limits, self.planner.endpoint_hold_s)
        plan = planner.plan(requested, note, self._gripper_target)
        plan['simulation_identity'] = {k:raw[k] for k in ('instance_id','connection_epoch','parameter_version')}
        plan['start_gripper_m'] = raw['robot']['gripper_width_m']
        return plan

    def _timeline(self, plan):
        m = self.settings.get('motion', {})
        return {**plan['simulation_identity'], 'start_joints_deg':np.rad2deg(plan['start_joint_positions_rad']).tolist(),
                'start_gripper_m':plan['start_gripper_m'], 'joints_deg':np.rad2deg(plan['joint_positions_rad']).tolist(),
                'relative_times_s':list(plan['relative_times_s']), 'note':plan['result']['note'],
                'control_hz':self.trajectory_hz,
                'settle_timeout_s':float(m.get('settle_timeout_s',3.)),
                'settle_position_rad':float(m.get('settle_position_tolerance_rad',.03)),
                'settle_velocity_rad_s':float(m.get('settle_velocity_tolerance_rad_s',.05)),
                'settle_samples':int(m.get('settle_samples',10)),
                'tracking_error_rad':float(m.get('tracking_error_limit_rad',.12))}

    def _submit(self, path, body, timeout):
        self.motion.check()
        request_id = 'gpt-policy-' + uuid.uuid4().hex
        body = {**body, 'request_id':request_id}
        try:
            job = self.motion.send(self.transport.call, 'POST', path, body)
        except BridgeError as exc:
            if exc.result.get('error',{}).get('code') not in ('transport_unknown','invalid_response'):
                raise
            # Read the same durable request only. A missing record is still an
            # unknown write outcome, never permission to POST it again.
            try:
                job = self.transport.call('GET', '/v1/requests/'+request_id)
            except Exception as lookup_error:
                self.fault_stop({'reason':'submission_outcome_unknown', 'request_id':request_id})
                raise MotionFault(self.motion.fault) from lookup_error
        self._active_job = job['job_id']
        deadline = time.monotonic()+timeout+3.
        while job['status'] in ('accepted','running'):
            self.motion.check()
            if time.monotonic() >= deadline:
                self.fault_stop({'reason':'execution_outcome_unknown', 'request_id':request_id})
                raise MotionFault(self.motion.fault)
            time.sleep(.02)
            try:
                job = self.transport.call('GET','/v1/requests/'+request_id)
            except BridgeError:
                continue  # read-only polling cannot replay a robot action
        self._active_job = None
        if job['status'] != 'succeeded':
            self.fault_stop({'reason':'simulation_action_failed', 'request_id':request_id,
                             'status':job['status'], 'error':job.get('error'), 'error_code':job.get('error_code')})
            raise MotionFault(self.motion.fault)
        self._last_settle_report = job.get('settle', {'settled':True, 'source':'middleware_fresh_feedback'})
        return job

    def send_eef_trajectory(self, plan, **kwargs):
        timeline = self._timeline(plan)
        job = self._submit('/v1/simulation/trajectory', {'trajectory':timeline},
            timeline['relative_times_s'][-1]+timeline['settle_timeout_s']+1.)
        self._arm_target = np.asarray(plan['joint_positions_rad'][-1]).copy()
        plan['execution_job'] = {k:job[k] for k in ('job_id','request_id','execution_timing','settle')}
        return job

    def execute(self, name, arguments):
        self.motion.check()
        if name == 'state':
            return self.state()
        if name in ('move_to','move_eef_chunk','check_path'):
            requested = [arguments['target']] if name == 'move_to' else arguments['poses']
            start = self.state()
            plan = self.plan_eef_trajectory(requested, arguments['note'])
            if name == 'check_path':
                self.transport.call('POST','/v1/simulation/trajectory/preview',self._timeline(plan))
                return path_check_result({'arm':plan})
            self.send_eef_trajectory(plan)
            result = self.state()
            target = np.asarray(plan['result']['_trace']['model_tcp_points_xyzrpy'][-1])
            actual = np.asarray(result['tcp_xyzrpy'])
            result['trajectory'] = plan['result']
            result['execution_feedback'] = {'settle':self._last_settle_report,
                'target_tcp_xyzrpy':target.tolist(), 'measured_tcp_xyzrpy':actual.tolist(),
                'tcp_translation_error_m':float(np.linalg.norm(target[:3]-actual[:3])),
                'tcp_rotation_error_rad':float((Rotation.from_euler('xyz',target[3:]).inv()*Rotation.from_euler('xyz',actual[3:])).magnitude()),
                'target_joint_positions_rad':self._arm_target.tolist(), 'execution_job':plan['execution_job'],
                'simulation':True, 'task_success_evaluated':False}
            return result
        if name == 'set_gripper':
            value = arguments.get('gripper', arguments.get('position', arguments.get('positions',{}).get('left')))
            if value is None or not np.isfinite(value) or not 0 <= float(value) <= 1:
                raise ValueError('Gripper position must be in [0,1]')
            width = float(value)*self.width_m
            job = self._submit('/v1/move', {'command':{'kind':'gripper','width_m':width,
                'effort_protocol':.5,'timeout_s':10.,
                'completion':'width_or_bilateral_contact' if width < self.state()['gripper_position_m'] else 'width'}}, 10.)
            self._gripper_target = float(value)
            result = self.state()
            result['execution_feedback'] = {'gripper':{'target_normalized':float(value),
                'measured_normalized':result['gripper_normalized'], 'completion':'width_feedback',
                'contact_confirmed':'contact_evidence' in job,
                'force_evidence':job.get('contact_evidence')}, 'job_id':job['job_id'], 'task_success_evaluated':False}
            return result
        if name == 'home':
            return self.return_home()
        raise ValueError('Unsupported Piper simulation tool: '+name)

    def return_home(self):
        raw = self._raw_state()
        start = np.deg2rad(raw['robot']['q_deg'])
        target = np.deg2rad([0.,30.,-30.,0.,0.,0.])
        fractions = np.linspace(0.,1.,max(3,int(np.max(np.abs(target-start))*self.trajectory_hz)+1))
        joints = start[None,:]+fractions[:,None]*(target-start)[None,:]
        timing = retime_path_segment(fractions,joints,0.,0.,self.motion_limits)
        plan = {'simulation_identity':{k:raw[k] for k in ('instance_id','connection_epoch','parameter_version')},
            'start_joint_positions_rad':start, 'start_gripper_m':raw['robot']['gripper_width_m'],
            'joint_positions_rad':joints[1:], 'relative_times_s':timing.times_s[1:].tolist(),
            'result':{'note':'Return to existing Piper simulation home pose'}}
        self.send_eef_trajectory(plan)
        arm_settle = dict(self._last_settle_report)
        self.execute('set_gripper',{'position':1.})
        result = self.state()
        result['home'] = {'source':'piper_simulation_default_pose','target_joint_positions_rad':target.tolist(),
            'settle':arm_settle,'capability_deviation':'Piper home replaces ARX-only home geometry'}
        return result

    def wait_reference_complete(self):
        self.motion.check()  # submissions above already wait for durable completion

    def clock(self):
        return float(self._raw_state()['robot']['feedback_stamp_s'])

    def cancel(self):
        self.motion.cancel(lambda:self.transport.call('POST','/v1/stop'))
        if self._active_job:
            deadline = time.monotonic()+3.
            while time.monotonic() < deadline:
                job = self.transport.call('GET','/v1/jobs/'+self._active_job)
                if job['status'] not in ('accepted','running','awaiting_approval'):
                    if job['status'] == 'outcome_unknown':
                        self.motion.fault = {'reason':'cancel_outcome_unknown','job_id':self._active_job}
                    self._active_job = None
                    return
                time.sleep(.02)
            raise RuntimeError('Previous simulation worker did not finish cancellation')

    def resume(self):
        if self._active_job:
            job = self.transport.call('GET','/v1/jobs/'+self._active_job)
            if job['status'] in ('accepted','running','awaiting_approval'):
                raise RuntimeError('Previous action has not finished cancellation')
            self._active_job = None
        self.motion.resume()

    def fault_stop(self, details):
        self.motion.fault = details
        self.cancel()

    def close(self):
        if getattr(self,'_owned_transport',False):
            self.transport.close()
