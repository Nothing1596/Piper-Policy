#!/usr/bin/env python3
"""ROS control admission, watchdog, explicit ownership and MoveIt validation."""
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, CompressedImage
from geometry_msgs.msg import TwistStamped
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import String
from std_srvs.srv import Empty, SetBool, Trigger
from moveit_msgs.srv import GetPositionFK, GetStateValidity, ServoCommandType
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import yaml
from agx_arm_msgs.msg import AgxArmStatus
from piperlab_msgs.msg import Command, Jog
from piperlab_msgs.srv import Control
from piperlab.safety import SafetyGate, Rejected, load_config

def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

class Runtime(Node):
    def __init__(self):
        super().__init__('piper_lab_runtime')
        self.declare_parameter('config', '/home/ros/piper-lab/config/hardware.yaml')
        self.c = load_config(self.get_parameter('config').value)
        self.gate = SafetyGate(self.c, wall_clock=lambda:self.get_clock().now().nanoseconds/1e9)
        self.group = ReentrantCallbackGroup()
        self.pending = None
        self.pending_since = 0.
        self.arming = False
        self.arm_status_at = -float('inf')
        self.jog_sequence=-1; self.servo_sequence=0; self.last_jog=-float('inf')
        self.gripper_target=.05
        self.out = self.create_publisher(JointState, '/piper/control/joint_states', 1)
        self.observation = self.create_publisher(JointState, '/lab/observation', 1)
        self.applied = self.create_publisher(JointState, '/lab/applied_action', 1)
        self.event = self.create_publisher(String, '/lab/events', 20)
        self.status = self.create_publisher(String, '/lab/status', 1)
        self.twist = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 1)
        self.create_subscription(JointState, '/piper/feedback/joint_states', self.feedback, 1)
        self.create_subscription(AgxArmStatus, '/piper/feedback/arm_status', self.arm_status, 1)
        self.create_subscription(CompressedImage, '/lab/rgb', self.camera, qos_profile_sensor_data)
        self.create_subscription(Command, '/lab/command', self.command, 1)
        self.create_subscription(Jog, '/lab/jog', self.jog, 1)
        self.create_subscription(JointTrajectory, '/lab/servo_targets', self.servo_target, 1)
        self.create_service(Control, '/lab/control', self.control, callback_group=self.group)
        self.stop_client=self.create_client(Empty,'/piper/emergency_stop',callback_group=self.group)
        self.enable_client=self.create_client(SetBool,'/piper/enable_agx_arm',callback_group=self.group)
        self.driver_gate=self.create_client(SetBool,'/piper/control_enable',callback_group=self.group)
        self.servo_pause=self.create_client(SetBool,'/servo_node/pause_servo',callback_group=self.group)
        self.servo_mode=self.create_client(ServoCommandType,'/servo_node/switch_command_type',callback_group=self.group)
        self.fk=self.create_client(GetPositionFK,'/compute_fk',callback_group=self.group)
        self.valid=self.create_client(GetStateValidity,'/check_state_validity',callback_group=self.group)
        self.scene=self.create_client(ApplyPlanningScene,'/apply_planning_scene',callback_group=self.group)
        self.create_timer(.02,self.watchdog)
        self.create_timer(.1,self.publish_status)
        self.get_logger().info(f'Piper Lab {self.c["mode"]}: control disabled')

    def feedback(self,msg):
        if self.gate.observe(list(msg.name),list(msg.position),stamp_s(msg.header.stamp)):
            out=JointState(); out.header=msg.header; out.name=self.gate.names
            out.position=self.gate.measured.tolist(); self.observation.publish(out)
    def camera(self,msg):
        self.gate.observe_camera(stamp_s(msg.header.stamp))
    def arm_status(self,msg):
        self.arm_status_at = time.monotonic()
        if msg.arm_status or msg.err_status or any(msg.joint_angle_limit) or any(msg.communication_status_joint):
            self.stop('hardware_fault')
    def stop(self,reason):
        self.gate.trip(reason)
        self.pending=None
    def watchdog(self):
        self.gate.tick()
        if self.c['mode']=='real' and self.gate.armed and time.monotonic()-self.arm_status_at > .2:
            self.stop('hardware_status_expired')
        if self.pending and time.monotonic()-self.pending_since > self.c['stale_timeout_s']:
            self.stop('moveit_validation_timeout')
        while self.gate.events:
            event=self.gate.events.pop(0)
            self.event.publish(String(data=json.dumps(event)))
            if event['kind']=='stop':
                self.pending=None
                zero=TwistStamped(); zero.header.stamp=self.get_clock().now().to_msg(); self.twist.publish(zero)
                if self.servo_pause.service_is_ready(): self.servo_pause.call_async(SetBool.Request(data=True))
                if self.stop_client.service_is_ready(): self.stop_client.call_async(Empty.Request())
                if self.driver_gate.service_is_ready(): self.driver_gate.call_async(SetBool.Request(data=False))
    def publish_status(self):
        self.status.publish(String(data=json.dumps({'mode':self.c['mode'],'armed':self.gate.armed,
            'owner':self.gate.owner,'fault':self.gate.fault,'state':None if self.gate.measured is None else self.gate.measured.tolist(),
            'wall_time':self.gate.wall_clock(),'synthetic':self.c['mode']=='simulation'})))

    async def call(self,client,request):
        if not client.service_is_ready(): raise Rejected('Required service unavailable: '+client.srv_name)
        future=client.call_async(request)
        timer=self.create_timer(5.,lambda:future.cancel(),callback_group=self.group)
        try:
            result=await future
            if result is None or (hasattr(result,'success') and not result.success):
                raise Rejected('Service rejected or timed out: '+client.srv_name)
            return result
        finally:
            timer.cancel(); self.destroy_timer(timer)

    async def control(self,request,response):
        try:
            if request.operation=='arm':
                if self.arming: raise Rejected('Arming is already in progress')
                self.arming=True
                if self.c['mode']=='real' and time.monotonic()-self.arm_status_at > .2:
                    raise Rejected('Fresh hardware status required')
                # Validate without enabling hardware; keep callbacks/watchdog active during service calls.
                self.gate.control('arm',request.owner,request.session_id)
                self.gate.armed=False
                if self.c['mode']=='real':
                    with open(self.c['collision_scene_file'],encoding='utf-8') as f: scene=yaml.safe_load(f)
                    req=ApplyPlanningScene.Request(); req.scene.is_diff=True
                    if not scene.get('boxes'): raise Rejected('Measured collision scene is empty')
                    for box in scene['boxes']:
                        size=box['size_m']; pos=box['position_m']
                        if len(size)!=3 or len(pos)!=3 or not np.all(np.isfinite(size+pos)) or min(size)<=0:
                            raise Rejected('Invalid collision box')
                        obj=CollisionObject(); obj.header.frame_id='base_link'; obj.id=str(box['name']); obj.operation=CollisionObject.ADD
                        primitive=SolidPrimitive(); primitive.type=SolidPrimitive.BOX; primitive.dimensions=[float(v) for v in size]
                        pose=Pose(); pose.position.x=float(pos[0]); pose.position.y=float(pos[1]); pose.position.z=float(pos[2]); pose.orientation.w=1.
                        obj.primitives=[primitive]; obj.primitive_poses=[pose]; req.scene.world.collision_objects.append(obj)
                    await self.call(self.scene,req)
                await self.call(self.enable_client,SetBool.Request(data=True))
                # Closing/releasing the session while enable was pending must prevent motion.
                if request.session_id != self.gate.session_id or self.gate.fault:
                    raise Rejected('Control session cancelled during enable')
                if request.owner=='teleop':
                    await self.call(self.servo_mode,ServoCommandType.Request(command_type=1))
                    await self.call(self.servo_pause,SetBool.Request(data=False))
                await self.call(self.driver_gate,SetBool.Request(data=True))
                self.jog_sequence=-1; self.servo_sequence=0; self.last_jog=-float('inf')
            response.session_id=self.gate.control(request.operation,request.owner,request.session_id)
            response.accepted=True; response.reason='ok'
            self.event.publish(String(data=json.dumps({'kind':'control','operation':request.operation,'owner':request.owner,'timestamp':self.gate.wall_clock()})))
        except Exception as exc:
            if request.operation=='arm': self.stop('arm_failed')
            response.accepted=False; response.reason=str(exc)
        finally:
            if request.operation=='arm': self.arming=False
        return response

    def jog(self,msg):
        if not self.gate.armed or self.gate.owner!='teleop' or msg.session_id!=self.gate.session_id:
            return
        if not msg.deadman:
            self.stop('deadman_released'); return
        if msg.sequence<=self.jog_sequence or not self.gate._fresh_stamp(stamp_s(msg.header.stamp)):
            self.stop('invalid_jog'); return
        values=[msg.twist.linear.x,msg.twist.linear.y,msg.twist.linear.z,msg.twist.angular.x,msg.twist.angular.y,msg.twist.angular.z,msg.gripper_width]
        if not np.all(np.isfinite(values)):
            self.stop('nonfinite_jog'); return
        self.jog_sequence=msg.sequence; self.last_jog=time.monotonic()
        self.gate.last_command_at=self.last_jog
        self.gripper_target=float(np.clip(msg.gripper_width,self.c['gripper']['min_m'],self.c['gripper']['max_m']))
        twist=TwistStamped(); twist.header.stamp=self.get_clock().now().to_msg(); twist.header.frame_id='base_link'
        for field,src,limit in ((twist.twist.linear,msg.twist.linear,.02),(twist.twist.angular,msg.twist.angular,.1)):
            field.x=float(np.clip(src.x,-limit,limit)); field.y=float(np.clip(src.y,-limit,limit)); field.z=float(np.clip(src.z,-limit,limit))
        self.twist.publish(twist)
        if not any(values[:6]) and self.gate.measured is not None:
            # Servo may produce no trajectory for zero twist; still permit slow gripper-only jogs.
            command=Command(); command.header.stamp=self.get_clock().now().to_msg()
            command.session_id=self.gate.session_id; command.source='teleop'
            command.sequence=self.servo_sequence; self.servo_sequence+=1
            command.target.name=self.gate.names
            command.target.position=self.gate.measured[:6].tolist()+[self.gripper_target]
            self.command(command)

    def servo_target(self,msg):
        if self.gate.owner!='teleop' or time.monotonic()-self.last_jog>.2 or not msg.points:
            return
        positions=dict(zip(msg.joint_names,msg.points[0].positions))
        if not all(n in positions for n in self.c['joint_names']):
            self.stop('servo_joint_mismatch'); return
        command=Command(); command.header.stamp=self.get_clock().now().to_msg()
        command.session_id=self.gate.session_id; command.source='teleop'
        command.sequence=self.servo_sequence; self.servo_sequence+=1
        command.target.name=self.gate.names
        command.target.position=[positions[n] for n in self.c['joint_names']]+[self.gripper_target]
        self.command(command)

    def command(self,msg):
        if self.pending: return # Drop, never queue actions behind validation.
        try:
            target=self.gate.admit(msg.source,msg.session_id,int(msg.sequence),list(msg.target.name),list(msg.target.position),stamp_s(msg.header.stamp))
        except Rejected as exc:
            self.get_logger().debug(str(exc)); return
        state=JointState(); state.header.stamp=self.get_clock().now().to_msg()
        state.name=self.gate.names; state.position=target.tolist()
        context=(msg.session_id,msg.sequence,stamp_s(msg.header.stamp),state)
        if self.c['mode']=='simulation':
            self.apply(context); return
        if not self.fk.service_is_ready() or not self.valid.service_is_ready():
            self.stop('moveit_unavailable'); return
        self.pending=context; self.pending_since=time.monotonic()
        req=GetPositionFK.Request(); req.header.frame_id='base_link'; req.fk_link_names=['tcp_link']
        req.robot_state.joint_state=state
        self.fk.call_async(req).add_done_callback(lambda f:self.fk_done(f,context))

    def current(self,ctx):
        return self.gate.armed and ctx[0]==self.gate.session_id and self.gate._fresh_stamp(ctx[2])
    def fk_done(self,future,ctx):
        if self.pending is not ctx or not self.current(ctx): return
        try:
            result=future.result()
            if result.error_code.val!=1 or not result.pose_stamped: raise Rejected('FK failed')
            p=result.pose_stamped[0].pose.position; xyz=np.array([p.x,p.y,p.z])
            if not np.all(np.isfinite(xyz)) or np.any(xyz<self.c['workspace_min_m']) or np.any(xyz>self.c['workspace_max_m']):
                raise Rejected('TCP workspace limit')
            req=GetStateValidity.Request(); req.group_name='arm'; req.robot_state.joint_state=ctx[3]
            self.valid.call_async(req).add_done_callback(lambda f:self.valid_done(f,ctx))
        except Exception as exc: self.stop(str(exc))
    def valid_done(self,future,ctx):
        if self.pending is not ctx or not self.current(ctx): return
        try:
            if not future.result().valid: raise Rejected('Collision or kinematic constraint')
            self.apply(ctx)
        except Exception as exc: self.stop(str(exc))
        finally: self.pending=None
    def apply(self,ctx):
        if not self.current(ctx): return
        self.gate.tick()
        if not self.gate.armed: return
        self.gate.mark_applied(ctx[3].position)
        ctx[3].header.stamp=self.get_clock().now().to_msg()
        self.out.publish(ctx[3]); self.applied.publish(ctx[3])

def main():
    rclpy.init(); node=Runtime()
    try: rclpy.spin(node)
    finally:
        node.stop('runtime_shutdown'); node.watchdog()
        node.destroy_node(); rclpy.shutdown()
if __name__=='__main__': main()
