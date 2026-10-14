"""Generic bounded tabletop skills; selection/order belong to the vision planner."""
from __future__ import annotations
import math
import time
from .grounding import require_workspace


class SkillRunner:
    def __init__(self, executor, *, downward_rpy=(180,0,0), hover_m=.07,
                 object_height_m=.03, grasp_width_m=.027, fingertip_clearance_m=.004,release_width_m=.04):
        if not all(math.isfinite(v) for v in (*downward_rpy,hover_m,object_height_m,grasp_width_m,fingertip_clearance_m)):
            raise ValueError('nonfinite_skill_configuration')
        if len(downward_rpy)!=3 or not .03<=hover_m<=.2 or not .005<=object_height_m<=.08 or not 0<=grasp_width_m<=.07 or not 0<fingertip_clearance_m<object_height_m:
            raise ValueError('unsupported_tabletop_skill_profile')
        self.executor=executor
        self.rpy=list(downward_rpy)
        self.hover_m=hover_m
        self.object_height_m=object_height_m
        self.grasp_width_m=grasp_width_m
        self.fingertip_clearance_m=fingertip_clearance_m
        if not math.isfinite(release_width_m) or not grasp_width_m<release_width_m<=.07:
            raise ValueError('invalid_release_width')
        self.release_width_m=release_width_m
        self.placement_half_size=release_width_m/2+.006+.001
        self.trace=[]
        self.visual_monitor=None
        self.expect_contact=False
        self.last_contact_s=0.
        self.prepared=None

    def prepare(self,action,source,destination):
        x,y,top=source.xyz_m;dx,dy,surface=destination.xyz_m
        hover=max(top,surface)+self.hover_m
        if action=='pick_place':
            targets=((x,y,hover),(x,y,top-self.object_height_m+self.fingertip_clearance_m),
                     (dx,dy,hover),(dx,dy,surface+self.fingertip_clearance_m+.002))
        elif action=='push':
            length=math.hypot(dx-x,dy-y)
            if not .01<=length<=.18:raise ValueError('push_distance_limit')
            ux,uy=(dx-x)/length,(dy-y)/length
            targets=((x-.04*ux,y-.04*uy,hover),(x-.04*ux,y-.04*uy,top-self.object_height_m/2),
                     (dx-.04*ux,dy-.04*uy,top-self.object_height_m/2))
        else:raise ValueError('unsupported_skill')
        for point in targets:
            self.executor.request('POST','/v1/primitives/preview',{'command':{
                'kind':'move_to','xyz_m':list(map(float,point)),'rpy_deg':self.rpy,'speed_percent':15}})
        self.prepared=(action,source.xyz_m,destination.xyz_m)

    def _monitor(self):
        if self.visual_monitor is not None:self.visual_monitor()
        state=self.executor.state()
        robot=state['robot']
        if not robot['connected'] or robot.get('feedback_age_s',math.inf)>.2:
            raise RuntimeError('stale_execution_feedback')
        if robot.get('collision_status') and any(robot['collision_status']):
            raise RuntimeError('collision_feedback')
        if self.expect_contact:
            if robot.get('diagnostics',{}).get('gripper_contact',{}).get('supported'):
                self.last_contact_s=time.monotonic()
            elif time.monotonic()-self.last_contact_s>.3:
                raise RuntimeError('grasp_contact_lost')

    def _move(self, xyz, *, linear=False):
        if any(not lo<=a<=hi for a,lo,hi in zip(xyz,(-.05,-.45,-.02),(.65,.45,.65))):
            raise ValueError('skill_workspace_limit')
        command={'kind':'move_linear' if linear else 'move_to','xyz_m':list(map(float,xyz)),
                 'speed_percent':15,'timeout_s':30.}
        if not linear:
            command['rpy_deg']=self.rpy
        result=self.executor.primitive(command,monitor=self._monitor)
        self.trace.append(result)

    def _grip(self,width,*,contact=False):
        result=self.executor.primitive({'kind':'set_gripper','width_m':float(width),
                                       'effort_protocol':.5,'timeout_s':10.,
                                       'completion':'bilateral_contact' if contact else 'width'},monitor=self._monitor)
        self.trace.append(result)

    def run(self, action, source, destination):
        require_workspace(source);require_workspace(destination)
        if min(source.valid_until,destination.valid_until)<time.monotonic():
            raise ValueError('grounded_goal_expired')
        if source.epoch!=destination.epoch or source.calibration_id!=destination.calibration_id:
            raise ValueError('grounded_goal_frame_mismatch')
        prepared=self.prepared;self.prepared=None
        if not prepared or prepared[0]!=action or any(abs(a-b)>.003 for old,new in
                zip(prepared[1:],(source.xyz_m,destination.xyz_m)) for a,b in zip(old,new)):
            raise ValueError('missing_or_changed_preflight')
        self.trace=[]
        self.expect_contact=False
        return_joints=self.executor.state()['robot']['q_deg']
        x,y,top=source.xyz_m
        dx,dy,surface=destination.xyz_m
        hover=max(top,surface)+self.hover_m
        if action=='pick_place':
            # First-version grasp profile is explicitly for 30 mm rigid blocks.
            # Width closure alone is never used as the final task success verdict.
            self._grip(.065)
            self._move((x,y,hover))
            self._move((x,y,top-self.object_height_m+self.fingertip_clearance_m),linear=True)
            self._grip(self.grasp_width_m,contact=True)
            self.expect_contact=True;self.last_contact_s=time.monotonic()
            self._move((x,y,hover),linear=True)
            self._move((dx,dy,hover))
            self._move((dx,dy,surface+self.fingertip_clearance_m+.002),linear=True)
            self.expect_contact=False
            self._grip(self.release_width_m)
            self._move((dx,dy,hover),linear=True)
        elif action=='push':
            length=math.hypot(dx-x,dy-y)
            if not .01<=length<=.18:
                raise ValueError('push_distance_limit')
            ux,uy=(dx-x)/length,(dy-y)/length
            offset=.04
            sx,sy=x-offset*ux,y-offset*uy
            self._grip(.005)
            self._move((sx,sy,hover))
            self._move((sx,sy,top-self.object_height_m/2),linear=True)
            self._move((dx-offset*ux,dy-offset*uy,top-self.object_height_m/2),linear=True)
            self._move((dx-offset*ux,dy-offset*uy,hover),linear=True)
        else:
            raise ValueError('unsupported_skill')
        # Return to the measured pre-skill pose so the fixed camera can see
        # the result; hovering directly over the tray occludes its contents.
        self.trace.append(self.executor.joint_move(return_joints,monitor=self._monitor))
        return {'action':action,'status':'executed_pending_visual_verification','jobs':self.trace}
