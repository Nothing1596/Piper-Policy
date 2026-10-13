"""Compile legal LongVIL primitives into real Piper CLI calls in simulation.

Object localisation uses the shared private simulator pose oracle, matching
LongVIL getpos semantics. This evaluates contact execution and video procedure
transfer; it does not validate a real-world perception system.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
import numpy as np

from plan_protocol import strict_program_actions


class PhysicsExecutionError(RuntimeError):
    def __init__(self,message,code=None):super().__init__(message);self.code=code


class Player:
    def __init__(self, session, read_state, capture, event, request_prefix, scene):
        self.session,self.read_state,self.capture,self.event=session,read_state,capture,event
        self.prefix,self.scene=request_prefix,scene
        self.sdk_calls=0;self.steps=[]
        self.execution_start=None
        self.carry_yaw=None
        self.rotation=np.asarray(scene['base_from_dataset'],float)[:3,:3]
        from piperx_middleware.kinematics import PiperKinematics
        self.kinematics=PiperKinematics([0.,0.,.1425])
        self.home_xy=np.asarray(self.kinematics.pose([0.,30.,-30.,0.,0.,0.])['xyz_m'][:2])

    async def call(self,name,arguments):
        if self.sdk_calls>=2000 or (self.execution_start and time.monotonic()-self.execution_start>2700):
            raise PhysicsExecutionError('Physical execution call/time budget exhausted','execution_budget')
        self.sdk_calls+=1
        arguments=dict(arguments,request_id=f'{self.prefix}-{self.sdk_calls:04d}')
        result=await self.session.call_tool(name,arguments)
        value=result.structuredContent or {}
        self.event({'kind':'sdk','name':name,'arguments':arguments,'result':value})
        if result.isError or value.get('status') not in ('succeeded',):
            raise PhysicsExecutionError('SDK action failed: '+json.dumps(value,ensure_ascii=False)[:700],
                                        value.get('status') or value.get('error',{}).get('code'))
        return value

    async def move(self,xyz,yaw=90,linear=False):
        if linear:
            state=await self.read_state();start=np.asarray(state['robot']['tcp_pos']);target=np.asarray(xyz,float)
            seed=state['robot']['q_deg'];rpy=self.kinematics.pose(seed)['rpy_deg']
            # Completed 10 mm Cartesian waypoints use the normal position
            # primitive. The continuously streamed line guard currently races
            # the simulator's acceleration filter; do not relax that guard or
            # relabel an unknown result as success. This is a sampled path, not
            # a verified continuous Cartesian trajectory.
            count=max(1,math.ceil(float(np.linalg.norm(target-start))/.010))
            result=None
            for index in range(1,count+1):
                point=(start+(target-start)*index/count).tolist()
                seed=(await self.read_state())['robot']['q_deg']
                solved=self.kinematics.solve(point,rpy,seed,nearby=True)['joints_deg']
                if max(abs(a-b) for a,b in zip(solved,seed))>5:
                    raise PhysicsExecutionError('Sampled path exceeds 5 degree local continuity')
                result=await self.call('move_to',{'xyz_m':point,'rpy_deg':rpy,'speed_percent':10,'timeout_s':60.})
            return result
        body={'xyz_m':[float(v) for v in xyz],'speed_percent':10,'timeout_s':60.}
        if not linear:
            seed=(await self.read_state())['robot']['q_deg']
            # Parallel fingers have a 180-degree equivalent grasp. Select a
            # legal wrist orientation before any SDK action, never widen limits.
            selected=None
            candidates=(self.carry_yaw,) if self.carry_yaw is not None else (yaw,yaw+180,yaw-180)
            for candidate in candidates:
                try:self.kinematics.solve(body['xyz_m'],[180.,0.,candidate],seed)
                except Exception:continue
                selected=float(candidate);break
            if selected is None:
                phi=float(np.degrees(np.arctan2(xyz[1],xyz[0])))
                for candidate in candidates:
                    warm=[phi,90.,-90.,30.,0.,math.remainder(phi-90-candidate,360)]
                    try:solved=self.kinematics.solve(body['xyz_m'],[180.,0.,candidate],warm)
                    except Exception:continue
                    # Numerical multistart failure with the default seed is
                    # not global unreachability. Use its independently checked
                    # legal solution through the unchanged joint primitive.
                    return await self.call('robot_move_joints',{'joints_deg':solved['joints_deg'],
                                                'speed_percent':10,'timeout_s':60.})
                raise PhysicsExecutionError('No legal IK for grasp-equivalent orientations')
            body['rpy_deg']=[180.,0.,selected]
        return await self.call('move_linear' if linear else 'move_to',body)

    async def grip(self,width,contact=False):
        return await self.call('set_gripper',{'width_m':float(width),'effort_protocol':.5,
                  'completion':'bilateral_contact' if contact else 'width','timeout_s':8.})

    async def settle(self,seconds=.6):
        await asyncio.sleep(seconds)
        return await self.read_state()

    async def return_route(self,waypoints):
        # Return through measured staging poses from this successful transfer.
        # A direct joint-space jump from a drawer lip to the next object can
        # sweep the empty fingers through the cabinet. This reuses the outward
        # route, instead of claiming endpoint IK is a collision-free planner.
        for joints in reversed(waypoints):
            await self.call('robot_move_joints',{'joints_deg':list(joints),
                                               'speed_percent':10,'timeout_s':60.})

    def target(self,state,source,reference,relation,grasp_offset):
        obj,ref=state['objects'][source],state['objects'][reference]
        box=obj['aabb'];rb=ref['aabb']
        center=(np.asarray(rb['lower'])+rb['upper'])/2
        size=np.asarray(box['upper'])-box['lower']
        source_center=(np.asarray(box['lower'])+box['upper'])/2
        xy_offset=source_center[:2]-np.asarray(state['robot']['tcp_pos'])[:2]
        if relation=='top':
            return [*(center[:2]-xy_offset),rb['upper'][2]-grasp_offset+.010]
        if relation=='into':
            bounds=ref.get('accessible_opening') or ref.get('container_bounds')
            if not bounds:raise PhysicsExecutionError('No hollow interior for '+reference)
            lo,hi=np.asarray(bounds['lower']),np.asarray(bounds['upper'])
            circle=ref.get('inner_circle')
            if circle:
                origin=np.asarray(circle['center_xy_m']);radius=circle['radius_m']
                lo[:2]=origin-radius;hi[:2]=origin+radius
            margin=size[:2]/2+.004
            if np.any(hi[:2]-lo[:2]<=2*margin):
                raise PhysicsExecutionError('Object does not fit container interior')
            candidates=[(lo[:2]+hi[:2])/2]
            candidates.extend([np.array([x,y]) for x in np.linspace(lo[0]+margin[0],hi[0]-margin[0],5)
                               for y in np.linspace(lo[1]+margin[1],hi[1]-margin[1],5)])
            others=[r for n,r in state['objects'].items() if n not in (source,reference) and r['kind']=='dynamic'
                    and lo[0]<=r['position'][0]<=hi[0] and lo[1]<=r['position'][1]<=hi[1]]
            def clearance(p):
                return min((math.dist(p,r['position'][:2]) for r in others),default=1.)
            legal=[];z=float(hi[2]-grasp_offset+.012)
            footprint=np.asarray(obj.get('grasp_frame',{}).get('footprint_xy_m',[]))
            for p in candidates:
                if circle and (not len(footprint) or np.any(np.linalg.norm(footprint-source_center[:2]+p-origin,axis=1)>radius+.003)):continue
                overlaps=[o for o in others if np.all(p+size[:2]/2>np.asarray(o['aabb']['lower'])[:2]-.004) and
                          np.all(p-size[:2]/2<np.asarray(o['aabb']['upper'])[:2]+.004)]
                # Contents may support one another. Do not require every
                # inserted object to occupy a disjoint patch of the floor;
                # reject stacking whose prospective centre is above the lip.
                if overlaps and max(o['aabb']['upper'][2] for o in overlaps)+size[2]/2>=hi[2]:continue
                hover=max(state['robot']['tcp_pos'][2],rb['upper'][2]-grasp_offset+.035,z+.04)
                try:
                    for height in (z,hover):self.kinematics.solve([*(p-xy_offset),height],[180.,0.,self.carry_yaw],state['robot']['q_deg'])
                except Exception:continue
                legal.append((p,len(overlaps)))
            if not legal:raise PhysicsExecutionError('No free, reachable container placement fits actual collision footprint')
            p,_=max(legal,key=lambda row:(-row[1],round(clearance(row[0]),3),-np.linalg.norm(row[0]-self.home_xy)))
            # Release just above the physical lip. Gravity seats the object;
            # burying the fingertips to the floor causes avoidable rim contact.
            p=p-xy_offset
            return [float(p[0]),float(p[1]),z]
        directions={'left':[-1,0,0],'right':[1,0,0],'front':[0,-1,0],'behind':[0,1,0]}
        if relation not in directions:raise PhysicsExecutionError('Unknown spatial relation')
        direction=self.rotation@directions[relation]
        source_half=size/2;reference_half=(np.asarray(rb['upper'])-rb['lower'])/2
        # A fixed 35 mm gap leaves pad/palm clearance for these task assets.
        # It is a shared placement heuristic, not a collision-proof path.
        distance=float(np.dot(np.abs(direction),source_half+reference_half))+.035
        tangent=np.asarray([-direction[1],direction[0],0.])
        candidates=[]
        for lateral in (0.,-.3,.3,-.6,.6):
            p=center+direction*distance+tangent*(distance*lateral)
            p[:2]-=xy_offset
            p[2]=self.scene.get('surface_z_m',0.)-grasp_offset+.015
            # Diagonal placement retains the dominant requested direction and
            # physical separation. Prefer lower reach, without using GT.
            candidates.append(p)
        seed=state['robot']['q_deg'];yaw=self.carry_yaw
        for p in sorted(candidates,key=lambda p:np.linalg.norm(p[:2]-self.home_xy)):
            try:self.kinematics.solve(p,[180.,0.,yaw],seed)
            except Exception:continue
            return p.tolist()
        return candidates[0].tolist()

    async def transfer(self,action):
        _,source,reference,relation=action
        before=await self.read_state();obj=before['objects'][source]
        route=[before['robot']['q_deg']]
        lo,hi=np.asarray(obj['aabb']['lower']),np.asarray(obj['aabb']['upper'])
        center=(lo+hi)/2
        frame=obj.get('grasp_frame')
        if frame:center[:2]=frame['center_xy_m']
        # Pad centre lies 12.5 mm above the configured fingertip TCP.
        grasp_z=float(center[2]-.0125)
        grasp_z=max(float(lo[2]+.0015),grasp_z)
        width=float(frame['closing_width_m'] if frame else min(hi[0]-lo[0],hi[1]-lo[1]))
        yaw=float(frame['yaw_deg'] if frame else 0. if hi[0]-lo[0]<hi[1]-lo[1] else 90.)
        if width+.010>.07:raise PhysicsExecutionError('Grasp width exceeds configured Piper gripper')
        await self.grip(max(.045,width+.010))
        above=[center[0],center[1],max(grasp_z+.06,hi[2]+.035)]
        await self.move(above,yaw)
        route.append((await self.read_state())['robot']['q_deg'])
        await self.move([center[0],center[1],grasp_z],yaw,linear=True)
        await self.grip(0.,contact=True)
        held=await self.read_state()
        if source not in held['bilateral_contact_objects']:
            raise PhysicsExecutionError('Requested object lacks bilateral finger contact')
        actual_tcp=held['robot']['tcp_pos']
        self.carry_yaw=float(self.kinematics.pose(held['robot']['q_deg'])['rpy_deg'][2])
        offset=float(held['objects'][source]['aabb']['lower'][2]-actual_tcp[2])
        await self.move([actual_tcp[0],actual_tcp[1],above[2]],yaw,linear=True)
        lifted=await self.read_state()
        lift=float(lifted['objects'][source]['position'][2]-obj['position'][2])
        if lift<.03:raise PhysicsExecutionError('Object did not physically lift by 30 mm')
        if source not in lifted['bilateral_contact_objects']:
            raise PhysicsExecutionError('Object slipped after lift')
        target=self.target(lifted,source,reference,relation,offset)
        safe_z=max(above[2],lifted['objects'][reference]['aabb']['upper'][2]-offset+.035,target[2]+.04)
        await self.move([actual_tcp[0],actual_tcp[1],safe_z],yaw,linear=True)
        route.append((await self.read_state())['robot']['q_deg'])
        await self.move([target[0],target[1],safe_z],yaw)
        if source not in (await self.read_state())['bilateral_contact_objects']:
            raise PhysicsExecutionError('Object slipped during transfer')
        await self.move(target,yaw,linear=True)
        await self.grip(max(.045,width+.010))
        self.carry_yaw=None
        released=await self.read_state()
        await self.move([target[0],target[1],safe_z],yaw,linear=True)
        await self.return_route(route)
        after=await self.settle()
        return {'action':list(action),'status':'executed','state_after':after,
                'grasp_evidence':{'bilateral_contact':True,'lift_height_m':lift,
                                  'released':source not in after['bilateral_contact_objects']},
                'state_before':before,'state_held':held,'state_lifted':lifted}

    async def drawer(self,action):
        if action not in (('open',),('close',)):
            raise PhysicsExecutionError('Illegal drawer primitive')
        state=await self.read_state()
        route=[state['robot']['q_deg']]
        if 'drawer' not in state:raise PhysicsExecutionError('Scene has no articulated drawer')
        goal=self.scene.get('drawer_travel_m',.17) if action[0]=='open' else 0.
        current=state['drawer']['qpos_m']
        if abs(current-goal)<.01:
            return {'action':list(action),'status':'executed','state_after':state,'already_at_target':True}
        # The scene builder records the actual handle frame in the moving body.
        handle=self.scene.get('drawer_handle_local')
        if handle is None:raise PhysicsExecutionError('Drawer handle geometry not registered')
        name=next(n for n,v in self.scene['objects'].items() if v['kind']=='drawer')
        pose=state['objects'][name]
        from scipy.spatial.transform import Rotation
        q=pose['quaternion'];rotation=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
        point=np.asarray(pose['position'])+rotation@handle
        axis=np.asarray(self.scene['drawer_axis_base'])
        yaw=float(np.degrees(np.arctan2(axis[1],axis[0]))+90)
        await self.grip(.07)
        await self.move([point[0],point[1],point[2]+.1],yaw)
        route.append((await self.read_state())['robot']['q_deg'])
        target=[point[0],point[1],point[2]-.0125]
        await self.move(target,yaw,linear=True)
        await self.grip(0.,contact=True)
        pulled=np.asarray(target)+axis*(goal-current)
        await self.move(pulled.tolist(),yaw,linear=True)
        await self.grip(.07)
        await self.move([pulled[0],pulled[1],pulled[2]+.1],yaw,linear=True)
        await self.return_route(route)
        return {'action':list(action),'status':'executed','state_after':await self.settle()}

    async def execute(self,code,objects):
        self.execution_start=time.monotonic()
        actions=strict_program_actions(code,objects)
        for index,action in enumerate(actions):
            try:
                row=await (self.transfer(action) if action[0]=='pick_place' else self.drawer(action))
            except Exception as exc:
                row={'action':list(action),'status':'failed','error_type':type(exc).__name__,
                     'error':str(exc)[:800],'error_code':getattr(exc,'code',None),'state_after':await self.read_state()}
            self.steps.append(row);self.event({'kind':'physical_step','index':index,**row})
            await self.capture(index)
            if row['status']!='executed':break
        return {'realized_steps':self.steps,'sdk_calls':self.sdk_calls,
                'execution_outcome_unknown':any(s.get('error_code')=='outcome_unknown' for s in self.steps),
                'path_semantics':'completed 10 mm Cartesian waypoints using move_to; no continuous path guarantee',
                'all_requested_steps_executed':len(self.steps)==len(actions) and all(r['status']=='executed' for r in self.steps)}
