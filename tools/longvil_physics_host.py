"""Experiment-only LongVIL scenes behind the normal Piper CLI/MCP lifecycle.

The production controller and robot servos are unchanged. Named scene assets,
private physical-state recording and operator scoring are confined to this job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import signal
import threading
import time

import numpy as np


def process_identity(pid,proc_root=Path('/proc')):
    """Linux process start identity prevents treating a reused PID as ours."""
    path=proc_root/str(pid)
    try:
        stat=(path/'stat').read_text().rsplit(') ',1)[1].split()
        command=(path/'cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:return None
    return {'pid':int(pid),'start_ticks':stat[19],'state':stat[0],
            'command':[p for p in command if p]}


def resource_check(profile):
    marker=profile/'physics-owner.json';owner=json.loads(marker.read_text()) if marker.exists() else None
    live=process_identity(owner['pid']) if owner else None
    verified=bool(owner and live and owner['start_ticks']==live['start_ticks'] and
                  owner['command']==live['command'])
    alive=bool(verified and live['state']!='Z')
    return {'runtime_remaining':(profile/'runtime.json').exists() or alive,
            'owned_process_alive':alive,'process_identity_recorded':owner is not None,
            'recording_error':json.loads((profile/'recording-error.json').read_text())
            if (profile/'recording-error.json').exists() else None}


def stop_owned(profile):
    """Fallback only for this isolated simulation service, never foreign PIDs."""
    marker=profile/'physics-owner.json'
    if not marker.exists():return dict(resource_check(profile),stop_status='no_owned_process_identity')
    owner=json.loads(marker.read_text());live=process_identity(owner['pid'])
    if live and live['state']!='Z':
        if owner['start_ticks']!=live['start_ticks'] or owner['command']!=live['command']:
            raise RuntimeError('Owned simulation PID identity changed; refusing to signal it')
        if 'serve' not in live['command'] or str(profile.resolve()) not in live['command']:
            raise RuntimeError('Process does not match this isolated simulation profile')
        os.kill(owner['pid'],signal.SIGTERM)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            identity=process_identity(owner['pid'])
            if not identity or identity['state']=='Z':break
            time.sleep(.1)
    return dict(resource_check(profile),stop_status='owned_simulation_shutdown_requested')


def scene_backend_class():
    import mujoco
    from piperx_middleware.mujoco_backend import MujocoBackend

    class LongVILBackend(MujocoBackend):
        def __init__(self, asset_path=None, seed=0):
            self.scene = json.loads(Path(asset_path).with_suffix('.json').read_text())
            if hashlib.sha256(Path(asset_path).read_bytes()).hexdigest() != self.scene['scene_sha256']:
                raise ValueError('Scene XML does not match its manifest')
            for path,expected in self.scene.get('asset_sha256s',{}).items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Scene asset changed: '+path)
            self.record_stop = threading.Event()
            self.recorder = None
            self.record_error = None
            super().__init__(asset_path, seed)
            # Recorder atomically pairs pixels and state; nested observe uses
            # the same renderer lock. No worker is started by base __init__.
            self.render_lock = threading.RLock()
            self.object_ids = {
                name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, entry['body'])
                for name, entry in self.scene['objects'].items()}
            if any(i < 0 for i in self.object_ids.values()):
                raise ValueError('Scene manifest names a missing object body')
            self.body_owners = {}
            for body in range(self.model.nbody):
                ancestors, current = set(), body
                while current:
                    ancestors.add(current)
                    current = int(self.model.body_parentid[current])
                # The moving drawer takes precedence over its static cabinet.
                for name, oid in self.object_ids.items():
                    if oid in ancestors:
                        self.body_owners[body] = name
            self.drawer_joint = None
            if self.scene.get('drawer_joint'):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.scene['drawer_joint'])
                if jid < 0:
                    raise ValueError('Missing passive drawer joint')
                self.drawer_joint = jid
                self.data.qpos[self.model.jnt_qposadr[jid]] = self.scene.get('drawer_initial_qpos', 0.)
                mujoco.mj_forward(self.model, self.data)
            if self.scene.get('approximate_recovery'):
                from approximate_physics_recovery import apply_recorded_pose
                receipt = apply_recorded_pose(self, self.scene['approximate_recovery'])
                profile = os.environ.get('PIPER_PHYSICS_PROFILE')
                if profile:
                    (Path(profile)/'recovery-initialization.private.json').write_text(json.dumps(receipt))

        def connect(self):
            super().connect()
            if self.recorder is None:
                self.recorder = threading.Thread(target=self._record, name='longvil-physics-record', daemon=True)
                self.recorder.start()

        def close(self):
            self.record_stop.set()
            try:
                if self.recorder and self.recorder is not threading.current_thread():
                    self.recorder.join(timeout=15)
                    if self.recorder.is_alive():
                        raise RuntimeError('Physics recorder did not release its resources')
            finally:
                super().close()

        def _bounds(self, name):
            ids = [g for g in range(self.model.ngeom)
                   if self.body_owners.get(int(self.model.geom_bodyid[g])) == name
                   and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
            if not ids:
                raise ValueError('Object has no physical collision geometry: '+name)
            lower, upper = [], []
            for g in ids:
                local = self.model.geom_aabb[g]
                rotation = self.data.geom_xmat[g].reshape(3, 3)
                center = self.data.geom_xpos[g] + rotation @ local[:3]
                extent = np.abs(rotation) @ local[3:]
                lower.append(center-extent);upper.append(center+extent)
            return {'lower': np.min(lower, axis=0).tolist(), 'upper': np.max(upper, axis=0).tolist()}

        def _grasp_frame(self,name):
            """Minimum-width horizontal envelope of actual collision geometry.

            A rotated banana/long prism can fit the fingers although its world
            AABB cannot. This is shared simulator geometry, not VLM perception.
            """
            from scipy.spatial import ConvexHull
            points=[]
            for g in range(self.model.ngeom):
                if self.body_owners.get(int(self.model.geom_bodyid[g]))!=name:continue
                if not(self.model.geom_contype[g] or self.model.geom_conaffinity[g]):continue
                if self.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                    mid=int(self.model.geom_dataid[g]);start=int(self.model.mesh_vertadr[mid]);count=int(self.model.mesh_vertnum[mid])
                    local=self.model.mesh_vert[start:start+count]
                else:
                    b=self.model.geom_aabb[g]
                    local=np.asarray([[x,y,z] for x in (-b[3],b[3]) for y in (-b[4],b[4])
                                      for z in (-b[5],b[5])])+b[:3]
                points.extend(local@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g])
            points=np.asarray(points);hull=ConvexHull(points[:,:2]);vertices=points[hull.vertices,:2]
            best=None
            for edge in np.roll(vertices,-1,axis=0)-vertices:
                normal=np.asarray([-edge[1],edge[0]])/np.linalg.norm(edge)
                along=np.asarray([-normal[1],normal[0]])
                proj=points[:,:2]@normal;side=points[:,:2]@along
                width=float(proj.max()-proj.min())
                if best is None or width<best[0]:
                    center=normal*((proj.min()+proj.max())/2)+along*((side.min()+side.max())/2)
                    # Measured pad separation follows TCP local X: yaw=0
                    # closes along base X, yaw=90 along base Y.
                    yaw=float(np.degrees(np.arctan2(normal[1],normal[0])))
                    yaw=(yaw+90)%180-90
                    best=(width,center,yaw)
            # Avoid arbitrary mesh-facet yaw for nearly circular/symmetric
            # shapes. Cardinal grasps are equivalent within 5 mm of minimum.
            for normal,yaw in ((np.array([1.,0.]),0.),(np.array([0.,1.]),90.)):
                proj=points[:,:2]@normal;along=np.asarray([-normal[1],normal[0]]);side=points[:,:2]@along
                width=float(proj.max()-proj.min())
                if width<=.060 and width<=best[0]+.005:
                    center=normal*((proj.min()+proj.max())/2)+along*((side.min()+side.max())/2)
                    best=(width,center,yaw);break
            return {'closing_width_m':best[0],'center_xy_m':best[1].tolist(),'yaw_deg':best[2],
                    'footprint_xy_m':vertices.tolist()}

        def _get_gripper_contact(self, stamp):
            # Use the same force criterion as the shipped backend for scene
            # objects, including the moving drawer's descendant handle.
            forces = {}
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                gids = [int(c.geom1),int(c.geom2)]
                names = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in gids]
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model,self.data,i,force)
                for side in (0,1):
                    owner = self.body_owners.get(int(self.model.geom_bodyid[gids[side]]))
                    if owner and self.scene['objects'][owner]['kind'] in ('dynamic','drawer') and force[0]>.005:
                        for finger in (1,2):
                            if names[1-side]==f'gripper_link{finger}_geom':
                                pair=forces.setdefault(owner,[0.,0.]);pair[finger-1]+=float(force[0])
            supported = next((f for f in forces.values() if min(f)>.01),None)
            pair=supported or [max((f[0] for f in forces.values()),default=0.),
                               max((f[1] for f in forces.values()),default=0.)]
            return {'supported':supported is not None,'finger1_force_n':pair[0],
                    'finger2_force_n':pair[1],'source_timestamp_s':float(stamp)}

        def evaluate(self):
            with self.lock:
                objects = {}
                for name, oid in self.object_ids.items():
                    entry = self.scene['objects'][name]
                    velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                             oid, velocity, 0)
                    row = {'position': self.data.xpos[oid].tolist(),
                           'quaternion': self.data.xquat[oid].tolist(),
                           'aabb': self._bounds(name), 'kind': entry['kind'],
                           'speed_m_s': float(np.linalg.norm(velocity[3:]))}
                    if entry['kind']=='dynamic':row['grasp_frame']=self._grasp_frame(name)
                    if entry.get('inner_circle_local'):
                        circle=entry['inner_circle_local'];rot=self.data.xmat[oid].reshape(3,3)
                        center=rot@np.asarray([*circle['center_xy_m'],0.])+self.data.xpos[oid]
                        row['inner_circle']={'center_xy_m':center[:2].tolist(),'radius_m':circle['radius_m']}
                    if entry.get('container_bounds_local'):
                        rotation = self.data.xmat[oid].reshape(3, 3)
                        bounds = entry['container_bounds_local']
                        corners = np.array([[x, y, z] for x in (bounds['lower'][0],bounds['upper'][0])
                                           for y in (bounds['lower'][1],bounds['upper'][1])
                                           for z in (bounds['lower'][2],bounds['upper'][2])])
                        points = corners @ rotation.T + self.data.xpos[oid]
                        row['container_bounds'] = {'lower': points.min(axis=0).tolist(),
                                                   'upper': points.max(axis=0).tolist()}
                        if entry['kind']=='drawer':
                            # Only the exposed drawer opening admits a vertical
                            # drop; its covered rear remains part of the interior.
                            opening={k:list(v) for k,v in row['container_bounds'].items()}
                            axis=np.asarray(self.scene['drawer_axis_base']);dim=int(np.argmax(np.abs(axis[:2])))
                            roof=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'container__drawer__top_wall__c0')
                            if roof>=0:
                                box=self.model.geom_aabb[roof];rot=self.data.geom_xmat[roof].reshape(3,3)
                                center=self.data.geom_xpos[roof]+rot@box[:3];extent=np.abs(rot)@box[3:]
                                if axis[dim]<0:opening['upper'][dim]=min(opening['upper'][dim],float(center[dim]-extent[dim])-.01)
                                else:opening['lower'][dim]=max(opening['lower'][dim],float(center[dim]+extent[dim])+.01)
                            row['accessible_opening']=opening
                    objects[name] = row
                contacts, fingers = [], {}
                for i in range(self.data.ncon):
                    contact = self.data.contact[i]
                    geoms = [int(contact.geom1), int(contact.geom2)]
                    owners = [self.body_owners.get(int(self.model.geom_bodyid[g])) for g in geoms]
                    names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in geoms]
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    contacts.append({'object1': owners[0] or names[0], 'object2': owners[1] or names[1],
                                     'normal_force_n': float(force[0])})
                    for side in (0,1):
                        if owners[side] and force[0] > .001:
                            for finger in (1,2):
                                if names[1-side] == f'gripper_link{finger}_geom':
                                    fingers.setdefault(owners[side],set()).add(finger)
                state = {'sim_time_s':float(self.data.time), 'epoch':self.epoch, 'objects':objects,
                         'contacts':contacts, 'base_from_dataset':self.scene['base_from_dataset'],
                         'bilateral_contact_objects':[n for n,f in fingers.items() if f == {1,2}],
                         'robot':{'tcp_pos':self.data.site_xpos[self.tcp_site_id].tolist(),
                                  'q_deg':np.rad2deg(self.data.qpos[self.arm_qpos_indices]).tolist(),
                                  'gripper_width_m':float(sum(self.data.qpos[self.gripper_qpos_indices]))}}
                if self.drawer_joint is not None:
                    state['drawer'] = {'qpos_m':float(self.data.qpos[self.model.jnt_qposadr[self.drawer_joint]]),
                                       'travel_m':self.scene.get('drawer_travel_m',.17)}
                    for row in objects.values():
                        if row['kind']=='drawer':row.update(state['drawer'])
                return state

        def _record(self):
            """Record rendered physics and state on this thread; no object updates."""
            import av
            profile = Path(os.environ['PIPER_PHYSICS_PROFILE'])
            container = stream = None
            try:
                container = av.open(str(profile/'execution.mp4'),'w')
                stream = container.add_stream('libx264',rate=10)
                stream.width=640;stream.height=480;stream.pix_fmt='yuv420p'
                stream.options={'crf':'25','preset':'ultrafast'}
                with (profile/'private-physics.jsonl').open('w') as log:
                    frame=0;deadline=time.monotonic()
                    while not self.record_stop.is_set():
                        # Match observe's lock order (renderer, then state).
                        with self.render_lock:
                            state=self.evaluate()
                            observation=self.observe()
                        state['video_frame']=frame
                        log.write(json.dumps(state,allow_nan=False)+'\n');log.flush()
                        picture=av.VideoFrame.from_ndarray(observation['rgb'],format='rgb24')
                        for packet in stream.encode(picture):container.mux(packet)
                        frame+=1;deadline+=.1
                        self.record_stop.wait(max(0,deadline-time.monotonic()))
            except Exception as exc:
                self.record_error={'type':type(exc).__name__,'message':str(exc)[:400]}
                (profile/'recording-error.json').write_text(json.dumps(self.record_error))
            finally:
                try:
                    if container:
                        if stream:
                            for packet in stream.encode():container.mux(packet)
                        container.close()
                except Exception as exc:
                    self.record_error={'type':type(exc).__name__,'message':str(exc)[:400]}
                    (profile/'recording-error.json').write_text(json.dumps(self.record_error))
                finally:self.close_renderer()

    return LongVILBackend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['start','serve','score','check','stop'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--scene',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.operation=='serve':
        os.environ['PIPER_PHYSICS_PROFILE']=str(root)
        if sys.platform.startswith('linux'):
            (root/'physics-owner.json').write_text(json.dumps(process_identity(os.getpid())))
        import piperx_middleware.mujoco_backend as module
        module.MujocoBackend=scene_backend_class()
        from piperx_middleware.cli import main as cli
        sys.argv=['piperx','--root',str(root),'serve','--managed'];cli()
        return
    profile=root/'profiles/simulation/local'
    if args.operation=='start':
        if args.scene is None:raise ValueError('An explicit LongVIL scene is required')
        from piperx_middleware.profiles import ensure_profile
        ensure_profile(root,'simulation',simulation_backend='mujoco')
        config=json.loads((profile/'config.json').read_text())
        config.update(simulation_asset=str(args.scene.resolve()),simulation_seed=0,
                      tcp_offset_m=[0.,0.,.1425],tcp_offset_rpy_deg=[0.,0.,0.])
        from piperx_middleware.models import Settings
        Settings.model_validate(config)
        (profile/'config.json').write_text(json.dumps(config))
        from piperx_middleware.managed_runtime import RuntimeManager
        def command(self,profile):
            return [sys.executable,str(Path(__file__).resolve()),'serve','--root',str(profile.root)]
        RuntimeManager._local_command=command
        # The standalone entry spawns another interpreter, which would discard
        # this experiment's runtime factory. Use its normal in-process CLI.
        from piperx_middleware.cli import main as cli
        sys.argv=['piper-robot','--root',str(root),'--mode','simulation',
                  '--simulation-backend','mujoco','--agent-mcp'];cli()
        return
    if args.operation=='check':
        print(json.dumps(resource_check(profile)))
        return
    if args.operation=='stop':
        print(json.dumps(stop_owned(profile)))
        return
    from piperx_middleware.client import RobotClient
    runtime=json.loads((profile/'runtime.json').read_text())
    if runtime['backend']!='mujoco':raise ValueError('Physics scoring requires MuJoCo')
    operator=RobotClient(runtime['url'],profile/'operator.token')
    try:print(json.dumps(operator.call('GET','/operator/simulation/evaluate')))
    finally:operator.close()


if __name__=='__main__':main()
