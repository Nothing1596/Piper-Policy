"""Bounded URDF-to-MJCF conversion for the nine reviewed LongVIL scenes.

Only initial scene fields are used. This does not reconstruct camera texture,
human hands, material compliance, or the real objects from video pixels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as E

import numpy as np
from scipy.spatial.transform import Rotation

COLORS={'red':[.85,.1,.1,1.],'green':[.2,.8,.1,1.],'purple':[.6,.2,.8,1.],
        'orange':[1.,.5,.05,1.],'blue':[.1,.3,.9,1.],'white':[.95,.95,.95,1.]}
BASE_FROM_DATASET=np.array([[0.,-1.,0.,-.12],[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])


def slug(value):return re.sub('[^A-Za-z0-9]+','_',value).strip('_')
def numbers(value,default):return np.array([float(x) for x in value.replace(',',' ').split()]) if value else np.array(default,float)
def text(values):return ' '.join(format(float(v),'.12g') for v in values)


def origin(element,scale=1.):
    value=element.find('origin')
    if value is None:return np.zeros(3),np.array([1.,0.,0.,0.])
    q=Rotation.from_euler('xyz',numbers(value.get('rpy'),[0,0,0])).as_quat()
    return numbers(value.get('xyz'),[0,0,0])*scale,q[[3,0,1,2]]


def mesh_bounds(path,scale):
    vertices=[]
    with path.open(errors='replace') as f:
        for line in f:
            if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:4]])
    if not vertices:raise ValueError('OBJ mesh has no vertices: '+str(path))
    points=np.asarray(vertices)*scale
    return points.min(axis=0),points.max(axis=0)


def parse_urdf(path):
    raw=Path(path).read_text()
    end=raw.find('</robot>')
    # Several upstream assets append inactive, nested XML comments after the
    # complete robot. TinyXML accepts them; Expat rejects those nested comments.
    # Keep the complete active robot, never remove any active link or joint.
    if end>=0 and raw[end+8:].lstrip().startswith('<!--'):raw=raw[:end+8]
    return E.fromstring(raw)


def build_scene(annotation,template,asset_root,output,*,translation_m=None):
    asset_root=Path(asset_root);output=Path(output)
    transform=BASE_FROM_DATASET.copy()
    if translation_m is not None:
        translation=np.asarray(translation_m,dtype=float)
        if translation.shape!=(3,) or not np.all(np.isfinite(translation)):
            raise ValueError('Scene translation must contain three finite metres')
        transform[:3,3]=translation
    surface_z=float(transform[2,3])
    tree=E.parse(template);root=tree.getroot();world=root.find('worldbody');assets=root.find('asset')
    root.find('compiler').set('meshdir',str(Path(template).parent/'meshes'))
    for body in list(world.findall('body')):
        if body.get('name') in ('cube_red','cube_blue','tray'):world.remove(body)
    table=world.find("body[@name='table']");table.set('pos',text([.35,0,surface_z-.20]))
    table.find('geom').set('size','.8 .75 .20')
    world.find("geom[@name='floor']").set('pos',text([0,0,surface_z-.45]))
    camera=world.find("camera[@name='overview_camera']")
    initial=np.asarray([[p['x'],p['y'],p['z']] for p in annotation['positions'].values()])
    center=transform[:3,:3]@((initial.min(axis=0)+initial.max(axis=0))/2)+transform[:3,3]
    camera.set('pos',text([center[0],center[1],surface_z+1.1]));camera.set('quat','.707106781 0 0 .707106781')
    checksums={}
    def track(path):
        path=Path(path).resolve();checksums[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for mesh in assets.findall('mesh'):
        track(Path(template).parent/'meshes'/mesh.get('file'))
    objects={};metadata={'base_from_dataset':transform.tolist(),'surface_z_m':surface_z,'objects':objects,'asset_sha256s':checksums,
        'limitations':['Task assets and initial annotation poses, not a photoreal digital twin.',
                      'Concave bowl/glass collision uses a declared hollow rim approximation.',
                      'Drawer travel is normalized to 0.17 m; no robot servo or joint limit changes.']}
    for name in annotation['object_list']:
        container=name in ('box','glass','white bowl','drawer')
        is_drawer=name=='drawer';kind='drawer' if is_drawer else 'container' if container else 'dynamic'
        basename=slug(name);body_name=('container__' if container else 'object__')+basename
        p=annotation['positions'][name];point=transform[:3,:3]@np.array([p['x'],p['y'],p['z']])+transform[:3,3]
        scale=.4 if is_drawer else 2. if name=='white bowl' else 1.
        if name=='white bowl':point[2]=surface_z+.045
        if container and not name=='white bowl':point[2]=surface_z
        if any(x in name for x in ('cuboid','cylinder','prism','cube')):
            directory=asset_root/'buildings';urdf=directory/(name.split()[-1]+'.urdf');yaw=math.pi/2
        else:
            directory=asset_root/('yellow_bowl' if name=='white bowl' else name)
            urdf=directory/('model.urdf' if name=='white bowl' else name+'.urdf');yaw=0.
        if not urdf.is_file():raise ValueError('Missing authoritative object asset: '+str(urdf))
        track(urdf)
        definition=parse_urdf(urdf);links={x.get('name'):x for x in definition.findall('link')}
        joints=definition.findall('joint');children={j.find('child').get('link') for j in joints}
        base=next(n for n in links if n not in children)
        quaternion=Rotation.from_euler('z',math.pi/2+yaw).as_quat()[[3,0,1,2]]
        body=E.SubElement(world,'body',name=body_name,pos=text(point),quat=text(quaternion))
        if not container:E.SubElement(body,'freejoint',name=body_name+'__free')
        own_geoms=[];moving_drawer=None;handle_local=None;inner_circle=None
        overrides=COLORS.get(name.split()[0]);default_color=[.5,.5,.5,1.]
        if is_drawer:default_color=[.65,.43,.25,1.]
        if name=='box':default_color=[.2,.65,.85,1.]
        if name=='white bowl':default_color=COLORS['white']

        def geometry(parent,element,label,visual,force_color=None):
            pos,quat=origin(element,scale);shape=element.find('geometry')
            attributes={'name':label,'pos':text(pos),'quat':text(quat)}
            color=force_color or overrides
            if color is None:
                c=element.find('material/color')
                color=numbers(c.get('rgba'),default_color).tolist() if c is not None else default_color
            attributes['rgba']=text(color)
            if visual:attributes.update(contype='0',conaffinity='0',mass='0',group='2')
            else:attributes.update(friction='1.2 .005 .0001',condim='4',solref='.004 1',solimp='.95 .99 .001')
            box=shape.find('box');cylinder=shape.find('cylinder');sphere=shape.find('sphere');mesh=shape.find('mesh')
            if box is not None:attributes.update(type='box',size=text(numbers(box.get('size'),[])*scale/2))
            elif cylinder is not None:attributes.update(type='cylinder',size=text([float(cylinder.get('radius'))*scale,float(cylinder.get('length'))*scale/2]))
            elif sphere is not None:attributes.update(type='sphere',size=str(float(sphere.get('radius'))*scale))
            elif mesh is not None:
                file=directory/mesh.get('filename');mesh_scale=numbers(mesh.get('scale'),[1,1,1])*scale
                if not file.is_file():raise ValueError('Missing mesh: '+str(file))
                track(file)
                mesh_name=label+'__mesh'
                E.SubElement(assets,'mesh',name=mesh_name,file=str(file.resolve()),scale=text(mesh_scale))
                attributes.update(type='mesh',mesh=mesh_name)
            else:raise ValueError('Unsupported asset geometry')
            geom=E.SubElement(parent,'geom',**attributes)
            if not visual:own_geoms.append(geom)
            return geom

        def add_link(link_name,parent):
            nonlocal moving_drawer,handle_local
            link=links[link_name]
            # Mesh visuals preserve identity; separate hollow collision below.
            for i,v in enumerate(link.findall('visual')):geometry(parent,v,body_name+'__'+slug(link_name)+f'__v{i}',True)
            for i,c in enumerate(link.findall('collision')):
                if name in ('glass','white bowl'):continue
                geometry(parent,c,body_name+'__'+slug(link_name)+f'__c{i}',False)
            if link_name=='handle':handle_local=parent.get('pos')
            for joint in joints:
                if joint.find('parent').get('link')!=link_name:continue
                child=joint.find('child').get('link');pos,quat=origin(joint,scale)
                branch=E.SubElement(parent,'body',name=body_name+'__'+slug(child),pos=text(pos),quat=text(quat))
                if joint.get('type')=='prismatic':
                    if not is_drawer:raise ValueError('Unexpected articulated object')
                    moving_drawer=branch
                    joint_name=body_name+'__slide'
                    # The upstream base link is rotated by pi: +local X is
                    # already outward in dataset -X. Preserve that direction.
                    E.SubElement(branch,'joint',name=joint_name,type='slide',axis='1 0 0',range='0 .17',damping='1',frictionloss='.05')
                    metadata.update(drawer_joint=joint_name,drawer_travel_m=.17,
                        drawer_axis_base=(transform[:3,:3]@[-1.,0.,0.]).tolist(),
                        drawer_initial_qpos=.17 if 'open()' in (annotation.get('pre_action_sequences') or []) else 0.)
                elif joint.get('type')!='fixed':raise ValueError('Unsupported object joint type')
                add_link(child,branch)
        add_link(base,body)
        if name in ('glass','white bowl'):
            first=next(v for l in links.values() for v in l.findall('visual'))
            m=first.find('geometry/mesh')
            if m is None:raise ValueError('Container mesh needed to derive hollow dimensions')
            lo,hi=mesh_bounds(directory/m.get('filename'),numbers(m.get('scale'),[1,1,1])*scale)
            pos,q=origin(first,scale);R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            points=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])@R.T+pos
            lo,hi=points.min(axis=0),points.max(axis=0);radius=float(max(hi[:2]-lo[:2])/2)
            center=(lo+hi)/2;bottom=float(lo[2]);height=float(hi[2]-lo[2]);thickness=.004
            g=E.SubElement(body,'geom',name=body_name+'__hollow_bottom',type='cylinder',size=text([radius,thickness/2]),
                           pos=text([center[0],center[1],bottom+thickness/2]),rgba=text(default_color),friction='1.2 .005 .0001')
            own_geoms.append(g)
            for i in range(16):
                angle=2*math.pi*i/16
                g=E.SubElement(body,'geom',name=body_name+f'__rim{i}',type='box',
                    size=text([thickness/2,radius*math.tan(math.pi/16),height/2]),
                    pos=text([center[0]+(radius-thickness/2)*math.cos(angle),center[1]+(radius-thickness/2)*math.sin(angle),bottom+height/2]),
                    euler=text([0,0,angle]),rgba=text(default_color))
                own_geoms.append(g)
            bounds={'lower':[center[0]-(radius-thickness)/math.sqrt(2),center[1]-(radius-thickness)/math.sqrt(2),bottom+thickness],
                    'upper':[center[0]+(radius-thickness)/math.sqrt(2),center[1]+(radius-thickness)/math.sqrt(2),bottom+height]}
            inner_circle={'center_xy_m':center[:2].tolist(),'radius_m':radius-thickness}
        elif container:
            # Bounds are refined from compiled primitive geometry during preflight.
            bounds=None
        else:bounds=None
        if not own_geoms:raise ValueError('No collision geometry for '+name)
        if not container:
            mass=sum(float(x.get('value')) for x in definition.findall('link/inertial/mass')) or .1
            for g in own_geoms:g.set('mass',str(mass/len(own_geoms)))
        entry={'body':moving_drawer.get('name') if moving_drawer is not None else body_name,
               'kind':kind,'half_extents':[.02,.02,.02], 'source_asset':str(urdf),
               'source_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest()}
        if bounds:entry['container_bounds_local']=bounds
        if inner_circle:entry['inner_circle_local']=inner_circle
        objects[name]=entry
        if is_drawer and handle_local is not None:metadata['drawer_handle_local']=numbers(handle_local,[0,0,0]).tolist()
    E.SubElement(root,'keyframe').append(E.Element('key',name='longvil_initial'))
    output.parent.mkdir(parents=True,exist_ok=True);tree.write(output,encoding='unicode')
    metadata.update(scene_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),object_count=len(objects))
    output.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    return metadata


def complete_scene_metadata(output):
    """Derive collision dimensions/interiors from the compiled initial model."""
    import mujoco
    output=Path(output);path=output.with_suffix('.json');metadata=json.loads(path.read_text())
    model=mujoco.MjModel.from_xml_path(str(output));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    # Authoritative assets have different origin conventions. Seat their
    # collision bottoms on the translated table, instead of starting buried
    # inside it. This changes only initial vertical placement, never geometry.
    tree=E.parse(output);world=tree.getroot().find('worldbody');seated=False
    for name,entry in metadata['objects'].items():
        root_name=('object__' if entry['kind']=='dynamic' else 'container__')+slug(name)
        oid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,root_name)
        bottoms=[]
        for g in range(model.ngeom):
            if not(model.geom_contype[g] or model.geom_conaffinity[g]):continue
            body=int(model.geom_bodyid[g])
            while body and body!=oid:body=int(model.body_parentid[body])
            if body!=oid:continue
            local=model.geom_aabb[g];rotation=data.geom_xmat[g].reshape(3,3)
            center=data.geom_xpos[g]+rotation@local[:3]
            extent=np.abs(rotation)@local[3:]
            bottoms.append(float(center[2]-extent[2]))
        if not bottoms:raise ValueError('No collision bottom for '+name)
        desired=metadata.get('surface_z_m',0.)+(.002 if entry['kind']=='dynamic' else 0.)
        delta=desired-min(bottoms)
        entry['vertical_seating_m']=delta
        if abs(delta)>1e-8:
            body=world.find(f"body[@name='{root_name}']")
            point=numbers(body.get('pos'),[0,0,0]);point[2]+=delta;body.set('pos',text(point));seated=True
    if seated:
        tree.write(output,encoding='unicode')
        metadata['scene_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
        model=mujoco.MjModel.from_xml_path(str(output));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    for name,entry in metadata['objects'].items():
        oid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,entry['body'])
        R=data.xmat[oid].reshape(3,3);position=data.xpos[oid]
        boxes=[];all_bounds=[]
        for g in range(model.ngeom):
            if not(model.geom_contype[g] or model.geom_conaffinity[g]):continue
            body=int(model.geom_bodyid[g])
            while body and body!=oid:body=int(model.body_parentid[body])
            if body!=oid:continue
            local=model.geom_aabb[g];rotation=R.T@data.geom_xmat[g].reshape(3,3)
            center=R.T@(data.geom_xpos[g]-position)+rotation@local[:3]
            extent=np.abs(rotation)@local[3:]
            lower,upper=center-extent,center+extent
            all_bounds.append((lower,upper))
            label=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:boxes.append((label,center,extent,lower,upper))
        if not all_bounds:raise ValueError('No compiled collision bounds for '+name)
        lower=np.min([b[0] for b in all_bounds],axis=0);upper=np.max([b[1] for b in all_bounds],axis=0)
        entry['half_extents']=((upper-lower)/2).tolist()
        if entry['kind'] in ('container','drawer') and 'container_bounds_local' not in entry:
            floors=[b for b in boxes if 'bottom' in b[0] or b[0].endswith('_base')]
            if len(floors)!=1:raise ValueError('One unambiguous container floor is required: '+name)
            _,center,extent,lo,hi=floors[0];inner_lo=lo.copy();inner_hi=hi.copy()
            floor_names={b[0] for b in floors}
            walls=[b for b in boxes if b[0] not in floor_names and b[2][2]>extent[2]*2
                   and max(b[2][:2])>=.7*max(extent[:2])]
            for axis in (0,1):
                negative=[b[4][axis] for b in walls if b[1][axis]<center[axis] and b[2][axis]<b[2][1-axis]]
                positive=[b[3][axis] for b in walls if b[1][axis]>center[axis] and b[2][axis]<b[2][1-axis]]
                if not negative or not positive:raise ValueError('Missing hollow container side wall: '+name)
                inner_lo[axis]=max(negative);inner_hi[axis]=min(positive)
            inner_lo[2]=hi[2]
            tall=[b[4][2] for b in walls if b[2][2]>extent[2]*2]
            if not tall:raise ValueError('Missing container lip: '+name)
            inner_hi[2]=min(tall)
            if np.any(inner_hi<=inner_lo):raise ValueError('Inverted hollow interior: '+name)
            entry['container_bounds_local']={'lower':inner_lo.tolist(),'upper':inner_hi.tolist()}
    if metadata.get('drawer_joint'):
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,metadata['drawer_joint'])
        metadata['drawer_axis_base']=data.xaxis[jid].tolist()
    path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    return metadata
