"""Independent image/depth check for the explicit colored-block first task profile."""
import numpy as np
from .grounding import ground_pixel


def recheck_latest_effects(observation, history):
    """Recheck each object's last executed target using current RGB-D only.

    A stage's historical success does not establish its effect still holds.
    Earlier targets for the same colored object are superseded by later moves.
    This first-version check is bounded to the existing colored-block profile.
    """
    from .grounding import GroundedPoint
    latest = {}
    for entry in history:
        decision = entry.get('decision', {})
        if decision.get('action') not in ('pick_place', 'push'):
            continue
        if entry.get('execution', {}).get('status') != 'executed_pending_visual_verification':
            continue
        label = decision.get('object_label', '').lower()
        key = 'red' if 'red' in label or '红' in label else 'blue' if 'blue' in label or '蓝' in label else label
        latest[key] = entry
    checks = []
    for entry in latest.values():
        decision = entry['decision']
        targets = entry.get('grounding_after_preflight', entry.get('grounding', []))
        if len(targets) != 2:
            result = {'verdict': 'unknown', 'reason': 'missing_executed_target'}
        elif (targets[1]['epoch'] != observation['epoch'] or
              targets[1]['calibration_id'] != observation['camera_info']['calibration_id']):
            result = {'verdict': 'unknown', 'reason': 'target_reference_changed'}
        else:
            result = verify_block_destination(observation, decision['object_label'], GroundedPoint(**targets[1]),
                destination_region=entry.get('destination_region') if decision['action']=='pick_place' else None)
        checks.append({'stage_id': decision['stage_id'], 'object_label': decision['object_label'], **result})
    return {'verdict': 'supported' if checks and all(c['verdict'] == 'supported' for c in checks) else 'unknown' if not checks or any(c['verdict'] == 'unknown' for c in checks) else 'refuted',
            'checks': checks, 'method': 'current_rgb_depth_latest_effects_v1'}


def verify_block_destination(observation, label, destination, *, tolerance_m=.04, destination_region=None):
    """No simulator body positions or contacts are read here."""
    image=np.asarray(observation['rgb']).astype(float)
    name=label.lower()
    if 'red' in name or '红' in name:
        mask=(image[:,:,0]>90)&(image[:,:,0]>image[:,:,1]*1.5)&(image[:,:,0]>image[:,:,2]*1.5)
    elif 'blue' in name or '蓝' in name:
        mask=(image[:,:,2]>80)&(image[:,:,2]>image[:,:,0]*1.4)&(image[:,:,2]>image[:,:,1]*1.2)
    else:
        return {'verdict':'unknown','reason':'unsupported_visual_label','method':'rgb_depth_block_v1'}
    # Connected components reject ambiguous same-color instances.
    import cv2
    count, components, stats, centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=25]
    points=[]
    for index in candidates:
        try:
            point=ground_pixel(centers[index],observation['depth'],observation['camera_info'],
                source_stamp_s=observation['source_stamp_s'],epoch=observation['epoch'],max_age_s=1)
            # This verifier is explicitly limited to 30 mm table blocks. A red
            # indicator on the robot well above the table is outside that profile.
            if -.02<=point.xyz_m[2]<=.10:points.append(point)
        except ValueError:continue
    if len(points)!=1:
        return {'verdict':'unknown','reason':'occluded_or_ambiguous','method':'rgb_depth_block_v1'}
    point=points[0]
    distance=float(np.linalg.norm(np.asarray(point.xyz_m[:2])-np.asarray(destination.xyz_m[:2])))
    result={'verdict':'supported' if distance<tolerance_m else 'refuted','distance_xy_m':distance,
            'measured_xyz_m':point.xyz_m,'target_xyz_m':destination.xyz_m,
            'method':'rgb_depth_block_v1','source_stamp_s':point.source_stamp_s,
            'calibration_id':point.calibration_id}
    if destination_region is not None:
        region_check=verify_region_occupancy(observation,point,destination_region)
        result['region_occupancy']=region_check
        if region_check['verdict']!='supported':result['verdict']=region_check['verdict']
    return result


def verify_region_occupancy(observation,point,region):
    """Current RGB-D support for a 30 mm block inside the convex green tray.

    Uses the observed green outline and an observed interior support point.
    The footprint conservatively covers in-plane cube rotation plus 3 mm
    measurement margin. It is not a contact or arbitrary-object verifier.
    """
    import cv2
    from .scene import color_candidates,free_region_pixels
    method='rgb_depth_green_region_v1'
    if region.get('color')!='green':return {'verdict':'unknown','reason':'unsupported_region','method':method}
    current=[c for c in color_candidates(observation['rgb']) if c['color']=='green']
    if len(current)!=1 or current[0]['id']!=region.get('id'):
        return {'verdict':'unknown','reason':'destination_region_missing_or_ambiguous','method':method}
    current=current[0]
    samples=free_region_pixels(observation['rgb'],current,limit=1)
    if not samples:return {'verdict':'unknown','reason':'no_visible_support_surface','method':method}
    try:
        surface=ground_pixel(samples[0],observation['depth'],observation['camera_info'],
            source_stamp_s=observation['source_stamp_s'],epoch=observation['epoch'],max_age_s=1)
    except ValueError:return {'verdict':'unknown','reason':'support_depth_unavailable','method':method}
    hsv=cv2.cvtColor(np.asarray(observation['rgb']),cv2.COLOR_RGB2HSV)
    mask=(hsv[:,:,0]>35)&(hsv[:,:,0]<90)&(hsv[:,:,1]>90)&(hsv[:,:,2]>75)
    x1,y1,x2,y2=current['bbox_xyxy']
    ys,xs=np.where(mask[y1:y2,x1:x2])
    hull=cv2.convexHull(np.column_stack((xs+x1,ys+y1)).astype(np.float32))
    info=observation['camera_info'];inverse=np.linalg.inv(np.asarray(info['base_from_camera']))
    k=np.asarray(info['intrinsics']);half=.03*np.sqrt(2)/2+.003
    pixels=[];inside=True
    for dx,dy in ((-half,-half),(-half,half),(half,half),(half,-half)):
        optical=inverse@np.array([point.xyz_m[0]+dx,point.xyz_m[1]+dy,surface.xyz_m[2],1])
        if optical[2]<=0:return {'verdict':'unknown','reason':'footprint_behind_camera','method':method}
        projected=k@optical[:3];pixel=projected[:2]/projected[2];pixels.append(pixel.tolist())
        inside=inside and cv2.pointPolygonTest(hull,tuple(map(float,pixel)),False)>=0
    height=float(point.xyz_m[2]-surface.xyz_m[2]);height_ok=abs(height-.03)<=.012
    return {'verdict':'supported' if inside and height_ok else 'refuted','method':method,
        'region_id':current['id'],'conservative_footprint_inside':bool(inside),'observed_height_m':height,
        'height_matches_block_profile':height_ok,'support_z_m':surface.xyz_m[2],'footprint_pixels':pixels,
        'source_stamp_s':surface.source_stamp_s}
