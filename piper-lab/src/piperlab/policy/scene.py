"""Measured candidates for the colored tabletop profile, from pixels only."""
from __future__ import annotations
import numpy as np


def color_candidates(rgb):
    import cv2
    hsv=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2HSV)
    hue,sat,val=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    masks={
        'red':((hue<12)|(hue>168))&(sat>110)&(val>65),
        'blue':(hue>95)&(hue<132)&(sat>100)&(val>60),
        'green':(hue>35)&(hue<90)&(sat>90)&(val>75)}
    records=[]
    for color,mask in masks.items():
        n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        order=sorted((i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=40),
                     key=lambda i:int(stats[i,cv2.CC_STAT_AREA]),reverse=True)
        for number,i in enumerate(order):
            x,y,w,h,area=map(int,stats[i])
            # Guarantee chosen pixel belongs to the region, even for a hollow shape.
            yy,xx=np.where(labels==i)
            target=np.argmin((xx-centers[i,0])**2+(yy-centers[i,1])**2)
            records.append({'id':f'{color}-{number}','color':color,'bbox_xyxy':[x,y,x+w,y+h],
                            'pixel_xy':[int(xx[target]),int(yy[target])],'area_px':area,
                            'method':'hsv_connected_component','semantic_class':'unknown'})
    return records


def measured_candidates(observation):
    """Colored regions plus bounded push targets derived from calibrated depth."""
    from .grounding import ground_pixel
    records=color_candidates(observation['rgb'])
    if 'camera_info' not in observation or 'depth' not in observation:return records
    info=observation['camera_info']
    transform=np.asarray(info['base_from_camera']);k=np.asarray(info['intrinsics'])
    inverse=np.linalg.inv(transform)
    kwargs=dict(source_stamp_s=observation['source_stamp_s'],epoch=observation['epoch'],
                now=observation['source_stamp_s'])
    for region in list(records):
        if region['color'] not in ('red','blue'):continue
        try:
            source=ground_pixel(region['pixel_xy'],observation['depth'],info,**kwargs)
            start=np.array(source.xyz_m)
            if not .02<start[2]<.08:continue
            start[2]-=.03  # Explicit 30 mm tabletop block profile.
            ray=start-transform[:3,3]
            for axis,sign,name in ((0,1,'right'),(0,-1,'left'),(1,1,'down'),(1,-1,'up')):
                tangent=transform[:3,axis]-ray*(transform[2,axis]/ray[2])
                tangent[2]=0;tangent=sign*tangent/np.linalg.norm(tangent)
                target=start+.08*tangent
                optical=(inverse@np.r_[target,1])[:3]
                uv=k@optical;pixel=(uv[:2]/uv[2]).tolist()
                try:
                    measured=ground_pixel(pixel,observation['depth'],info,**kwargs)
                    if np.linalg.norm(np.array(measured.xyz_m)-target)>.008:continue
                except ValueError:continue
                records.append({'id':region['id']+'-push-'+name,'color':'free',
                    'pixel_xy':pixel,'source_candidate_id':region['id'],'image_direction':name,
                    'displacement_m':.08,'method':'rgb_depth_table_push_proposal',
                    'semantic_class':'free_table_destination'})
        except ValueError:continue
    return records


def resolve_object_point(rgb,label,proposed_pixel):
    names={'red':('red','红'),'blue':('blue','蓝'),'green':('green','绿')}
    colors=[c for c,aliases in names.items() if any(a in label.lower() for a in aliases)]
    if len(colors)!=1:
        raise ValueError('object_color_unsupported_or_ambiguous')
    candidates=[c for c in color_candidates(rgb) if c['color']==colors[0]]
    # Labels guide which measured region; the model's coarse point cannot create geometry.
    h,w=np.asarray(rgb).shape[:2]
    if not candidates:
        raise ValueError('object_not_observed')
    distance=lambda c:float(np.linalg.norm((np.asarray(c['pixel_xy'])-np.asarray(proposed_pixel))/np.array([w,h])))
    candidates.sort(key=distance)
    if len(candidates)>1 and distance(candidates[1])<2*distance(candidates[0])+.02:
        raise ValueError('object_instances_ambiguous')
    result=candidates[0]
    delta=np.asarray(result['pixel_xy'])-np.asarray(proposed_pixel)
    if np.linalg.norm(delta/np.array([w,h]))>.30:
        raise ValueError('semantic_point_and_visual_region_conflict')
    return result


def free_region_pixels(rgb,region,*,limit=48,include_grid=False):
    """Interior free-space proposals; never place onto a differently colored block."""
    import cv2
    hsv=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2HSV)
    if region['color']!='green':return [region['pixel_xy']]
    mask=((hsv[:,:,0]>35)&(hsv[:,:,0]<90)&(hsv[:,:,1]>90)&(hsv[:,:,2]>75)).astype(np.uint8)
    x1,y1,x2,y2=region['bbox_xyxy']
    bounded=np.zeros_like(mask);bounded[y1:y2,x1:x2]=mask[y1:y2,x1:x2]
    distance=cv2.distanceTransform(bounded,cv2.DIST_L2,5)
    points=[]
    for _ in range(limit):
        _,radius,_,pixel=cv2.minMaxLoc(distance)
        if radius<12:break
        points.append(list(pixel))
        cv2.circle(distance,pixel,8,0,thickness=-1)
    if include_grid:
        # A circle-distance ranking can miss a metric square footprint under
        # perspective, especially beside an already placed block. Preserve the
        # ranked proposals, then cover the region with a bounded pixel grid.
        # All proposals still require live depth, footprint and IK validation.
        stride=max(4,int(np.ceil(np.sqrt((x2-x1)*(y2-y1)/1024))))
        seen={tuple(p) for p in points}
        for y in range(y1,y2,stride):
            for x in range(x1,x2,stride):
                if bounded[y,x] and (x,y) not in seen:
                    points.append([x,y])
    return points


def footprint_is_free(rgb,region,point,camera_info,half_size=.030):
    """Project a metric footprint into the chosen region to reject occupied pixels."""
    import cv2
    hsv=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2HSV)
    mask=(hsv[:,:,0]>35)&(hsv[:,:,0]<90)&(hsv[:,:,1]>90)&(hsv[:,:,2]>75)
    transform=np.asarray(camera_info['base_from_camera']);k=np.asarray(camera_info['intrinsics'])
    inv=np.linalg.inv(transform)
    for dx in np.linspace(-half_size,half_size,5):
        for dy in np.linspace(-half_size,half_size,5):
            xyz=np.array(point.xyz_m)+[dx,dy,0]
            optical=(inv@np.r_[xyz,1])[:3]
            if optical[2]<=0:return False
            pixel=k@optical;u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
            if not 0<=v<mask.shape[0] or not 0<=u<mask.shape[1] or not mask[v,u]:return False
    return True
