"""Operator-only physical scoring helpers; never imported into model context."""
import numpy as np


def placed_block(obj,tray,size_m=.03):
    try:
        center=np.asarray(obj['position'],dtype=float)
        q=np.asarray(obj['quaternion'],dtype=float)
        if center.shape!=(3,) or q.shape!=(4,) or not np.isfinite(center).all() or not np.isfinite(q).all():
            raise ValueError('invalid_pose')
        if abs(np.linalg.norm(q)-1)>.001:raise ValueError('invalid_orientation')
        if not np.isfinite(size_m) or not 0<size_m<=.1:raise ValueError('unsupported_block_size')
        w,x,y,z=q/np.linalg.norm(q)
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        extent=np.abs(rotation)@np.full(3,size_m/2)
        low,high=center-extent,center+extent
        x1,x2,y1,y2=tray['bounds_xy']
        inside=bool(x1<=low[0] and high[0]<=x2 and y1<=low[1] and high[1]<=y2)
        resting=abs(low[2]-tray['z_surface'])<=.004
        released=not obj.get('grasped',True)
        return {'success':bool(inside and resting and released),'entire_footprint_inside':inside,
            'bottom_at_support_surface':bool(resting),'released':released,
            'bounds_min_m':low.tolist(),'bounds_max_m':high.tolist()}
    except (KeyError,TypeError,ValueError):return {'success':False,'reason':'missing_or_invalid_physical_scoring_evidence'}
