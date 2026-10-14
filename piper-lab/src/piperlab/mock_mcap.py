"""Portable ROS 2 CDR/MCAP fixture. Explicitly synthetic; requires no ROS daemon."""
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from mcap_ros2.writer import Writer
from . import data

HEADER = '''
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec'''
SCHEMAS = {
    'joint': ('sensor_msgs/msg/JointState', 'std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort' + HEADER),
    'rgb': ('sensor_msgs/msg/CompressedImage', 'std_msgs/Header header\nstring format\nuint8[] data' + HEADER),
    'depth': ('sensor_msgs/msg/Image', 'std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data' + HEADER),
    'info': ('sensor_msgs/msg/CameraInfo', '''std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi''' + HEADER + '''
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify'''),
    'event': ('std_msgs/msg/String', 'string data'),
}

def generate(output, episodes=5, frames=40, fps=20):
    path=Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(data.MOCK_SEED); size=data.MOCK_IMAGE_SIZE
    with path.open('xb') as stream:
        writer=Writer(stream)
        schemas={key:writer.register_msgdef(*value) for key,value in SCHEMAS.items()}
        def put(topic, kind, t, message):
            writer.write_message(topic, schemas[kind], message, log_time=t, publish_time=t)
        def event(t, kind, ep):
            put('/lab/events','event',t,{'data':json.dumps({'event':kind,'episode_index':ep,'task':'synthetic pick and place','synthetic':True})})
        for ep in range(episodes):
            base=data.MOCK_BASE_STAMP_NS + ep*(frames+5)*int(1e9/fps)
            states, actions=data._mock_joint_trajectories(ep,frames,fps,rng)
            event(base,'episode_start',ep)
            for k in range(frames):
                t=base+k*int(1e9/fps)
                header={'stamp':{'sec':t//10**9,'nanosec':t%10**9},'frame_id':'synthetic_camera'}
                for topic, q in [('/lab/observation',states[k]),('/lab/applied_action',actions[k])]:
                    put(topic,'joint',t,{'header':header,'name':data.JOINT_NAMES,'position':q.tolist(),'velocity':[],'effort':[]})
                image=io.BytesIO(); Image.fromarray(data._mock_rgb_frame(ep,k,size)).save(image,format='JPEG',quality=95)
                put('/lab/rgb','rgb',t,{'header':header,'format':'jpeg','data':image.getvalue()})
                put('/lab/depth','depth',t,{'header':header,'height':size,'width':size,'encoding':'16UC1','is_bigendian':0,'step':size*2,'data':data._mock_depth_frame(k,size).astype('<u2').tobytes()})
                put('/lab/camera_info','info',t,{'header':header,'height':size,'width':size,'distortion_model':'plumb_bob','d':[0.]*5,'k':[200.,0.,112.,0.,200.,112.,0.,0.,1.],'r':np.eye(3).reshape(-1).tolist(),'p':[200.,0.,112.,0.,0.,200.,112.,0.,0.,0.,1.,0.],'binning_x':0,'binning_y':0,'roi':{'x_offset':0,'y_offset':0,'height':0,'width':0,'do_rectify':False}})
            event(t,'episode_end',ep)
        writer.finish()
    return {'output':str(path),'episodes':episodes,'frames':frames*episodes,'synthetic':True,'encoding':'ROS2 CDR / MCAP'}
