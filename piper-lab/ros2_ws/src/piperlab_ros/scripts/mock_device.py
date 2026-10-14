#!/usr/bin/env python3
"""Synthetic device transport fixture, not a physics or grasp simulator."""
import io
import numpy as np
from PIL import Image as PILImage, ImageDraw
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage, Image, CameraInfo
from std_msgs.msg import String
from std_srvs.srv import Empty, SetBool

class MockDevice(Node):
    def __init__(self):
        super().__init__('piper_mock_device')
        self.q = [0., 1., -1., 0., 0., 0., .05]
        self.names = [f'joint{i}' for i in range(1,7)] + ['gripper']
        self.enabled = False
        self.feedback = self.create_publisher(JointState, '/piper/feedback/joint_states', 1)
        self.rgb = self.create_publisher(CompressedImage, '/lab/rgb', 1)
        self.depth = self.create_publisher(Image, '/lab/depth', 1)
        self.info = self.create_publisher(CameraInfo, '/lab/camera_info', 1)
        self.events = self.create_publisher(String, '/lab/events', 10)
        self.create_subscription(JointState, '/piper/control/joint_states', self.command, 1)
        self.create_service(SetBool, '/piper/enable_agx_arm', self.enable)
        self.create_service(SetBool, '/piper/control_enable', self.enable)
        self.create_service(Empty, '/piper/emergency_stop', lambda req, res: res)
        self.create_timer(.02, self.state)
        self.create_timer(1/30, self.camera)
        self.create_timer(1, lambda: self.events.publish(String(data='{"kind":"provenance","synthetic":true,"depth_scale_m":0.001}')))
    def enable(self, req, res):
        self.enabled=req.data; res.success=True; return res
    def command(self, msg):
        if self.enabled and msg.name == self.names and len(msg.position)==7:
            self.q=list(msg.position)
    def state(self):
        msg=JointState(); msg.header.stamp=self.get_clock().now().to_msg()
        msg.name=self.names; msg.position=self.q; self.feedback.publish(msg)
    def camera(self):
        stamp=self.get_clock().now().to_msg()
        frame=PILImage.new('RGB',(640,480),(30,40,55)); draw=ImageDraw.Draw(frame)
        x=320+int(self.q[0]*70); draw.rectangle((x-25,200,x+25,250), fill=(220,100,40))
        draw.text((12,12),'SYNTHETIC - NOT REAL ROBOT DATA',fill=(255,255,255))
        buffer=io.BytesIO(); frame.save(buffer,format='JPEG',quality=90)
        rgb=CompressedImage(); rgb.header.stamp=stamp; rgb.header.frame_id='camera_color_optical_frame'
        rgb.format='jpeg'; rgb.data=buffer.getvalue(); self.rgb.publish(rgb)
        depth=Image(); depth.header=rgb.header; depth.width=640; depth.height=480
        depth.encoding='16UC1'; depth.is_bigendian=0; depth.step=1280
        depth.data=np.full((480,640),600,dtype='<u2').tobytes(); self.depth.publish(depth)
        info=CameraInfo(); info.header=rgb.header; info.width=640; info.height=480
        info.distortion_model='plumb_bob'; info.d=[0.]*5
        info.k=[600.,0.,320.,0.,600.,240.,0.,0.,1.]
        info.r=[1.,0.,0.,0.,1.,0.,0.,0.,1.]
        info.p=[600.,0.,320.,0.,0.,600.,240.,0.,0.,0.,1.,0.]
        self.info.publish(info)

def main():
    rclpy.init(); node=MockDevice()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()
if __name__=='__main__': main()
