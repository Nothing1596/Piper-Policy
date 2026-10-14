"""Camera-only qualification. Does not create an arm driver or any motion client."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('serial',description='D435 serial; prefix with underscore to force string'),
        Node(package='realsense2_camera',executable='realsense2_camera_node',name='camera',namespace='camera',output='screen',
             parameters=[{'serial_no':ParameterValue(LaunchConfiguration('serial'),value_type=str),'enable_color':True,'enable_depth':True,
                          'rgb_camera.color_profile':'640x480x30','depth_module.depth_profile':'640x480x30','enable_sync':True}],
             remappings=[('~/color/image_raw/compressed','/lab/rgb'),('~/depth/image_rect_raw','/lab/depth'),
                         ('~/color/camera_info','/lab/camera_info'),('~/depth/camera_info','/lab/depth_camera_info')])])
