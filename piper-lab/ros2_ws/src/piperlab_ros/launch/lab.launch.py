"""Bring up the Piper Lab ROS stack in simulation or on real hardware.

Provenance: the structure below is the exact shape verified to launch in this
environment (bisection step x3 plus the simulation/real branches). Two earlier
revisions aborted with ``TypeError: Expected 'value' to be one of [...] but got
'()' of type 'tuple'`` inside launch_ros parameter evaluation, before any node
started; docs/acceptance.md records what was ruled out and what remains an open
question. Keep this file close to the verified shape when editing.

MoveIt nodes can be skipped with ``include_moveit:=false`` so the ROS I/O layer
(runtime + mock device + bridges) can be verified on its own; that path is also
available as ``lab-io.launch.py``.
"""
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from piperlab.safety import load_config, commissioning_errors, verify_can_identity

CONFIG_ARG = DeclareLaunchArgument(
    'config', default_value='/home/ros/piper-lab/config/hardware.yaml')
INCLUDE_MOVEIT_ARG = DeclareLaunchArgument('include_moveit', default_value='true')


def _moveit(context, config):
    moveit_dir = Path(get_package_share_directory('agx_arm_moveit'))
    sys.path.insert(0, str(moveit_dir / 'launch'))
    from _moveit_config_builder import build_moveit_config
    context.launch_configurations.update(
        arm_type='piper_x', effector_type='agx_gripper', revo2_type='left',
        tcp_offset=str(config.get('tcp_m_rad') or [0.0] * 6), namespace='', follow='true',
        feedback_topic='/lab/observation', control_topic='/lab/moveit_unused')
    return build_moveit_config(context)


def setup(context):
    config_path = LaunchConfiguration('config').perform(context)
    config = load_config(config_path)
    real = config['mode'] == 'real'
    if real:
        errors = commissioning_errors(config)
        if errors:
            raise RuntimeError('Real bringup blocked: ' + ', '.join(errors))
        verify_can_identity(config)
        if config['gripper']['model'] != 'agx_gripper':
            raise RuntimeError(
                'First release supports agx_gripper only; other effectors need '
                'validated mapping')

    include_moveit = str(
        LaunchConfiguration('include_moveit').perform(context)).lower() not in (
            'false', '0', 'no')
    if not include_moveit:
        nodes = [
            Node(package='piperlab_ros', executable='runtime.py',
                 parameters=[{'config': config_path}], output='screen'),
            Node(package='piperlab_ros', executable='mock_device.py', output='screen'),
            Node(package='rosbridge_server', executable='rosbridge_websocket',
                 parameters=[{'address': '127.0.0.1', 'port': 9090}], output='screen'),
            Node(package='foxglove_bridge', executable='foxglove_bridge',
                 parameters=[{'address': '127.0.0.1', 'port': 8765}], output='screen'),
        ]
        return nodes

    moveit = _moveit(context, config)
    servo = {'publish_period': .05, 'move_group_name': 'arm'}
    nodes = [
        Node(package='piperlab_ros', executable='runtime.py',
             parameters=[{'config': config_path}], output='screen'),
        Node(package='piperlab_ros', executable='mock_device.py', output='screen'),
        Node(package='moveit_ros_move_group', executable='move_group',
             parameters=[moveit.to_dict(), {'allow_trajectory_execution': False}],
             remappings=[('joint_states', '/lab/observation')], output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[moveit.robot_description],
             remappings=[('joint_states', '/lab/observation')]),
        Node(package='moveit_servo', executable='servo_node', name='servo_node',
             parameters=[moveit.robot_description, moveit.robot_description_semantic,
                         moveit.robot_description_kinematics, moveit.joint_limits,
                         {'moveit_servo': servo}], output='screen'),
        Node(package='rosbridge_server', executable='rosbridge_websocket',
             parameters=[{'address': '127.0.0.1', 'port': 9090}], output='screen'),
        Node(package='foxglove_bridge', executable='foxglove_bridge',
             parameters=[{'address': '127.0.0.1', 'port': 8765}], output='screen'),
    ]
    if real:
        nodes = nodes[:2] + [
            Node(package='agx_arm_ctrl', executable='agx_arm_ctrl_single',
                 namespace='piper',
                 parameters=[{'arm_type': 'piper_x',
                              'can_port': config['can']['interface'],
                              'effector_type': 'agx_gripper',
                              'auto_enable': False,
                              'reconnect_auto_enable': False,
                              'control_enabled': False,
                              'fast_mode': False,
                              'command_timeout_s': 0.2,
                              'speed_percent': 5,
                              'fw_version': config['can']['firmware'],
                              'tcp_offset': config['tcp_m_rad']}],
                 output='screen'),
        ] + nodes[2:]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        CONFIG_ARG, INCLUDE_MOVEIT_ARG, OpaqueFunction(function=setup)])
