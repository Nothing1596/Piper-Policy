"""Bring up the lab's ROS I/O layer only: runtime, mock device, bridges.

Why this file exists
--------------------
``lab.launch.py`` aborts in this environment before the bridge nodes start:

    TypeError: Expected 'value' to be one of [float, int, str, bool, bytes],
               but got '()' of type 'tuple'

launch_ros raises that from ``evaluate_parameter_dict`` and the message does not
name the offending node or parameter. It happens with and without MoveIt, so it
is not caused by the MoveIt config; it is unresolved and tracked in
docs/acceptance.md.

This launch file therefore starts the subset that the ROS acceptance test needs
(runtime -> /lab/observation, mock_device -> simulated arm, rosbridge, foxglove)
and deliberately contains no nested parameter dicts, so it cannot hit the same
unknown code path. MoveIt bring-up stays in lab.launch.py.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from piperlab.safety import load_config


def setup(context):
    config_path = LaunchConfiguration('config').perform(context)
    config = load_config(config_path)
    if config['mode'] != 'simulation':
        # This file never starts a hardware driver; refuse rather than pretend.
        raise RuntimeError(
            f"lab-io.launch.py is simulation-only, config mode is {config['mode']!r}"
        )
    return [
        Node(package='piperlab_ros', executable='runtime.py',
             parameters=[{'config': config_path}], output='screen'),
        Node(package='piperlab_ros', executable='mock_device.py', output='screen'),
        Node(package='rosbridge_server', executable='rosbridge_websocket',
             parameters=[{'address': '127.0.0.1', 'port': 9090}], output='screen'),
        Node(package='foxglove_bridge', executable='foxglove_bridge',
             parameters=[{'address': '127.0.0.1', 'port': 8765}], output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='/home/ros/piper-lab/config/hardware.yaml',
        ),
        OpaqueFunction(function=setup),
    ])
