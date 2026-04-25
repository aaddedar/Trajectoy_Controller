from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='control_car',
                executable='pure_persuit_node',
                name='pure_pursuit_node',
                output='screen',
            ),
        ]
    )
