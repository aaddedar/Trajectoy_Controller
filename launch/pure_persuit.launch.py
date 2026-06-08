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
                parameters=[{
                    'wheelbase': 0.5,   # TODO: set to actual front-to-rear axle distance (m)
                    'L_d':       0.2,   # base look-ahead distance (m)
                    'k':         0.1,   # speed-proportional look-ahead gain
                }],
            ),
        ]
    )
