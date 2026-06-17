from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='control_car',
            executable='pure_persuit_node',
            name='pure_pursuit_node',
            output='screen',
            parameters=[{
                'command_speed':        1.4,
                'min_curve_speed':      0.9,
                'L_d':                  0.30,
                'invert_steering':      False,
                'person_stop_dist':     0.4,
                'person_slowdown_dist': 1.5,
                'avoidance_clearance':  0.3,
                'avoidance_predict_t':  1.5,
                'avoidance_start_dist': 1.5,
                'avoidance_clear_hold': 6.0,
            }],
        ),
    ])
