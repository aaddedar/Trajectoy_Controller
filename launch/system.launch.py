from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value='/dev/ttyACM0',
        description='Serial port for ros2arduino',
    )

    # ── Node 1: Path Planner (starts immediately) ──────────────────────────────
    path_planner = Node(
        package='path_planner',
        executable='astar_path_planner',
        name='astar_path_planner_node',
        output='screen',
        parameters=[{
            'half_car_width':             0.2,
            'clearance_m':                0.01,
            'step_size_cells':            22,
            'goal_search_tolerance_m':    0.1,
            'goal_reached_tolerance_m':   0.2,
            'treat_unknown_as_obstacle':  True,
        }],
    )

    # ── Node 2: Pure Pursuit controller (starts after 2 s) ─────────────────────
    pure_pursuit = TimerAction(
        period=2.0,
        actions=[
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
                    'avoidance_start_dist': 2.5,
                }],
            )
        ],
    )

    # ── Node 3: Decision Core (starts after 4 s) ───────────────────────────────
    decision_core = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='decision_core',
                executable='decision_core',
                name='decision_core_node',
                output='screen',
            )
        ],
    )

    # ── Node 4: ros2arduino (starts after 6 s) ─────────────────────────────────
    ros2arduino = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='ros2arduino',
                executable='ros2arduino_node',
                name='ros2arduino_node',
                output='screen',
                parameters=[{
                    'serial_port': LaunchConfiguration('serial_port'),
                }],
                arguments=['--serial_port', LaunchConfiguration('serial_port')],
            )
        ],
    )

    return LaunchDescription([
        serial_port_arg,
        path_planner,
        pure_pursuit,
        decision_core,
        ros2arduino,
    ])
