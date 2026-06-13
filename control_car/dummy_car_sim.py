#!/usr/bin/env python3
"""
Closed-loop car simulator for testing pure pursuit + obstacle avoidance
without real hardware or MOCAP.

What this node does:
  1. Publishes a /path (L-shape: east then north, configurable via params)
  2. Reads /ackermann_drive from pure_pursuit_node and integrates the bicycle
     kinematic model to move the simulated car
  3. Publishes /kinematic_state with the simulated position, velocity, heading
  4. Publishes /allowed_to_move True

MOCAP quaternion convention used here:
  quaternion encodes actual_yaw directly (yaw_mocap = actual_yaw).
  pure_pursuit adds π → theta = actual_yaw + π (backward convention).
  This matches the assumption in _map_to_car_frame and pure_pursuit_control.

Run order:
  Terminal 1: ros2 run control_car pure_persuit_node
  Terminal 2: ros2 run control_car dummy_car_sim
  Terminal 3: ros2 run control_car dummy_tracked_objects --ros-args -p scenario:=stationary

Verify in RViz:
  - /path         : planned path (green line)
  - /ackermann_drive : speed/steering commands
  - /pure_pursuit/lookahead_marker : yellow sphere (look-ahead point)
  - /dummy_obstacle_markers : green cube (simulated person)
  - /obstacle_information : True when hard-stop triggered
"""

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from curobot_msgs.msg import KinematicState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

DT    = 1.0 / 30.0   # 30 Hz — same as pure_pursuit control loop
L_CAR = 0.3           # wheelbase (m) — must match HagenRobot.L in pure_persuit_node


class DummyCarSim(Node):

    def __init__(self):
        super().__init__('dummy_car_sim')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('start_x',    0.5)
        self.declare_parameter('start_y',    1.0)
        self.declare_parameter('start_yaw',  0.0)   # radians; 0 = east
        # Waypoints: list of (x, y) the path goes through
        # Default: east 3.5 m then north 3 m (L-shape)
        self.declare_parameter('wp_x', [0.5, 4.0, 4.0])
        self.declare_parameter('wp_y', [1.0, 1.0, 4.0])

        sx  = float(self.get_parameter('start_x').value)
        sy  = float(self.get_parameter('start_y').value)
        yaw = float(self.get_parameter('start_yaw').value)
        wp_x = list(self.get_parameter('wp_x').value)
        wp_y = list(self.get_parameter('wp_y').value)

        # ── Car state ────────────────────────────────────────────────────────
        self._x     = sx
        self._y     = sy
        self._yaw   = yaw    # actual forward heading in map frame (radians)
        self._v     = 0.0    # speed from /ackermann_drive
        self._delta = 0.0    # physics steering angle (after undoing invert_steering_sign)

        # ── QoS profiles ────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self._kin_pub  = self.create_publisher(KinematicState, '/kinematic_state', best_effort_qos)
        self._path_pub = self.create_publisher(Path, '/path', latched_qos)
        self._atm_pub  = self.create_publisher(Bool, '/allowed_to_move', reliable_qos)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(
            AckermannDrive, '/ackermann_drive',
            self._ackermann_cb, reliable_qos,
        )

        # ── Publish static data immediately (latched /path) ──────────────────
        self._build_and_publish_path(wp_x, wp_y)
        self._publish_allowed_to_move()
        # Re-publish every 5 s so late-starting nodes receive them
        self.create_timer(5.0, self._periodic_announce)

        # ── Main sim loop ────────────────────────────────────────────────────
        self.create_timer(DT, self._step)

        self.get_logger().info(
            f"DummyCarSim started\n"
            f"  start=({sx:.2f}, {sy:.2f})  yaw={math.degrees(yaw):.1f}°\n"
            f"  path waypoints: {list(zip(wp_x, wp_y))}\n"
            "Waiting for /ackermann_drive from pure_pursuit_node..."
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _ackermann_cb(self, msg: AckermannDrive):
        """
        pure_pursuit_node applies invert_steering_sign=True:
          cmd = -delta_physics
        Undo it here so the bicycle model uses the correct physics steering.
        """
        self._v     = float(msg.speed)
        self._delta = -float(msg.steering_angle)   # undo invert_steering_sign

    # ── Simulation step ───────────────────────────────────────────────────────

    def _step(self):
        if self._v > 0.01:
            # Bicycle kinematic model in map frame (forward integration)
            self._x   += self._v * math.cos(self._yaw) * DT
            self._y   += self._v * math.sin(self._yaw) * DT
            self._yaw += (self._v / L_CAR) * math.tan(self._delta) * DT
            self._yaw  = math.atan2(math.sin(self._yaw), math.cos(self._yaw))

        self._publish_kinematic_state()

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_kinematic_state(self):
        msg = KinematicState()
        now = self.get_clock().now().to_msg()
        msg.header.stamp    = now
        msg.header.frame_id = 'map'
        msg.child_frame_id  = 'base_link'

        msg.pose_with_covariance.pose.position.x = self._x
        msg.pose_with_covariance.pose.position.y = self._y
        msg.pose_with_covariance.pose.position.z = 0.0

        # Simulate MOCAP 180° rotation: publish quaternion at actual_yaw + π.
        # pure_pursuit reads yaw_mocap = actual + π and uses it directly as theta
        # (backward convention), so _map_to_car_frame gives cx > 0 for objects ahead.
        mocap_yaw = self._yaw + math.pi
        half = mocap_yaw / 2.0
        msg.pose_with_covariance.pose.orientation.z = math.sin(half)
        msg.pose_with_covariance.pose.orientation.w = math.cos(half)

        # Velocity in map frame (actual direction, NOT backward convention)
        msg.twist_with_covariance.twist.linear.x = self._v * math.cos(self._yaw)
        msg.twist_with_covariance.twist.linear.y = self._v * math.sin(self._yaw)

        self._kin_pub.publish(msg)

    def _build_and_publish_path(self, wp_x, wp_y):
        if len(wp_x) < 2 or len(wp_x) != len(wp_y):
            self.get_logger().error("wp_x / wp_y must have matching length ≥ 2")
            return

        # Densify to 5 cm spacing (same as pure_pursuit _densify_path)
        xs, ys = [], []
        for i in range(len(wp_x) - 1):
            x0, y0 = wp_x[i], wp_y[i]
            x1, y1 = wp_x[i + 1], wp_y[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            n = max(2, int(seg_len / 0.05))
            xs.extend(np.linspace(x0, x1, n, endpoint=False).tolist())
            ys.extend(np.linspace(y0, y1, n, endpoint=False).tolist())
        xs.append(float(wp_x[-1]))
        ys.append(float(wp_y[-1]))

        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for x, y in zip(xs, ys):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)
        self.get_logger().info(
            f"Published /path with {len(path_msg.poses)} waypoints "
            f"({wp_x[0]:.1f},{wp_y[0]:.1f}) → ... → ({wp_x[-1]:.1f},{wp_y[-1]:.1f})"
        )

    def _publish_allowed_to_move(self):
        msg = Bool()
        msg.data = True
        self._atm_pub.publish(msg)

    def _periodic_announce(self):
        # Re-publish /allowed_to_move so nodes that start late receive it
        self._publish_allowed_to_move()


def main(args=None):
    rclpy.init(args=args)
    node = DummyCarSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
