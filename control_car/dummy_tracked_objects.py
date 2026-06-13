#!/usr/bin/env python3
"""
Dummy publisher for /tracked_objects to test DWA obstacle avoidance
without needing the real camera → detectnet → localizer → KF pipeline.

Publishes a simulated person in the EXACT same format the KF tracker uses:
  - bbox.center.position  = map-frame position
  - bbox.size             = physical dimensions
  - results[0].class_id  = "person"
  - results[0].pose.pose.position = (vx, vy, vz) map-frame velocity

Subscribes to /kinematic_state so the person is placed relative to the
car's actual position (works wherever the car starts on the map).

Scenarios (set via 'scenario' ROS parameter):
  stationary  — person 2.0 m ahead, stationary           → TTC=1.5s → hard stop
  approaching — person 4.0 m ahead, walks toward car      → higher closing speed → stop
  crossing    — person 3.0 m ahead-right, walks left      → DWA should steer around
  zigzag      — person alternates sides every 4 s          → stress-tests DWA replanning
  away        — person 3.0 m ahead, moves same direction  → TTC=inf → no reaction
  fast_cross  — person 2.0 m ahead, runs fast laterally   → DWA short window avoidance

Usage:
  ros2 run control_car dummy_tracked_objects --ros-args -p scenario:=stationary
  ros2 run control_car dummy_tracked_objects --ros-args -p scenario:=crossing
  ros2 run control_car dummy_tracked_objects --ros-args -p scenario:=away
"""

import math

import rclpy
from curobot_msgs.msg import KinematicState
from geometry_msgs.msg import Quaternion
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


class DummyTrackedObjects(Node):

    def __init__(self):
        super().__init__('dummy_tracked_objects')

        self.declare_parameter('scenario',     'stationary')
        self.declare_parameter('publish_hz',   10.0)

        self._scenario = self.get_parameter('scenario').value
        rate           = float(self.get_parameter('publish_hz').value)
        self._dt       = 1.0 / rate

        # Car state (updated from /kinematic_state)
        self._car_x     = 0.0
        self._car_y     = 0.0
        self._car_theta = 0.0
        self._car_known = False

        # Person state (map frame, set once car position is known)
        self._px   = 0.0
        self._py   = 0.0
        self._pvx  = 0.0
        self._pvy  = 0.0
        self._initialized = False
        self._sim_t = 0.0   # simulation time since person was placed

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

        self._pub_tracked = self.create_publisher(
            Detection3DArray, '/tracked_objects', reliable_qos)
        self._pub_markers = self.create_publisher(
            MarkerArray, '/dummy_obstacle_markers', 10)

        self.create_subscription(
            KinematicState, '/kinematic_state',
            self._kin_cb, best_effort_qos)

        self.create_timer(self._dt, self._publish)

        VALID = ['stationary', 'approaching', 'crossing', 'zigzag', 'away', 'fast_cross']
        if self._scenario not in VALID:
            self.get_logger().warn(
                f"Unknown scenario '{self._scenario}'. Valid: {VALID}. "
                "Defaulting to 'stationary'."
            )
            self._scenario = 'stationary'

        self.get_logger().info(
            f"Dummy object publisher ready — scenario='{self._scenario}' "
            f"at {rate:.0f} Hz on /tracked_objects\n"
            "Waiting for first /kinematic_state to place person..."
        )

    # ------------------------------------------------------------------

    def _kin_cb(self, msg: KinematicState):
        self._car_x = msg.pose_with_covariance.pose.position.x
        self._car_y = msg.pose_with_covariance.pose.position.y
        vx = msg.twist_with_covariance.twist.linear.x
        vy = msg.twist_with_covariance.twist.linear.y
        if math.hypot(vx, vy) > 0.05:
            # Use actual forward direction so car_to_map places persons correctly.
            # (pure_pursuit uses backward convention internally, but here we need
            # forward so that car_to_map(fwd=2.5) puts the person AHEAD of the car.)
            self._car_theta = math.atan2(vy, vx)
        self._car_known = True

    def _init_person(self):
        """Place person at scenario-specific position relative to car."""
        ct = math.cos(self._car_theta)
        st = math.sin(self._car_theta)

        def car_to_map(fwd, lat):
            """Convert (forward, left-lateral) offsets to map frame."""
            return (
                self._car_x + fwd * ct - lat * st,
                self._car_y + fwd * st + lat * ct,
            )

        s = self._scenario

        if s == 'stationary':
            # 2.0 m ahead, not moving.
            # At car speed 1.3 m/s: TTC = 2.0/1.3 = 1.54 s ≈ TTC_HARD_STOP (1.5 s) → stop
            self._px, self._py = car_to_map(2.0, 0.0)
            self._pvx, self._pvy = 0.0, 0.0

        elif s == 'approaching':
            # 4.0 m ahead, walking toward car at 0.4 m/s.
            # Closing speed = 1.3 + 0.4 = 1.7 m/s → TTC = 4.0/1.7 = 2.35 s → DWA avoid
            self._px, self._py = car_to_map(4.0, 0.0)
            self._pvx = -ct * 0.4   # toward car = opposite of car forward
            self._pvy = -st * 0.4

        elif s == 'crossing':
            # 3.0 m ahead-right, walks LEFT at 0.5 m/s (crosses car's path).
            # DWA should steer right to avoid.
            self._px, self._py = car_to_map(3.0, -1.5)
            self._pvx = -st * 0.5   # left = 90° CCW from forward
            self._pvy =  ct * 0.5

        elif s == 'zigzag':
            # Alternates crossing direction every 4 s — stress-tests DWA replanning.
            self._px, self._py = car_to_map(3.0, -0.8)
            self._pvx = -st * 0.4
            self._pvy =  ct * 0.4

        elif s == 'away':
            # 3.0 m ahead, moving SAME direction as car at 0.6 m/s (faster).
            # Closing speed = 1.3 - 0.6 = 0.7 m/s → TTC = 3.0/0.7 = 4.3 s > TTC_AVOID
            # → NO reaction expected (person running away faster than car)
            self._px, self._py = car_to_map(3.0, 0.0)
            self._pvx = ct * 0.6    # same direction as car
            self._pvy = st * 0.6

        elif s == 'fast_cross':
            # 2.0 m ahead, running fast laterally (1.2 m/s) from right to left.
            # Short time window for DWA to react before person clears corridor.
            self._px, self._py = car_to_map(2.0, -0.6)
            self._pvx = -st * 1.2
            self._pvy =  ct * 1.2

        self._initialized = True
        self.get_logger().info(
            f"Person placed: scenario='{self._scenario}' "
            f"map=({self._px:.2f},{self._py:.2f}) "
            f"vel=({self._pvx:.2f},{self._pvy:.2f}) m/s"
        )

    # ------------------------------------------------------------------

    def _publish(self):
        if not self._car_known:
            return

        if not self._initialized:
            self._init_person()
            return

        # Advance person
        self._px  += self._pvx * self._dt
        self._py  += self._pvy * self._dt
        self._sim_t += self._dt

        # Zigzag: flip lateral direction every 4 s
        # ct/st are actual forward direction, so perpendicular left = (-st, ct)
        if self._scenario == 'zigzag':
            ct = math.cos(self._car_theta)
            st = math.sin(self._car_theta)
            if int(self._sim_t / 4.0) % 2 == 1:
                self._pvx =  st * 0.4    # right = (st, -ct)
                self._pvy = -ct * 0.4
            else:
                self._pvx = -st * 0.4    # left = (-st, ct)
                self._pvy =  ct * 0.4

        now = self.get_clock().now().to_msg()
        arr = Detection3DArray()
        arr.header.stamp    = now
        arr.header.frame_id = 'map'

        det = Detection3D()
        det.header = arr.header
        det.id     = '99'

        # Position (map frame) — matches KF tracker bbox.center.position
        det.bbox.center.position.x    = self._px
        det.bbox.center.position.y    = self._py
        det.bbox.center.position.z    = 0.85   # person mid-height
        det.bbox.center.orientation   = Quaternion(w=1.0)
        det.bbox.size.x               = 0.5    # width
        det.bbox.size.y               = 0.5    # depth
        det.bbox.size.z               = 1.7    # height

        # Hypothesis + velocity — MUST match KF tracker convention:
        #   results[0].hypothesis.class_id   = label
        #   results[0].pose.pose.position     = (vx, vy, vz) in map frame
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id     = 'person'
        hyp.hypothesis.score        = 1.0
        hyp.pose.pose.position.x    = self._pvx
        hyp.pose.pose.position.y    = self._pvy
        hyp.pose.pose.position.z    = 0.0
        hyp.pose.pose.orientation   = Quaternion(w=1.0)
        det.results.append(hyp)

        arr.detections.append(det)
        self._pub_tracked.publish(arr)
        self._publish_marker(now)

        if self._car_known:
            dist = math.hypot(self._px - self._car_x, self._py - self._car_y)
            self.get_logger().info(
                f"[dummy] person at map=({self._px:.2f},{self._py:.2f}) "
                f"vel=({self._pvx:.2f},{self._pvy:.2f}) "
                f"dist_to_car={dist:.2f}m t={self._sim_t:.1f}s",
                throttle_duration_sec=1.0,
            )

    # ------------------------------------------------------------------

    def _publish_marker(self, stamp):
        ma = MarkerArray()

        # Bounding box cube
        cube = Marker()
        cube.header.frame_id = 'map'
        cube.header.stamp    = stamp
        cube.ns              = 'dummy_person'
        cube.id              = 0
        cube.type            = Marker.CUBE
        cube.action          = Marker.ADD
        cube.pose.position.x = self._px
        cube.pose.position.y = self._py
        cube.pose.position.z = 0.85
        cube.pose.orientation = Quaternion(w=1.0)
        cube.scale.x = 0.5
        cube.scale.y = 0.5
        cube.scale.z = 1.7
        cube.color.r = 0.1
        cube.color.g = 0.9
        cube.color.b = 0.2
        cube.color.a = 0.7
        ma.markers.append(cube)

        # Velocity arrow
        spd = math.hypot(self._pvx, self._pvy)
        if spd > 0.05:
            arrow = Marker()
            arrow.header = cube.header
            arrow.ns     = 'dummy_person_vel'
            arrow.id     = 1
            arrow.type   = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = self._px
            arrow.pose.position.y = self._py
            arrow.pose.position.z = 0.85
            yaw = math.atan2(self._pvy, self._pvx)
            arrow.pose.orientation.z = math.sin(yaw / 2.0)
            arrow.pose.orientation.w = math.cos(yaw / 2.0)
            arrow.scale.x = max(0.3, spd * 1.2)
            arrow.scale.y = 0.08
            arrow.scale.z = 0.08
            arrow.color.r = 1.0
            arrow.color.g = 1.0
            arrow.color.b = 0.0
            arrow.color.a = 1.0
            ma.markers.append(arrow)

        # Label
        txt = Marker()
        txt.header = cube.header
        txt.ns     = 'dummy_person_label'
        txt.id     = 2
        txt.type   = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = self._px
        txt.pose.position.y = self._py
        txt.pose.position.z = 2.1
        txt.pose.orientation = Quaternion(w=1.0)
        txt.scale.z = 0.22
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f"DUMMY PERSON\n{self._scenario}\nv={spd:.2f}m/s"
        ma.markers.append(txt)

        self._pub_markers.publish(ma)


# ----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DummyTrackedObjects()
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
