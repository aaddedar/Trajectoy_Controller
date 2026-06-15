#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from curobot_msgs.msg import KinematicState
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

# Must match Arduino WHEELSTEERLIMIT (0.5235987756 rad = 30°).
STEER_LIMIT = 0.5236


class HagenRobot:
    def __init__(self, x=0, y=0, theta=0, v=0, L=0.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L


class PurePurSuitController:
    def __init__(self, robot_model, ref_path_x, ref_path_y, L_d=0.5, k=0.0):
        self.hagen_robot = robot_model
        self.L_d = L_d
        self.ref_path_x = ref_path_x
        self.ref_path_y = ref_path_y
        self.k = k
        self.last_index = 0

    def pure_pursuit_control(self):
        target_index = self.look_ahead_point_index()
        t_x = self.ref_path_x[target_index]
        t_y = self.ref_path_y[target_index]
        alpha = (
            np.arctan2(t_y - self.hagen_robot.y, t_x - self.hagen_robot.x)
            - self.hagen_robot.theta
        )
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        LF = self.k * abs(self.hagen_robot.v) + self.L_d
        delta = np.arctan(2 * self.hagen_robot.L * np.sin(alpha) / (LF + 1e-5))
        delta = np.clip(delta, -STEER_LIMIT, STEER_LIMIT)
        return delta, target_index

    def look_ahead_point_index(self):
        n = len(self.ref_path_x)
        self.last_index = min(self.last_index, n - 1)
        BACK = 30
        search_start = max(0, self.last_index - BACK)
        dx = self.hagen_robot.x - self.ref_path_x[search_start:]
        dy = self.hagen_robot.y - self.ref_path_y[search_start:]
        index = search_start + int(np.argmin(dx**2 + dy**2))
        self.last_index = max(search_start, index)

        L = 0.0
        LF = self.k * abs(self.hagen_robot.v) + self.L_d
        while LF > L and (index + 1) < n:
            step_x = self.ref_path_x[index + 1] - self.ref_path_x[index]
            step_y = self.ref_path_y[index + 1] - self.ref_path_y[index]
            L += math.hypot(step_x, step_y)
            index += 1
        return index


class PurePursuitNode(Node):

    def __init__(self):
        super().__init__("pure_pursuit_node")

        # ── Tunable ROS params ──────────────────────────────────────────────
        self.declare_parameter('command_speed',         1.4)
        self.declare_parameter('min_curve_speed',       0.9)
        self.declare_parameter('L_d',                   0.30)
        self.declare_parameter('k',                     0.3)
        self.declare_parameter('invert_steering',       False)
        self.declare_parameter('person_stop_dist',      0.4)
        self.declare_parameter('person_slowdown_dist',  1.5)
        self.declare_parameter('avoidance_clearance',   0.5)
        self.declare_parameter('avoidance_predict_t',   1.5)
        self.declare_parameter('avoidance_start_dist',  1.5)

        L_d                       = float(self.get_parameter('L_d').value)
        k                         = float(self.get_parameter('k').value)
        self.command_speed        = float(self.get_parameter('command_speed').value)
        self.min_curve_speed      = float(self.get_parameter('min_curve_speed').value)
        self.invert_steering_sign = bool(self.get_parameter('invert_steering').value)
        self.person_stop_dist     = float(self.get_parameter('person_stop_dist').value)
        self.person_slowdown_dist = float(self.get_parameter('person_slowdown_dist').value)
        self.avoidance_clearance  = float(self.get_parameter('avoidance_clearance').value)
        self.avoidance_predict_t  = float(self.get_parameter('avoidance_predict_t').value)
        self.avoidance_start_dist = float(self.get_parameter('avoidance_start_dist').value)

        self.Ts                = 1.0 / 30.0
        self.goal_tolerance    = 0.02   # stop within 2 cm of goal
        self.goal_slowdown_dist = 0.5   # start decelerating 0.5 m before goal

        self._base_L_d   = L_d
        self._k          = k

        self.hagen_robot = HagenRobot(x=0.0, y=0.0, theta=0.0, v=0.0, L=0.3)
        self.controller  = PurePurSuitController(
            self.hagen_robot,
            np.array([], dtype=float),
            np.array([], dtype=float),
            L_d, k,
        )

        self.path_received            = False
        self.kinematic_state_received = False
        self.goal_reached             = False
        self.allowed_to_move_stop     = False
        self.stop_requested           = False
        self.obstacle_stop_requested  = False

        self.closest_person_dist  = float('inf')
        self.car_width            = 0.40
        self.path_corridor_margin = 1.0
        self.avoidance_target     = 0.0   # desired lateral offset (set by detection callback)
        self.avoidance_offset     = 0.0   # actual applied offset, ramped toward target each tick
        self._cte                 = 0.0   # cross-track error: + = car left of path, - = car right
        self.closest_vlong        = 0.0   # person longitudinal velocity in car frame (+ = moving away)

        self.last_detection_time     = None
        self.obstacle_timeout        = 1.0
        self.last_kinematic_time     = None
        self.kinematic_state_timeout = 3.0

        self.smoothed_speed = 0.0
        self.smoothed_delta = 0.0

        # ── QoS ─────────────────────────────────────────────────────────────
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
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self.ackermann_pub            = self.create_publisher(AckermannDrive, '/ackermann_drive', reliable_qos)
        self.obstacle_information_pub = self.create_publisher(Bool, '/obstacle_information', reliable_qos)
        self.lookahead_marker_pub     = self.create_publisher(Marker, '/pure_pursuit/lookahead_marker', 10)
        self.avoidance_path_pub       = self.create_publisher(Path,        '/pure_pursuit/avoidance_path',   10)
        self.original_path_ahead_pub  = self.create_publisher(Path,        '/pure_pursuit/original_path_ahead', 10)
        self.avoidance_viz_pub        = self.create_publisher(MarkerArray, '/pure_pursuit/avoidance_viz',    10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Path,             '/path',                    self.path_callback,                 latched_qos)
        self.create_subscription(AckermannDrive,   '/ackermann_drive_feedback', self.ackermann_feedback_callback,  best_effort_qos)
        self.create_subscription(Bool,             '/allowed_to_move',         self.allowed_to_move_callback,      best_effort_qos)
        self.create_subscription(Detection3DArray, '/tracked_objects',         self.objects_in_map_frame_callback, best_effort_qos)
        self.create_subscription(KinematicState,   '/kinematic_state',         self.kinematic_state_callback,      best_effort_qos)

        self.create_timer(self.Ts, self.control_loop)
        self.create_timer(0.5,    self._check_obstacle_timeout)

        self.get_logger().info(
            f"\nPurePursuitNode started.\n"
            f"  L_d={L_d:.2f}m  k={k:.2f}  speed={self.command_speed:.1f}m/s  "
            f"min_speed={self.min_curve_speed:.1f}m/s  invert={self.invert_steering_sign}\n"
            f"  person_stop_dist={self.person_stop_dist:.1f}m  "
            f"person_slowdown_dist={self.person_slowdown_dist:.1f}m\n"
            f"  REQUIRED topics for obstacle avoidance:\n"
            f"    REAL car : camera → detectnet → 3D localizer → KF tracker → /tracked_objects\n"
            f"    TESTING  : ros2 run control_car dummy_tracked_objects --ros-args -p scenario:=stationary"
        )

    # ── Main control loop ─────────────────────────────────────────────────

    def control_loop(self):
        self.publish_obstacle_information()

        if self.stop_requested:
            self.publish_stop_command()
            self.get_logger().warn(
                "STOPPED: /allowed_to_move=False",
                throttle_duration_sec=2.0,
            )
            return

        if self.obstacle_stop_requested:
            self.publish_stop_command()
            self.get_logger().warn(
                f"STOPPED: person at {self.closest_person_dist:.2f}m "
                f"(hard-stop zone={self.person_stop_dist:.1f}m)",
                throttle_duration_sec=0.5,
            )
            return

        if not self.path_received or len(self.controller.ref_path_x) < 2:
            self.publish_stop_command()
            self.get_logger().warn("STOPPED: waiting for /path...", throttle_duration_sec=2.0)
            return

        if not self.kinematic_state_received:
            self.publish_stop_command()
            self.get_logger().warn("STOPPED: waiting for /kinematic_state...", throttle_duration_sec=2.0)
            return

        goal_distance = float(np.hypot(
            self.controller.ref_path_x[-1] - self.hagen_robot.x,
            self.controller.ref_path_y[-1] - self.hagen_robot.y,
        ))

        if goal_distance <= self.goal_slowdown_dist:
            self.get_logger().info(
                f"Approaching goal: {goal_distance:.2f}m remaining",
                throttle_duration_sec=0.5,
            )

        if goal_distance <= self.goal_tolerance:
            if not self.goal_reached:
                self.goal_reached = True
                self.path_received = False
                self.controller.ref_path_x = np.array([], dtype=float)
                self.controller.ref_path_y = np.array([], dtype=float)
                self.controller.last_index = 0
                self.closest_person_dist = float('inf')
                self.obstacle_stop_requested = False
                self.get_logger().info(f"Goal reached ({goal_distance:.3f}m). Path cleared.")
            self.publish_stop_command()
            return

        self.controller.L_d = self._compute_adaptive_L_d(self.smoothed_delta, self.hagen_robot.v)

        raw_delta, target_index = self.controller.pure_pursuit_control()

        # Smooth ramp: avoidance_offset moves toward avoidance_target at 0.5 m/s
        # (at 30 Hz that's 0.5/30 ≈ 0.017 m per tick — ~1 s to reach full 0.5 m offset)
        RAMP_RATE = 0.5 / 30.0
        if abs(self.avoidance_offset - self.avoidance_target) < RAMP_RATE:
            self.avoidance_offset = self.avoidance_target
        elif self.avoidance_offset < self.avoidance_target:
            self.avoidance_offset += RAMP_RATE
        else:
            self.avoidance_offset -= RAMP_RATE

        # Lateral avoidance: offset lookahead point perpendicular to PATH direction.
        # Using path tangent (not car heading) so curves don't push toward walls.
        #
        # effective_offset is reduced in two situations:
        #   1. Curves: scale linearly to 0 as steering approaches STEER_LIMIT —
        #              avoidance on a tight curve compounds steering and exits road.
        #   2. CTE clamp: if the car is already displaced in the avoidance direction,
        #              subtract that CTE so total path deviation ≤ avoidance_clearance.
        effective_offset = self.avoidance_offset
        if abs(effective_offset) > 0.01:
            # 1. Reduce on curves (raw_delta = pure-pursuit angle before avoidance)
            curve_factor     = min(1.0, abs(raw_delta) / STEER_LIMIT)
            effective_offset *= (1.0 - curve_factor)

            # 2. CTE clamp: available space in avoidance direction
            cte_in_dir   = self._cte * math.copysign(1.0, effective_offset)
            available    = max(0.0, self.avoidance_clearance - max(0.0, cte_in_dir))
            if abs(effective_offset) > available:
                effective_offset = math.copysign(available, effective_offset)

        if abs(effective_offset) > 0.01:
            px_a = self.controller.ref_path_x
            py_a = self.controller.ref_path_y
            t_x  = float(px_a[target_index])
            t_y  = float(py_a[target_index])
            # Path tangent at lookahead point
            if target_index + 1 < len(px_a):
                atdx = float(px_a[target_index + 1] - px_a[target_index])
                atdy = float(py_a[target_index + 1] - py_a[target_index])
            elif target_index > 0:
                atdx = float(px_a[target_index] - px_a[target_index - 1])
                atdy = float(py_a[target_index] - py_a[target_index - 1])
            else:
                atdx = math.cos(self.hagen_robot.theta)
                atdy = math.sin(self.hagen_robot.theta)
            atlen = math.hypot(atdx, atdy) + 1e-9
            # Path left perpendicular (+offset → left of path, -offset → right of path)
            t_x += effective_offset * (-atdy / atlen)
            t_y += effective_offset * ( atdx / atlen)
            alpha = math.atan2(t_y - self.hagen_robot.y, t_x - self.hagen_robot.x) - self.hagen_robot.theta
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            lf    = self.controller.L_d
            raw_delta = math.atan(2.0 * self.hagen_robot.L * math.sin(alpha) / (lf + 1e-5))
            raw_delta = float(np.clip(raw_delta, -STEER_LIMIT, STEER_LIMIT))

        self.smoothed_delta = 0.8 * self.smoothed_delta + 0.2 * float(raw_delta)
        delta = self.smoothed_delta

        goal_speed = self._compute_goal_speed(goal_distance)

        raw_speed = float(min(
            self._compute_curve_speed(delta),
            self._compute_ahead_curvature_speed(),
            self._compute_safe_speed(),
            goal_speed,
        ))
        # Brake fast, accelerate slowly — keeps path tight on curves
        if raw_speed < self.smoothed_speed:
            self.smoothed_speed = raw_speed                               # instant brake
        else:
            self.smoothed_speed = 0.75 * self.smoothed_speed + 0.25 * raw_speed  # ramp-up
        cmd_speed = self.smoothed_speed

        # ── Logging ──────────────────────────────────────────────────────────
        px = self.controller.ref_path_x
        py = self.controller.ref_path_y
        ni = self.controller.last_index
        t_x = float(px[target_index]) if target_index < len(px) else float('nan')
        t_y = float(py[target_index]) if target_index < len(py) else float('nan')
        n_x = float(px[ni]) if ni < len(px) else float('nan')
        n_y = float(py[ni]) if ni < len(py) else float('nan')

        cte_now = 0.0
        if ni + 1 < len(px):
            tx_seg = px[ni + 1] - n_x
            ty_seg = py[ni + 1] - n_y
            seg_len = math.hypot(tx_seg, ty_seg)
            if seg_len > 1e-6:
                ln_x = -ty_seg / seg_len
                ln_y =  tx_seg / seg_len
                cte_now = (self.hagen_robot.x - n_x) * ln_x + (self.hagen_robot.y - n_y) * ln_y
        self._cte = cte_now   # + = car left of path, - = car right of path

        steering_cmd_deg = math.degrees(-delta if self.invert_steering_sign else delta)
        avoid_str = ""
        if abs(self.avoidance_offset) > 0.01:
            avoid_str = (
                f" | AVOID={'L' if self.avoidance_offset > 0 else 'R'}"
                f"  target={self.avoidance_offset:+.2f}m  eff={effective_offset:+.2f}m"
            )
        self.get_logger().info(
            f"pos=({self.hagen_robot.x:.2f},{self.hagen_robot.y:.2f}) "
            f"θ={math.degrees(self.hagen_robot.theta):.0f}° "
            f"v={self.hagen_robot.v:.2f}m/s | "
            f"CTE={cte_now:+.3f}m | "
            f"steer={steering_cmd_deg:+.1f}°({'L' if steering_cmd_deg>0 else 'R'}) "
            f"spd={cmd_speed:.2f}m/s | "
            f"person={self.closest_person_dist:.1f}m{avoid_str}",
            throttle_duration_sec=0.5,
        )

        # ── Avoidance trajectory visualisation ───────────────────────────────
        self._publish_avoidance_viz(self.controller.last_index)

        # ── RViz markers ─────────────────────────────────────────────────────
        from geometry_msgs.msg import Point
        stamp = self.get_clock().now().to_msg()

        mk = Marker()
        mk.header.frame_id = 'map'; mk.header.stamp = stamp
        mk.ns = 'pure_pursuit'; mk.id = 0; mk.type = Marker.SPHERE; mk.action = Marker.ADD
        mk.pose.position.x = t_x; mk.pose.position.y = t_y; mk.pose.position.z = 0.1
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.15
        mk.color.r = 1.0; mk.color.g = 1.0; mk.color.a = 1.0
        self.lookahead_marker_pub.publish(mk)

        mk2 = Marker()
        mk2.header = mk.header; mk2.ns = 'pure_pursuit'; mk2.id = 1
        mk2.type = Marker.SPHERE; mk2.action = Marker.ADD
        mk2.pose.position.x = n_x; mk2.pose.position.y = n_y; mk2.pose.position.z = 0.1
        mk2.pose.orientation.w = 1.0
        mk2.scale.x = mk2.scale.y = mk2.scale.z = 0.10
        mk2.color.g = 1.0; mk2.color.b = 1.0; mk2.color.a = 1.0
        self.lookahead_marker_pub.publish(mk2)

        arrow = Marker()
        arrow.header = mk.header; arrow.ns = 'pure_pursuit'; arrow.id = 2
        arrow.type = Marker.ARROW; arrow.action = Marker.ADD
        p0 = Point(); p0.x = self.hagen_robot.x; p0.y = self.hagen_robot.y; p0.z = 0.1
        p1 = Point(); p1.x = t_x; p1.y = t_y; p1.z = 0.1
        arrow.points = [p0, p1]
        arrow.scale.x = 0.04; arrow.scale.y = 0.08; arrow.scale.z = 0.10
        arrow.color.r = 1.0; arrow.color.g = 1.0; arrow.color.a = 1.0
        self.lookahead_marker_pub.publish(arrow)

        # ── Publish drive command ─────────────────────────────────────────────
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = float(-delta if self.invert_steering_sign else delta)
        drive_msg.speed = cmd_speed
        self.ackermann_pub.publish(drive_msg)

    # ── Timeout watchdog ──────────────────────────────────────────────────

    def _check_obstacle_timeout(self):
        now = self.get_clock().now()

        if self.last_kinematic_time is not None and self.kinematic_state_received:
            if (now - self.last_kinematic_time).nanoseconds / 1e9 > self.kinematic_state_timeout:
                self.get_logger().warn("No /kinematic_state — stopping.", throttle_duration_sec=2.0)
                self.kinematic_state_received = False

        if self.last_detection_time is None:
            return
        elapsed = (now - self.last_detection_time).nanoseconds / 1e9
        if elapsed > self.obstacle_timeout and (
            self.obstacle_stop_requested or self.closest_person_dist < float('inf')
        ):
            self.get_logger().info(f"No detections for {elapsed:.1f}s — clearing, resuming.")
            self.obstacle_stop_requested = False
            self.closest_person_dist     = float('inf')
            self.publish_obstacle_information()

    # ── Path processing ───────────────────────────────────────────────────

    @staticmethod
    def _smooth_path(x, y, window=7, iterations=3):
        pad = window // 2
        kernel = np.ones(window) / window
        for _ in range(iterations):
            x = np.convolve(np.pad(x, pad, mode='edge'), kernel, mode='valid')
            y = np.convolve(np.pad(y, pad, mode='edge'), kernel, mode='valid')
        return x, y

    @staticmethod
    def _densify_path(x, y, spacing=0.05):
        if len(x) < 2:
            return x, y
        seg_len = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        if s[-1] < spacing:
            return x, y
        s_new = np.arange(0.0, s[-1], spacing)
        x_new = np.append(np.interp(s_new, s, x), x[-1])
        y_new = np.append(np.interp(s_new, s, y), y[-1])
        return x_new, y_new

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            self.get_logger().warn(f"Received /path with {len(msg.poses)} poses — need ≥ 2.")
            return

        ref_x = np.array([p.pose.position.x for p in msg.poses], dtype=float)
        ref_y = np.array([p.pose.position.y for p in msg.poses], dtype=float)
        start_x, start_y = ref_x[0], ref_y[0]
        end_x,   end_y   = ref_x[-1], ref_y[-1]

        ref_x, ref_y = self._smooth_path(ref_x, ref_y)
        ref_x, ref_y = self._densify_path(ref_x, ref_y, spacing=0.05)
        ref_x[0], ref_y[0]   = start_x, start_y
        ref_x[-1], ref_y[-1] = end_x,   end_y

        self.controller.ref_path_x = ref_x
        self.controller.ref_path_y = ref_y
        self.controller.last_index = 0
        self.path_received = True
        self.goal_reached  = False
        self.get_logger().info(f"Path received: {len(msg.poses)} poses → {len(ref_x)} pts after smooth+densify")

    # ── Callbacks ─────────────────────────────────────────────────────────

    def kinematic_state_callback(self, msg):
        pose = msg.pose_with_covariance.pose
        self.hagen_robot.x = pose.position.x
        self.hagen_robot.y = pose.position.y

        vx = msg.twist_with_covariance.twist.linear.x
        vy = msg.twist_with_covariance.twist.linear.y
        self.hagen_robot.v = math.hypot(vx, vy)

        q = pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.hagen_robot.theta = math.atan2(siny, cosy)

        self.kinematic_state_received = True
        self.last_kinematic_time = self.get_clock().now()
        self.get_logger().info(
            f"kin: pos=({self.hagen_robot.x:.3f},{self.hagen_robot.y:.3f}) "
            f"θ={math.degrees(self.hagen_robot.theta):.1f}° v={self.hagen_robot.v:.3f}m/s",
            throttle_duration_sec=1.0,
        )

    def ackermann_feedback_callback(self, msg):
        self.get_logger().debug(f"feedback: spd={msg.speed:.3f} steer={msg.steering_angle:.3f}")

    def allowed_to_move_callback(self, msg):
        self.allowed_to_move_stop = not bool(msg.data)
        self.stop_requested = self.allowed_to_move_stop
        self.get_logger().info(f"/allowed_to_move={msg.data}  stop={self.stop_requested}")

    @staticmethod
    def _is_person(class_id: str) -> bool:
        s = class_id.strip().lower()
        return s == 'person' or s == '1'

    def objects_in_map_frame_callback(self, msg):
        self.last_detection_time = self.get_clock().now()

        if not msg.detections:
            self.get_logger().info("[tracked] 0 objects — clear", throttle_duration_sec=2.0)
            self._clear_obstacle()
            return

        if not self.kinematic_state_received:
            self.get_logger().warn("[tracked] no kinematic state yet — skipping", throttle_duration_sec=2.0)
            return

        corridor_half = self.car_width / 2.0 + self.path_corridor_margin
        closest_cx    = float('inf')
        closest_cy    = 0.0
        closest_vlat  = 0.0
        closest_vlong = 0.0

        cos_t = math.cos(self.hagen_robot.theta)
        sin_t = math.sin(self.hagen_robot.theta)

        for det in msg.detections:
            if not det.results:
                continue
            cid = det.results[0].hypothesis.class_id
            if not self._is_person(cid):
                self.get_logger().info(
                    f"[tracked] skipping class='{cid}'", throttle_duration_sec=2.0)
                continue

            x_map = float(det.bbox.center.position.x)
            y_map = float(det.bbox.center.position.y)
            cx, cy = self._map_to_car_frame(x_map, y_map)

            # KF velocity → lateral and longitudinal components in car frame
            vx_map = float(det.results[0].pose.pose.position.x)
            vy_map = float(det.results[0].pose.pose.position.y)
            vlat   = -vx_map * sin_t + vy_map * cos_t   # + = moving left
            vlong  =  vx_map * cos_t + vy_map * sin_t   # + = moving away from car

            self.get_logger().warn(
                f"[PERSON] class='{cid}' map=({x_map:.2f},{y_map:.2f}) "
                f"→ car fwd={cx:.2f}m lat={cy:+.2f}m "
                f"vlat={vlat:+.2f}m/s vlong={vlong:+.2f}m/s",
                throttle_duration_sec=0.3,
            )

            if cx <= 0.1:
                self.get_logger().info(f"[person] behind car (cx={cx:.2f}) — skip", throttle_duration_sec=1.0)
                continue
            if abs(cy) > corridor_half:
                self.get_logger().info(
                    f"[person] outside corridor (|cy|={abs(cy):.2f} > {corridor_half:.2f}) — skip",
                    throttle_duration_sec=1.0)
                continue
            if cx < closest_cx:
                closest_cx    = cx
                closest_cy    = cy
                closest_vlat  = vlat
                closest_vlong = vlong

        if closest_cx == float('inf'):
            self.get_logger().info("[person] none in corridor ahead — clear", throttle_duration_sec=2.0)
            self._clear_obstacle()
            return

        self.closest_person_dist = closest_cx
        self.closest_vlong       = closest_vlong

        obstacle_stop, avoidance_target, early_trigger_dist = \
            self._decide_avoidance_action(closest_cx, closest_cy, closest_vlat, closest_vlong)

        self.get_logger().warn(
            f"[person] raw_cx={closest_cx:.2f}m  vlong={closest_vlong:+.2f}m/s  "
            f"avoidance_triggers@{early_trigger_dist:.2f}m  target={avoidance_target:+.2f}m",
            throttle_duration_sec=0.3,
        )

        self.avoidance_target        = avoidance_target
        self.obstacle_stop_requested = obstacle_stop

        if obstacle_stop:
            self.get_logger().warn(
                f"PERSON HARD STOP: {closest_cx:.2f}m <= stop_dist={self.person_stop_dist:.1f}m",
                throttle_duration_sec=0.3,
            )

        self.publish_obstacle_information()

    def _clear_obstacle(self):
        if self.obstacle_stop_requested or self.closest_person_dist < float('inf'):
            self.get_logger().info("Person cleared — resuming normal path.")
        self.obstacle_stop_requested = False
        self.closest_person_dist     = float('inf')
        self.closest_vlong           = 0.0
        self.avoidance_target        = 0.0   # ramps smoothly to 0 in control_loop
        self.publish_obstacle_information()

    # ── Car-frame transform ───────────────────────────────────────────────

    def _map_to_car_frame(self, x_map, y_map):
        dx = x_map - self.hagen_robot.x
        dy = y_map - self.hagen_robot.y
        cos_t = math.cos(self.hagen_robot.theta)
        sin_t = math.sin(self.hagen_robot.theta)
        cx =  dx * cos_t + dy * sin_t
        cy = -dx * sin_t + dy * cos_t
        return cx, cy

    # ── Speed scaling ─────────────────────────────────────────────────────

    def _decide_avoidance_action(self, cx, cy, vlat, vlong):
        """Return (obstacle_stop, avoidance_target, early_trigger_dist) for one person."""
        person_approach    = max(0.0, -vlong)
        early_trigger_dist = self.avoidance_start_dist + person_approach * self.avoidance_predict_t

        lat_predicted = cy + vlat * self.avoidance_predict_t
        CENTERED = 0.20
        if abs(lat_predicted) > CENTERED:
            avoid_dir = -math.copysign(self.avoidance_clearance, lat_predicted)
        else:
            if self._cte < -0.05:
                avoid_dir = self.avoidance_clearance
            else:
                avoid_dir = -self.avoidance_clearance

        if cx <= self.person_stop_dist:
            return True, 0.0, early_trigger_dist
        if cx <= early_trigger_dist:
            return False, avoid_dir, early_trigger_dist
        return False, 0.0, early_trigger_dist

    def _compute_goal_speed(self, goal_distance):
        if goal_distance < self.goal_slowdown_dist:
            ramp = goal_distance / self.goal_slowdown_dist
            return self.command_speed * max(0.0, ramp)
        return self.command_speed

    def _compute_adaptive_L_d(self, prev_delta, speed):
        curve_factor = min(1.0, abs(prev_delta) / STEER_LIMIT)
        adaptive = (self._base_L_d + self._k * abs(speed)) * (1.0 - 0.6 * curve_factor)
        return max(self._base_L_d * 0.5, adaptive)

    def _compute_curve_speed(self, delta):
        # sqrt mapping: speed drops steeply even at small steer angles
        ratio = min(1.0, abs(delta) / STEER_LIMIT) ** 0.5
        return self.command_speed - (self.command_speed - self.min_curve_speed) * ratio

    def _compute_ahead_curvature_speed(self):
        px = self.controller.ref_path_x
        py = self.controller.ref_path_y
        n  = len(px)
        if n < 3:
            return self.command_speed

        idx     = self.controller.last_index
        R_min   = self.hagen_robot.L / math.tan(STEER_LIMIT)
        k_max   = 1.0 / R_min
        worst_k = 0.0

        for i in range(idx, min(idx + 30, n - 2)):
            p1x, p1y = px[i],            py[i]
            p2x, p2y = px[i+1],          py[i+1]
            p3x, p3y = px[min(i+2,n-1)], py[min(i+2,n-1)]
            area = abs((p2x-p1x)*(p3y-p1y) - (p3x-p1x)*(p2y-p1y)) / 2.0
            d12  = math.hypot(p2x-p1x, p2y-p1y)
            d23  = math.hypot(p3x-p2x, p3y-p2y)
            d13  = math.hypot(p3x-p1x, p3y-p1y)
            if d12 * d23 * d13 > 1e-10:
                worst_k = max(worst_k, 4.0 * area / (d12 * d23 * d13))

        ratio = min(1.0, worst_k / k_max)
        return self.command_speed - (self.command_speed - self.min_curve_speed) * ratio

    def _compute_safe_speed(self):
        dist = self.closest_person_dist
        if dist == float('inf'):
            return self.command_speed

        # Extend the slowdown zone when person walks toward car.
        # Uses only person velocity — no car-velocity term avoids stop/accelerate oscillation.
        person_approach      = max(0.0, -self.closest_vlong)   # + when person walks toward car
        effective_slow_dist  = self.person_slowdown_dist + person_approach * self.avoidance_predict_t

        if dist <= self.person_stop_dist:
            return 0.0
        if dist <= effective_slow_dist:
            factor = (dist - self.person_stop_dist) / (effective_slow_dist - self.person_stop_dist)
            return self.command_speed * factor
        return self.command_speed

    # ── Avoidance trajectory visualisation ───────────────────────────────

    def _publish_avoidance_viz(self, start_index: int):
        px  = self.controller.ref_path_x
        py  = self.controller.ref_path_y
        n   = len(px)
        now = self.get_clock().now().to_msg()

        AHEAD = 60   # points to show (~3 m at 5 cm spacing)
        end   = min(start_index + AHEAD, n)

        # ── Deviated path (yellow when avoiding, green when back on path) ──
        dev_path = Path()
        dev_path.header.stamp    = now
        dev_path.header.frame_id = 'map'

        orig_path = Path()
        orig_path.header.stamp    = now
        orig_path.header.frame_id = 'map'

        for i in range(start_index, end):
            # Path tangent at point i
            if i + 1 < n:
                pdx, pdy = float(px[i+1] - px[i]), float(py[i+1] - py[i])
            else:
                pdx, pdy = float(px[i] - px[i-1]), float(py[i] - py[i-1])
            plen = math.hypot(pdx, pdy) + 1e-9

            # Original path pose
            op = PoseStamped()
            op.header = orig_path.header
            op.pose.position.x = float(px[i])
            op.pose.position.y = float(py[i])
            op.pose.position.z = 0.02
            op.pose.orientation.w = 1.0
            orig_path.poses.append(op)

            # Deviated path pose (shifted by current avoidance_offset)
            dp = PoseStamped()
            dp.header = dev_path.header
            dp.pose.position.x = float(px[i]) + self.avoidance_offset * (-pdy / plen)
            dp.pose.position.y = float(py[i]) + self.avoidance_offset * ( pdx / plen)
            dp.pose.position.z = 0.05
            dp.pose.orientation.w = 1.0
            dev_path.poses.append(dp)

        self.avoidance_path_pub.publish(dev_path)
        self.original_path_ahead_pub.publish(orig_path)

        # ── Marker: coloured line strips ──────────────────────────────────
        ma = MarkerArray()

        def _line_strip(ns, mid, pts, r, g, b, z_off=0.0, width=0.04):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns     = ns
            m.id     = mid
            m.type   = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = width
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.85
            m.pose.orientation.w = 1.0
            from geometry_msgs.msg import Point
            for p in pts:
                pt = Point(); pt.x = p[0]; pt.y = p[1]; pt.z = p[2] + z_off
                m.points.append(pt)
            return m

        orig_pts = [(p.pose.position.x, p.pose.position.y, 0.0) for p in orig_path.poses]
        dev_pts  = [(p.pose.position.x, p.pose.position.y, 0.0) for p in dev_path.poses]

        # Original path ahead — white
        if orig_pts:
            ma.markers.append(_line_strip('orig_ahead', 0, orig_pts, 1.0, 1.0, 1.0, width=0.03))

        # Deviated trajectory — yellow while avoiding, cyan while returning
        if dev_pts and abs(self.avoidance_offset) > 0.02:
            returning = abs(self.avoidance_offset) < abs(self.avoidance_target) + 0.01 \
                        and self.avoidance_target == 0.0
            r, g, b = (0.0, 1.0, 1.0) if returning else (1.0, 1.0, 0.0)
            ma.markers.append(_line_strip('deviated', 1, dev_pts, r, g, b, z_off=0.03, width=0.05))

            # Lateral offset arrows at start and end to show how far off path
            from geometry_msgs.msg import Point
            arr = Marker()
            arr.header.frame_id = 'map'; arr.header.stamp = now
            arr.ns = 'offset_arrow'; arr.id = 2
            arr.type = Marker.ARROW; arr.action = Marker.ADD
            arr.scale.x = 0.03; arr.scale.y = 0.06; arr.scale.z = 0.06
            arr.color.r = 1.0; arr.color.g = 0.4; arr.color.b = 0.0; arr.color.a = 1.0
            arr.pose.orientation.w = 1.0
            mid = len(orig_pts) // 2
            if mid < len(orig_pts) and mid < len(dev_pts):
                p0 = Point(); p0.x = orig_pts[mid][0]; p0.y = orig_pts[mid][1]; p0.z = 0.06
                p1 = Point(); p1.x = dev_pts[mid][0];  p1.y = dev_pts[mid][1];  p1.z = 0.06
                arr.points = [p0, p1]
                ma.markers.append(arr)
        else:
            # Clear deviated markers when no longer avoiding
            for mid in [1, 2]:
                clr = Marker()
                clr.header.frame_id = 'map'; clr.header.stamp = now
                clr.ns = 'deviated' if mid == 1 else 'offset_arrow'
                clr.id = mid; clr.action = Marker.DELETE
                ma.markers.append(clr)

        self.avoidance_viz_pub.publish(ma)

    # ── Publish helpers ───────────────────────────────────────────────────

    def publish_obstacle_information(self):
        msg = Bool()
        msg.data = bool(self.obstacle_stop_requested)
        self.obstacle_information_pub.publish(msg)

    def publish_stop_command(self):
        self.smoothed_speed   = 0.0
        self.smoothed_delta   = 0.0
        self.avoidance_target = 0.0
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = 0.0
        drive_msg.speed = 0.0
        try:
            if rclpy.ok():
                self.ackermann_pub.publish(drive_msg)
        except Exception as exc:
            self.get_logger().debug(f"Stop command error: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop_command()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
