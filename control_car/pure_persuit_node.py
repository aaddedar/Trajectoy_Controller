#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from curobot_msgs.msg import KinematicState
from enum import Enum, auto
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection3DArray


class DriveState(Enum):
    FOLLOWING_PATH     = auto()   # normal pure pursuit on planner path
    OBSTACLE_WAITING   = auto()   # stopped, waiting for dynamic obstacle to cross
    OBSTACLE_AVOIDANCE = auto()   # executing local bypass trajectory
    OBSTACLE_STOPPED   = auto()   # static obstacle or emergency — hard stop


class HagenRobot:
    def __init__(self, Ts, x=0.0, y=0.0, theta=0.0, v=0.0, L=0.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L        # wheelbase [m]
        self.Ts = Ts      # control timestep [s]


class PurePurSuitController:
    def __init__(self, robot, ref_x, ref_y, L_d=0.4, k=0.35, max_speed=1.0, max_steer=0.698):
        self.hagen_robot = robot
        self.L_d         = L_d
        self.k           = k
        self.max_speed   = max_speed
        self.max_steer   = max_steer
        self.ref_path_x  = ref_x
        self.ref_path_y  = ref_y
        self._last_idx   = 0

    def pure_pursuit_control(self):
        tx, ty, seg_idx = self._find_lookahead_point()
        self._last_idx = seg_idx
        alpha = (
            np.arctan2(ty - self.hagen_robot.y, tx - self.hagen_robot.x)
            - self.hagen_robot.theta
        )
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # wrap to [-π, π]
        LF = self.k * self.max_speed + self.L_d            # speed-adaptive lookahead distance
        delta = np.arctan(2.0 * self.hagen_robot.L * np.sin(alpha) / max(LF, 0.1))
        return float(np.clip(delta, -self.max_steer, self.max_steer)), seg_idx

    def reset_index(self):
        """Reset to path start — use only when switching to a fresh path that starts at robot."""
        self._last_idx = 0

    def reset_index_to_nearest(self, rx: float, ry: float):
        """Seek to the closest path point to (rx, ry) — use when rejoining mid-path."""
        n = len(self.ref_path_x)
        if n == 0:
            self._last_idx = 0
            return
        best, best_d = 0, float("inf")
        for i in range(n):
            d = math.hypot(self.ref_path_x[i] - rx, self.ref_path_y[i] - ry)
            if d < best_d:
                best_d, best = d, i
        self._last_idx = best

    def _find_lookahead_point(self) -> tuple:
        """
        Interpolate along path segments to find the exact point at distance LF
        from the robot (circle-segment intersection). This avoids snapping to
        waypoints, which causes wide turns when waypoint spacing ≈ LF.
        Returns (x, y, segment_index).
        """
        rx, ry, th = self.hagen_robot.x, self.hagen_robot.y, self.hagen_robot.theta
        ct, st = math.cos(th), math.sin(th)
        n = len(self.ref_path_x)
        LF = self.k * self.max_speed + self.L_d

        self._last_idx = min(self._last_idx, n - 1)

        # Walk segments forward from _last_idx looking for circle intersection
        for i in range(self._last_idx, n - 1):
            x1, y1 = self.ref_path_x[i],     self.ref_path_y[i]
            x2, y2 = self.ref_path_x[i + 1], self.ref_path_y[i + 1]

            # Circle-segment intersection: solve ||(x1+t*d) - robot||^2 = LF^2
            dx, dy = x2 - x1, y2 - y1
            fx, fy = x1 - rx, y1 - ry
            a = dx * dx + dy * dy
            if a < 1e-10:
                continue
            b = 2.0 * (fx * dx + fy * dy)
            c = fx * fx + fy * fy - LF * LF
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                continue

            sqrt_disc = math.sqrt(disc)
            # Prefer t2 (further along segment = forward on path)
            for t in ((-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)):
                if 0.0 <= t <= 1.0:
                    ix = x1 + t * dx
                    iy = y1 + t * dy
                    # Accept only points ahead of robot
                    if (ix - rx) * ct + (iy - ry) * st >= 0.0:
                        return ix, iy, i

        # Fallback: goal point or last reachable waypoint ahead
        for i in range(n - 1, self._last_idx - 1, -1):
            px, py = self.ref_path_x[i], self.ref_path_y[i]
            if (px - rx) * ct + (py - ry) * st >= 0.0:
                return float(px), float(py), i

        return float(self.ref_path_x[-1]), float(self.ref_path_y[-1]), n - 1


class PurePursuitNode(Node):

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #
    def __init__(self):
        super().__init__("pure_pursuit_node")

        self.declare_parameter("L_d",                         0.1)
        self.declare_parameter("k",                           0.35)
        self.declare_parameter("max_speed",                   1.0)
        self.declare_parameter("max_steer_deg",               30.0)
        self.declare_parameter("Ts",                          0.05)
        self.declare_parameter("wheelbase",                   0.5)
        self.declare_parameter("goal_tolerance",              0.3)
        self.declare_parameter("publish_steering_in_degrees", False)
        self.declare_parameter("invert_steering_sign",        False)
        self.declare_parameter("robot_width",                 0.6)
        self.declare_parameter("avoidance_clearance",         0.35)
        self.declare_parameter("obstacle_stop_distance",      0.8)
        self.declare_parameter("obstacle_avoidance_trigger",  2.0)
        self.declare_parameter("obstacle_timeout",            3.0)
        self.declare_parameter("static_speed_threshold",      0.15)
        self.declare_parameter("time_horizon",                4.0)
        self.declare_parameter("wait_timeout",                8.0)

        def _p(name):
            return self.get_parameter(name).value

        self.L_d                         = _p("L_d")
        self.k                           = _p("k")
        self.max_speed                   = _p("max_speed")
        self.max_steer                   = math.radians(_p("max_steer_deg"))
        self.Ts                          = _p("Ts")
        self.goal_tolerance              = _p("goal_tolerance")
        self.publish_steering_in_degrees = _p("publish_steering_in_degrees")
        self.invert_steering_sign        = _p("invert_steering_sign")
        self.robot_width                 = _p("robot_width")
        self.avoidance_clearance         = _p("avoidance_clearance")
        self.obstacle_stop_distance      = _p("obstacle_stop_distance")
        self.obstacle_avoidance_trigger  = _p("obstacle_avoidance_trigger")
        self.obstacle_timeout            = _p("obstacle_timeout")
        self.static_speed_threshold      = _p("static_speed_threshold")
        self.time_horizon                = _p("time_horizon")
        self.wait_timeout                = _p("wait_timeout")

        self.hagen_robot = HagenRobot(Ts=self.Ts, L=_p("wheelbase"))
        self.controller  = PurePurSuitController(
            self.hagen_robot,
            np.empty(0, dtype=float), np.empty(0, dtype=float),
            self.L_d, self.k, self.max_speed, self.max_steer,
        )

        self.main_path_x = np.empty(0, dtype=float)
        self.main_path_y = np.empty(0, dtype=float)

        self.drive_state              = DriveState.FOLLOWING_PATH
        self.path_received            = False
        self.kinematic_state_received = False
        self.goal_reached             = False

        self.allowed_to_move = True
        self.traffic_stop    = False

        self.current_detections:      list = []
        self.last_detection_time            = None
        self.obstacle_velocity_cache: dict  = {}   # track_id → {x, y, vx, vy, time}
        self.waiting_since                  = None
        self._avoidance_cooldown_until      = None  # suppress re-trigger after rejoining

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

        self.ackermann_pub     = self.create_publisher(AckermannDrive, "/ackermann_drive",      reliable_qos)
        self.obstacle_info_pub = self.create_publisher(Bool,           "/obstacle_information",  reliable_qos)
        self.goal_reached_pub  = self.create_publisher(Bool,           "/goal_reached",          reliable_qos)

        self.create_subscription(Path,             "/path",                     self.path_callback,               reliable_qos)
        self.create_subscription(KinematicState,   "/kinematic_state",          self.kinematic_state_callback,    best_effort_qos)
        self.create_subscription(AckermannDrive,   "/ackermann_drive_feedback", self.ackermann_feedback_callback, best_effort_qos)
        self.create_subscription(Bool,             "/allowed_to_move",          self.allowed_to_move_callback,    reliable_qos)
        self.create_subscription(Detection3DArray, "/tracked_objects",          self.tracked_objects_callback,    reliable_qos)
        self.create_subscription(Bool,             "/traffic_decision",         self.traffic_decision_callback,   reliable_qos)
        self.create_subscription(String,           "/robot_description",        self.robot_description_callback,  10)

        self.create_timer(self.Ts, self.control_loop)
        self.create_timer(0.1,     self.obstacle_monitor)

        self.get_logger().info(
            "PurePursuitNode started — constant speed, static/dynamic obstacle handling."
        )

    # ================================================================== #
    #  CONTROL LOOP                                                        #
    # ================================================================== #
    def control_loop(self):
        if not self.allowed_to_move or self.traffic_stop:
            self.publish_stop_command()
            return

        if not self.path_received or not self.kinematic_state_received:
            self.publish_stop_command()
            self.get_logger().warn(
                "Waiting for path / kinematic state.", throttle_duration_sec=5.0
            )
            return

        if len(self.controller.ref_path_x) < 2:
            self.publish_stop_command()
            return

        if self.drive_state in (DriveState.OBSTACLE_STOPPED, DriveState.OBSTACLE_WAITING):
            self.publish_stop_command()
            return

        if self.drive_state == DriveState.FOLLOWING_PATH:
            goal_dist = float(np.hypot(
                self.main_path_x[-1] - self.hagen_robot.x,
                self.main_path_y[-1] - self.hagen_robot.y,
            ))
            if goal_dist <= self.goal_tolerance:
                if not self.goal_reached:
                    self.goal_reached = True
                    self.get_logger().info(
                        f"Goal reached ({goal_dist:.3f} m from final waypoint)."
                    )
                    self.goal_reached_pub.publish(Bool(data=True))
                self.publish_stop_command()
                return

        delta, idx = self.controller.pure_pursuit_control()
        radius = self.hagen_robot.L / math.tan(abs(delta)) if abs(delta) > 1e-6 else float("inf")
        self.get_logger().info(
            f"state={self.drive_state.name} "
            f"look_ahead_idx={idx}/{len(self.controller.ref_path_x)-1} "
            f"delta={delta:.4f} rad  delta_deg={math.degrees(delta):.1f}  "
            f"turn_radius={radius:.2f} m",
            throttle_duration_sec=1.0,
        )
        self._publish_drive(delta, self.max_speed)

    def _publish_drive(self, delta: float, speed: float):
        msg = AckermannDrive()
        steer = -delta if self.invert_steering_sign else delta
        msg.steering_angle = (
            float(math.degrees(steer)) if self.publish_steering_in_degrees else float(steer)
        )
        msg.speed        = speed
        msg.acceleration = 0.0
        self.ackermann_pub.publish(msg)

    # ================================================================== #
    #  OBSTACLE MONITOR  (10 Hz)                                          #
    # ================================================================== #
    def obstacle_monitor(self):
        if not self.kinematic_state_received or not self.path_received:
            return

        # Suppress obstacle detection briefly after rejoining the main path to
        # prevent immediately re-entering avoidance for the same obstacle.
        if self._avoidance_cooldown_until is not None:
            if self.get_clock().now() < self._avoidance_cooldown_until:
                return
            self._avoidance_cooldown_until = None
        if self.goal_reached and self.drive_state == DriveState.FOLLOWING_PATH:
            return

        # Stale-detection timeout — clear cache and resume if objects disappeared
        if self.last_detection_time is not None:
            elapsed = (
                self.get_clock().now() - self.last_detection_time
            ).nanoseconds / 1e9
            if elapsed > self.obstacle_timeout:
                self.current_detections      = []
                self.obstacle_velocity_cache = {}
                if self.drive_state != DriveState.FOLLOWING_PATH:
                    self.get_logger().info(
                        f"No detections for {elapsed:.1f}s — resuming main path."
                    )
                    self._resume_main_path()
                self._pub_obstacle_info(False)
                return

        blocking = self._find_blocking_obstacle()

        # ── No obstacle in path ──────────────────────────────────────── #
        if blocking is None:
            self._pub_obstacle_info(False)
            if self.drive_state == DriveState.OBSTACLE_AVOIDANCE:
                if self._avoidance_end_reached() or self._close_to_main_path():
                    self.get_logger().info("Obstacle cleared — rejoining main path.")
                    self._resume_main_path()
            elif self.drive_state in (DriveState.OBSTACLE_STOPPED,
                                      DriveState.OBSTACLE_WAITING):
                self.get_logger().info("Path clear — resuming.")
                self._resume_main_path()
            return

        # ── Obstacle present ─────────────────────────────────────────── #
        ox, oy, hw, hl, dist, vx, vy = blocking
        self._pub_obstacle_info(True)

        # Emergency stop — too close to do anything else
        if dist <= self.obstacle_stop_distance:
            if self.drive_state != DriveState.OBSTACLE_STOPPED:
                self.get_logger().warn(
                    f"Emergency stop: obstacle {dist:.2f} m ahead."
                )
                self.drive_state = DriveState.OBSTACLE_STOPPED
            return

        # Already executing avoidance path
        if self.drive_state == DriveState.OBSTACLE_AVOIDANCE:
            if self._avoidance_end_reached():
                self.get_logger().info(
                    "Avoidance waypoints consumed — rejoining main path."
                )
                self._resume_main_path()
            return

        # Already waiting — check if timeout expired → switch to avoidance
        if self.drive_state == DriveState.OBSTACLE_WAITING:
            elapsed = (
                self.get_clock().now() - self.waiting_since
            ).nanoseconds / 1e9
            if elapsed > self.wait_timeout:
                self.get_logger().warn(
                    f"Wait timeout ({elapsed:.1f}s) — switching to avoidance."
                )
                result = self._plan_avoidance_path(ox, oy, hw, hl)
                if result is not None:
                    self.controller.ref_path_x, self.controller.ref_path_y = result
                    self.controller.reset_index()   # avoidance path starts at robot
                    self.drive_state = DriveState.OBSTACLE_AVOIDANCE
                else:
                    self.drive_state = DriveState.OBSTACLE_STOPPED
            return

        # ── Classify and decide (FOLLOWING_PATH or OBSTACLE_STOPPED) ─── #
        obs_speed = math.hypot(vx, vy)

        if obs_speed < self.static_speed_threshold:
            # Static obstacle — hard stop, never try to avoid
            if self.drive_state != DriveState.OBSTACLE_STOPPED:
                self.get_logger().warn(
                    f"Static obstacle at ({ox:.2f}, {oy:.2f}) "
                    f"dist={dist:.2f}m — stopping."
                )
                self.drive_state = DriveState.OBSTACLE_STOPPED
            return

        # Dynamic obstacle — predict if it will cross and clear
        if self._will_obstacle_clear(ox, oy, hw, hl, vx, vy):
            if self.drive_state != DriveState.OBSTACLE_WAITING:
                self.get_logger().info(
                    f"Dynamic obstacle ({obs_speed:.2f} m/s) crossing path — waiting."
                )
                self.drive_state  = DriveState.OBSTACLE_WAITING
                self.waiting_since = self.get_clock().now()
        else:
            result = self._plan_avoidance_path(ox, oy, hw, hl)
            if result is not None:
                self.controller.ref_path_x, self.controller.ref_path_y = result
                self.controller.reset_index()       # avoidance path starts at robot
                self.drive_state = DriveState.OBSTACLE_AVOIDANCE
                self.get_logger().info(
                    f"Dynamic obstacle ({obs_speed:.2f} m/s) blocking path — deviating."
                )
            else:
                self.get_logger().warn("Cannot plan avoidance path — stopping.")
                self.drive_state = DriveState.OBSTACLE_STOPPED

    # ------------------------------------------------------------------ #
    #  Obstacle helpers                                                    #
    # ------------------------------------------------------------------ #
    def _find_blocking_obstacle(self):
        """
        Returns (ox, oy, hw, hl, dist, vx, vy) for the nearest object
        inside the robot's forward path corridor, or None.
        """
        rx, ry, th = self.hagen_robot.x, self.hagen_robot.y, self.hagen_robot.theta
        ct, st = math.cos(th), math.sin(th)

        nearest      = None
        nearest_dist = float("inf")

        for det in self.current_detections:
            ox = float(det.bbox.center.position.x)
            oy = float(det.bbox.center.position.y)
            hw = float(det.bbox.size.x) / 2.0
            hl = float(det.bbox.size.y) / 2.0

            dx, dy    = ox - rx, oy - ry
            local_fwd = dx * ct + dy * st
            local_lat = -dx * st + dy * ct

            if local_fwd <= 0.0:
                continue                    # behind robot

            # Conservative lateral extent: max of both bbox dimensions
            corridor_half = max(hw, hl) + self.robot_width / 2.0
            if abs(local_lat) > corridor_half:
                continue                    # outside corridor

            if local_fwd > self.obstacle_avoidance_trigger:
                continue                    # too far ahead to react yet

            dist = math.hypot(dx, dy)
            if dist < nearest_dist:
                nearest_dist = dist
                cached = self.obstacle_velocity_cache.get(det.id, {})
                vx = cached.get('vx', 0.0)
                vy = cached.get('vy', 0.0)
                nearest = (ox, oy, hw, hl, dist, vx, vy)

        return nearest

    def _will_obstacle_clear(self, ox, oy, hw, hl, vx, vy) -> bool:
        """
        Linear-prediction check: returns True if the obstacle will leave
        the path corridor within time_horizon seconds.
        """
        if math.hypot(vx, vy) < 1e-3:
            return False

        rx, ry, th = self.hagen_robot.x, self.hagen_robot.y, self.hagen_robot.theta
        ct, st = math.cos(th), math.sin(th)
        corridor_half = max(hw, hl) + self.robot_width / 2.0

        for i in range(1, 21):
            t = self.time_horizon * i / 20.0
            pred_x = ox + vx * t
            pred_y = oy + vy * t
            dx, dy = pred_x - rx, pred_y - ry
            local_fwd = dx * ct + dy * st
            local_lat = -dx * st + dy * ct
            if local_fwd < 0.0 or abs(local_lat) > corridor_half:
                return True                 # will leave corridor

        return False                        # stays in path for entire horizon

    def _plan_avoidance_path(self, ox, oy, hw, hl):
        """
        4-waypoint bypass: robot → beside obstacle → past obstacle → rejoin main path.
        """
        rx, ry, th = self.hagen_robot.x, self.hagen_robot.y, self.hagen_robot.theta
        ct, st = math.cos(th), math.sin(th)
        px, py = -st, ct                    # unit vector perpendicular (left of heading)

        dx, dy    = ox - rx, oy - ry
        local_lat = -dx * st + dy * ct
        side = -1.0 if local_lat >= 0.0 else 1.0   # go right if obstacle left, left if right

        lateral = max(hw, hl) + self.robot_width / 2.0 + self.avoidance_clearance

        bypass_x = ox + side * lateral * px
        bypass_y = oy + side * lateral * py

        fwd_margin = hl + 1.0
        merge_x = ox + fwd_margin * ct + side * (lateral * 0.4) * px
        merge_y = oy + fwd_margin * st + side * (lateral * 0.4) * py

        rj_x, rj_y = self._rejoin_point(merge_x, merge_y)

        path_x = np.array([rx, bypass_x, merge_x, rj_x], dtype=float)
        path_y = np.array([ry, bypass_y, merge_y, rj_y], dtype=float)
        return path_x, path_y

    def _rejoin_point(self, from_x: float, from_y: float):
        """Nearest main-path point ahead of robot and closest to (from_x, from_y)."""
        if len(self.main_path_x) == 0:
            return from_x, from_y

        th = self.hagen_robot.theta
        ct, st = math.cos(th), math.sin(th)
        best_i = len(self.main_path_x) - 1
        best_d = float("inf")

        for i, (mpx, mpy) in enumerate(zip(self.main_path_x, self.main_path_y)):
            if (mpx - self.hagen_robot.x) * ct + (mpy - self.hagen_robot.y) * st < 0.0:
                continue
            d = math.hypot(mpx - from_x, mpy - from_y)
            if d < best_d:
                best_d, best_i = d, i

        return float(self.main_path_x[best_i]), float(self.main_path_y[best_i])

    def _avoidance_end_reached(self) -> bool:
        ax = self.controller.ref_path_x
        ay = self.controller.ref_path_y
        if len(ax) == 0:
            return False
        return math.hypot(ax[-1] - self.hagen_robot.x, ay[-1] - self.hagen_robot.y) < 0.5

    def _close_to_main_path(self) -> bool:
        if len(self.main_path_x) == 0:
            return False
        th = self.hagen_robot.theta
        ct, st = math.cos(th), math.sin(th)
        rx, ry = self.hagen_robot.x, self.hagen_robot.y
        for mpx, mpy in zip(self.main_path_x, self.main_path_y):
            if (mpx - rx) * ct + (mpy - ry) * st < 0.0:
                continue
            if math.hypot(mpx - rx, mpy - ry) < 0.5:
                return True
        return False

    def _resume_main_path(self):
        self.drive_state           = DriveState.FOLLOWING_PATH
        self.controller.ref_path_x = self.main_path_x
        self.controller.ref_path_y = self.main_path_y
        self.waiting_since         = None
        # Seek to nearest main-path point, NOT index 0 — avoids backtracking to start
        self.controller.reset_index_to_nearest(self.hagen_robot.x, self.hagen_robot.y)
        # Suppress re-triggering avoidance for the same just-passed obstacle
        from rclpy.duration import Duration
        self._avoidance_cooldown_until = self.get_clock().now() + Duration(seconds=2)

    def _pub_obstacle_info(self, detected: bool):
        self.obstacle_info_pub.publish(Bool(data=detected))

    # ================================================================== #
    #  CALLBACKS                                                           #
    # ================================================================== #
    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            self.get_logger().warn(f"Path too short ({len(msg.poses)} poses).")
            return

        xs = np.array([p.pose.position.x for p in msg.poses], dtype=float)
        ys = np.array([p.pose.position.y for p in msg.poses], dtype=float)

        self.main_path_x = xs
        self.main_path_y = ys
        self.path_received = True
        self.goal_reached  = False

        if self.drive_state != DriveState.OBSTACLE_AVOIDANCE:
            self.controller.ref_path_x = xs
            self.controller.ref_path_y = ys
            self.controller.reset_index()

        self.get_logger().info(
            f"New path: {len(msg.poses)} poses.", throttle_duration_sec=2.0
        )

    def kinematic_state_callback(self, msg: KinematicState):
        pose = msg.pose_with_covariance.pose
        self.hagen_robot.x = pose.position.x
        self.hagen_robot.y = pose.position.y
        q = pose.orientation
        # Standard quaternion → yaw (2D heading) extraction
        self.hagen_robot.theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.hagen_robot.v = msg.twist_with_covariance.twist.linear.x
        self.kinematic_state_received = True

    def tracked_objects_callback(self, msg: Detection3DArray):
        """
        Update detection list and estimate per-track velocity from position history.
        Subscribes to /tracked_objects (KF-filtered, stable track IDs) instead of
        raw /objects_in_map_frame so velocity estimation is reliable.
        """
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.last_detection_time = self.get_clock().now()

        new_cache: dict = {}
        for det in msg.detections:
            track_id = det.id
            ox = float(det.bbox.center.position.x)
            oy = float(det.bbox.center.position.y)

            old = self.obstacle_velocity_cache.get(track_id)
            if old is not None:
                dt = now_sec - old['time']
                if dt > 1e-3:
                    vx = (ox - old['x']) / dt
                    vy = (oy - old['y']) / dt
                else:
                    vx, vy = old['vx'], old['vy']
            else:
                vx, vy = 0.0, 0.0          # new track — assume static until next update

            new_cache[track_id] = {'x': ox, 'y': oy, 'vx': vx, 'vy': vy, 'time': now_sec}

        self.obstacle_velocity_cache = new_cache
        self.current_detections      = msg.detections

    def ackermann_feedback_callback(self, msg: AckermannDrive):
        self.hagen_robot.v = float(msg.speed)

    def allowed_to_move_callback(self, msg: Bool):
        val = bool(msg.data)
        if val != self.allowed_to_move:
            self.get_logger().info(f"/allowed_to_move → {val}")
        self.allowed_to_move = val

    def traffic_decision_callback(self, msg: Bool):
        stop = not bool(msg.data)
        if stop != self.traffic_stop:
            self.get_logger().info(f"/traffic_decision → stop={stop}")
        self.traffic_stop = stop

    def robot_description_callback(self, msg: String):
        self.get_logger().info(f"/robot_description received (len={len(msg.data)})")

    def publish_stop_command(self):
        msg = AckermannDrive()
        msg.steering_angle = 0.0
        msg.speed          = 0.0
        msg.acceleration   = -5.0   # hard braking deceleration [m/s²]
        try:
            if rclpy.ok():
                self.ackermann_pub.publish(msg)
        except Exception as exc:
            self.get_logger().debug(f"Stop publish failed: {exc}")


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


if __name__ == "__main__":
    main()
