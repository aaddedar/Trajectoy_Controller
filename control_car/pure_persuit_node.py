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
from visualization_msgs.msg import Marker

# Must match Arduino WHEELSTEERLIMIT (0.5235987756 rad = 30°).
# Arduino clips silently at this value, so commanding more is wasted range.
STEER_LIMIT   = 0.5236
TTC_HARD_STOP = 1.5   # seconds — stop if collision is this imminent (at 1.1 m/s → 1.65 m)
TTC_AVOID     = 3.5   # seconds — steer around person if collision is within this time

# ── DWA trajectory evaluation parameters ────────────────────────────────
DWA_PREDICT_STEPS = 20       # number of future timesteps to simulate
DWA_DT            = 0.1      # seconds per step  (matches KF tracker predict_dt)
DWA_CAR_RADIUS    = 0.45     # m — circle approximation of car footprint (half-diagonal + margin)
DWA_OBS_MARGIN    = 0.15     # m — extra safety margin added to each obstacle radius
DWA_SPEEDS        = [1.0, 1.1, 1.2]          # m/s candidates — must be above ESC dead-band (~0.9 m/s)
DWA_STEERS        = [-0.5236, -0.35, -0.175, 0.0, 0.175, 0.35, 0.5236]  # rad candidates


class DWAObstacle:
    """Lightweight container for a tracked obstacle used in DWA collision checking."""
    __slots__ = ('x', 'y', 'vx', 'vy', 'size_x', 'size_y')

    def __init__(self, x, y, vx, vy, size_x, size_y):
        self.x      = x
        self.y      = y
        self.vx     = vx
        self.vy     = vy
        self.size_x = size_x
        self.size_y = size_y


class HagenRobot:
    """
    Simple robot state container.
    """

    def __init__(self, x=0, y=0, theta=0, v=0, L=0.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L

class PurePurSuitController:
    """
    Implements the pure pursuit path tracking controller.
    """

    def __init__(self, robot_model, ref_path_x, ref_path_y, L_d=2.0, k=0.3):
        self.hagen_robot = robot_model
        self.L_d = L_d
        self.ref_path_x = ref_path_x
        self.ref_path_y = ref_path_y
        self.k = k
        self.last_index = 0

    def pure_pursuit_control(self):
        """
        Compute steering angle combining pure pursuit look-ahead with a cross-track
        error (CTE) correction term.

        Pure pursuit alone recovers from deviations slowly once last_index advances
        past a corner apex.  The CTE term actively steers back toward the nearest
        path point so the car does not drift further after going slightly wide.
        """
        target_index = self.look_ahead_point_index()
        nearest_idx  = self.last_index          # set as side-effect of look_ahead_point_index

        # ── Pure pursuit steering ─────────────────────────────────────────────
        t_x, t_y = self.ref_path_x[target_index], self.ref_path_y[target_index]
        alpha = (
            np.arctan2(t_y - self.hagen_robot.y, t_x - self.hagen_robot.x)
            - self.hagen_robot.theta
        )
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        LF = self.k * abs(self.hagen_robot.v) + self.L_d
        delta = np.arctan(2 * self.hagen_robot.L * np.sin(alpha) / (LF + 1e-5))

        # ── Cross-track error correction ──────────────────────────────────────
        # CTE = signed lateral distance from car to nearest path point.
        # Positive = car is LEFT of path → steer right (subtract from delta).
        n = len(self.ref_path_x)
        if nearest_idx + 1 < n:
            nx, ny = self.ref_path_x[nearest_idx], self.ref_path_y[nearest_idx]
            tx = self.ref_path_x[nearest_idx + 1] - nx
            ty = self.ref_path_y[nearest_idx + 1] - ny
            tlen = np.hypot(tx, ty)
            if tlen > 1e-6:
                # Left-normal = rotate tangent 90° CCW
                ln_x, ln_y = -ty / tlen, tx / tlen
                cte = ((self.hagen_robot.x - nx) * ln_x
                       + (self.hagen_robot.y - ny) * ln_y)
                # Stanley-style correction: atan(k_cte * cte / speed)
                k_cte = 2.5   # strong recovery — pulls car back within ~2 path steps
                delta_cte = -np.arctan2(k_cte * cte,
                                        abs(self.hagen_robot.v) + 0.1)
                delta += delta_cte

        delta = np.clip(delta, -STEER_LIMIT, STEER_LIMIT)
        return delta, target_index

    def look_ahead_point_index(self):
        """
        Find the index of the look-ahead point on the reference path.
        Returns:
            index: Index of the look-ahead point
        """
        n = len(self.ref_path_x)
        self.last_index = min(self.last_index, n - 1)
        # Allow a small backward window (15 pts = 0.75 m) so the nearest-point
        # index doesn't jump past the apex when the car goes slightly wide.
        # 0.75 m < half U-turn arc (~1.36 m), so this won't snap to the wrong leg.
        BACK = 15
        search_start = max(0, self.last_index - BACK)
        search_x = self.ref_path_x[search_start:]
        search_y = self.ref_path_y[search_start:]
        dx = self.hagen_robot.x - search_x
        dy = self.hagen_robot.y - search_y
        d = np.sqrt(dx**2 + dy**2)
        index = search_start + int(np.argmin(d))
        # Only advance last_index — allow at most BACK pts of retreat
        self.last_index = max(search_start, index)

        L = 0
        LF = self.k * abs(self.hagen_robot.v) + self.L_d
        while LF > L and (index + 1) < n:
            step_x = self.ref_path_x[index + 1] - self.ref_path_x[index]
            step_y = self.ref_path_y[index + 1] - self.ref_path_y[index]
            L += np.sqrt(step_x**2 + step_y**2)
            index += 1
        return index

class PurePursuitNode(Node):
    """
    ROS 2 Node for pure pursuit path tracking.
    Subscribes to /robot_description, /path, and /kinematic_state, publishes robot state.
    """

    def __init__(self):
        super().__init__("pure_pursuit_node")
        # Parameters (could be loaded from ROS params)
        L_d = 0.16   # base look-ahead — e_ss=Ld²/(2R): 0.2m→4cm, 1.0m→58cm on min curve
        k   = 0.05  # small speed scaling keeps LF tight even at high speed
        self.command_speed = 1.2      # m/s on straight
        self.min_curve_speed = 1.0   # m/s at tightest curve — ESC dead-band is ~0.9 m/s
        self.Ts = 1.0 / 30.0  # 30 Hz — matches kinematic_state rate, avoids single-thread starvation
        # Goal tolerance aligned with planner (typically 0.10 m),
        # set higher to prevent early stopping and allow planner to advance.
        self.goal_tolerance = 0.3   # tighter for small model city map
        self.publish_steering_in_degrees = False
        self.invert_steering_sign = True
        self.hagen_robot = HagenRobot(x=0.0, y=0.0, theta=0.0, v=0.0, L=0.5)
        self.controller = PurePurSuitController(
            self.hagen_robot,
            np.array([], dtype=float),
            np.array([], dtype=float),
            L_d,
            k,
        )
        self.path_received = False
        self.kinematic_state_received = False
        self.goal_reached = False
        self._using_velocity_heading = False  # hysteresis flag for theta source
        self.allowed_to_move_stop = False
        self.traffic_stop = False
        self.stop_requested = False  # combined: allowed_to_move_stop OR traffic_stop
        self.obstacle_stop_requested = False
        # Distances derived from TTC thresholds × command_speed (1.1 m/s):
        #   TTC_HARD_STOP=1.5s → 1.65m,  TTC_AVOID=3.5s → 3.85m
        self.obstacle_distance_threshold = 4.5   # outer slow-down boundary (m)
        self.obstacle_avoidance_distance = 3.85  # inside here: steer around obstacle (must match TTC_AVOID * speed)
        self.obstacle_hard_stop_distance = 1.65  # inside here: hard stop (must match TTC_HARD_STOP * speed)
        self.avoidance_gain = 0.4                # rad of steering per meter of lateral offset
        self.closest_obstacle_distance = float('inf')
        self.closest_obstacle_cy = 0.0           # lateral position of closest obstacle (car frame)
        self.car_width = 0.40                    # physical car width (m)
        self.path_corridor_margin = 1.0          # wide corridor — catches persons up to 1.2m off-center
        # Obstacle timeout: clear obstacle flag if no detections arrive within this window
        self.last_detection_time = None
        self.obstacle_timeout = 5.0
        # Kinematic state staleness: stop if /kinematic_state goes silent
        self.last_kinematic_time = None
        self.kinematic_state_timeout = 1.0
        # DWA: list of DWAObstacle objects, populated from /tracked_objects persons.
        # Non-empty → control_loop runs DWA trajectory evaluation instead of TTC nudge.
        self.dwa_obstacles: list = []

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
        self.ackermann_pub = self.create_publisher(
            AckermannDrive, "/ackermann_drive", reliable_qos
        )
        self.obstacle_information_pub = self.create_publisher(
            Bool, "/obstacle_information", reliable_qos
        )
        self.lookahead_marker_pub = self.create_publisher(
            Marker, "/pure_pursuit/lookahead_marker", 10
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.path_sub = self.create_subscription(
            Path, "/path", self.path_callback, latched_qos
        )
        self.ackermann_feedback_sub = self.create_subscription(
            AckermannDrive,
            "/ackermann_drive_feedback",
            self.ackermann_feedback_callback,
            best_effort_qos,
        )
        self.allowed_to_move_sub = self.create_subscription(
            Bool,
            "/allowed_to_move",
            self.allowed_to_move_callback,
            reliable_qos,
        )
        self.objects_in_map_frame_sub = self.create_subscription(
            Detection3DArray,
            "/tracked_objects",
            self.objects_in_map_frame_callback,
            reliable_qos,
        )
        self.kinematic_state_sub = self.create_subscription(
            KinematicState,
            "/kinematic_state",
            self.kinematic_state_callback,
            best_effort_qos,
        )
        self.traffic_decision_sub = self.create_subscription(
            Bool,
            "/traffic_decision",
            self.traffic_decision_callback,
            reliable_qos,
        )
        # Control loop timer
        self.timer = self.create_timer(self.Ts, self.control_loop)
        # Obstacle timeout timer: checks every 0.5s if detections have gone stale
        
        self.obstacle_timeout_timer = self.create_timer(0.5, self._check_obstacle_timeout)
        self.get_logger().info("PurePursuitNode started.")

    def control_loop(self):
        """
        Main control loop: computes control, updates robot, publishes state.
        """
        self.publish_obstacle_information()

        if self.stop_requested:
            self.publish_stop_command()
            self.get_logger().warn(
                f"STOPPED: /allowed_to_move={not self.allowed_to_move_stop} "
                f"traffic_stop={self.traffic_stop}",
                throttle_duration_sec=2.0,
            )
            return

        if self.obstacle_stop_requested:
            self.publish_stop_command()
            self.get_logger().warn(
                f"STOPPED: obstacle hard-stop dist={self.closest_obstacle_distance:.2f}m "
                f"(threshold={self.obstacle_hard_stop_distance}m)",
                throttle_duration_sec=2.0,
            )
            return

        if (
            not self.path_received
            or len(self.controller.ref_path_x) < 2
            or len(self.controller.ref_path_y) < 2
        ):
            self.publish_stop_command()
            self.get_logger().warn(
                "STOPPED: no valid path yet — waiting for /path...",
                throttle_duration_sec=2.0,
            )
            return

        if not self.kinematic_state_received:
            self.publish_stop_command()
            self.get_logger().warn(
                "STOPPED: waiting for /kinematic_state...",
                throttle_duration_sec=2.0,
            )
            return

        goal_dx = self.controller.ref_path_x[-1] - self.hagen_robot.x
        goal_dy = self.controller.ref_path_y[-1] - self.hagen_robot.y
        goal_distance = float(np.hypot(goal_dx, goal_dy))

        if goal_distance <= self.goal_tolerance:
            if not self.goal_reached:
                self.goal_reached = True
                self.path_received = False
                self.controller.ref_path_x = np.array([], dtype=float)
                self.controller.ref_path_y = np.array([], dtype=float)
                self.controller.last_index = 0
                self.closest_obstacle_distance = float('inf')
                self.closest_obstacle_cy = 0.0
                self.obstacle_stop_requested = False
                self.get_logger().info(
                    f"Goal reached at distance {goal_distance:.3f} m. Path cleared. Steering straight."
                )
            self.publish_stop_command()
            return

        delta, target_index = self.controller.pure_pursuit_control()

        if self.dwa_obstacles:
            # ── DWA ACTIVE — log every obstacle before evaluating ────────────
            obs_lines = []
            for i, obs in enumerate(self.dwa_obstacles):
                cx, cy = self._map_to_car_frame(obs.x, obs.y)
                dist   = math.hypot(cx, cy)
                spd    = math.hypot(obs.vx, obs.vy)
                obs_lines.append(
                    f"  obs[{i}] map=({obs.x:.2f},{obs.y:.2f}) "
                    f"car=(fwd={cx:.2f}m lat={cy:.2f}m) dist={dist:.2f}m "
                    f"vel=({obs.vx:.2f},{obs.vy:.2f}) spd={spd:.2f}m/s"
                )
            self.get_logger().warn(
                f"\n{'='*55}\n"
                f"DWA ACTIVATED — {len(self.dwa_obstacles)} obstacle(s)\n"
                + "\n".join(obs_lines) + "\n"
                f"  PP steering: {math.degrees(delta):.1f}deg\n"
                f"{'='*55}",
                throttle_duration_sec=0.5,
            )

            v_dwa, delta_dwa = self._run_dwa(delta)

            if v_dwa == 0.0:
                self.publish_stop_command()
                self.get_logger().warn(
                    f"DWA: *** NO SAFE TRAJECTORY *** — stopping car\n"
                    f"  {len(self.dwa_obstacles)} obstacle(s) block all {len(DWA_SPEEDS)*len(DWA_STEERS)} candidates",
                    throttle_duration_sec=1.0,
                )
                return

            curve_cap = self._compute_ahead_curvature_speed()
            cmd_speed = min(v_dwa, curve_cap)
            delta     = delta_dwa
            # NOTE: do NOT call pure_pursuit_control() again here — it advances
            # last_index as a side-effect, which would corrupt path tracking.
            self.get_logger().warn(
                f"DWA RESULT: v={v_dwa:.2f}m/s (curve_cap={curve_cap:.2f}) "
                f"→ cmd={cmd_speed:.2f}m/s  steer={math.degrees(delta_dwa):.1f}deg "
                f"(PP was {math.degrees(delta):.1f}deg before DWA override)",
                throttle_duration_sec=0.5,
            )
        else:
            delta     = self._apply_avoidance(delta)
            cmd_speed = float(min(
                self._compute_curve_speed(delta),       # reactive: current steering angle
                self._compute_ahead_curvature_speed(),  # proactive: upcoming path curvature
                self._compute_safe_speed(),             # obstacle proximity
            ))

        # ── Look-ahead target info ────────────────────────────────────────────
        px = self.controller.ref_path_x
        py = self.controller.ref_path_y
        nearest_idx = self.controller.last_index
        t_x = float(px[target_index]) if len(px) > target_index else float('nan')
        t_y = float(py[target_index]) if len(py) > target_index else float('nan')
        n_x = float(px[nearest_idx]) if len(px) > nearest_idx else float('nan')
        n_y = float(py[nearest_idx]) if len(py) > nearest_idx else float('nan')
        dist_to_target = math.hypot(t_x - self.hagen_robot.x,
                                    t_y - self.hagen_robot.y)
        cte_now = 0.0
        if nearest_idx + 1 < len(px):
            tx_seg = px[nearest_idx + 1] - n_x
            ty_seg = py[nearest_idx + 1] - n_y
            seg_len = math.hypot(tx_seg, ty_seg)
            if seg_len > 1e-6:
                ln_x, ln_y = -ty_seg / seg_len, tx_seg / seg_len
                cte_now = ((self.hagen_robot.x - n_x) * ln_x +
                           (self.hagen_robot.y - n_y) * ln_y)

        self.get_logger().info(
            f"\n"
            f"  CURRENT : pos=({self.hagen_robot.x:.3f},{self.hagen_robot.y:.3f}) "
            f"theta={math.degrees(self.hagen_robot.theta):.1f}deg "
            f"v={self.hagen_robot.v:.2f}m/s\n"
            f"  NEAREST : path[{nearest_idx}]=({n_x:.3f},{n_y:.3f})  "
            f"CTE={cte_now:+.3f}m ({'LEFT' if cte_now>0 else 'RIGHT'} of path)\n"
            f"  LOOKAHEAD: path[{target_index}]=({t_x:.3f},{t_y:.3f})  "
            f"dist={dist_to_target:.3f}m\n"
            f"  CMD     : delta={math.degrees(delta):.1f}deg  speed={cmd_speed:.2f}m/s  "
            f"dwa_obs={len(self.dwa_obstacles)}",
            throttle_duration_sec=0.5,
        )

        # ── RViz marker: look-ahead target (yellow sphere) ───────────────────
        mk = Marker()
        mk.header.frame_id = 'map'
        mk.header.stamp    = self.get_clock().now().to_msg()
        mk.ns     = 'pure_pursuit'
        mk.id     = 0
        mk.type   = Marker.SPHERE
        mk.action = Marker.ADD
        mk.pose.position.x    = t_x
        mk.pose.position.y    = t_y
        mk.pose.position.z    = 0.1
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.15
        mk.color.r = 1.0; mk.color.g = 1.0; mk.color.b = 0.0; mk.color.a = 1.0
        self.lookahead_marker_pub.publish(mk)

        # ── RViz marker: nearest path point (cyan sphere) ────────────────────
        mk2 = Marker()
        mk2.header = mk.header
        mk2.ns     = 'pure_pursuit'
        mk2.id     = 1
        mk2.type   = Marker.SPHERE
        mk2.action = Marker.ADD
        mk2.pose.position.x    = n_x
        mk2.pose.position.y    = n_y
        mk2.pose.position.z    = 0.1
        mk2.pose.orientation.w = 1.0
        mk2.scale.x = mk2.scale.y = mk2.scale.z = 0.10
        mk2.color.r = 0.0; mk2.color.g = 1.0; mk2.color.b = 1.0; mk2.color.a = 1.0
        self.lookahead_marker_pub.publish(mk2)

        # ── RViz marker: arrow from car to look-ahead ─────────────────────────
        arrow = Marker()
        arrow.header = mk.header
        arrow.ns     = 'pure_pursuit'
        arrow.id     = 2
        arrow.type   = Marker.ARROW
        arrow.action = Marker.ADD
        from geometry_msgs.msg import Point
        p_start          = Point()
        p_start.x        = self.hagen_robot.x
        p_start.y        = self.hagen_robot.y
        p_start.z        = 0.1
        p_end            = Point()
        p_end.x          = t_x
        p_end.y          = t_y
        p_end.z          = 0.1
        arrow.points     = [p_start, p_end]
        arrow.scale.x    = 0.04   # shaft diameter
        arrow.scale.y    = 0.08   # head diameter
        arrow.scale.z    = 0.10   # head length
        arrow.color.r    = 1.0; arrow.color.g = 1.0; arrow.color.b = 0.0
        arrow.color.a    = 1.0
        self.lookahead_marker_pub.publish(arrow)

        # Publish Ackermann command
        drive_msg = AckermannDrive()
        steering_cmd = -delta if self.invert_steering_sign else delta
        if self.publish_steering_in_degrees:
            drive_msg.steering_angle = float(np.degrees(steering_cmd))
        else:
            drive_msg.steering_angle = float(steering_cmd)
        drive_msg.speed = cmd_speed
        self.ackermann_pub.publish(drive_msg)

    def _apply_avoidance(self, delta: float) -> float:
        """
        Blend path-following steering with obstacle avoidance steering.

        Zones (by forward distance to closest dynamic obstacle):
          dist > avoidance_distance : pure path steering, no nudge
          hard_stop < dist ≤ avoidance_distance : path + lateral nudge AWAY from obstacle
          dist ≤ hard_stop : handled upstream (hard stop), not reached here

        Avoidance nudge:
          obstacle at cy > 0 (LEFT)  → steer right → subtract from delta
          obstacle at cy < 0 (RIGHT) → steer left  → add to delta
          nudge = -avoidance_gain * cy * proximity_factor
        """
        dist = self.closest_obstacle_distance
        if (dist > self.obstacle_avoidance_distance
                or dist <= self.obstacle_hard_stop_distance
                or self.obstacle_stop_requested):
            return delta

        # Proximity factor: 0.0 at avoidance boundary, 1.0 at hard-stop boundary
        proximity = 1.0 - (dist - self.obstacle_hard_stop_distance) / (
            self.obstacle_avoidance_distance - self.obstacle_hard_stop_distance
        )

        nudge = -self.avoidance_gain * self.closest_obstacle_cy * proximity
        combined = float(np.clip(delta + nudge, -STEER_LIMIT, STEER_LIMIT))

        self.get_logger().info(
            f"AVOIDANCE: cy={self.closest_obstacle_cy:.2f}m dist={dist:.2f}m "
            f"proximity={proximity:.2f} nudge={math.degrees(nudge):.1f}deg "
            f"delta={math.degrees(delta):.1f}→{math.degrees(combined):.1f}deg",
            throttle_duration_sec=1.0,
        )
        return combined

    def _check_obstacle_timeout(self):
        """
        Runs every 0.5 s. Clears stale obstacle state when detections stop arriving.
        Also resets kinematic_state_received if /kinematic_state goes silent (MOCAP dropout),
        which causes the control loop to stop the car until state resumes.
        """
        now = self.get_clock().now()

        # Kinematic state watchdog: stop the car if /kinematic_state goes silent
        if (self.last_kinematic_time is not None and self.kinematic_state_received):
            kin_elapsed = (now - self.last_kinematic_time).nanoseconds / 1e9
            if kin_elapsed > self.kinematic_state_timeout:
                self.get_logger().warn(
                    f"No /kinematic_state for {kin_elapsed:.1f}s — stopping.",
                    throttle_duration_sec=2.0,
                )
                self.kinematic_state_received = False

        if self.last_detection_time is None:
            return
        elapsed = (now - self.last_detection_time).nanoseconds / 1e9
        if elapsed > self.obstacle_timeout and (
            self.obstacle_stop_requested
            or self.closest_obstacle_distance < float('inf')
            or self.dwa_obstacles
        ):
            self.get_logger().info(
                f"No detections for {elapsed:.1f}s — clearing obstacle state, resuming."
            )
            self.obstacle_stop_requested = False
            self.closest_obstacle_distance = float('inf')
            self.closest_obstacle_cy = 0.0
            self.dwa_obstacles = []
            self.publish_obstacle_information()

    @staticmethod
    def _smooth_path(x: np.ndarray, y: np.ndarray,
                     window: int = 7, iterations: int = 3) -> tuple:
        """
        Iterative moving-average smoothing — rounds off sharp A* grid corners.
        Uses edge-replication padding so boundary points are not pulled toward (0,0)
        as they would be with the default zero-padding in np.convolve mode='same'.
        """
        pad = window // 2
        kernel = np.ones(window) / window
        for _ in range(iterations):
            x = np.convolve(np.pad(x, pad, mode='edge'), kernel, mode='valid')
            y = np.convolve(np.pad(y, pad, mode='edge'), kernel, mode='valid')
        return x, y

    @staticmethod
    def _densify_path(x: np.ndarray, y: np.ndarray, spacing: float = 0.05) -> tuple:
        """
        Resample path at uniform arc-length spacing (default 5 cm).

        A* step_size_cells=34 × 7.69 mm ≈ 0.26 m per waypoint.
        With L_d=0.1 m the while-loop look-ahead exits after a single step, giving
        zero curve anticipation.  Densifying to 5 cm gives 4–5 points of preview
        so the controller can actually see the upcoming curve direction.
        """
        if len(x) < 2:
            return x, y
        dx = np.diff(x)
        dy = np.diff(y)
        seg_len = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = s[-1]
        if total < spacing:
            return x, y
        s_new = np.arange(0.0, total, spacing)
        x_new = np.interp(s_new, s, x)
        y_new = np.interp(s_new, s, y)
        # Always include the exact goal endpoint
        x_new = np.append(x_new, x[-1])
        y_new = np.append(y_new, y[-1])
        return x_new, y_new

    def path_callback(self, msg):
        """
        Callback for /path topic.
        """
        pose_count = len(msg.poses)
        if pose_count < 2:
            self.get_logger().warn(
                f"Received /path with {pose_count} pose(s). Need at least 2 poses."
            )
            return

        ref_x = np.array([pose.pose.position.x for pose in msg.poses], dtype=float)
        ref_y = np.array([pose.pose.position.y for pose in msg.poses], dtype=float)

        # 1. Smooth: round off sharp A* staircase corners into arcs.
        # 2. Densify: resample to 5 cm spacing so the look-ahead has fine resolution.
        #    Without densification, LF (0.21 m) < step size (0.26 m) → the while-loop
        #    always exits after 1 step → zero curve anticipation → car goes wide.
        start_x, start_y = ref_x[0], ref_y[0]
        end_x,   end_y   = ref_x[-1], ref_y[-1]
        ref_x, ref_y = self._smooth_path(ref_x, ref_y)
        ref_x, ref_y = self._densify_path(ref_x, ref_y, spacing=0.05)
        ref_x[0],  ref_y[0]  = start_x, start_y
        ref_x[-1], ref_y[-1] = end_x,   end_y

        self.controller.ref_path_x = ref_x
        self.controller.ref_path_y = ref_y
        self.controller.last_index = 0
        self.path_received = True
        self.goal_reached = False
        # Do NOT clear obstacle state here — obstacle may still be present
        # when path planner replans around it. Obstacle callback will update correctly.

        # Seed heading from first path segment only when stationary.
        # If the car is already moving, velocity-based theta is more accurate.
        if pose_count >= 2 and self.hagen_robot.v < 0.05:
            dx = ref_x[1] - ref_x[0]
            dy = ref_y[1] - ref_y[0]
            self.hagen_robot.theta = math.atan2(dy, dx)

        self.get_logger().info(
            f"Updated controller path from /path with {pose_count} poses (smoothed)",
             throttle_duration_sec=2.0,
        )
    
    def traffic_decision_callback(self, msg: Bool):
        """
        Stop when traffic_decision is False (red), move when True (green).
        """
        self.traffic_stop = not bool(msg.data)
        self.stop_requested = self.allowed_to_move_stop or self.traffic_stop
        self.get_logger().info(
            f"/traffic_decision={msg.data} traffic_stop={self.traffic_stop} "
            f"stop_requested={self.stop_requested}"
        )

    def kinematic_state_callback(self, msg):
        """
        Callback for /kinematic_state topic. Updates robot pose from message.
        Heading is derived from map-frame velocity so it is correct regardless of
        MOCAP quaternion convention. Falls back to MOCAP yaw when stationary.
        """
        pose = msg.pose_with_covariance.pose
        self.hagen_robot.x = pose.position.x
        self.hagen_robot.y = pose.position.y

        # Map-frame velocity
        vx_map = msg.twist_with_covariance.twist.linear.x
        vy_map = msg.twist_with_covariance.twist.linear.y
        map_speed = math.hypot(vx_map, vy_map)
        self.hagen_robot.v = map_speed

        # ── Heading: MOCAP quaternion + 180° Z correction ────────────────────
        # Velocity-based heading lags during curves (velocity vector trails the
        # car body angle). MOCAP quaternion gives instantaneous body heading.
        # The MOCAP frame is 180° rotated around Z vs map frame → add π to yaw.
        q = pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_mocap = math.atan2(siny, cosy)
        theta_quat = math.atan2(math.sin(yaw_mocap + math.pi),
                                math.cos(yaw_mocap + math.pi))

        if map_speed > 0.05:
            # Blend: 70% quaternion (accurate heading) + 30% velocity (removes
            # any quaternion drift by anchoring to actual motion direction).
            vel_theta = math.atan2(-vy_map, -vx_map)
            # Circular-mean blend (handles wrap-around correctly)
            dx = 0.7 * math.cos(theta_quat) + 0.3 * math.cos(vel_theta)
            dy = 0.7 * math.sin(theta_quat) + 0.3 * math.sin(vel_theta)
            self.hagen_robot.theta = math.atan2(dy, dx)
        else:
            # Stationary: quaternion only (no velocity to blend)
            self.hagen_robot.theta = theta_quat

        self.kinematic_state_received = True
        self.last_kinematic_time = self.get_clock().now()
        self.get_logger().info(
            f"kin: pos=({self.hagen_robot.x:.3f},{self.hagen_robot.y:.3f}) "
            f"theta={math.degrees(self.hagen_robot.theta):.1f}deg "
            f"v={map_speed:.3f}m/s vel_heading={self._using_velocity_heading}",
            throttle_duration_sec=1.0,
        )

    def ackermann_feedback_callback(self, msg):
        self.get_logger().debug(
            f"Ackermann feedback: speed={msg.speed:.3f}, steering={msg.steering_angle:.3f}"
        )

    def allowed_to_move_callback(self, msg):
        """
        Stop vehicle when /allowed_to_move is False, allow movement when True.
        """
        self.allowed_to_move_stop = not bool(msg.data)
        self.stop_requested = self.allowed_to_move_stop or self.traffic_stop
        self.get_logger().info(
            f"/allowed_to_move={msg.data} allowed_stop={self.allowed_to_move_stop} "
            f"stop_requested={self.stop_requested}"
        )

    def objects_in_map_frame_callback(self, msg):
        """
        Callback for /tracked_objects (KF tracker output).
        Only reacts to 'person' class. All other objects are static and handled by the path planner.
        Uses Time-To-Collision (TTC) to decide: stop vs steer around vs ignore.
        """
        self.last_detection_time = self.get_clock().now()

        # Log every object so we can trace exactly where in the pipeline detections are lost.
        # Shows class + map-frame position for each tracked object.
        if msg.detections:
            obj_summary = " | ".join(
                f"{d.results[0].hypothesis.class_id if d.results else '?'}"
                f"@({d.bbox.center.position.x:.1f},{d.bbox.center.position.y:.1f})"
                for d in msg.detections
            )
            self.get_logger().warn(
                f"[tracked] {len(msg.detections)} obj(s): {obj_summary}",
                throttle_duration_sec=1.0,
            )
        else:
            self.get_logger().info(
                "[tracked] 0 objects — clear",
                throttle_duration_sec=2.0,
            )

        if not self.kinematic_state_received:
            self.closest_obstacle_distance = float('inf')
            self.obstacle_stop_requested = False
            return

        best_ttc = float('inf')
        best_cy  = 0.0

        cos_t = math.cos(self.hagen_robot.theta)
        sin_t = math.sin(self.hagen_robot.theta)

        # Persons that passed both the ahead-check and corridor-check.
        # DWA uses this same set — avoids acting on persons already cleared by TTC.
        corridor_half = self.car_width / 2.0 + self.path_corridor_margin
        threat_persons: list = []

        for detection in msg.detections:
            if not detection.results:
                continue
            class_name = detection.results[0].hypothesis.class_id.lower()
            if class_name != "person":
                continue  # static objects handled by path planner, ignore here

            # Position in map frame (bounding box center)
            x_map = float(detection.bbox.center.position.x)
            y_map = float(detection.bbox.center.position.y)
            cx, cy = self._map_to_car_frame(x_map, y_map)

            if cx <= 0.1:
                continue  # person is behind or at car — ignore

            # Corridor check: ignore person far to the side (not in car's path)
            if abs(cy) > corridor_half:
                self.get_logger().warn(
                    f"[person] outside corridor cy={cy:.2f}m half={corridor_half:.2f}m — ignored",
                    throttle_duration_sec=1.0,
                )
                continue

            # KF velocity embedded in results[0].pose.pose.position (map frame)
            vx_p = float(detection.results[0].pose.pose.position.x)
            vy_p = float(detection.results[0].pose.pose.position.y)

            # Person velocity components in car frame
            v_cx =  vx_p * cos_t + vy_p * sin_t   # forward component (+ve = same dir as car)
            v_cy = -vx_p * sin_t + vy_p * cos_t   # lateral component (+ve = left of car)

            # Closing speed: how fast the gap is shrinking
            closing_speed = self.hagen_robot.v - v_cx
            ttc = (cx / closing_speed) if closing_speed > 0.05 else float('inf')

            direction = self._person_direction(v_cx, v_cy)
            self.get_logger().warn(
                f"[person] fwd={cx:.2f}m lat={cy:.2f}m "
                f"v_fwd={v_cx:.2f} v_lat={v_cy:.2f} "
                f"closing={closing_speed:.2f}m/s TTC={ttc:.1f}s dir={direction}",
                throttle_duration_sec=0.5,
            )

            if ttc < best_ttc:
                best_ttc = ttc
                best_cy  = cy

            # Collect for DWA — same gate as TTC (ahead + in corridor)
            threat_persons.append(detection)

        # ── Decision ──────────────────────────────────────────────────────────
        if best_ttc == float('inf'):
            # No person on collision course → clear everything including DWA
            if self.obstacle_stop_requested or self.closest_obstacle_distance < float('inf'):
                self.get_logger().info("Person cleared — resuming normal drive")
            self.obstacle_stop_requested    = False
            self.closest_obstacle_distance  = float('inf')
            self.closest_obstacle_cy        = 0.0
            self.dwa_obstacles              = []   # clear so DWA doesn't run on stale data

        elif best_ttc < TTC_HARD_STOP:
            # Too close to steer around — STOP
            self.obstacle_stop_requested    = True
            self.closest_obstacle_distance  = 0.0
            self.closest_obstacle_cy        = best_cy
            self.dwa_obstacles              = []   # hard-stop takes over; DWA not needed
            self.get_logger().warn(
                f"HARD STOP: TTC={best_ttc:.1f}s < {TTC_HARD_STOP}s — collision imminent",
                throttle_duration_sec=0.5,
            )

        else:
            # TTC manageable — steer around person via DWA
            virtual_dist = best_ttc * self.hagen_robot.v
            self.obstacle_stop_requested   = False
            self.closest_obstacle_distance = virtual_dist
            self.closest_obstacle_cy       = best_cy
            # Populate DWA with only the qualified (ahead + corridor) persons
            self.dwa_obstacles = [
                DWAObstacle(
                    x=float(det.bbox.center.position.x),
                    y=float(det.bbox.center.position.y),
                    vx=float(det.results[0].pose.pose.position.x),
                    vy=float(det.results[0].pose.pose.position.y),
                    size_x=float(det.bbox.size.x) if det.bbox.size.x > 0 else 0.5,
                    size_y=float(det.bbox.size.y) if det.bbox.size.y > 0 else 0.5,
                )
                for det in threat_persons
            ]

        self.publish_obstacle_information()

    @staticmethod
    def _person_direction(v_cx: float, v_cy: float) -> str:
        """Person movement direction relative to the car's heading."""
        speed = math.hypot(v_cx, v_cy)
        if speed < 0.1:
            return "stationary"
        if abs(v_cx) >= abs(v_cy):
            return "toward_car" if v_cx < 0 else "away_from_car"
        return "left" if v_cy > 0 else "right"

    # ── DWA: Dynamic Window Approach ─────────────────────────────────────────

    @staticmethod
    def _simulate_car_trajectory(x0, y0, theta0, v, delta, L,
                                 n_steps=DWA_PREDICT_STEPS, dt=DWA_DT):
        """
        Integrate the bicycle kinematic model for n_steps × dt seconds.
        Returns list of (x, y, theta) — one entry per future timestep.
        """
        states = []
        x, y, theta = x0, y0, theta0
        for _ in range(n_steps):
            x     += v * math.cos(theta) * dt
            y     += v * math.sin(theta) * dt
            theta += (v / L) * math.tan(delta) * dt
            states.append((x, y, theta))
        return states

    def _cte_to_path(self, x: float, y: float) -> float:
        """
        Signed lateral distance from (x, y) to the nearest reference path segment.
        Positive = point is LEFT of path direction.
        Returns 0 if path is not available.
        """
        px = self.controller.ref_path_x
        py = self.controller.ref_path_y
        if len(px) < 2:
            return 0.0
        dx  = x - px
        dy  = y - py
        idx = int(np.argmin(dx ** 2 + dy ** 2))
        if idx + 1 >= len(px):
            idx = len(px) - 2
        tx   = px[idx + 1] - px[idx]
        ty   = py[idx + 1] - py[idx]
        tlen = math.hypot(tx, ty)
        if tlen < 1e-6:
            return 0.0
        # Left-normal of path segment
        ln_x, ln_y = -ty / tlen, tx / tlen
        return float((x - px[idx]) * ln_x + (y - py[idx]) * ln_y)

    def _run_dwa(self, pp_delta: float) -> tuple:
        """
        Dynamic Window Approach obstacle avoidance.

        Evaluates all (speed, steering) candidate trajectories against
        constant-velocity projections of every obstacle in self.dwa_obstacles.
        Collision detection uses circle approximations for speed.

        Returns (v_cmd, delta_cmd) for the best collision-free trajectory,
        or (0.0, 0.0) if every moving trajectory collides (car should stop).

        Strategy:
          1. Always try the pure-pursuit trajectory first (cheapest path).
          2. If PP is safe, use it — no unnecessary avoidance manoeuvre.
          3. Otherwise search all (speed × steering) candidates, rank by:
               cost = 2.0 × |CTE at end| + 0.5 × |delta - pp_delta| + 1.0 × (v_max - v)
             → prefers staying on path, minimises steering change, maximises speed.
        """
        x0  = self.hagen_robot.x
        y0  = self.hagen_robot.y
        th0 = self.hagen_robot.theta
        L   = self.hagen_robot.L

        # Pre-project every obstacle forward at constant velocity
        obs_futures = []
        for obs in self.dwa_obstacles:
            r_obs = max(obs.size_x, obs.size_y) / 2.0 + DWA_OBS_MARGIN
            future = [
                (obs.x + obs.vx * (k + 1) * DWA_DT,
                 obs.y + obs.vy * (k + 1) * DWA_DT)
                for k in range(DWA_PREDICT_STEPS)
            ]
            obs_futures.append((future, r_obs))

        def collides(states):
            for k, (cx, cy, _) in enumerate(states):
                for fut, r_obs in obs_futures:
                    ox, oy = fut[k]
                    if math.hypot(cx - ox, cy - oy) < (DWA_CAR_RADIUS + r_obs):
                        return True
            return False

        # Step 1: check whether pure-pursuit trajectory is already safe
        pp_speed  = self._compute_curve_speed(pp_delta)
        pp_states = self._simulate_car_trajectory(x0, y0, th0, pp_speed, pp_delta, L)
        if not collides(pp_states):
            self.get_logger().info(
                f"DWA: PP trajectory safe (v={pp_speed:.2f} delta={math.degrees(pp_delta):.1f}deg) — no avoidance",
                throttle_duration_sec=1.0,
            )
            return (pp_speed, pp_delta)

        self.get_logger().warn(
            f"DWA: PP trajectory COLLIDES — searching {len(DWA_SPEEDS)*len(DWA_STEERS)} candidates ...",
            throttle_duration_sec=0.5,
        )

        # Step 2: search all candidates for the lowest-cost safe trajectory
        best_v, best_delta, best_cost = None, None, float('inf')
        safe_count = 0

        for v in DWA_SPEEDS:
            for delta in DWA_STEERS:
                states = self._simulate_car_trajectory(x0, y0, th0, v, delta, L)
                if collides(states):
                    continue
                safe_count += 1
                end_x, end_y, _ = states[-1]
                cte  = abs(self._cte_to_path(end_x, end_y))
                cost = (2.0 * cte
                        + 0.5 * abs(delta - pp_delta)
                        + 1.0 * (self.command_speed - v))
                if cost < best_cost:
                    best_cost  = cost
                    best_v     = v
                    best_delta = delta

        if best_v is None:
            self.get_logger().warn(
                "DWA: 0 safe candidates found — all trajectories collide",
                throttle_duration_sec=0.5,
            )
            return (0.0, 0.0)

        best_end = self._simulate_car_trajectory(x0, y0, th0, best_v, best_delta, L)[-1]
        best_cte = abs(self._cte_to_path(best_end[0], best_end[1]))
        self.get_logger().warn(
            f"DWA: {safe_count}/{len(DWA_SPEEDS)*len(DWA_STEERS)} safe  "
            f"→ best v={best_v:.2f}m/s steer={math.degrees(best_delta):.1f}deg "
            f"cost={best_cost:.2f} cte={best_cte:.2f}m",
            throttle_duration_sec=0.5,
        )
        return (best_v, best_delta)

    def _map_to_car_frame(self, x_map: float, y_map: float) -> tuple:
        """Transform a point from map frame to car frame (base_link)."""
        dx = x_map - self.hagen_robot.x
        dy = y_map - self.hagen_robot.y
        cos_t = math.cos(self.hagen_robot.theta)
        sin_t = math.sin(self.hagen_robot.theta)
        cx =  dx * cos_t + dy * sin_t   # forward (+X in car frame)
        cy = -dx * sin_t + dy * cos_t   # lateral (+Y = left in car frame)
        return cx, cy


    def _compute_curve_speed(self, delta: float) -> float:
        """
        Scale speed with CURRENT steering angle (reactive).
          delta = 0.00 rad       → command_speed
          delta = STEER_LIMIT rad → min_curve_speed
        """
        ratio = min(1.0, abs(delta) / STEER_LIMIT)
        return self.command_speed - (self.command_speed - self.min_curve_speed) * ratio

    def _compute_ahead_curvature_speed(self) -> float:
        """
        Proactive speed reduction: look 1.5 m ahead on the reference path
        and find the tightest curve coming up. Slow down BEFORE entering it,
        not after the car is already going wide.

        Car minimum turning radius = L / tan(STEER_LIMIT) = 0.5 / 0.577 = 0.866 m
        Curvature k = 1/R.  k_max = 1/0.866 = 1.154 rad/m.
        At k_max → min_curve_speed.  At k=0 → command_speed.
        """
        px = self.controller.ref_path_x
        py = self.controller.ref_path_y
        n  = len(px)
        if n < 3:
            return self.command_speed

        idx   = self.controller.last_index
        R_min = self.hagen_robot.L / math.tan(STEER_LIMIT)   # 0.866 m
        k_max = 1.0 / R_min                                   # 1.154 rad/m

        LOOK_STEPS = 60   # 60 × 5 cm = 3.0 m ahead — slow down earlier before curve entry

        worst_k = 0.0
        for i in range(idx, min(idx + LOOK_STEPS, n - 2)):
            p1x, p1y = px[i],   py[i]
            p2x, p2y = px[i+1], py[i+1]
            p3x, p3y = px[min(i+2, n-1)], py[min(i+2, n-1)]
            # Menger curvature = 4 × triangle_area / (d12 × d23 × d13)
            area = abs((p2x-p1x)*(p3y-p1y) - (p3x-p1x)*(p2y-p1y)) / 2.0
            d12  = math.hypot(p2x-p1x, p2y-p1y)
            d23  = math.hypot(p3x-p2x, p3y-p2y)
            d13  = math.hypot(p3x-p1x, p3y-p1y)
            if d12 * d23 * d13 > 1e-10:
                worst_k = max(worst_k, 4.0 * area / (d12 * d23 * d13))

        # Linearly map [0, k_max] → [command_speed, min_curve_speed]
        ratio = min(1.0, worst_k / k_max)
        return self.command_speed - (self.command_speed - self.min_curve_speed) * ratio

    def _compute_safe_speed(self) -> float:
        """
        Scales command speed based on distance to the closest obstacle.
          dist > slow zone  : full speed
          hard_stop < dist <= slow zone : linear ramp down toward zero
          dist <= hard_stop : zero (hard stop handled before this is called)
        """
        dist = self.closest_obstacle_distance
        if dist <= self.obstacle_hard_stop_distance:
            return 0.0
        if dist <= self.obstacle_distance_threshold:
            factor = (dist - self.obstacle_hard_stop_distance) / (
                self.obstacle_distance_threshold - self.obstacle_hard_stop_distance
            )
            # Floor at min_curve_speed so the ESC always gets an actionable command
            return max(self.min_curve_speed, self.command_speed * factor)
        return self.command_speed
    
    def publish_obstacle_information(self):
        info_msg = Bool()
        info_msg.data = bool(self.obstacle_stop_requested)
        self.obstacle_information_pub.publish(info_msg)

    def publish_stop_command(self):
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = 0.0
        drive_msg.speed = 0.0
        try:
            if rclpy.ok():
                self.ackermann_pub.publish(drive_msg)
        except Exception as exc:
            self.get_logger().debug(
                f"Could not publish stop command during shutdown: {exc}"
            )

def main(args=None):
    """
    ROS 2 entry point.
    """
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