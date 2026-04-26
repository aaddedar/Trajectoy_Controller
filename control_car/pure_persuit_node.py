#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive
from curobot_msgs.msg import KinematicState
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection3DArray


class HagenRobot:
    """
    Simple robot state container.
    """

    def __init__(self, Ts, x=0, y=0, theta=0, v=0, L=2.4):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L
        self.Ts = Ts

class PurePurSuitController:
    """
    Implements the pure pursuit path tracking controller.
    """

    def __init__(self, robot_model, ref_path_x, ref_path_y, L_d=2.0, k=0.3, kp=1):
        self.hagen_robot = robot_model
        self.L_d = L_d
        self.ref_path_x = ref_path_x
        self.ref_path_y = ref_path_y
        self.k = k
        self.kp = kp

    def pure_pursuit_control(self):
        """
        Compute steering angle (delta) to follow the path using pure pursuit.
        Returns:
            delta: Steering angle
            target_index: Index of the look-ahead point
        """
        target_index = self.look_ahead_point_index()
        t_x, t_y = self.ref_path_x[target_index], self.ref_path_y[target_index]
        alpha = (
            np.arctan2(t_y - self.hagen_robot.y, t_x - self.hagen_robot.x)
            - self.hagen_robot.theta
        )
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        delta = np.arctan(
            2
            * self.hagen_robot.L
            * np.sin(alpha)
            / (self.k * (self.hagen_robot.v + 1e-5) + self.L_d)
        )
        delta_min = -np.pi / 6
        delta_max = np.pi / 6
        delta = np.clip(delta, delta_min, delta_max)
        return delta, target_index

    def look_ahead_point_index(self):
        """
        Find the index of the look-ahead point on the reference path.
        Returns:
            index: Index of the look-ahead point
        """
        dx = [self.hagen_robot.x - t_x for t_x in self.ref_path_x]
        dy = [self.hagen_robot.y - t_y for t_y in self.ref_path_y]
        d = [
            np.abs(np.sqrt(idx**2 + idy**2)) for (idx, idy) in zip(dx, dy, strict=False)
        ]
        index = d.index(min(d))
        L = 0
        LF = self.k * abs(self.hagen_robot.v) + self.L_d
        while LF > L and (index + 1) < len(self.ref_path_x):
            dx = self.ref_path_x[index + 1] - self.ref_path_x[index]
            dy = self.ref_path_y[index + 1] - self.ref_path_y[index]
            L += np.sqrt(dx**2 + dy**2)
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
        self.L_d = 0.8
        self.k = 0.35
        self.kp = 0.5
        self.command_speed = 0.1
        self.Ts = 0.01
        # Goal tolerance aligned with planner (typically 0.10 m),
        # set higher to prevent early stopping and allow planner to advance.
        self.goal_tolerance = 0.3
        self.publish_steering_in_degrees = True
        self.invert_steering_sign = False
        self.hagen_robot = HagenRobot(Ts=self.Ts, x=0.0, y=0.0, theta=0.0, v=0.0, L=0.5)
        self.controller = PurePurSuitController(
            self.hagen_robot,
            np.array([], dtype=float),
            np.array([], dtype=float),
            self.L_d,
            self.k,
            self.kp,
        )
        self.path_received = False
        self.kinematic_state_received = False
        self.goal_reached = False
        self.feedback_speed = 0.0
        self.feedback_steering = 0.0
        self.stop_requested = False
        self.object_detections_count = 0
        self.obstacle_stop_requested = False
        self.obstacle_distance_threshold = 2.0 #1.5 
        # Obstacle timeout: clear obstacle flag if no detections arrive within this window
        self.last_detection_time = None
        self.obstacle_timeout = 5.0 #1.0  # seconds

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
        # Subscribers
        self.robot_description_sub = self.create_subscription(
            String, "/robot_description", self.robot_description_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, "/path", self.path_callback, reliable_qos
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
            "/objects_in_map_frame",
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

        # self.traffic_decision_sub = self.create_subscription(
        #     String,
        #     "/traffic_decision",
        #     self.traffic_decision_callback,
        #     reliable_qos,
        # )
        # self.traffic_stop_requested = False
        # self.traffic_state = "UNKNOWN"
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
                "Stop requested from /allowed_to_move. Holding vehicle stopped.",
                throttle_duration_sec=5.0,
            )
            return

        if self.obstacle_stop_requested:
            self.publish_stop_command()
            self.get_logger().warn(
                "Obstacle too close from /objects_in_map_frame. Holding vehicle stopped.",
                throttle_duration_sec=5.0,
            )
            return

        if (
            not self.path_received
            or len(self.controller.ref_path_x) < 2
            or len(self.controller.ref_path_y) < 2
        ):
            self.publish_stop_command()
            self.get_logger().warn(
                "No valid path available yet. Waiting for /path...",
                throttle_duration_sec=5.0,
            )
            return

        if not self.kinematic_state_received:
            self.publish_stop_command()
            self.get_logger().warn(
                "Waiting for /kinematic_state before sending control commands.",
                throttle_duration_sec=5.0,
            )
            return

        goal_dx = self.controller.ref_path_x[-1] - self.hagen_robot.x
        goal_dy = self.controller.ref_path_y[-1] - self.hagen_robot.y
        goal_distance = float(np.hypot(goal_dx, goal_dy))

        if goal_distance <= self.goal_tolerance:
            if not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info(
                    f"Goal reached. Stopping vehicle at distance {goal_distance:.3f} m from final waypoint.",
                    throttle_duration_sec = 5.0
                )
            self.publish_stop_command()
            return

        delta, _ = self.controller.pure_pursuit_control()
        self.goal_reached = False
        # Publish Ackermann command
        drive_msg = AckermannDrive()
        steering_cmd = -delta if self.invert_steering_sign else delta
        if self.publish_steering_in_degrees:
            drive_msg.steering_angle = float(np.degrees(steering_cmd))
        else:
            drive_msg.steering_angle = float(steering_cmd)
        drive_msg.speed = float(self.command_speed)
        drive_msg.acceleration = 0.0
        self.ackermann_pub.publish(drive_msg)

    def _check_obstacle_timeout(self):
        """
        Clears obstacle_stop_requested if no detection messages have arrived recently.
        This allows the robot to resume automatically when an obstacle leaves the frame.
        """
        if self.last_detection_time is None:
            return
        elapsed = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
        if elapsed > self.obstacle_timeout and self.obstacle_stop_requested:
            self.get_logger().info(
                f"No detections for {elapsed:.1f}s — clearing obstacle, resuming operation."
            )
            self.obstacle_stop_requested = False
            self.publish_obstacle_information()

    def robot_description_callback(self, msg):
        """
        Callback for /robot_description topic.
        """
        self.get_logger().info(f"Received /robot_description (length={len(msg.data)})")

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

        ref_x = [pose.pose.position.x for pose in msg.poses]
        ref_y = [pose.pose.position.y for pose in msg.poses]

        self.ref_path_x = np.array(ref_x, dtype=float)
        self.ref_path_y = np.array(ref_y, dtype=float)
        self.controller.ref_path_x = self.ref_path_x
        self.controller.ref_path_y = self.ref_path_y
        self.path_received = True
        self.goal_reached = False

        self.get_logger().info(
            f"Updated controller path from /path with {pose_count} poses",
             throttle_duration_sec=2.0,
        )
    
    def traffic_decision_callback(self, msg: String):
        """
        Parses traffic_decision string and stop/resumes based on cmd field.
        Expected format: cmd=STOP signal=RED sg=1 int=2 lane=3 dist=1.2m ...
        """
        new_stop_requested = not bool(msg.data)  # False = stop, True = move
        if new_stop_requested != self.stop_requested:
            self.get_logger().info(
                f"/tarffic_decision_callback={msg.data}. stop_requested={new_stop_requested}"
            )
        self.stop_requested = new_stop_requested
        # data = msg.data
        # params = {}
        # for part in data.split():
        #     if "=" in part:
        #         k, v = part.split("=", 1)
        #         params[k] = v.strip('"')
            
        #     cmd = params.get("cmd", "UNKNOWN")
        #     signal = params.get("signal", "-")
        #     dist = params.get("lane", "-")
        #     lane = params.get("lane", "-")

        #     new_stop = cmd == "STOP"

        #     if new_stop != self.traffic_stop_requested:
        #         self.get_logger().info(
        #             f"Traffic signal changed: cmd={cmd}, signal={signal}, "
        #             f"lane={lane}, dist={dist}m -> stop_requested={new_stop}"
        #         )
        #     self.traffic_stop_requested = new_stop
        #     self.traffic_signal_state = signal

    def kinematic_state_callback(self, msg):
        """
        Callback for /kinematic_state topic. Updates robot pose from message.
        """
        pose = msg.pose_with_covariance.pose
        self.hagen_robot.x = pose.position.x
        self.hagen_robot.y = pose.position.y
        q = pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.hagen_robot.theta = math.atan2(siny_cosp, cosy_cosp)
        self.hagen_robot.v = msg.twist_with_covariance.twist.linear.x
        self.kinematic_state_received = True
        self.get_logger().debug(
            f"Updated robot pose from /kinematic_state: x={self.hagen_robot.x}, y={self.hagen_robot.y}, "
            f"theta={self.hagen_robot.theta}, v={self.hagen_robot.v}"
        )

    def ackermann_feedback_callback(self, msg):
        """
        Callback for /ackermann_drive_feedback topic.
        """
        self.feedback_speed = float(msg.speed)
        self.feedback_steering = float(msg.steering_angle)
        self.get_logger().debug(
            f"Ackermann feedback: speed={self.feedback_speed:.3f}, steering={self.feedback_steering:.3f}"
        )

    def allowed_to_move_callback(self, msg):
        """
        Callback for /allowed_to_move topic.
        Stop vehicle when value is False, allow movement when True.
        """
        new_stop_requested = not bool(msg.data)  # False = stop, True = move
        if new_stop_requested != self.stop_requested:
            self.get_logger().info(
                f"/allowed_to_move={msg.data}. stop_requested={new_stop_requested}"
            )
        self.stop_requested = new_stop_requested

    def objects_in_map_frame_callback(self, msg):
        """
        Callback for /objects_in_map_frame topic.
        Stamps the last detection time on every message so the timeout timer
        can detect when the obstacle has left the frame.
        """
        self.last_detection_time = self.get_clock().now()
        self.object_detections_count = len(msg.detections)

        if not self.kinematic_state_received:
            self.obstacle_stop_requested = False
            return

        obstacle_close = False
        for detection in msg.detections:
            if detection.results:
                class_id = detection.results[0].hypothesis.class_id
                score = detection.results[0].hypothesis.score
                cx = float(detection.bbox.center.position.x)
                cy = float(detection.bbox.center.position.y)
                self.get_logger().info(
                    f"center=({cx:.2f}, {cy:.2f}), "
                    f"threshold={self.obstacle_distance_threshold:.2f}m"
                )
            if self._is_obstacle_too_close(detection):
                obstacle_close = True
                break

        if obstacle_close != self.obstacle_stop_requested:
            self.get_logger().info(
                f"Obstacle stop changed to {obstacle_close} from /objects_in_map_frame "
                f"(detections={self.object_detections_count})"
            )
        self.obstacle_stop_requested = obstacle_close
        self.publish_obstacle_information()

    def _is_obstacle_too_close(self, detection):
        """
        Returns True when obstacle center.x is within obstacle_distance_threshold.
        Only considers objects of class: car, person, plant.
        """
        allowed_classes = {"car", "person", "plant"}

        if not detection.results:
            return False

        class_name = detection.results[0].hypothesis.class_id.lower()
        if class_name not in allowed_classes:
            self.get_logger().debug(f"Ignoring object class: {class_name}")
            return False

        bbox = detection.bbox
        center = bbox.center.position
        cx = float(center.x)

        is_too_close = cx < self.obstacle_distance_threshold

        self.get_logger().debug(
            f"[{class_name}] center.x={cx:.2f}m, threshold={self.obstacle_distance_threshold}m, too_close={is_too_close}"
        )
        if is_too_close:
            self.get_logger().warn(
                f"Obstacle too close! Class={class_name}, center.x={cx:.2f}m",
                throttle_duration_sec=1.0,
            )

        return is_too_close
    
    def publish_obstacle_information(self):
        info_msg = Bool()
        info_msg.data = bool(self.obstacle_stop_requested)
        self.obstacle_information_pub.publish(info_msg)

    @staticmethod
    def _yaw_from_quaternion(x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_stop_command(self):
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = 0.0
        drive_msg.speed = 0.0
        drive_msg.acceleration = -5.0 #0.0
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