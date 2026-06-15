#!/usr/bin/env python3
"""
Unit tests for PurePurSuitController and PurePursuitNode methods.

Run with:
    cd /home/af/ros2_ws
    python3 -m pytest src/control_car/test/test_pure_pursuit.py -s --no-header -q
"""

import math
import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock

# ── Mock all ROS 2 / message packages before importing the node ──────────────
for _mod in [
    'rclpy', 'rclpy.node', 'rclpy.qos',
    'ackermann_msgs', 'ackermann_msgs.msg',
    'curobot_msgs',   'curobot_msgs.msg',
    'nav_msgs',       'nav_msgs.msg',
    'std_msgs',       'std_msgs.msg',
    'vision_msgs',    'vision_msgs.msg',
    'geometry_msgs',  'geometry_msgs.msg',
    'visualization_msgs', 'visualization_msgs.msg',
]:
    sys.modules[_mod] = MagicMock()

class _RosNode:
    def __init__(self, *args, **kwargs): pass

sys.modules['rclpy.node'].Node = _RosNode

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'control_car')
)
from pure_persuit_node import (
    HagenRobot, PurePurSuitController, PurePursuitNode, STEER_LIMIT
)


# ── Minimal stand-in: borrow instance methods without a real ROS 2 node ──────
class _FakeNode:
    command_speed        = 1.4
    min_curve_speed      = 0.9
    person_stop_dist     = 0.4
    person_slowdown_dist = 1.5
    avoidance_predict_t  = 1.5
    avoidance_start_dist = 1.5
    avoidance_clearance  = 0.3
    closest_person_dist  = float('inf')
    closest_vlong        = 0.0
    hagen_robot          = HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.4, L=0.3)

    _compute_safe_speed            = PurePursuitNode._compute_safe_speed
    _compute_curve_speed           = PurePursuitNode._compute_curve_speed
    _compute_ahead_curvature_speed = PurePursuitNode._compute_ahead_curvature_speed
    _map_to_car_frame              = PurePursuitNode._map_to_car_frame
    _clear_obstacle                = PurePursuitNode._clear_obstacle

    obstacle_stop_requested = False
    avoidance_target        = 0.0

    def publish_obstacle_information(self): pass
    def get_logger(self): return MagicMock()


def _make_ctrl(xs, ys, last_index=0):
    return type('C', (), {
        'ref_path_x': np.asarray(xs, dtype=float),
        'ref_path_y': np.asarray(ys, dtype=float),
        'last_index': last_index,
    })()


# ═════════════════════════════════════════════════════════════════════════════
# 1. HagenRobot
# ═════════════════════════════════════════════════════════════════════════════
class TestHagenRobot(unittest.TestCase):

    def test_defaults(self):
        r = HagenRobot()
        self.assertEqual(r.x, 0)
        self.assertEqual(r.y, 0)
        self.assertEqual(r.theta, 0)
        self.assertEqual(r.v, 0)

    def test_custom_values(self):
        r = HagenRobot(x=1.0, y=2.0, theta=math.pi / 4, v=1.4, L=0.3)
        self.assertAlmostEqual(r.x,     1.0)
        self.assertAlmostEqual(r.y,     2.0)
        self.assertAlmostEqual(r.theta, math.pi / 4)
        self.assertAlmostEqual(r.v,     1.4)
        self.assertAlmostEqual(r.L,     0.3)


# ═════════════════════════════════════════════════════════════════════════════
# 2. PurePurSuitController — steering and lookahead
# ═════════════════════════════════════════════════════════════════════════════
class TestPurePursuitController(unittest.TestCase):

    @staticmethod
    def _straight(length=3.0, spacing=0.05):
        xs = np.arange(0.0, length, spacing)
        return xs, np.zeros_like(xs)

    def _ctrl(self, robot, xs, ys):
        return PurePurSuitController(robot, xs, ys, L_d=0.30, k=0.0)

    def test_straight_path_zero_steering(self):
        """Car on straight path produces near-zero steering."""
        xs, ys = self._straight()
        delta, _ = self._ctrl(HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        self.assertAlmostEqual(delta, 0.0, places=3)

    def test_car_left_of_path_steers_right(self):
        """Car displaced left of path → negative (right) steering."""
        xs, ys = self._straight()
        delta, _ = self._ctrl(HagenRobot(x=0.0, y=0.05, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        self.assertLess(delta, 0.0)

    def test_car_right_of_path_steers_left(self):
        """Car displaced right of path → positive (left) steering."""
        xs, ys = self._straight()
        delta, _ = self._ctrl(HagenRobot(x=0.0, y=-0.05, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        self.assertGreater(delta, 0.0)

    def test_steering_saturates_at_steer_limit(self):
        """Large lateral error saturates delta at ±STEER_LIMIT (30°)."""
        xs, ys = self._straight()
        delta, _ = self._ctrl(HagenRobot(x=0.0, y=5.0, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        self.assertAlmostEqual(abs(delta), STEER_LIMIT, places=4)

    def test_larger_lateral_error_larger_delta(self):
        """Greater lateral offset produces larger steering angle."""
        xs, ys = self._straight()
        d1, _ = self._ctrl(HagenRobot(x=0.0, y=0.02, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        d2, _ = self._ctrl(HagenRobot(x=0.0, y=0.10, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        self.assertGreater(abs(d2), abs(d1))

    def test_lookahead_index_advances_from_start(self):
        """Lookahead index must be ahead of the car start position."""
        xs, ys = self._straight()
        _, idx = self._ctrl(HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        self.assertGreaterEqual(idx, 5)

    def test_lookahead_stays_within_path_bounds(self):
        """Lookahead index never exceeds last path index."""
        xs, ys = self._straight(length=0.5)
        _, idx = self._ctrl(HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3), xs, ys).pure_pursuit_control()
        self.assertLess(idx, len(xs))

    def test_lookahead_never_goes_backward(self):
        """last_index must not decrease between consecutive calls."""
        xs, ys = self._straight()
        robot = HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3)
        ctrl  = self._ctrl(robot, xs, ys)
        ctrl.pure_pursuit_control()
        saved = ctrl.last_index
        robot.x = 0.3
        ctrl.pure_pursuit_control()
        self.assertGreaterEqual(ctrl.last_index, saved)


# ═════════════════════════════════════════════════════════════════════════════
# 3. PurePursuitNode — static path utilities
# ═════════════════════════════════════════════════════════════════════════════
class TestPathProcessing(unittest.TestCase):

    def test_smooth_preserves_point_count(self):
        """_smooth_path must return same number of points."""
        xs, ys = np.linspace(0, 5, 100), np.sin(np.linspace(0, 5, 100))
        xs_s, ys_s = PurePursuitNode._smooth_path(xs, ys)
        self.assertEqual(len(xs_s), 100)

    def test_smooth_leaves_straight_path_unchanged(self):
        """Smoothing a perfectly straight path leaves y = 0."""
        _, ys_s = PurePursuitNode._smooth_path(np.linspace(0, 1, 50), np.zeros(50))
        np.testing.assert_allclose(ys_s, 0.0, atol=1e-10)

    def test_smooth_reduces_noise(self):
        """Smoothed path must have smaller std-dev than noisy input."""
        rng = np.random.default_rng(42)
        xs  = np.linspace(0, 5, 100)
        ys  = rng.normal(0, 1, 100)
        _, ys_s = PurePursuitNode._smooth_path(xs, ys)
        self.assertLess(np.std(ys_s), np.std(ys))

    def test_densify_point_spacing_at_most_target(self):
        """All inter-point distances must be ≤ target spacing."""
        xd, yd = PurePursuitNode._densify_path(np.array([0.0, 1.0, 2.0]), np.zeros(3), spacing=0.05)
        self.assertTrue(np.all(np.hypot(np.diff(xd), np.diff(yd)) <= 0.051))

    def test_densify_increases_point_count(self):
        """Densifying 2 points 1 m apart at 5 cm spacing yields > 2 points."""
        xd, _ = PurePursuitNode._densify_path(np.array([0.0, 1.0]), np.zeros(2), spacing=0.05)
        self.assertGreater(len(xd), 2)

    def test_densify_preserves_endpoint(self):
        """Last point of densified path matches original endpoint."""
        xd, yd = PurePursuitNode._densify_path(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(xd[-1], 1.0, places=5)
        self.assertAlmostEqual(yd[-1], 1.0, places=5)

    def test_densify_short_segment_not_split(self):
        """Segment shorter than target spacing is kept as-is."""
        xd, _ = PurePursuitNode._densify_path(np.array([0.0, 0.02]), np.zeros(2), spacing=0.05)
        self.assertEqual(len(xd), 2)


# ═════════════════════════════════════════════════════════════════════════════
# 4. PurePursuitNode._is_person
# ═════════════════════════════════════════════════════════════════════════════
class TestIsPersonClassifier(unittest.TestCase):

    def test_lowercase_person(self):    self.assertTrue(PurePursuitNode._is_person('person'))
    def test_uppercase_person(self):    self.assertTrue(PurePursuitNode._is_person('PERSON'))
    def test_class_id_1(self):          self.assertTrue(PurePursuitNode._is_person('1'))
    def test_whitespace_trimmed(self):  self.assertTrue(PurePursuitNode._is_person('  person  '))
    def test_car_is_not_person(self):   self.assertFalse(PurePursuitNode._is_person('car'))
    def test_traffic_light(self):       self.assertFalse(PurePursuitNode._is_person('traffic light'))
    def test_empty_string(self):        self.assertFalse(PurePursuitNode._is_person(''))
    def test_class_id_3(self):          self.assertFalse(PurePursuitNode._is_person('3'))


# ═════════════════════════════════════════════════════════════════════════════
# 5. PurePursuitNode._map_to_car_frame
# ═════════════════════════════════════════════════════════════════════════════
class TestMapToCarFrame(unittest.TestCase):

    def _n(self, x=0.0, y=0.0, theta=0.0):
        n = _FakeNode()
        n.hagen_robot = HagenRobot(x=x, y=y, theta=theta)
        return n

    def test_point_directly_ahead(self):
        """Point on +x axis appears as (cx>0, cy=0) when car faces east."""
        cx, cy = self._n()._map_to_car_frame(1.0, 0.0)
        self.assertAlmostEqual(cx, 1.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_point_to_the_left(self):
        """Point on +y axis appears as cy>0 when car faces east."""
        cx, cy = self._n()._map_to_car_frame(0.0, 1.0)
        self.assertAlmostEqual(cx, 0.0, places=5)
        self.assertAlmostEqual(cy, 1.0, places=5)

    def test_point_to_the_right(self):
        """Point on -y axis appears as cy<0 when car faces east."""
        _, cy = self._n()._map_to_car_frame(0.0, -1.0)
        self.assertAlmostEqual(cy, -1.0, places=5)

    def test_point_behind_car(self):
        """Point behind car appears as cx<0."""
        cx, _ = self._n()._map_to_car_frame(-1.0, 0.0)
        self.assertAlmostEqual(cx, -1.0, places=5)

    def test_car_facing_north_point_ahead(self):
        """Car facing north (+y), point on +y axis → cx=1, cy=0."""
        cx, cy = self._n(theta=math.pi / 2)._map_to_car_frame(0.0, 1.0)
        self.assertAlmostEqual(cx, 1.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_car_facing_north_point_to_right(self):
        """Car facing north, point on +x (east) → cy<0 (right)."""
        _, cy = self._n(theta=math.pi / 2)._map_to_car_frame(1.0, 0.0)
        self.assertAlmostEqual(cy, -1.0, places=5)

    def test_car_at_map_offset(self):
        """Transformation accounts for car position offset."""
        cx, cy = self._n(x=2.0, y=3.0)._map_to_car_frame(3.0, 3.0)
        self.assertAlmostEqual(cx, 1.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_left_right_symmetry(self):
        """Points equidistant left/right have equal |cy|."""
        _, cy_l = self._n()._map_to_car_frame(0.0,  0.5)
        _, cy_r = self._n()._map_to_car_frame(0.0, -0.5)
        self.assertAlmostEqual(cy_l, -cy_r, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 6. PurePursuitNode._compute_safe_speed
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeSafeSpeed(unittest.TestCase):

    def _n(self, dist=float('inf'), vlong=0.0):
        n = _FakeNode()
        n.closest_person_dist = dist
        n.closest_vlong       = vlong
        return n

    def test_no_person_returns_full_speed(self):
        """No person detected → command_speed returned."""
        self.assertAlmostEqual(self._n()._compute_safe_speed(), 1.4)

    def test_at_stop_distance_returns_zero(self):
        """Person exactly at stop_dist → speed = 0."""
        self.assertAlmostEqual(self._n(dist=0.4)._compute_safe_speed(), 0.0)

    def test_inside_stop_distance_returns_zero(self):
        """Person closer than stop_dist → speed = 0."""
        self.assertAlmostEqual(self._n(dist=0.1)._compute_safe_speed(), 0.0)

    def test_midpoint_of_ramp_zone(self):
        """Person at midpoint of slow zone → proportional speed."""
        dist = 0.95
        expected = 1.4 * (dist - 0.4) / (1.5 - 0.4)
        self.assertAlmostEqual(self._n(dist=dist)._compute_safe_speed(), expected, places=4)

    def test_beyond_slowdown_dist_full_speed(self):
        """Person beyond slowdown distance → full speed."""
        self.assertAlmostEqual(self._n(dist=2.0)._compute_safe_speed(), 1.4)

    def test_approaching_person_extends_slowdown_zone(self):
        """Person walking toward car slows it down at greater distance."""
        s_still  = self._n(dist=1.8, vlong= 0.0)._compute_safe_speed()
        s_toward = self._n(dist=1.8, vlong=-0.5)._compute_safe_speed()
        self.assertLess(s_toward, s_still)

    def test_receding_person_no_zone_extension(self):
        """Person walking away from car does not extend slowdown zone."""
        s_still = self._n(dist=1.8, vlong= 0.0)._compute_safe_speed()
        s_away  = self._n(dist=1.8, vlong= 0.5)._compute_safe_speed()
        self.assertAlmostEqual(s_still, s_away, places=4)

    def test_speed_never_negative(self):
        """Speed output must never be negative at any distance."""
        for d in [0.0, 0.1, 0.4, 0.9, 1.5, 3.0]:
            self.assertGreaterEqual(self._n(dist=d)._compute_safe_speed(), 0.0)

    def test_speed_never_exceeds_command_speed(self):
        """Speed output must never exceed command_speed."""
        for d in [0.5, 1.0, 2.0, float('inf')]:
            self.assertLessEqual(self._n(dist=d)._compute_safe_speed(), 1.4)

    def test_ramp_monotonically_increasing_with_distance(self):
        """Speed must increase monotonically as person distance increases."""
        dists  = np.linspace(0.4, 1.5, 20)
        speeds = [self._n(dist=d)._compute_safe_speed() for d in dists]
        for i in range(len(speeds) - 1):
            self.assertLessEqual(speeds[i], speeds[i + 1])


# ═════════════════════════════════════════════════════════════════════════════
# 7. PurePursuitNode._compute_curve_speed
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeCurveSpeed(unittest.TestCase):

    def _n(self): return _FakeNode()

    def test_zero_steering_full_speed(self):
        """No steering → command_speed returned."""
        self.assertAlmostEqual(self._n()._compute_curve_speed(0.0), 1.4)

    def test_max_steering_min_speed(self):
        """Full steering angle → min_curve_speed returned."""
        self.assertAlmostEqual(self._n()._compute_curve_speed(STEER_LIMIT), 0.9)

    def test_left_right_steering_symmetric(self):
        """Equal magnitude left/right steering produces identical speed."""
        n = self._n()
        self.assertAlmostEqual(
            n._compute_curve_speed( STEER_LIMIT / 2),
            n._compute_curve_speed(-STEER_LIMIT / 2),
        )

    def test_intermediate_steering_in_range(self):
        """Partial steering angle → speed between min and max."""
        s = self._n()._compute_curve_speed(STEER_LIMIT / 2)
        self.assertGreater(s, 0.9)
        self.assertLess(s,    1.4)

    def test_sqrt_mapping_drops_faster_than_linear(self):
        """Sqrt speed mapping causes faster speed reduction than linear."""
        quarter = STEER_LIMIT * 0.25
        actual  = self._n()._compute_curve_speed(quarter)
        linear  = 1.4 - 0.25 * (1.4 - 0.9)
        self.assertLess(actual, linear)

    def test_speed_monotonically_decreasing_with_steering(self):
        """Speed must decrease monotonically as steering angle increases."""
        n      = self._n()
        speeds = [n._compute_curve_speed(d) for d in np.linspace(0, STEER_LIMIT, 20)]
        for i in range(len(speeds) - 1):
            self.assertGreaterEqual(speeds[i], speeds[i + 1])


# ═════════════════════════════════════════════════════════════════════════════
# 8. PurePursuitNode._compute_ahead_curvature_speed
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeAheadCurvatureSpeed(unittest.TestCase):

    def _n(self, xs, ys, last_index=0):
        n = _FakeNode()
        n.hagen_robot = HagenRobot(v=1.4, L=0.3)
        n.controller  = _make_ctrl(xs, ys, last_index)
        return n

    def test_straight_path_returns_full_speed(self):
        """No curvature ahead → full command_speed."""
        xs = np.arange(0.0, 3.0, 0.05)
        self.assertAlmostEqual(
            self._n(xs, np.zeros_like(xs))._compute_ahead_curvature_speed(), 1.4, places=3
        )

    def test_tight_circle_returns_min_speed(self):
        """Radius smaller than min turning radius → min_curve_speed."""
        t = np.linspace(0, math.pi, 60)
        speed = self._n(0.15 * np.cos(t), 0.15 * np.sin(t))._compute_ahead_curvature_speed()
        self.assertAlmostEqual(speed, 0.9, places=2)

    def test_gentle_curve_speed_between_min_and_max(self):
        """Moderate curve → speed between min and max."""
        t = np.linspace(0, math.pi / 2, 60)
        speed = self._n(np.cos(t), np.sin(t))._compute_ahead_curvature_speed()
        self.assertGreater(speed, 0.9)
        self.assertLessEqual(speed, 1.4)

    def test_short_path_returns_full_speed(self):
        """Path with < 3 points → full speed (no curvature to compute)."""
        speed = self._n(np.array([0.0, 1.0]), np.array([0.0, 0.0]))._compute_ahead_curvature_speed()
        self.assertAlmostEqual(speed, 1.4)

    def test_scans_from_last_index_not_origin(self):
        """Curvature scan starts at last_index, not path index 0."""
        xs = np.arange(0.0, 3.0, 0.05)
        # car near end of straight path → no curvature ahead → full speed
        speed = self._n(xs, np.zeros_like(xs), last_index=len(xs) - 3)._compute_ahead_curvature_speed()
        self.assertAlmostEqual(speed, 1.4, places=3)


# ═════════════════════════════════════════════════════════════════════════════
# 9. PurePursuitNode._clear_obstacle
# ═════════════════════════════════════════════════════════════════════════════
class TestClearObstacle(unittest.TestCase):

    def _n(self):
        n = _FakeNode()
        n.obstacle_stop_requested = True
        n.closest_person_dist     = 0.8
        n.closest_vlong           = -0.3
        n.avoidance_target        = 0.3
        return n

    def test_clears_obstacle_stop_flag(self):
        """_clear_obstacle resets obstacle_stop_requested to False."""
        n = self._n(); n._clear_obstacle()
        self.assertFalse(n.obstacle_stop_requested)

    def test_clears_person_distance(self):
        """_clear_obstacle resets closest_person_dist to inf."""
        n = self._n(); n._clear_obstacle()
        self.assertEqual(n.closest_person_dist, float('inf'))

    def test_clears_longitudinal_velocity(self):
        """_clear_obstacle resets KF longitudinal velocity to 0."""
        n = self._n(); n._clear_obstacle()
        self.assertAlmostEqual(n.closest_vlong, 0.0)

    def test_clears_avoidance_target(self):
        """_clear_obstacle resets lateral avoidance target to 0."""
        n = self._n(); n._clear_obstacle()
        self.assertAlmostEqual(n.avoidance_target, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
