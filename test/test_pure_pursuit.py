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
    _cte                    = 0.0

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
    def test_car_is_not_person(self):   self.assertFalse(PurePursuitNode._is_person('car'))
    def test_empty_string(self):        self.assertFalse(PurePursuitNode._is_person(''))


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

    def test_point_behind_car(self):
        """Point behind car appears as cx<0."""
        cx, _ = self._n()._map_to_car_frame(-1.0, 0.0)
        self.assertAlmostEqual(cx, -1.0, places=5)

    def test_car_facing_north_point_ahead(self):
        """Car facing north (+y), point on +y axis → cx=1, cy=0."""
        cx, cy = self._n(theta=math.pi / 2)._map_to_car_frame(0.0, 1.0)
        self.assertAlmostEqual(cx, 1.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_car_at_map_offset(self):
        """Transformation accounts for car position offset."""
        cx, cy = self._n(x=2.0, y=3.0)._map_to_car_frame(3.0, 3.0)
        self.assertAlmostEqual(cx, 1.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)


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
        n = self._n(); n._clear_obstacle()
        self.assertFalse(n.obstacle_stop_requested)

    def test_clears_person_distance(self):
        n = self._n(); n._clear_obstacle()
        self.assertEqual(n.closest_person_dist, float('inf'))

    def test_clears_longitudinal_velocity(self):
        n = self._n(); n._clear_obstacle()
        self.assertAlmostEqual(n.closest_vlong, 0.0)

    def test_clears_avoidance_target(self):
        n = self._n(); n._clear_obstacle()
        self.assertAlmostEqual(n.avoidance_target, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Goal speed ramp — stopping at target location
# ═════════════════════════════════════════════════════════════════════════════
class TestGoalSpeedRamp(unittest.TestCase):

    def _n(self):
        n = _FakeNode()
        n.goal_slowdown_dist = 0.5
        n._compute_goal_speed = PurePursuitNode._compute_goal_speed.__get__(n)
        return n

    def test_beyond_slowdown_dist_full_speed(self):
        """Beyond 0.5 m from goal → full command_speed."""
        self.assertAlmostEqual(self._n()._compute_goal_speed(1.0), 1.4)

    def test_halfway_to_goal_half_speed(self):
        """At 0.25 m (halfway into ramp) → 50% of command_speed."""
        self.assertAlmostEqual(self._n()._compute_goal_speed(0.25), 0.7, places=4)

    def test_at_zero_distance_zero_speed(self):
        """At exact goal position → speed = 0."""
        self.assertAlmostEqual(self._n()._compute_goal_speed(0.0), 0.0)

    def test_ramp_monotonically_decreasing(self):
        """Speed must decrease as car gets closer to goal."""
        n = self._n()
        dists  = np.linspace(0.5, 0.0, 20)
        speeds = [n._compute_goal_speed(d) for d in dists]
        for i in range(len(speeds) - 1):
            self.assertGreaterEqual(speeds[i], speeds[i + 1])


# ═════════════════════════════════════════════════════════════════════════════
# 11. Curvature-adaptive lookahead distance
# ═════════════════════════════════════════════════════════════════════════════
class TestAdaptiveLookahead(unittest.TestCase):

    def _n(self, base_L_d=0.30, k=0.3):
        n = _FakeNode()
        n._base_L_d = base_L_d
        n._k        = k
        n._compute_adaptive_L_d = PurePursuitNode._compute_adaptive_L_d.__get__(n)
        return n

    def test_straight_expands_lookahead(self):
        """Zero steering on straight → lookahead = base + k*v."""
        L = self._n()._compute_adaptive_L_d(0.0, 1.4)
        self.assertAlmostEqual(L, 0.30 + 0.3 * 1.4, places=4)

    def test_tight_curve_reduces_lookahead(self):
        """Max steering angle → lookahead reduced toward floor."""
        L = self._n()._compute_adaptive_L_d(STEER_LIMIT, 1.4)
        self.assertLess(L, 0.30 + 0.3 * 1.4)

    def test_tight_curve_not_below_floor(self):
        """Lookahead must never go below 50% of base L_d."""
        L = self._n()._compute_adaptive_L_d(STEER_LIMIT, 1.4)
        self.assertGreaterEqual(L, 0.30 * 0.5)

    def test_zero_speed_base_lookahead(self):
        """At v=0 on straight, lookahead equals base L_d."""
        L = self._n()._compute_adaptive_L_d(0.0, 0.0)
        self.assertAlmostEqual(L, 0.30, places=4)

    def test_higher_speed_larger_lookahead(self):
        """Higher speed on straight produces larger lookahead."""
        L_slow = self._n()._compute_adaptive_L_d(0.0, 0.5)
        L_fast = self._n()._compute_adaptive_L_d(0.0, 1.4)
        self.assertGreater(L_fast, L_slow)

    def test_lookahead_decreases_with_curve_factor(self):
        """Lookahead must decrease monotonically as steering increases."""
        n = self._n()
        deltas = np.linspace(0.0, STEER_LIMIT, 20)
        L_vals = [n._compute_adaptive_L_d(d, 1.4) for d in deltas]
        for i in range(len(L_vals) - 1):
            self.assertGreaterEqual(L_vals[i], L_vals[i + 1])


# ═════════════════════════════════════════════════════════════════════════════
# 12. Corridor filter — persons behind or outside corridor are ignored
# ═════════════════════════════════════════════════════════════════════════════
class TestCorridorFilter(unittest.TestCase):

    def _car_frame(self, x_map, y_map, car_x=0.0, car_y=0.0, theta=0.0):
        n = _FakeNode()
        n.hagen_robot = HagenRobot(x=car_x, y=car_y, theta=theta)
        return PurePursuitNode._map_to_car_frame(n, x_map, y_map)

    def test_person_ahead_in_corridor(self):
        """Person 2 m ahead, centered → cx > 0.1, |cy| < corridor."""
        cx, cy = self._car_frame(2.0, 0.0)
        self.assertGreater(cx, 0.1)
        self.assertLess(abs(cy), 1.2)

    def test_person_behind_car_excluded(self):
        """Person behind car → cx < 0 → must be excluded (cx <= 0.1)."""
        cx, _ = self._car_frame(-1.0, 0.0)
        self.assertLessEqual(cx, 0.1)

    def test_person_far_left_outside_corridor(self):
        """Person 3 m to the left → |cy| > corridor_half (1.2 m)."""
        _, cy = self._car_frame(0.0, 3.0)
        self.assertGreater(abs(cy), 1.2)

    def test_person_beside_car_excluded(self):
        """Person directly to the side (cx=0) → behind threshold, excluded."""
        cx, _ = self._car_frame(0.0, 1.0)
        self.assertLessEqual(cx, 0.1)


# ═════════════════════════════════════════════════════════════════════════════
# 13. Avoidance action decision — direction, hard stop, early trigger
# ═════════════════════════════════════════════════════════════════════════════
class TestDecideAvoidanceAction(unittest.TestCase):

    def _n(self, cte=0.0):
        n = _FakeNode()
        n._cte = cte
        n._decide_avoidance_action = PurePursuitNode._decide_avoidance_action.__get__(n)
        return n

    def test_person_on_left_avoids_right(self):
        """Person clearly to the left → avoidance target is negative (right)."""
        _, target, _ = self._n()._decide_avoidance_action(cx=1.0, cy=0.5, vlat=0.0, vlong=0.0)
        self.assertLess(target, 0.0)

    def test_person_on_right_avoids_left(self):
        """Person clearly to the right → avoidance target is positive (left)."""
        _, target, _ = self._n()._decide_avoidance_action(cx=1.0, cy=-0.5, vlat=0.0, vlong=0.0)
        self.assertGreater(target, 0.0)

    def test_lateral_velocity_shifts_predicted_position(self):
        """Person near center moving left → predicted left → avoids right."""
        # cy=0.1 but vlat=+0.5 → lat_predicted = 0.1 + 0.5*1.5 = 0.85 (left)
        _, target, _ = self._n()._decide_avoidance_action(cx=1.0, cy=0.1, vlat=0.5, vlong=0.0)
        self.assertLess(target, 0.0)

    def test_person_at_stop_distance_triggers_hard_stop(self):
        """Person at exactly stop_dist (0.4 m) → hard stop, avoidance target = 0."""
        stop, target, _ = self._n()._decide_avoidance_action(cx=0.4, cy=0.0, vlat=0.0, vlong=0.0)
        self.assertTrue(stop)
        self.assertEqual(target, 0.0)

    def test_person_beyond_stop_distance_no_hard_stop(self):
        """Person at 0.6 m (beyond stop zone) → no hard stop."""
        stop, _, _ = self._n()._decide_avoidance_action(cx=0.6, cy=0.0, vlat=0.0, vlong=0.0)
        self.assertFalse(stop)

    def test_static_person_avoids_within_start_dist(self):
        """Static person at 1.0 m (inside 1.5 m zone) → avoidance active."""
        _, target, _ = self._n()._decide_avoidance_action(cx=1.0, cy=0.4, vlat=0.0, vlong=0.0)
        self.assertNotEqual(target, 0.0)

    def test_static_person_beyond_start_dist_no_avoidance(self):
        """Static person at 2.0 m (beyond 1.5 m zone) → no avoidance."""
        _, target, _ = self._n()._decide_avoidance_action(cx=2.0, cy=0.4, vlat=0.0, vlong=0.0)
        self.assertEqual(target, 0.0)

    def test_approaching_person_extends_trigger_distance(self):
        """Person at -0.5 m/s toward car → trigger fires at 1.5 + 0.5*1.5 = 2.25 m."""
        _, _, trigger = self._n()._decide_avoidance_action(cx=2.0, cy=0.4, vlat=0.0, vlong=-0.5)
        self.assertAlmostEqual(trigger, 1.5 + 0.5 * 1.5, places=4)

    def test_receding_person_no_trigger_extension(self):
        """Person moving away → no extension, trigger stays at base 1.5 m."""
        _, _, trigger = self._n()._decide_avoidance_action(cx=2.0, cy=0.4, vlat=0.0, vlong=0.5)
        self.assertAlmostEqual(trigger, 1.5, places=4)

    def test_centered_person_car_on_path_avoids_right(self):
        """Person centered, car on path (CTE≈0) → default right (target < 0)."""
        _, target, _ = self._n(cte=0.0)._decide_avoidance_action(cx=1.0, cy=0.0, vlat=0.0, vlong=0.0)
        self.assertLess(target, 0.0)

    def test_centered_person_car_right_of_path_avoids_left(self):
        """Person centered, car right of path (CTE=-0.1) → go left (target > 0)."""
        _, target, _ = self._n(cte=-0.10)._decide_avoidance_action(cx=1.0, cy=0.0, vlat=0.0, vlong=0.0)
        self.assertGreater(target, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
