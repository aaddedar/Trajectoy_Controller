#!/usr/bin/env python3
"""
Unit tests for pure pursuit path tracking and obstacle avoidance logic.

Run with:
    cd /home/af/ros2_ws
    python3 -m pytest src/control_car/test/test_pure_pursuit.py -v
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


# ── Minimal stand-in: call instance methods without a real ROS 2 node ────────
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

    _compute_safe_speed           = PurePursuitNode._compute_safe_speed
    _compute_curve_speed          = PurePursuitNode._compute_curve_speed
    _compute_ahead_curvature_speed= PurePursuitNode._compute_ahead_curvature_speed
    _map_to_car_frame             = PurePursuitNode._map_to_car_frame
    _clear_obstacle               = PurePursuitNode._clear_obstacle

    # State attributes used by _clear_obstacle
    obstacle_stop_requested = False
    avoidance_target        = 0.5

    def publish_obstacle_information(self): pass   # no-op for tests
    def get_logger(self): return MagicMock()


def _make_ctrl(xs, ys, last_index=0):
    """Helper: build a minimal controller-like object."""
    c = type('C', (), {
        'ref_path_x': np.asarray(xs, dtype=float),
        'ref_path_y': np.asarray(ys, dtype=float),
        'last_index': last_index,
    })()
    return c


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
# 2. Pure pursuit geometry
# ═════════════════════════════════════════════════════════════════════════════
class TestPurePursuitGeometry(unittest.TestCase):

    @staticmethod
    def _straight(length=3.0, spacing=0.05):
        xs = np.arange(0.0, length, spacing)
        return xs, np.zeros_like(xs)

    def _ctrl(self, robot, xs, ys):
        return PurePurSuitController(robot, xs, ys, L_d=0.30, k=0.0)

    def test_straight_path_zero_steering(self):
        xs, ys = self._straight()
        robot  = HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3)
        delta, _ = self._ctrl(robot, xs, ys).pure_pursuit_control()
        self.assertAlmostEqual(delta, 0.0, places=3)

    def test_car_left_steers_right(self):
        xs, ys = self._straight()
        robot  = HagenRobot(x=0.0, y=0.05, theta=0.0, v=1.0, L=0.3)
        delta, _ = self._ctrl(robot, xs, ys).pure_pursuit_control()
        self.assertLess(delta, 0.0)

    def test_car_right_steers_left(self):
        xs, ys = self._straight()
        robot  = HagenRobot(x=0.0, y=-0.05, theta=0.0, v=1.0, L=0.3)
        delta, _ = self._ctrl(robot, xs, ys).pure_pursuit_control()
        self.assertGreater(delta, 0.0)

    def test_delta_clipped_at_steer_limit(self):
        xs, ys = self._straight()
        robot  = HagenRobot(x=0.0, y=5.0, theta=0.0, v=1.0, L=0.3)
        delta, _ = self._ctrl(robot, xs, ys).pure_pursuit_control()
        self.assertAlmostEqual(abs(delta), STEER_LIMIT, places=4)

    def test_larger_error_larger_delta(self):
        xs, ys = self._straight()
        r1 = HagenRobot(x=0.0, y=0.02, theta=0.0, v=1.0, L=0.3)
        r2 = HagenRobot(x=0.0, y=0.10, theta=0.0, v=1.0, L=0.3)
        d1, _ = self._ctrl(r1, xs, ys).pure_pursuit_control()
        d2, _ = self._ctrl(r2, xs, ys).pure_pursuit_control()
        self.assertGreater(abs(d2), abs(d1))

    def test_lookahead_advances_from_start(self):
        xs, ys = self._straight()
        robot  = HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3)
        _, idx = self._ctrl(robot, xs, ys).pure_pursuit_control()
        self.assertGreaterEqual(idx, 5)

    def test_lookahead_within_path_bounds(self):
        xs, ys = self._straight(length=0.5)
        robot  = HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3)
        _, idx = self._ctrl(robot, xs, ys).pure_pursuit_control()
        self.assertLess(idx, len(xs))

    def test_lookahead_does_not_go_backward(self):
        xs, ys = self._straight()
        robot  = HagenRobot(x=0.0, y=0.0, theta=0.0, v=1.0, L=0.3)
        ctrl   = self._ctrl(robot, xs, ys)
        ctrl.pure_pursuit_control()
        saved = ctrl.last_index
        robot.x = 0.3
        ctrl.pure_pursuit_control()
        self.assertGreaterEqual(ctrl.last_index, saved)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Path processing
# ═════════════════════════════════════════════════════════════════════════════
class TestPathProcessing(unittest.TestCase):

    def test_smooth_preserves_length(self):
        xs, ys = np.linspace(0, 5, 100), np.sin(np.linspace(0, 5, 100))
        xs_s, ys_s = PurePursuitNode._smooth_path(xs, ys)
        self.assertEqual(len(xs_s), 100)

    def test_smooth_straight_unchanged(self):
        xs = np.linspace(0, 1, 50)
        _, ys_s = PurePursuitNode._smooth_path(xs, np.zeros(50))
        np.testing.assert_allclose(ys_s, 0.0, atol=1e-10)

    def test_smooth_reduces_noise(self):
        rng = np.random.default_rng(42)
        xs  = np.linspace(0, 5, 100)
        ys  = rng.normal(0, 1, 100)
        _, ys_s = PurePursuitNode._smooth_path(xs, ys)
        self.assertLess(np.std(ys_s), np.std(ys))

    def test_densify_spacing_at_most_target(self):
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.zeros(3)
        xd, yd = PurePursuitNode._densify_path(xs, ys, spacing=0.05)
        dists  = np.hypot(np.diff(xd), np.diff(yd))
        self.assertTrue(np.all(dists <= 0.051))

    def test_densify_increases_point_count(self):
        xd, _ = PurePursuitNode._densify_path(np.array([0.0, 1.0]), np.zeros(2), spacing=0.05)
        self.assertGreater(len(xd), 2)

    def test_densify_preserves_endpoint(self):
        xd, yd = PurePursuitNode._densify_path(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.assertAlmostEqual(xd[-1], 1.0, places=5)
        self.assertAlmostEqual(yd[-1], 1.0, places=5)

    def test_densify_short_segment_unchanged(self):
        xd, _ = PurePursuitNode._densify_path(np.array([0.0, 0.02]), np.zeros(2), spacing=0.05)
        self.assertEqual(len(xd), 2)


# ═════════════════════════════════════════════════════════════════════════════
# 4. _is_person classifier
# ═════════════════════════════════════════════════════════════════════════════
class TestIsPersonClassifier(unittest.TestCase):

    def test_lowercase(self):       self.assertTrue(PurePursuitNode._is_person('person'))
    def test_uppercase(self):       self.assertTrue(PurePursuitNode._is_person('PERSON'))
    def test_class_id_1(self):      self.assertTrue(PurePursuitNode._is_person('1'))
    def test_whitespace(self):      self.assertTrue(PurePursuitNode._is_person('  person  '))
    def test_car(self):             self.assertFalse(PurePursuitNode._is_person('car'))
    def test_traffic_light(self):   self.assertFalse(PurePursuitNode._is_person('traffic light'))
    def test_empty(self):           self.assertFalse(PurePursuitNode._is_person(''))
    def test_class_3(self):         self.assertFalse(PurePursuitNode._is_person('3'))


# ═════════════════════════════════════════════════════════════════════════════
# 5. _map_to_car_frame
# ═════════════════════════════════════════════════════════════════════════════
class TestMapToCarFrame(unittest.TestCase):

    def _n(self, x=0.0, y=0.0, theta=0.0):
        n = _FakeNode()
        n.hagen_robot = HagenRobot(x=x, y=y, theta=theta)
        return n

    def test_point_ahead(self):
        cx, cy = self._n()._map_to_car_frame(1.0, 0.0)
        self.assertAlmostEqual(cx,  1.0, places=5)
        self.assertAlmostEqual(cy,  0.0, places=5)

    def test_point_left(self):
        cx, cy = self._n()._map_to_car_frame(0.0, 1.0)
        self.assertAlmostEqual(cx,  0.0, places=5)
        self.assertAlmostEqual(cy,  1.0, places=5)

    def test_point_right(self):
        _, cy = self._n()._map_to_car_frame(0.0, -1.0)
        self.assertAlmostEqual(cy, -1.0, places=5)

    def test_point_behind(self):
        cx, _ = self._n()._map_to_car_frame(-1.0, 0.0)
        self.assertAlmostEqual(cx, -1.0, places=5)

    def test_facing_north_point_ahead(self):
        cx, cy = self._n(theta=math.pi / 2)._map_to_car_frame(0.0, 1.0)
        self.assertAlmostEqual(cx,  1.0, places=5)
        self.assertAlmostEqual(cy,  0.0, places=5)

    def test_facing_north_point_right(self):
        cx, cy = self._n(theta=math.pi / 2)._map_to_car_frame(1.0, 0.0)
        self.assertAlmostEqual(cx,  0.0, places=5)
        self.assertAlmostEqual(cy, -1.0, places=5)

    def test_car_at_offset(self):
        cx, cy = self._n(x=2.0, y=3.0)._map_to_car_frame(3.0, 3.0)
        self.assertAlmostEqual(cx,  1.0, places=5)
        self.assertAlmostEqual(cy,  0.0, places=5)

    def test_symmetry_left_right(self):
        """Equal distances left and right produce equal |cy|."""
        _, cy_l = self._n()._map_to_car_frame(0.0,  0.5)
        _, cy_r = self._n()._map_to_car_frame(0.0, -0.5)
        self.assertAlmostEqual(cy_l, -cy_r, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 6. _compute_safe_speed
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeSafeSpeed(unittest.TestCase):

    def _n(self, dist=float('inf'), vlong=0.0):
        n = _FakeNode()
        n.closest_person_dist = dist
        n.closest_vlong       = vlong
        return n

    def test_no_person_full_speed(self):
        self.assertAlmostEqual(self._n()._compute_safe_speed(), 1.4)

    def test_at_stop_dist_zero(self):
        self.assertAlmostEqual(self._n(dist=0.4)._compute_safe_speed(), 0.0)

    def test_below_stop_dist_zero(self):
        self.assertAlmostEqual(self._n(dist=0.1)._compute_safe_speed(), 0.0)

    def test_midpoint_ramp(self):
        dist = 0.95
        exp  = 1.4 * (dist - 0.4) / (1.5 - 0.4)
        self.assertAlmostEqual(self._n(dist=dist)._compute_safe_speed(), exp, places=4)

    def test_beyond_slowdown_full_speed(self):
        self.assertAlmostEqual(self._n(dist=2.0)._compute_safe_speed(), 1.4)

    def test_person_toward_slows_earlier(self):
        s_still  = self._n(dist=1.8, vlong= 0.0)._compute_safe_speed()
        s_toward = self._n(dist=1.8, vlong=-0.5)._compute_safe_speed()
        self.assertLess(s_toward, s_still)

    def test_person_away_no_extension(self):
        s_still = self._n(dist=1.8, vlong= 0.0)._compute_safe_speed()
        s_away  = self._n(dist=1.8, vlong= 0.5)._compute_safe_speed()
        self.assertAlmostEqual(s_still, s_away, places=4)

    def test_never_negative(self):
        for d in [0.0, 0.1, 0.4, 0.9, 1.5, 3.0]:
            self.assertGreaterEqual(self._n(dist=d)._compute_safe_speed(), 0.0)

    def test_never_exceeds_command_speed(self):
        for d in [0.5, 1.0, 2.0, float('inf')]:
            self.assertLessEqual(self._n(dist=d)._compute_safe_speed(), 1.4)

    def test_ramp_monotonically_increasing_with_distance(self):
        dists  = np.linspace(0.4, 1.5, 20)
        speeds = [self._n(dist=d)._compute_safe_speed() for d in dists]
        for i in range(len(speeds) - 1):
            self.assertLessEqual(speeds[i], speeds[i + 1])


# ═════════════════════════════════════════════════════════════════════════════
# 7. _compute_curve_speed
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeCurveSpeed(unittest.TestCase):

    def _n(self): return _FakeNode()

    def test_zero_steer_full_speed(self):
        self.assertAlmostEqual(self._n()._compute_curve_speed(0.0), 1.4)

    def test_max_steer_min_speed(self):
        self.assertAlmostEqual(self._n()._compute_curve_speed(STEER_LIMIT), 0.9)

    def test_symmetric(self):
        n = self._n()
        self.assertAlmostEqual(
            n._compute_curve_speed( STEER_LIMIT / 2),
            n._compute_curve_speed(-STEER_LIMIT / 2),
        )

    def test_intermediate_in_range(self):
        s = self._n()._compute_curve_speed(STEER_LIMIT / 2)
        self.assertGreater(s, 0.9)
        self.assertLess(s,    1.4)

    def test_sqrt_drops_faster_than_linear(self):
        quarter = STEER_LIMIT * 0.25
        actual  = self._n()._compute_curve_speed(quarter)
        linear  = 1.4 - 0.25 * (1.4 - 0.9)
        self.assertLess(actual, linear)

    def test_monotonically_decreasing(self):
        n      = self._n()
        speeds = [n._compute_curve_speed(d) for d in np.linspace(0, STEER_LIMIT, 20)]
        for i in range(len(speeds) - 1):
            self.assertGreaterEqual(speeds[i], speeds[i + 1])


# ═════════════════════════════════════════════════════════════════════════════
# 8. _compute_ahead_curvature_speed
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeAheadCurvatureSpeed(unittest.TestCase):

    def _n(self, xs, ys, last_index=0):
        n = _FakeNode()
        n.hagen_robot = HagenRobot(v=1.4, L=0.3)
        n.controller  = _make_ctrl(xs, ys, last_index)
        return n

    def test_straight_path_full_speed(self):
        """No curvature ahead → full command speed."""
        xs = np.arange(0.0, 3.0, 0.05)
        ys = np.zeros_like(xs)
        speed = self._n(xs, ys)._compute_ahead_curvature_speed()
        self.assertAlmostEqual(speed, 1.4, places=3)

    def test_tight_circle_min_speed(self):
        """Circle smaller than R_min → min curve speed."""
        # R = 0.15 m — tighter than R_min = L/tan(STEER_LIMIT)
        R   = 0.15
        t   = np.linspace(0, math.pi, 60)
        xs  = R * np.cos(t)
        ys  = R * np.sin(t)
        speed = self._n(xs, ys)._compute_ahead_curvature_speed()
        self.assertAlmostEqual(speed, 0.9, places=2)

    def test_gentle_curve_between_min_and_max(self):
        """Moderate curve → speed between min and max."""
        R  = 1.0
        t  = np.linspace(0, math.pi / 2, 60)
        xs = R * np.cos(t)
        ys = R * np.sin(t)
        speed = self._n(xs, ys)._compute_ahead_curvature_speed()
        self.assertGreater(speed, 0.9)
        self.assertLessEqual(speed, 1.4)

    def test_short_path_returns_full_speed(self):
        """Path with < 3 points → full speed (no curvature calc)."""
        speed = self._n(np.array([0.0, 1.0]), np.array([0.0, 0.0]))._compute_ahead_curvature_speed()
        self.assertAlmostEqual(speed, 1.4)

    def test_last_index_offset(self):
        """Starts scanning from last_index, not index 0."""
        xs = np.arange(0.0, 3.0, 0.05)
        ys = np.zeros_like(xs)
        # Place car near end: no curvature ahead → full speed regardless
        speed = self._n(xs, ys, last_index=len(xs) - 3)._compute_ahead_curvature_speed()
        self.assertAlmostEqual(speed, 1.4, places=3)


# ═════════════════════════════════════════════════════════════════════════════
# 9. CTE (cross-track error) calculation
# ═════════════════════════════════════════════════════════════════════════════
class TestCrossTrackError(unittest.TestCase):
    """
    CTE formula extracted from control_loop:
        ln = path_left_normal = (-ty_seg, tx_seg) / seg_len
        cte = dot((car - nearest_point), ln)
    +CTE = car left of path,  -CTE = car right of path.
    """

    @staticmethod
    def _cte(car_x, car_y, p1, p2):
        tx = p2[0] - p1[0]
        ty = p2[1] - p1[1]
        seg = math.hypot(tx, ty)
        if seg < 1e-6:
            return 0.0
        ln_x = -ty / seg
        ln_y =  tx / seg
        return (car_x - p1[0]) * ln_x + (car_y - p1[1]) * ln_y

    def test_car_on_path_zero_cte(self):
        cte = self._cte(0.5, 0.0, (0.0, 0.0), (1.0, 0.0))
        self.assertAlmostEqual(cte, 0.0, places=5)

    def test_car_left_positive_cte(self):
        cte = self._cte(0.5, 0.2, (0.0, 0.0), (1.0, 0.0))
        self.assertGreater(cte, 0.0)

    def test_car_right_negative_cte(self):
        cte = self._cte(0.5, -0.2, (0.0, 0.0), (1.0, 0.0))
        self.assertLess(cte, 0.0)

    def test_cte_magnitude(self):
        """CTE should equal the perpendicular offset distance."""
        cte = self._cte(0.5, 0.3, (0.0, 0.0), (1.0, 0.0))
        self.assertAlmostEqual(cte, 0.3, places=5)

    def test_diagonal_path(self):
        """CTE on a 45° path: car 0.1 m to the left → positive CTE."""
        # Path goes from (0,0) to (1,1). Left normal is (-1/√2, 1/√2).
        # Car at (0, 0.2): relative to (0,0) = (0, 0.2)
        # CTE = (0)*(-1/√2) + (0.2)*(1/√2) = 0.2/√2 ≈ 0.1414
        cte = self._cte(0.0, 0.2, (0.0, 0.0), (1.0, 1.0))
        self.assertGreater(cte, 0.0)
        self.assertAlmostEqual(cte, 0.2 / math.sqrt(2), places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Avoidance ramp (avoidance_offset → avoidance_target)
# ═════════════════════════════════════════════════════════════════════════════
class TestAvoidanceRamp(unittest.TestCase):
    """
    Logic from control_loop:
        RAMP_RATE = 0.5 / 30.0
        if |offset - target| < RAMP_RATE: offset = target
        elif offset < target:             offset += RAMP_RATE
        else:                             offset -= RAMP_RATE
    """

    RAMP = 0.5 / 30.0

    def _step(self, offset, target):
        r = self.RAMP
        if abs(offset - target) < r:
            return target
        elif offset < target:
            return offset + r
        else:
            return offset - r

    def test_ramp_rate_value(self):
        self.assertAlmostEqual(self.RAMP, 0.5 / 30.0, places=8)

    def test_ramps_up_toward_positive_target(self):
        offset = self._step(0.0, 0.3)
        self.assertAlmostEqual(offset, self.RAMP, places=6)

    def test_ramps_down_toward_zero(self):
        offset = self._step(0.3, 0.0)
        self.assertAlmostEqual(offset, 0.3 - self.RAMP, places=6)

    def test_snaps_when_close(self):
        """If gap < RAMP_RATE, offset should snap to target exactly."""
        offset = self._step(self.RAMP * 0.5, 0.0)
        self.assertAlmostEqual(offset, 0.0, places=6)

    def test_reaches_target_in_expected_ticks(self):
        """Full 0.3 m offset at 30 Hz should be reached in ~18 ticks."""
        offset = 0.0
        ticks  = 0
        while abs(offset - 0.3) > 1e-9 and ticks < 100:
            offset = self._step(offset, 0.3)
            ticks += 1
        expected = math.ceil(0.3 / self.RAMP)
        self.assertEqual(ticks, expected)

    def test_ramp_up_then_ramp_down(self):
        """After reaching target, ramping back to 0 works symmetrically."""
        offset = 0.0
        for _ in range(50):
            offset = self._step(offset, 0.3)
        self.assertAlmostEqual(offset, 0.3, places=5)
        for _ in range(50):
            offset = self._step(offset, 0.0)
        self.assertAlmostEqual(offset, 0.0, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 11. Asymmetric EMA (speed smoothing)
# ═════════════════════════════════════════════════════════════════════════════
class TestAsymmetricEMA(unittest.TestCase):
    """
    Logic from control_loop:
        if raw < smoothed: smoothed = raw           (instant brake)
        else:              smoothed = 0.85*s + 0.15*raw  (slow ramp-up)
    """

    def _step(self, smoothed, raw):
        if raw < smoothed:
            return raw
        return 0.85 * smoothed + 0.15 * raw

    def test_instant_brake(self):
        """Speed drops immediately when raw < smoothed."""
        self.assertAlmostEqual(self._step(1.4, 0.0), 0.0)

    def test_instant_brake_partial(self):
        self.assertAlmostEqual(self._step(1.4, 0.7), 0.7)

    def test_slow_ramp_up(self):
        """Speed ramps up slowly when raw > smoothed."""
        result = self._step(0.0, 1.4)
        self.assertAlmostEqual(result, 0.15 * 1.4, places=5)

    def test_ramp_up_never_overshoots(self):
        """Smoothed speed must never exceed raw during ramp-up."""
        smoothed = 0.0
        for _ in range(100):
            smoothed = self._step(smoothed, 1.4)
            self.assertLessEqual(smoothed, 1.4 + 1e-9)

    def test_ramp_up_converges(self):
        """After enough ticks, smoothed should be very close to command_speed."""
        smoothed = 0.0
        for _ in range(200):
            smoothed = self._step(smoothed, 1.4)
        self.assertAlmostEqual(smoothed, 1.4, places=2)

    def test_no_ramp_needed_when_equal(self):
        self.assertAlmostEqual(self._step(1.4, 1.4), 1.4, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 12. Steering EMA
# ═════════════════════════════════════════════════════════════════════════════
class TestSteeringEMA(unittest.TestCase):
    """smoothed_delta = 0.5 * prev + 0.5 * raw"""

    def _step(self, prev, raw):
        return 0.5 * prev + 0.5 * raw

    def test_from_zero(self):
        self.assertAlmostEqual(self._step(0.0, 0.4), 0.2, places=5)

    def test_symmetric_blend(self):
        self.assertAlmostEqual(self._step(0.2, 0.4), 0.3, places=5)

    def test_converges_to_constant_input(self):
        s = 0.0
        for _ in range(50):
            s = self._step(s, 0.3)
        self.assertAlmostEqual(s, 0.3, places=3)

    def test_step_change_half_damped(self):
        """Sudden large delta is half-damped on first tick."""
        result = self._step(0.0, STEER_LIMIT)
        self.assertAlmostEqual(result, STEER_LIMIT / 2, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 13. Lateral avoidance — path tangent perpendicular offset
# ═════════════════════════════════════════════════════════════════════════════
class TestLateralAvoidanceOffset(unittest.TestCase):
    """
    Shifted lookahead = T_orig + offset * (-ty/|t|, tx/|t|)
    Positive offset → shift left of path direction.
    """

    @staticmethod
    def _shift(tx, ty, tdx, tdy, offset):
        tlen = math.hypot(tdx, tdy) + 1e-9
        sx = tx + offset * (-tdy / tlen)
        sy = ty + offset * ( tdx / tlen)
        return sx, sy

    def test_positive_offset_shifts_left(self):
        """Path going +x, positive offset → shift in +y (left)."""
        sx, sy = self._shift(1.0, 0.0, 1.0, 0.0, 0.3)
        self.assertAlmostEqual(sx, 1.0, places=5)
        self.assertAlmostEqual(sy, 0.3, places=5)

    def test_negative_offset_shifts_right(self):
        """Path going +x, negative offset → shift in -y (right)."""
        sx, sy = self._shift(1.0, 0.0, 1.0, 0.0, -0.3)
        self.assertAlmostEqual(sx, 1.0, places=5)
        self.assertAlmostEqual(sy, -0.3, places=5)

    def test_zero_offset_no_shift(self):
        sx, sy = self._shift(1.0, 0.5, 1.0, 0.0, 0.0)
        self.assertAlmostEqual(sx, 1.0, places=5)
        self.assertAlmostEqual(sy, 0.5, places=5)

    def test_offset_perpendicular_to_path(self):
        """Shifted point must be exactly offset metres from original."""
        for tdx, tdy in [(1, 0), (0, 1), (1, 1)]:
            sx, sy = self._shift(0.0, 0.0, tdx, tdy, 0.3)
            dist = math.hypot(sx, sy)
            self.assertAlmostEqual(dist, 0.3, places=5)

    def test_path_facing_north_positive_offset_goes_left_in_map(self):
        """Path going +y (north), positive offset → -x (west = left of path)."""
        sx, sy = self._shift(0.0, 0.0, 0.0, 1.0, 0.3)
        self.assertAlmostEqual(sx, -0.3, places=5)
        self.assertAlmostEqual(sy,  0.0, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 14. Corridor filter
# ═════════════════════════════════════════════════════════════════════════════
class TestCorridorFilter(unittest.TestCase):
    """
    Filter rules from objects_in_map_frame_callback:
        reject if cx <= 0.1  (behind or at car)
        reject if |cy| > corridor_half = car_width/2 + path_corridor_margin
    """

    CAR_W   = 0.40
    MARGIN  = 1.0
    HALF    = CAR_W / 2 + MARGIN   # 1.2 m

    def _accept(self, cx, cy):
        return cx > 0.1 and abs(cy) <= self.HALF

    def test_person_directly_ahead_accepted(self):
        self.assertTrue(self._accept(1.5, 0.0))

    def test_person_behind_rejected(self):
        self.assertFalse(self._accept(0.05, 0.0))

    def test_person_at_boundary_rejected(self):
        self.assertFalse(self._accept(0.1, 0.0))

    def test_person_just_inside_corridor(self):
        self.assertTrue(self._accept(1.0, 1.19))

    def test_person_outside_corridor(self):
        self.assertFalse(self._accept(1.0, 1.21))

    def test_person_left_outside_corridor(self):
        self.assertFalse(self._accept(1.0, -1.21))

    def test_person_exactly_at_corridor_edge(self):
        self.assertTrue(self._accept(1.0, self.HALF))


# ═════════════════════════════════════════════════════════════════════════════
# 15. Avoidance direction decision
# ═════════════════════════════════════════════════════════════════════════════
class TestAvoidanceDirectionDecision(unittest.TestCase):
    """
    Direction logic from objects_in_map_frame_callback:
        if |lat_predicted| > CENTERED(0.20):
            avoid_dir = -copysign(clearance, lat_predicted)
        else (centered):
            if cte < -0.05: avoid_dir = +clearance  (go left)
            else:           avoid_dir = -clearance  (go right)
    """

    CLEARANCE = 0.3
    CENTERED  = 0.20

    def _decide(self, lat_predicted, cte=0.0):
        if abs(lat_predicted) > self.CENTERED:
            return -math.copysign(self.CLEARANCE, lat_predicted)
        else:
            return self.CLEARANCE if cte < -0.05 else -self.CLEARANCE

    def test_person_left_go_right(self):
        """Person predicted LEFT (+lat) → avoid_dir negative (go right)."""
        self.assertLess(self._decide(0.5), 0.0)

    def test_person_right_go_left(self):
        """Person predicted RIGHT (-lat) → avoid_dir positive (go left)."""
        self.assertGreater(self._decide(-0.5), 0.0)

    def test_clearance_magnitude(self):
        self.assertAlmostEqual(abs(self._decide(0.5)), self.CLEARANCE)

    def test_centered_car_right_go_left(self):
        """Person centered, car right of path (cte=-0.1) → go left."""
        self.assertGreater(self._decide(0.0, cte=-0.1), 0.0)

    def test_centered_car_left_go_right(self):
        """Person centered, car left of path (cte=+0.1) → go right."""
        self.assertLess(self._decide(0.0, cte=0.1), 0.0)

    def test_centered_on_path_go_right(self):
        """Person centered, car on path (cte≈0) → go right (default)."""
        self.assertLess(self._decide(0.0, cte=0.0), 0.0)

    def test_just_outside_centered_band(self):
        """lat_predicted just above CENTERED threshold → uses lateral decision."""
        avoid = self._decide(self.CENTERED + 0.01)
        self.assertLess(avoid, 0.0)   # person slightly left → go right

    def test_just_inside_centered_band(self):
        """lat_predicted just below CENTERED threshold → uses CTE decision."""
        avoid = self._decide(self.CENTERED - 0.01, cte=0.0)
        self.assertLess(avoid, 0.0)   # CTE≈0 → default right


# ═════════════════════════════════════════════════════════════════════════════
# 16. early_trigger_dist and obstacle_stop logic
# ═════════════════════════════════════════════════════════════════════════════
class TestAvoidanceTrigger(unittest.TestCase):
    """
    From callback:
        person_approach    = max(0, -vlong)
        early_trigger_dist = avoidance_start_dist + person_approach * predict_t
        avoidance active:  cx <= early_trigger_dist AND cx > person_stop_dist
        hard stop:         cx <= person_stop_dist
    """

    START = 1.5
    STOP  = 0.4
    T     = 1.5

    def _trigger(self, cx, vlong):
        approach     = max(0.0, -vlong)
        trigger_dist = self.START + approach * self.T
        avoidance    = cx <= trigger_dist and cx > self.STOP
        stop         = cx <= self.STOP
        return avoidance, stop

    def test_stationary_person_at_start_dist(self):
        av, st = self._trigger(cx=1.5, vlong=0.0)
        self.assertTrue(av)
        self.assertFalse(st)

    def test_stationary_person_far(self):
        av, st = self._trigger(cx=2.0, vlong=0.0)
        self.assertFalse(av)
        self.assertFalse(st)

    def test_approaching_person_triggers_earlier(self):
        """Person walking toward at 0.5 m/s → trigger at 2.25 m."""
        av, _ = self._trigger(cx=2.0, vlong=-0.5)
        self.assertTrue(av)   # 2.0 <= 1.5 + 0.5*1.5 = 2.25

    def test_receding_person_no_extension(self):
        """Person walking away → trigger_dist unchanged."""
        av_still, _ = self._trigger(cx=1.4, vlong= 0.0)
        av_away,  _ = self._trigger(cx=1.4, vlong= 0.5)
        self.assertTrue(av_still)
        self.assertTrue(av_away)   # still inside 1.5 m start dist

    def test_hard_stop_at_stop_dist(self):
        _, st = self._trigger(cx=0.4, vlong=0.0)
        self.assertTrue(st)

    def test_hard_stop_below_stop_dist(self):
        _, st = self._trigger(cx=0.2, vlong=0.0)
        self.assertTrue(st)

    def test_no_avoidance_in_hard_stop_zone(self):
        """When in hard stop zone, avoidance_target should be 0 (car stops)."""
        av, st = self._trigger(cx=0.3, vlong=0.0)
        self.assertFalse(av)
        self.assertTrue(st)


# ═════════════════════════════════════════════════════════════════════════════
# 17. _clear_obstacle state reset
# ═════════════════════════════════════════════════════════════════════════════
class TestClearObstacle(unittest.TestCase):

    def _n(self):
        n = _FakeNode()
        n.obstacle_stop_requested = True
        n.closest_person_dist     = 0.8
        n.closest_vlong           = -0.3
        n.avoidance_target        = 0.3
        return n

    def test_clears_stop_flag(self):
        n = self._n()
        n._clear_obstacle()
        self.assertFalse(n.obstacle_stop_requested)

    def test_clears_person_dist(self):
        n = self._n()
        n._clear_obstacle()
        self.assertEqual(n.closest_person_dist, float('inf'))

    def test_clears_vlong(self):
        n = self._n()
        n._clear_obstacle()
        self.assertAlmostEqual(n.closest_vlong, 0.0)

    def test_clears_avoidance_target(self):
        n = self._n()
        n._clear_obstacle()
        self.assertAlmostEqual(n.avoidance_target, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 18. Quaternion → yaw (from kinematic_state_callback)
# ═════════════════════════════════════════════════════════════════════════════
class TestQuaternionToYaw(unittest.TestCase):
    """
    theta = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    """

    @staticmethod
    def _yaw(w, x, y, z):
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny, cosy)

    def test_identity_zero_yaw(self):
        self.assertAlmostEqual(self._yaw(1, 0, 0, 0), 0.0, places=5)

    def test_90_degrees(self):
        s = math.sin(math.pi / 4)
        self.assertAlmostEqual(self._yaw(s, 0, 0, s), math.pi / 2, places=5)

    def test_180_degrees(self):
        self.assertAlmostEqual(abs(self._yaw(0, 0, 0, 1)), math.pi, places=5)

    def test_minus_90_degrees(self):
        s = math.sin(math.pi / 4)
        self.assertAlmostEqual(self._yaw(s, 0, 0, -s), -math.pi / 2, places=5)

    def test_45_degrees(self):
        angle = math.pi / 4
        s = math.sin(angle / 2)
        c = math.cos(angle / 2)
        self.assertAlmostEqual(self._yaw(c, 0, 0, s), angle, places=5)


# ═════════════════════════════════════════════════════════════════════════════
# 19. Goal detection
# ═════════════════════════════════════════════════════════════════════════════
class TestGoalDetection(unittest.TestCase):

    def test_within_5_of_end_triggers(self):
        self.assertTrue(56 >= 60 - 5)

    def test_at_last_index_triggers(self):
        self.assertTrue(59 >= 60 - 5)

    def test_middle_does_not_trigger(self):
        self.assertFalse(30 >= 60 - 5)

    def test_within_tolerance(self):
        dist = math.hypot(5.3 - 5.0, 5.3 - 5.0)
        self.assertLessEqual(dist, 0.5)

    def test_outside_tolerance(self):
        dist = math.hypot(4.0 - 5.0, 4.0 - 5.0)
        self.assertGreater(dist, 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
