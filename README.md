# control_car

Pure Pursuit Path Tracking Controller with Obstacle Avoidance for Autonomous Vehicles.

## Overview

The `control_car` package implements a **Pure Pursuit Path Tracking Controller** that receives a reference path and kinematic state, then computes steering commands to make the vehicle follow the path with real-time obstacle avoidance capabilities.

**Key Features:**
- Pure pursuit geometric path tracking algorithm
- Curvature-adaptive look-ahead distance (shrinks on curves, expands on straights)
- Velocity-adaptive lookahead via gain `k`
- Real-time person detection and lateral obstacle avoidance
- Goal approach speed ramp — stops within 2 cm of target
- Safety interlock system with external stop signals
- 30 Hz control loop
- Ackermann steering model support

---

## Mathematical Foundation

### Pure Pursuit Algorithm

Pure pursuit is a geometric path tracking method that calculates the required steering angle based on the look-ahead point on the reference path.

#### 1. Look-ahead Point Selection

Find the closest point on the reference path to the robot position, then move forward until cumulative distance reaches:

$$L_f = (k \cdot |v| + L_d) \cdot (1 - 0.6 \cdot c)$$

Where:
- $k = 0.3$ (velocity gain coefficient)
- $L_d = 0.30$ m (base look-ahead distance)
- $v$ = current robot velocity (m/s)
- $c = \min(1,\, |\delta_{prev}| / \delta_{max})$ = curvature factor from previous steering angle

**Curvature-adaptive behaviour:**

| Situation | Curve factor $c$ | Effective $L_f$ at 1.4 m/s |
|---|---|---|
| Straight | 0.0 | 0.72 m |
| Gentle curve (10°) | 0.33 | 0.57 m |
| Medium curve (20°) | 0.67 | 0.43 m |
| Tight curve (30°) | 1.0 | 0.30 m → floor 0.15 m |

This prevents the car from going wide on curves while suppressing oscillation on straights.

#### 2. Steering Angle Computation

Using the Ackermann steering kinematic model:

$$\delta = \arctan\!\left(\frac{2L \sin\alpha}{L_f + \epsilon}\right), \quad \delta \in [-30°,\, 30°]$$

Where:
- $L = 0.30$ m (vehicle wheelbase)
- $\alpha$ = angle from car heading to lookahead point
- $\epsilon = 10^{-5}$ — numerical stability

#### 3. Steering Smoothing

Raw steering is EMA-filtered to damp rapid corrections:

```
smoothed_delta = 0.8 × prev + 0.2 × raw
```

#### 4. Goal Stopping

$$d_{goal} = \sqrt{(x_{goal} - x)^2 + (y_{goal} - y)^2}$$

Car stops when $d_{goal} \leq 0.02$ m. Speed ramps linearly to zero over the last 0.5 m.

---

## Speed Control

Four independent speed limiters applied as `min(v1, v2, v3, v4)`:

### 1. Curve speed (current steering angle)

$$v_{curve} = v_{max} - (v_{max} - v_{min}) \cdot \left(\frac{|\delta|}{\delta_{max}}\right)^{0.5}$$

### 2. Ahead curvature speed

Scans next 30 path points (≈1.5 m ahead), pre-brakes before curves are reached.

### 3. Safe speed (obstacle distance)

$$v_{safe} = \begin{cases} 0 & d < d_{stop} \\ v_{max} \cdot \dfrac{d - d_{stop}}{d_{slow} - d_{stop}} & d_{stop} \le d < d_{slow} \\ v_{max} & d \ge d_{slow} \end{cases}$$

### 4. Goal approach ramp

$$v_{goal} = v_{max} \cdot \frac{d_{goal}}{0.5}, \quad d_{goal} < 0.5 \text{ m}$$

### Asymmetric EMA (anti-jerk)

```python
if raw_speed < smoothed_speed:
    smoothed_speed = raw_speed                               # instant brake
else:
    smoothed_speed = 0.75 * smoothed_speed + 0.25 * raw_speed   # ramp-up (~0.5 s)
```

---

## Obstacle Avoidance

Geometric avoidance: shifts the pure pursuit lookahead point laterally to steer around detected persons.

```
Detection → Predict position → Choose side → Offset lookahead → Pure pursuit steers
```

**State machine:**

```
person dist > avoidance_start_dist (1.5 m)  →  Normal path following
avoidance_start_dist ≥ dist > person_stop_dist (0.4 m)  →  Steer around person
dist ≤ person_stop_dist (0.4 m)  →  HARD STOP
No person for 1.0 s  →  Resume, ramp back to path
```

---

## ROS 2 Interface

### Node Information

**Node Name:** `pure_pursuit_node`
**Executable:** `ros2 launch control_car pure_persuit.launch.py`
**Control Loop:** 30 Hz

### Publishers

| Topic | Message Type | QoS | Description |
|-------|--------------|-----|-------------|
| `/ackermann_drive` | `AckermannDrive` | RELIABLE | Steering angle & speed commands |
| `/obstacle_information` | `Bool` | RELIABLE | True when hard-stopped for obstacle |
| `/pure_pursuit/lookahead_marker` | `Marker` | 10 | Lookahead sphere + nearest point |
| `/pure_pursuit/avoidance_path` | `Path` | 10 | Shifted avoidance trajectory |
| `/pure_pursuit/original_path_ahead` | `Path` | 10 | Original path ahead |
| `/pure_pursuit/avoidance_viz` | `MarkerArray` | 10 | Coloured visualisation markers |

### Subscribers

| Topic | Message Type | QoS | Description |
|-------|--------------|-----|-------------|
| `/path` | `Path` | TRANSIENT_LOCAL | Reference waypoints from planner |
| `/kinematic_state` | `KinematicState` | BEST_EFFORT | Robot pose, orientation, velocity |
| `/ackermann_drive_feedback` | `AckermannDrive` | BEST_EFFORT | Actual steering/speed feedback |
| `/allowed_to_move` | `Bool` | RELIABLE | Safety signal (False=STOP) |
| `/tracked_objects` | `Detection3DArray` | BEST_EFFORT | KF-tracked persons with velocity |

---

## Control Parameters

### Path Following

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `command_speed` | 1.4 | m/s | Maximum forward speed |
| `min_curve_speed` | 0.9 | m/s | Minimum speed in tight curves |
| `L_d` | 0.30 | m | Base look-ahead distance |
| `k` | 0.3 | — | Velocity gain for adaptive lookahead |
| `invert_steering` | False | — | Negate steering output |

### Obstacle Safety

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `person_stop_dist` | 0.4 | m | Hard stop distance |
| `person_slowdown_dist` | 1.5 | m | Begin speed reduction |
| `avoidance_start_dist` | 1.5 | m | Begin lateral avoidance |
| `avoidance_clearance` | 0.3 | m | Lateral offset magnitude |
| `avoidance_predict_t` | 1.5 | s | KF velocity prediction horizon |

### Internal Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `STEER_LIMIT` | 0.5236 rad (30°) | Max steering angle |
| Control loop | 30 Hz | `Ts = 1/30` s |
| Wheelbase `L` | 0.30 m | Bicycle model |
| Car width | 0.40 m | Corridor filter |
| Goal tolerance | 0.02 m | Stop within 2 cm of goal |
| Goal slowdown dist | 0.5 m | Begin speed ramp to zero |
| Obstacle timeout | 1.0 s | Clear if no detection for this long |
| Kinematic timeout | 3.0 s | Stop if no localization for this long |
| Path smoothing | window=7, iter=3 | Moving average |
| Path spacing | 0.05 m | Densification step |

---

## Class Architecture

### HagenRobot
Robot state container with kinematic properties.
```python
HagenRobot(x=0, y=0, theta=0, v=0, L=0.3)
```
- `x, y`: Position (m)
- `theta`: Heading (rad)
- `v`: Velocity (m/s)
- `L`: Wheelbase (m)

### PurePurSuitController
Implements pure pursuit algorithm with curvature-adaptive lookahead.
- `pure_pursuit_control()` → Compute steering angle
- `look_ahead_point_index()` → Find look-ahead point

### PurePursuitNode (ROS 2 Node)
Main control node with callbacks:
- `control_loop()` — 30 Hz control execution
- `path_callback()` — Receives and preprocesses reference path
- `kinematic_state_callback()` — Updates robot pose and velocity
- `objects_in_map_frame_callback()` — Person detection and avoidance logic
- `allowed_to_move_callback()` — Safety interlock
- `ackermann_feedback_callback()` — Hardware feedback
- `publish_stop_command()` — Emergency stop

---

## Installation & Build

```bash
cd ~/ros2_ws
colcon build --packages-select control_car
source install/setup.bash
```

---

## Launch Sequence

```bash
# Terminal 1 — System (localization, nav2, arduino, camera TF)
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch ~/ros2_ws/system_startup.launch.py

# Terminal 2 — Camera pipeline
pkill -9 -f "detectnet"; pkill -9 -f "object_localizer"; pkill -9 -f "realsense2_camera"; sleep 2
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch object_localizer camera.launch.py

# Terminal 3 — KF object tracker
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 run kf_object_tracking kf_object_tracker

# Terminal 4 — ros2arduino (start BEFORE pure pursuit — give Arduino 2 s to boot)
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 run ros2arduino ros2arduino_node &
sleep 2

# Terminal 5 — Pure pursuit controller
ros2 launch control_car pure_persuit.launch.py
```

### Monitor Topics

```bash
ros2 topic echo /ackermann_drive        # steering commands
ros2 topic echo /obstacle_information   # obstacle status
ros2 topic echo /kinematic_state        # robot state
ros2 topic hz /tracked_objects          # expect ~10 Hz
```

---

## Safety Features

1. **External Stop Signal** — `/allowed_to_move=False` stops vehicle immediately
2. **Person Detection** — Hard stop at 0.4 m, lateral avoidance from 1.5 m
3. **Goal Tolerance** — Stops within 2 cm of target waypoint
4. **Goal Speed Ramp** — Linearly decelerates to zero over last 0.5 m
5. **Path Validation** — Requires ≥2 waypoints before moving
6. **State Validation** — Waits for `/kinematic_state`; stops if lost for >3 s
7. **Obstacle Timeout** — Resumes after 1 s with no detections

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Vehicle not moving | Check `/allowed_to_move=False`, verify `/path` received, confirm `/kinematic_state` |
| Steering oscillates left–right | Tune `L_d` (increase) or `k` (decrease); check localization noise |
| Car goes wide on curves | Curvature-adaptive L_d handles this; if still wide, reduce `k` |
| Car stops short of goal | Check localization accuracy; goal tolerance is 2 cm |
| Unexpected obstacle stop | Check `/tracked_objects`; obstacle clears after 1.0 s timeout |
| ros2arduino crashes | Power brownout: Arduino + ESC share power rail. Start ros2arduino before pure pursuit. Sustained speed >1.0 m/s may cause USB disconnect — power Arduino via USB separately from motor ESC |

---

## Performance Specifications

| Metric | Value |
|--------|-------|
| Control Loop | 30 Hz |
| Sampling Time | 33 ms |
| Max Steering | ±30° |
| Command Speed | 1.4 m/s |
| Look-ahead (straight, 1.4 m/s) | 0.72 m |
| Look-ahead (tight curve) | 0.15 m (floor) |
| Goal Tolerance | 0.02 m |
| Obstacle hard stop | 0.4 m |

---

## Dependencies

**ROS 2 packages:**
- `ackermann_msgs` — drive commands
- `curobot_msgs` — KinematicState
- `nav_msgs` — Path
- `vision_msgs` — Detection3DArray
- `visualization_msgs` — Marker, MarkerArray
- `tf2_ros` — TF tree

**Python:** `numpy`, `rclpy`, `math`

---

## References

1. Coulter, R. C. (1992). *Implementation of the Pure Pursuit Path Tracking Algorithm*. CMU-RI-TR-92-01.
2. ROS 2 Humble Documentation — https://docs.ros.org/en/humble/
3. Ackermann Steering Model & Quaternion Conversions

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-25 | Initial pure pursuit, basic obstacle stop |
| v2.0 | 2026-06-14 | Tuned path following (L_d, speeds, EMA, sqrt curve mapping) |
| v3.0 | 2026-06-14 | Full camera pipeline, KF tracker, lateral avoidance, trajectory visualisation |
| v4.0 | 2026-06-16 | Curvature-adaptive L_d (k=0.3); faster ramp-up (0.75/0.25); tighter steering EMA (0.8/0.2); goal tolerance 0.02 m; obstacle timeout 1.0 s; ros2arduino power limit documented |
