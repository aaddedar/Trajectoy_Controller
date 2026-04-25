# control_car

Pure Pursuit Path Tracking Controller with Obstacle Avoidance for Autonomous Vehicles.

## Overview

The `control_car` package implements a **Pure Pursuit Path Tracking Controller** that receives a reference path and kinematic state, then computes steering commands to make the vehicle follow the path with real-time obstacle avoidance capabilities.

**Key Features:**
- Pure pursuit geometric path tracking algorithm
- Dynamic look-ahead distance based on velocity
- Real-time obstacle detection and avoidance
- Safety interlock system with external stop signals
- 100 Hz control loop (10 ms sampling)
- Ackermann steering model support

---

## Mathematical Foundation

### Pure Pursuit Algorithm

Pure pursuit is a geometric path tracking method that calculates the required steering angle based on the look-ahead point on the reference path.

#### 1. Look-ahead Point Selection

Find the closest point on the reference path to the robot position, then move forward until cumulative distance reaches:
$$L_f = k \cdot |v| + L_d$$

Where:
- $k = 0.35$ (velocity gain coefficient)
- $L_d = 0.8$ m (fixed look-ahead distance)
- $v$ = current robot velocity (m/s)

#### 2. Steering Angle Computation

Using the Ackermann steering kinematic model:
$$\delta = \arctan\left(\frac{2L \sin\alpha}{k(v + \epsilon) + L_d}\right)$$

Where:
- $L = 0.5$ m (vehicle wheelbase)
- $\alpha$ = cross-track error angle
- $\delta \in [-30°, 30°]$ (steering angle limits)

#### 3. Obstacle Detection

Transform robot position to obstacle's local frame and check if inside expanded bounding box with safety margins:
- Length margin: $+0.20$ m
- Width margin: $+0.05$ m

---

## ROS 2 Integration

### Node Information

**Node Name:** `pure_pursuit_node`  
**Executable:** `ros2 run control_car pure_pursuit_node`  
**Control Loop:** 100 Hz (10 ms sampling)

---

## Publishers

| Topic | Message Type | QoS | Description |
|-------|--------------|-----|-------------|
| `/ackermann_drive` | `AckermannDrive` | RELIABLE | Steering angle & speed commands |
| `/obstacle_information` | `Bool` | RELIABLE | Obstacle detection status (True=stop) |

---

## Subscribers

| Topic | Message Type | QoS | Description |
|-------|--------------|-----|-------------|
| `/path` | `Path` | RELIABLE | Reference waypoints from planner |
| `/kinematic_state` | `KinematicState` | BEST_EFFORT | Robot pose, orientation, velocity |
| `/ackermann_drive_feedback` | `AckermannDrive` | BEST_EFFORT | Actual steering/speed feedback |
| `/allowed_to_move` | `Bool` | RELIABLE | Safety signal (True=STOP) |
| `/objects_in_map_frame` | `Detection3DArray` | BEST_EFFORT | Detected obstacles with 3D boxes |
| `/robot_description` | `String` | DEFAULT | URDF robot model |

---

## Control Loop Logic

**Execution Flow (100 Hz):**

1. Check safety conditions (stop signals, obstacles, valid path/state)
2. Calculate distance to goal: $d_{goal} = \sqrt{(x_{goal} - x)^2 + (y_{goal} - y)^2}$
3. If $d_{goal} \leq 0.2$ m → Stop (goal reached)
4. Compute steering angle via pure pursuit
5. Publish steering + speed command

**Transitions:**
- `stop_requested=True` → Emergency stop
- Obstacle detected → Obstacle stop
- $d_{goal} \leq 0.2$ m → Goal reached, stop
- No path/state → Hold stopped

---

## Control Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `L_d` | 0.8 | m | Fixed look-ahead distance |
| `k` | 0.35 | - | Velocity gain |
| `command_speed` | 0.1 | m/s | Forward velocity |
| `Ts` | 0.01 | s | Control sampling time |
| `goal_tolerance` | 0.2 | m | Goal reached threshold |
| `object_length_margin` | 0.20 | m | Safety margin (length) |
| `object_width_margin` | 0.05 | m | Safety margin (width) |

---

## Class Architecture

### HagenRobot
Robot state container with kinematic properties.
```python
HagenRobot(Ts, x=0, y=0, theta=0, v=0, L=2.4)
```
- `Ts`: Sampling time (required)
- `x, y`: Position (m)
- `theta`: Heading (rad)
- `v`: Velocity (m/s)
- `L`: Wheelbase (m)

### PurePurSuitController
Implements pure pursuit algorithm.
- `pure_pursuit_control()` → Compute steering angle
- `look_ahead_point_index()` → Find look-ahead point

### PurePursuitNode (ROS 2 Node)
Main control system with callbacks:
- `control_loop()` - 100 Hz control execution
- `path_callback()` - Receives reference path
- `kinematic_state_callback()` - Updates robot state
- `objects_in_map_frame_callback()` - Detects obstacles
- `allowed_to_move_callback()` - Safety interlock
- `ackermann_feedback_callback()` - Hardware feedback
- `publish_stop_command()` - Emergency stop

---

## Installation & Build

```bash
# Build
cd ~/ros2_ws
colcon build --packages-select control_car

# Source
source install/setup.bash

# Run
ros2 run control_car pure_pursuit_node

# Check linting
python3 -m ruff check src/control_car/control_car/pure_persuit_node.py
```

---

## Usage Example

### System Launch

```bash
# Terminal 1: Localization & path planner
ros2 launch path_planner astar_planner.launch.py

# Terminal 2: Hardware interface
ros2 run ros2arduino ros2arduino_node

# Terminal 3: Controller
ros2 run control_car pure_pursuit_node
```

### Monitor Topics

```bash
# Steering commands
ros2 topic echo /ackermann_drive

# Obstacle status
ros2 topic echo /obstacle_information

# Robot state
ros2 topic echo /kinematic_state

# Reference path
ros2 topic echo /path
```

---

## Safety Features

1. **External Stop Signal** - `/allowed_to_move` topic stops vehicle
2. **Obstacle Avoidance** - Detects 3D bounding boxes with safety margins
3. **Goal Tolerance** - Stops at 0.2 m from waypoint
4. **Path Validation** - Requires ≥2 waypoints
5. **State Validation** - Waits for `/kinematic_state` feedback

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Vehicle not moving | Check `/allowed_to_move=False`, verify `/path` waypoints, confirm `/kinematic_state` |
| Erratic steering | Verify quaternion in `/kinematic_state`, check path sufficiency, tune `L_d` and `k` |
| Unexpected stops | Check `/objects_in_map_frame` for obstacles, verify `/allowed_to_move` signal |
| Steering limits exceeded | Reduce `command_speed`, increase `L_d`, reduce `k` |

---

## Performance Specifications

| Metric | Value |
|--------|-------|
| Control Loop | 100 Hz |
| Sampling Time | 10 ms |
| Max Steering | ±30° |
| Command Speed | 0.1 m/s |
| Look-ahead | 0.8 + 0.35×v |
| Goal Tolerance | 0.2 m |

---

## Dependencies

**ROS 2 Messages:**
- `ackermann_msgs/AckermannDrive`
- `curobot_msgs/KinematicState`
- `nav_msgs/Path`
- `std_msgs/Bool, String`
- `vision_msgs/Detection3DArray`

**Python Libraries:**
- numpy ≥ 1.20
- rclpy ≥ 3.0

---

## References

1. Coulter, R. C. (1992). "Implementation of the Pure Pursuit Path Tracking Algorithm"
2. https://github.com/GPrathap/autonomous_mobile_robots/blob/master/hagen_control/hagen_control/pure_pursuit_control.py
3. ROS 2 Documentation: https://docs.ros.org/en/humble/
4. Ackermann Steering Model & Quaternion Conversions

---

## Version

**v1.0** - 2026-04-25 - Pure pursuit path tracking with obstacle avoidance
