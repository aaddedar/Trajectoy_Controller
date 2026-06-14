# control_car

Pure Pursuit Path Tracking Controller with Real-Time Obstacle Detection and Avoidance.

Developed and evaluated on a 1:10 scale model city track (≈5 m lap) on a Jetson AGX Orin running ROS 2 Humble.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Hardware](#hardware)
3. [Software Architecture](#software-architecture)
4. [Full Pipeline](#full-pipeline)
5. [Pure Pursuit Controller](#pure-pursuit-controller)
6. [Speed Control](#speed-control)
7. [Obstacle Detection Pipeline](#obstacle-detection-pipeline)
8. [Obstacle Avoidance Algorithm](#obstacle-avoidance-algorithm)
9. [Trajectory Visualisation](#trajectory-visualisation)
10. [ROS 2 Interface](#ros-2-interface)
11. [Parameters](#parameters)
12. [Launch Sequence](#launch-sequence)
13. [Evaluation & Tuning Log](#evaluation--tuning-log)
14. [Known Limitations](#known-limitations)
15. [Future Work](#future-work)

---

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   Jetson AGX Orin                   │
│  ROS_DOMAIN_ID=11  │  ROS 2 Humble  │  Linux Tegra  │
└─────────────────────────────────────────────────────┘

Localization (MOCAP/OptiTrack KF8)
        │  /kinematic_state
        ▼
Path Planner (A*)
        │  /path
        ▼
┌──────────────────────────┐        Camera Pipeline
│   pure_pursuit_node      │◄───── /tracked_objects
│  - Path following        │
│  - Speed control         │
│  - Obstacle avoidance    │
│  - Trajectory viz        │
└──────────┬───────────────┘
           │  /ackermann_drive
           ▼
      ros2arduino → Car hardware
```

---

## Hardware

| Component | Specification |
|---|---|
| Platform | Jetson AGX Orin |
| Car | 1:10 scale, Ackermann steering |
| Wheelbase | 0.30 m |
| Car width | 0.40 m |
| Localization | OptiTrack MOCAP → KF8 node |
| Camera | Intel RealSense (RGB + aligned depth) |
| Map | Model city, ≈5 m lap track |

---

## Software Architecture

```
system_startup.launch.py
├── localization (KF8/MOCAP) → /kinematic_state, TF: odom→11/base_link
├── nav2 (map server + AMCL) → TF: map→odom
├── car_description             → URDF, TF: 11/base_link→...
├── ros2arduino                 → /ackermann_drive → hardware
└── static_transform_publisher  → TF: 11/base_link→camera_link

camera.launch.py  (object_localizer package)
├── realsense2_camera           → RGB + aligned depth streams
├── static_transform_publisher  → TF: 11/base_link→camera_link (redundant safety)
├── detectnet (t+3 s)           → /detectnet/detections  [ssd-mobilenet-v2]
└── object_localizer (t+5 s)    → /objects_in_map_frame

kf_object_tracker               → /tracked_objects  [Kalman Filter]

pure_persuit.launch.py
└── pure_pursuit_node           → /ackermann_drive
```

### TF Tree

```
map → odom → 11/base_link → camera_link → camera_depth_frame
                                                  └→ camera_depth_optical_frame
                                         → camera_color_frame
                                               └→ camera_color_optical_frame
```

**Critical:** `11/base_link → camera_link` must be published before `object_localizer` starts.
Static transform: `x=0.367 y=-0.0115 z=0.190` (camera mounted on car front).

---

## Full Pipeline

```
RealSense RGB  ──────────────────────────────────────────────────────►
                                                                       │
                                                                   detectnet
                                                              (ssd-mobilenet-v2)
                                                              class \x01 = person
                                                                       │
                                                              /detectnet/detections
                                                                       │
RealSense aligned depth ──────────────────────────────────────►        │
                                                              object_localizer
                                                        (depth + TF → 3D map pos)
                                                                       │
                                                          /objects_in_map_frame
                                                          (Detection3DArray, map frame)
                                                                       │
                                                           kf_object_tracker
                                                    (Kalman Filter: position + velocity)
                                                                       │
                                                            /tracked_objects
                                                     bbox.center = position (map)
                                                     results[0].pose = velocity (vx,vy)
                                                                       │
OptiTrack MOCAP ──► localization ──► /kinematic_state                  │
                                             │                          │
                                             ▼                          ▼
                                    pure_pursuit_node ◄─────────────────┘
                                             │
                                    /ackermann_drive
                                             │
                                       ros2arduino
                                             │
                                       Car hardware
```

---

## Pure Pursuit Controller

### Algorithm

Pure pursuit is a geometric path tracking method using the bicycle kinematic model.

**1. Path preprocessing** (on `/path` received):
- Smooth: moving-average kernel (window=7, iterations=3)
- Densify: linear interpolation to 5 cm point spacing
- Preserves original start and end points

**2. Lookahead point selection:**

$$L_f = k \cdot |v| + L_d \quad (k=0,\ L_d=0.30\text{ m})$$

Search backward 30 points from last index to handle circular/re-entrant paths.

**3. Steering angle:**

$$\delta = \arctan\!\left(\frac{2 L \sin\alpha}{L_f + \epsilon}\right), \quad \delta \in [-30°,\, 30°]$$

Where:
- $L = 0.30$ m — wheelbase
- $\alpha$ — angle from car heading to lookahead point
- $\epsilon = 10^{-5}$ — numerical stability

**4. Steering inversion:**
`ros2arduino` negates the steering angle internally → `invert_steering = False`

---

## Speed Control

Three independent speed limiters applied as `min(v1, v2, v3)`:

### 1. Curve speed (current steering angle)

$$v_{curve} = v_{max} - (v_{max} - v_{min}) \cdot \left(\frac{|\delta|}{\delta_{max}}\right)^{0.5}$$

Sqrt mapping: speed drops steeply even at small steer angles, keeping corners tight.

### 2. Ahead curvature speed (path lookahead)

Scans next 30 path points (≈1.5 m), computes circumradius at each triplet:

$$k_{path} = \frac{4 \cdot \text{Area}}{d_{12} \cdot d_{23} \cdot d_{13}}$$

Speed reduced proportionally to worst curvature found ahead.

### 3. Safe speed (person distance)

$$v_{safe} = \begin{cases}
0.0 & d < d_{stop} \\
v_{max} \cdot \dfrac{d - d_{stop}}{d_{slow} - d_{stop}} & d_{stop} \le d < d_{slow} \\
v_{max} & d \ge d_{slow}
\end{cases}$$

### Asymmetric EMA (anti-jerk)

```python
if raw_speed < smoothed_speed:
    smoothed_speed = raw_speed           # instant brake
else:
    smoothed_speed = 0.85 * smoothed_speed + 0.15 * raw_speed  # slow ramp-up
```

Steering also EMA-smoothed: `smoothed_delta = 0.5 * prev + 0.5 * raw`

---

## Obstacle Detection Pipeline

### 1. detectnet — Person detection

- Model: `ssd-mobilenet-v2` (TensorRT, FP16)
- Input: `/camera/camera/color/image_raw` (640×480 @ 15 Hz)
- Output: `/detectnet/detections` — class byte `\x01` = person
- Threshold: 0.5 confidence
- Launch: `my_detectnet.launch.py` (NOT `detectnet.ros2.launch` — that crashes headlessly)

### 2. object_localizer — 2D → 3D

- Subscribes: `/detectnet/detections` + `/camera/camera/aligned_depth_to_color/image_raw`
- Uses TF `camera_depth_optical_frame → 11/base_link → map`
- Publishes: `/objects_in_map_frame` (`Detection3DArray`, frame=map)
- Class resolution: byte `\x01` → string `'person'`

**Critical startup requirement:** `align_depth.enable: true` must be set as a node
parameter at launch (not via `ros2 param set` at runtime — that does not restart streaming).

### 3. kf_object_tracker — Tracking & velocity

- Kalman Filter tracks each detected person across frames
- Publishes `/tracked_objects` (`Detection3DArray`, RELIABLE QoS, depth=10)
  - `bbox.center.position` = estimated map-frame position
  - `results[0].pose.pose.position` = estimated velocity (vx, vy) in map frame
- `min_hits=1`, `max_misses=10`

### QoS compatibility

| Node | Topic | QoS |
|---|---|---|
| kf_object_tracker | `/tracked_objects` publish | RELIABLE |
| pure_pursuit_node | `/tracked_objects` subscribe | BEST_EFFORT |

BEST_EFFORT subscriber is compatible with RELIABLE publisher in ROS 2.

---

## Obstacle Avoidance Algorithm

### Overview

Geometric avoidance: shifts the pure pursuit lookahead point laterally to steer the car around the detected person. No ML model required.

```
Detection → Predict position → Choose side → Offset lookahead → Pure pursuit steers
```

### Step 1 — Corridor filter

Only persons within the car's forward corridor are considered:

```
corridor_half = car_width/2 + path_corridor_margin = 0.2 + 1.0 = 1.2 m
```

Persons with `cx ≤ 0.1 m` (behind car) or `|cy| > corridor_half` are ignored.

### Step 2 — Direction decision (car-frame)

KF velocity `(vx_map, vy_map)` is transformed to car-frame lateral velocity:

$$v_{lat} = -v_x \sin\theta + v_y \cos\theta$$

Predicted lateral position:

$$\hat{y} = c_y + v_{lat} \cdot T_{predict}$$

**Person to one side** (`|ŷ| > 0.20 m`):
- Person predicted LEFT (`ŷ ≥ 0`) → offset lookahead RIGHT (`offset = -clearance`)
- Person predicted RIGHT (`ŷ < 0`) → offset lookahead LEFT (`offset = +clearance`)

**Person centered** (`|ŷ| ≤ 0.20 m`): use cross-track error (CTE) to pick the side
with more road space:
- CTE < −0.05 m (car right of path) → go LEFT (offset = +clearance)
- CTE ≥ −0.05 m (car left or on center) → go RIGHT (offset = −clearance)

### Step 3 — Apply offset perpendicular to PATH tangent

Path tangent at lookahead index `i`:
$$\hat{t} = \frac{(p_{i+1} - p_i)}{|p_{i+1} - p_i|}$$

Path left perpendicular:
$$\hat{n} = (-t_y,\ t_x)$$

Shifted lookahead:
$$\mathbf{T}_{shifted} = \mathbf{T}_{orig} + \text{offset} \cdot \hat{n}$$

Using PATH tangent (not car heading) ensures the offset stays within road boundaries on curves.

### Step 4 — Smooth ramp

`avoidance_offset` ramps toward `avoidance_target` at 0.5 m/s:

```
RAMP_RATE = 0.5 / 30.0 ≈ 0.017 m per tick
```

~1 second to reach full offset, ~1 second to return to path after person clears.

### State machine

```
Person dist > avoidance_start_dist (1.5 m)
    → Normal path following, no avoidance

avoidance_start_dist ≥ person dist > person_stop_dist (0.4 m)
    → avoidance_target = ±clearance, speed ramps down
    → Steer around person

person dist ≤ person_stop_dist (0.4 m)
    → HARD STOP, avoidance_target = 0

No person in corridor (timeout 5.0 s)
    → avoidance_target = 0, offset ramps to 0, normal driving resumes
```

---

## Trajectory Visualisation

Three topics published every control tick for RViz:

| Topic | Type | Description |
|---|---|---|
| `/pure_pursuit/original_path_ahead` | `nav_msgs/Path` | Original path ahead (next 3 m) |
| `/pure_pursuit/avoidance_path` | `nav_msgs/Path` | Shifted avoidance trajectory |
| `/pure_pursuit/avoidance_viz` | `MarkerArray` | Coloured line strips + offset arrow |
| `/pure_pursuit/lookahead_marker` | `Marker` | Yellow sphere = lookahead point |

**Colour convention in RViz (MarkerArray):**

| Colour | Meaning |
|---|---|
| White line | Original path ahead |
| Yellow line | Active avoidance trajectory |
| Cyan line | Return trajectory (ramping back to path) |
| Orange arrow | Lateral offset magnitude at path midpoint |

---

## ROS 2 Interface

### Publishers

| Topic | Type | QoS | Description |
|---|---|---|---|
| `/ackermann_drive` | `AckermannDrive` | RELIABLE | Speed + steering command |
| `/obstacle_information` | `Bool` | RELIABLE | True when hard-stopped for obstacle |
| `/pure_pursuit/lookahead_marker` | `Marker` | 10 | Lookahead sphere + nearest point |
| `/pure_pursuit/avoidance_path` | `Path` | 10 | Shifted avoidance trajectory |
| `/pure_pursuit/original_path_ahead` | `Path` | 10 | Original path ahead |
| `/pure_pursuit/avoidance_viz` | `MarkerArray` | 10 | Coloured viz markers |

### Subscribers

| Topic | Type | QoS | Description |
|---|---|---|---|
| `/path` | `Path` | TRANSIENT_LOCAL | Reference path from planner |
| `/kinematic_state` | `KinematicState` | BEST_EFFORT | Car pose + velocity |
| `/tracked_objects` | `Detection3DArray` | BEST_EFFORT | Tracked persons with velocity |
| `/allowed_to_move` | `Bool` | RELIABLE | External stop signal |
| `/ackermann_drive_feedback` | `AckermannDrive` | BEST_EFFORT | Hardware feedback |

---

## Parameters

### Path Following

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `command_speed` | 1.4 | m/s | Maximum forward speed |
| `min_curve_speed` | 0.9 | m/s | Minimum speed in tight curves |
| `L_d` | 0.30 | m | Fixed lookahead distance |
| `invert_steering` | False | — | Negate steering output (False: ros2arduino negates internally) |

### Obstacle Safety

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `person_stop_dist` | 0.4 | m | Hard stop distance |
| `person_slowdown_dist` | 1.5 | m | Begin speed reduction distance |

### Avoidance

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `avoidance_start_dist` | 1.5 | m | Begin lateral steering distance |
| `avoidance_clearance` | 0.5 | m | Lateral offset magnitude |
| `avoidance_predict_t` | 1.5 | s | KF velocity prediction horizon |

### Internal Constants

| Constant | Value | Description |
|---|---|---|
| `STEER_LIMIT` | 0.5236 rad (30°) | Max steering angle |
| Control loop | 30 Hz | `Ts = 1/30` s |
| Wheelbase `L` | 0.30 m | Bicycle model |
| Car width | 0.40 m | Corridor filter |
| Path corridor margin | 1.0 m | Extra width beyond car half-width |
| Obstacle timeout | 5.0 s | Clear if no detection for this long |
| Kinematic timeout | 3.0 s | Stop if no localization for this long |
| Path smoothing | window=7, iter=3 | Moving average |
| Path spacing | 0.05 m | Densification step |
| Ahead curvature scan | 30 points (1.5 m) | Speed pre-reduction |

---

## Launch Sequence

```bash
# Terminal 1 — System (localization, nav2, arduino, camera TF)
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch ~/ros2_ws/system_startup.launch.py

# Terminal 2 — Camera pipeline (kill old processes first)
pkill -9 -f "detectnet"; pkill -9 -f "object_localizer"; pkill -9 -f "realsense2_camera"; sleep 2
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch object_localizer camera.launch.py
# Wait ~10 s for camera → detectnet (t+3s) → object_localizer (t+5s)

# Terminal 3 — KF object tracker
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 run kf_object_tracking kf_object_tracker

# Terminal 4 — Pure pursuit controller
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch control_car pure_persuit.launch.py

# Health check
ros2 topic hz /tracked_objects   # expect ~10 Hz
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select control_car
source install/setup.bash
```

---

## Evaluation & Tuning Log

### Path Following

| Issue | Root Cause | Fix Applied |
|---|---|---|
| Corner cutting | Lookahead too large (0.80 m) → car aims past the corner | Reduced `L_d` to 0.30 m |
| Steering oscillation left–right | No smoothing on raw delta | Added steering EMA: `0.5×prev + 0.5×raw` |
| Car jerking and stopping mid-run | Speed command jumped from 0 → target instantly | Asymmetric EMA: instant brake, 0.85/0.15 ramp-up |
| Still cutting corners | Linear speed-vs-steer mapping too gentle | Changed to sqrt mapping — steeper drop even at small angles |
| Path deviation in curves | L_d=0.30 m still slightly large at curves | Ahead-curvature pre-reduction (30-point lookahead scan) |

**Tuned values:** `command_speed=1.4 m/s`, `min_curve_speed=0.9 m/s`, `L_d=0.30 m`

---

### Obstacle Detection Pipeline

| Issue | Root Cause | Fix Applied |
|---|---|---|
| `aligned_depth` not streaming | `align_depth.enable` set via `ros2 param set` at runtime — does not restart stream | Passed as launch argument in `camera.launch.py` at node startup |
| `objects_in_map_frame` always empty (length 0) | `11/base_link → camera_link` static TF missing — camera started without `camera.launch.py` | Added TF publisher to `system_startup.launch.py` permanently |
| detectnet crash "context is invalid" | `detectnet.ros2.launch` (XML) includes `video_output` with `display://0` → crashes headlessly → invalidates all ROS contexts | Switched to `my_detectnet.launch.py` (Python, no video output) |
| detectnet crash repeatedly | Another detectnet still running from previous session | Added `pkill -9 -f "detectnet"` to launch procedure |
| Launch crashes everything at t=7 s | `environmental_model` package not installed — launch exception kills all processes | Removed `environmental_model` from `camera.launch.py` |
| `/tracked_objects` QoS mismatch | pure_pursuit subscribed RELIABLE, tracker publishes RELIABLE — actually worked, but changed to BEST_EFFORT | Changed subscription to BEST_EFFORT for robustness |
| Class ID not matching | detectnet encodes class as raw byte `\x01` → object_localizer resolves to string `'person'` → `_is_person()` must accept both | `_is_person()` accepts `'person'` and `'1'` |

---

### Obstacle Avoidance

| Issue | Root Cause | Fix Applied |
|---|---|---|
| Car not stopping at all | TF chain broken — object_localizer could not project to map | Fixed TF (see above) |
| Car stopping but not avoiding | `avoidance_start_dist = 1.0 m` = `person_stop_dist` — hard stop fired simultaneously | Separated: `avoidance_start_dist=1.5 m`, `person_stop_dist=0.4 m` |
| Car avoiding in wrong direction (into wall) | Offset applied using car heading (`sin_t, cos_t`) — wrong at curves | Changed to path tangent at lookahead point |
| Avoidance pushed car off road at curves | Direction decision used path-relative lateral — inconsistent with any path | Reverted direction to car-relative (`cy`, always valid); kept path-tangent for application |
| Car not returning to path after avoidance | `avoidance_offset` reset to 0 instantly on clear → jerk | Smooth ramp at 0.5 m/s via `avoidance_target` state |
| Car going further off path while stopped beside person | Avoidance offset kept applied at cx < stop_dist | Clear `avoidance_target = 0` when `cx < person_stop_dist` |
| Person centered — wrong side chosen | `lat_predicted ≈ 0` → defaulted to RIGHT always | Use CTE: car left of path → go right (more room); car right → go left |
| Fixed 0.8 m offset too large for bounded road | Inflated costmap gives ≈0.5–0.7 m clearance each side | Reduced `avoidance_clearance` to 0.5 m |

---

## Known Limitations

1. **Single-person avoidance only** — corridor filter picks the closest person; no simultaneous multi-person planning.
2. **No map boundary awareness** — avoidance offset is fixed magnitude; very narrow road sections could push car into inflation zone.
3. **Linear velocity prediction** — KF extrapolation assumes constant velocity; sharp turns by pedestrians will mislead prediction.
4. **No avoidance if person appears within `avoidance_start_dist`** — if person walks in from the side at close range, avoidance window is very short.

---

## Future Work

| Priority | Algorithm | Benefit |
|---|---|---|
| Near-term | **ORCA** (Optimal Reciprocal Collision Avoidance) | Multi-person, reciprocal velocity obstacle, single Python file |
| Near-term | **Potential Fields** | Drop-in repulsive force, continuous avoidance without discrete zones |
| Medium-term | **MPC** (Model Predictive Control) | Unified path tracking + avoidance over N-step horizon; better on curves |
| Long-term | **Social LSTM / Trajectron++** | Learned pedestrian trajectory prediction replacing linear KF extrapolation |
| Long-term | **MLP trajectory scorer** | Sample K avoidance candidates, MLP scores each; trained in simulation |

---

## Dependencies

**ROS 2 packages:**
- `ackermann_msgs` — drive commands
- `curobot_msgs` — KinematicState
- `nav_msgs` — Path
- `vision_msgs` — Detection3DArray
- `visualization_msgs` — Marker, MarkerArray
- `tf2_ros` — TF tree
- `realsense2_camera` — depth + RGB
- `ros_deep_learning` — detectnet TensorRT wrapper
- `object_localizer` — 2D→3D projection
- `kf_object_tracking` — Kalman Filter tracker
- `nav2_bringup` — map server + AMCL

**Python:** `numpy`, `rclpy`, `math`

---

## References

1. Coulter, R. C. (1992). *Implementation of the Pure Pursuit Path Tracking Algorithm*. CMU-RI-TR-92-01.
2. van den Berg et al. (2011). *Reciprocal n-Body Collision Avoidance*. Springer Tracts in Advanced Robotics.
3. Intel RealSense ROS2 wrapper — `realsense2_camera`
4. NVIDIA Jetson Inference — `ros_deep_learning`, `detectnet`
5. ROS 2 Humble Documentation — https://docs.ros.org/en/humble/

---

## Version History

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-04-25 | Initial pure pursuit, basic obstacle stop |
| v2.0 | 2026-06-14 | Tuned path following (L_d, speeds, EMA, sqrt curve mapping) |
| v3.0 | 2026-06-14 | Full camera pipeline, KF tracker, lateral avoidance, trajectory visualisation |
