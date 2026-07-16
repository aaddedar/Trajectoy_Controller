# control_car

Pure Pursuit Path Tracking Controller with Obstacle Avoidance for Autonomous Vehicles.

---

## Overview

The `control_car` package implements a **Pure Pursuit Path Tracking Controller** that receives a
reference path and kinematic state, then computes steering commands to make the vehicle follow the
path with real-time obstacle avoidance capabilities.

**Key Features:**
- Pure pursuit geometric path tracking algorithm
- Curvature-adaptive look-ahead distance (shrinks on curves, expands on straights)
- Velocity-adaptive lookahead via gain `k`
- Adaptive steering EMA filter: heavier smoothing on straights, lighter during avoidance
- Real-time person detection with map-aware lateral obstacle avoidance
- Inflated OccupancyGrid probed to select the roomier side and cap offset to available space
- Speed-adaptive avoidance magnitude: larger offset when person is approaching fast or crossing
- Direction lock prevents oscillation once avoidance side is committed
- Time-based avoidance hold: car stays offset for 6 s + 4 s ramp after person leaves corridor
- Full speed maintained during active avoidance (slowdown suppressed while passing)
- Hard stop suppressed while `_avoidance_active` car does not stop beside the obstacle
- Goal approach speed ramp - stops within 2 cm of target
- Safety interlock system with external stop signals
- 30 Hz control loop
- Ackermann steering model support

---

## Requirements

### US1.2 - Path Following

> **As a trajectory controller, I want the robot to follow straight and curved paths,
> so that motion is stable and predictable.**

#### AC01 - Command computation and output

The trajectory controller shall compute steering and target velocity commands from
`/path` and `/kinematic_state` and publish them on `/ackermann_drive`.


---

#### AC02 - Required subscriptions

The trajectory controller shall subscribe to the following ROS 2 topics:

| Topic | Message type | QoS | Purpose |
|---|---|---|---|
| `/path` | `Path` | TRANSIENT\_LOCAL | Reference waypoints from path planner |
| `/kinematic_state` | `KinematicState` | BEST\_EFFORT | Pose, heading, velocity from localization |
| `/ackermann_drive_feedback` | `AckermannDrive` | BEST\_EFFORT | Actual steering/speed from ROS2ARDUINO (logged only) |
| `/allowed_to_move` | `Bool` | RELIABLE | External safety interlock |
| `/tracked_objects` | `Detection3DArray` | BEST\_EFFORT | KF-tracked persons with velocity |
| `/map` | `OccupancyGrid` | TRANSIENT\_LOCAL | Inflated static map for avoidance direction probing |



---

#### AC03 - Target velocity

The trajectory controller shall command a target velocity of **1.4 m/s** on `/ackermann_drive`
when following straight, unobstructed segments of `/path`.

Speed control is **open-loop** - `/ackermann_drive_feedback` is received but not used as a
closed-loop feedback signal. No steady-state error guarantee applies without encoder-based
closed-loop speed control on the motor driver.

---

#### AC04 - Steering on straight segments

The trajectory controller shall produce near-zero steering commands when following a straight
segment of `/path` in the absence of localization noise. No hard ±2° limit is enforced in
software; the steering saturates at **±30°** (`STEER_LIMIT`). An adaptive EMA filter
(`α = 0.75` on normal driving) damps micro-corrections on straight sections.


---

#### AC05 - Control loop rate

The trajectory controller shall publish driving commands on `/ackermann_drive` at **30 Hz**
(`Ts = 1/30 s`).

---

#### AC06 - Safety stop conditions

The trajectory controller shall publish a command with target velocity **0 m/s** on
`/ackermann_drive` if any of the following conditions is true:

- `/allowed_to_move = False`
- A person is detected within **0.4 m** (`person_stop_dist`) in the forward corridor
  **and** `_avoidance_active` is `False`

Additional speed behaviour:

- Speed reduction (proportional slowdown to 0) begins at **1.5 m** (`person_slowdown_dist`)
- Both speed reduction and hard stop are **suppressed** while `_avoidance_active = True` so
  the car passes alongside an obstacle rather than stopping beside it

---

#### AC07 - Stale or missing topic handling

The trajectory controller shall publish a command with target velocity **0 m/s** on
`/ackermann_drive` if any of the following is true:

- `/path` has not been received, or contains fewer than 2 waypoints
- `/kinematic_state` has not been received since node startup
- `/kinematic_state` has not been received for more than **3 s** (`kinematic_state_timeout`)


---

### US3.9 - Tracked Object Response

> **As a trajectory controller, I want to receive tracked objects from the object tracker
> component and respond with progressive speed reduction, lateral avoidance steering, or a
> full stop based on their distance and movement, so that the robot navigates safely around
> people without unnecessary interruptions to reaching its target location.**

#### AC01 - Detection input and corridor filter

The trajectory controller shall subscribe to `/tracked_objects` (QoS: BEST\_EFFORT) and
process only detections where `class_id` is `'person'` or `'1'`, located within the forward
corridor:

- **longitudinal:** `cx > 0.1 m` (strictly ahead of the car)
- **lateral:** `|cy| ≤ car_width/2 + path_corridor_margin` (0.20 + 1.0 = **1.2 m** half-width)

All other detections are silently ignored.

---

#### AC02 - Velocity-adjusted distances

The trajectory controller shall extend both the speed-reduction distance and the avoidance
trigger distance by `max(0, −person_vlong) × avoidance_predict_t` when the person is moving
toward the robot. Only the person's **longitudinal velocity** is used - the car's own velocity
is excluded to prevent stop/accelerate oscillation.

```
person_approach     = max(0, −person_vlong)
effective_slow_dist = person_slowdown_dist + person_approach × avoidance_predict_t
early_trigger_dist  = avoidance_start_dist + person_approach × avoidance_predict_t
```

---

#### AC03 - Progressive speed reduction

The trajectory controller shall reduce speed linearly from `command_speed` to `0 m/s` as the
person's distance decreases from `effective_slow_dist` to `person_stop_dist`:

```
factor    = (dist − person_stop_dist) / (effective_slow_dist − person_stop_dist)
cmd_speed = command_speed × factor
```

Speed reduction is **bypassed** while `_avoidance_active = True` - the car runs at full
`command_speed` while steering around a person to avoid stopping beside them.


---

#### AC04 - Full stop

The trajectory controller shall command **0 m/s** and set `obstacle_stop_requested = True`
when a person's longitudinal distance is ≤ `person_stop_dist` (default 0.4 m) **and**
`_avoidance_active = False`. When `_avoidance_active = True` the hard stop is suppressed so
the car passes alongside the person rather than stopping beside them.


---

#### AC05 - Lateral avoidance direction

The trajectory controller shall activate lateral avoidance when the person's distance is between
`person_stop_dist` and `early_trigger_dist`. Direction is selected in priority order:

**Step 1 - Map clearance dominates:** if one side of the path has ≥ 0.30 m more free space
(probed on the inflated OccupancyGrid perpendicular to the path tangent at the predicted
obstacle position) the car steers toward the roomier side regardless of where the person is.

**Step 2 - Person position tiebreaker** (both sides within 0.30 m of each other):

| Predicted lateral position `lat_predicted = cy + vlat × predict_t` | Direction |
|---|---|
| `> +0.15 m` (person predicted left) | Steer **right** |
| `< −0.15 m` (person predicted right) | Steer **left** |

---

#### AC06 - Avoidance offset ramp rates

The lateral offset ramps using **asymmetric rates** to reach full offset quickly and return
to path gradually:

```
RAMP_UP   = 2.0 / 30  m per tick  →  full offset in ≈ 0.3 s  (urgent)
RAMP_DOWN = 0.15 / 30 m per tick  →  back to path in ≈ 4 s   (gradual)
```

`RAMP_DOWN` applies only when `avoidance_target = 0` (person cleared). Direction changes use
`RAMP_UP` in both directions.

> **Note - correction to original AC06:** The original specified a single symmetric rate of
> 0.5/30 m per tick. The implementation uses **2.0/30 ramp-up** (safety-critical, fast
> response) and **0.15/30 ramp-down** (prevents snap-back before the person has fully passed).


---

#### AC07 - Detection timeout and state reset

If no `/tracked_objects` message is received for **1 second** (`obstacle_timeout`) while any
obstacle state is active, the trajectory controller shall reset all obstacle state and resume
normal path following:

1. `obstacle_stop_requested = False`
2. `closest_person_dist = inf`
3. `_avoidance_active = False`, `_clear_since_time = None`, `avoidance_target = 0.0`
4. Publish cleared state on `/obstacle_information` (Bool)

`/obstacle_information` is also published after every detection update so external nodes
always have the current stop state.

---

## Development History

### Phase 1 - Basic Path Following (v1.0 – v2.0)
- Implemented pure pursuit algorithm with bicycle model kinematics
- Added curvature-adaptive lookahead: shorter `L_d` on curves prevents overshoot
- Added sqrt curve-speed mapping: drops speed more steeply on curves to tighten turns
- Goal approach ramp: linear deceleration to zero over last 0.5 m

### Phase 2 - Camera Pipeline and Obstacle Avoidance (v3.0)
- Integrated DetectNet → 3D localizer → Kalman Filter tracker pipeline
- Lookahead-point offset method: shifts pure-pursuit waypoint laterally to steer around persons
- Avoidance direction derived from predicted lateral position (`vlat × predict_t`)
- Hard stop at 0.4 m; slowdown from 1.5 m

### Phase 3 - Avoidance Robustness (v4.0)
- Curvature-adaptive `L_d` prevents wide turns at speed
- Obstacle timeout: resumes after 1.0 s with no detections

### Phase 4 - Avoidance Correctness and Stability (current)

**Path following fixes:**
- Fixed lookahead double-count bug (`LF` was computing `k×v` twice; now `LF = L_d` as already adaptive)
- Adaptive steering EMA: `alpha = 0.75` on normal driving (smooth straight), `alpha = 0.50` during avoidance (fast response)

**Map-aware avoidance direction:**
- Subscribes to `/map` (inflated OccupancyGrid, latched QoS)
- `_map_free_clearance()` probes left and right perpendicular to the path tangent at the predicted obstacle position, scanning up to 1.0 m
- **Clearance dominates direction**: if one side has ≥ 0.30 m more clearance than the other, always go to the roomier side regardless of where the person is standing
- Person lateral position (`lat_predicted`) is only a tiebreaker when both sides are within 0.30 m of each other
- Prevents the car steering toward a wall just because the person is slightly on the opposite side

**Map-capped adaptive avoidance offset:**
- `ideal_offset = max(avoidance_clearance, car_width/2 + 0.40 m) + speed_bonus`
- `speed_bonus = person_approach × 0.30 + |vlat| × 0.20` (larger offset when person moves toward car or crosses laterally; capped at 1.2 m total)
- Actual commanded offset `= min(ideal_offset, map_clearance − 0.05 m)` - car uses all available corridor space without hitting walls

**Direction lock:**
- Once avoidance side is committed, the sign of `avoidance_target` is locked
- Side only flips if `|lat_predicted| > 0.25 m` on the new side AND the new side has map clearance
- Prevents noisy `lat_predicted` from flipping direction each frame, which previously kept `avoidance_offset` near zero

**Avoidance state machine:**
- `_avoidance_active` flag set on first avoidance trigger; cleared only after confirmed clear
- While `_avoidance_active`: hard stop suppressed, speed slowdown bypassed, direction locked
- Clear confirmed by time: `avoidance_clear_hold` (default 6.0 s) of continuous absence from corridor
- If person re-enters corridor before timer expires, timer resets and avoidance resumes immediately
- Asymmetric ramp: fast ramp-up (0.3 s to full offset), slow ramp-down (≈ 4 s back to path)

**Stop logic fixes:**
- Hard stop suppressed while `_avoidance_active` - car passes beside person rather than stopping next to them
- Speed limiter (`_compute_safe_speed`) returns `command_speed` during active avoidance - prevents near-zero speed while alongside person

---

## Mathematical Foundation

### Pure Pursuit Algorithm

Pure pursuit is a geometric path tracking method that calculates the required steering angle
based on the look-ahead point on the reference path.

#### 1. Look-ahead Point Selection

Find the closest point on the reference path to the robot position, then move forward until
cumulative distance reaches:

```
L_f = (L_d + k × |v|) × (1 − 0.6 × c),   floored at L_d × 0.5
```

Where:
- `L_d = 0.30` m - base look-ahead distance
- `k = 0.3` - velocity gain coefficient
- `v` - current robot velocity (m/s)
- `c = min(1, |δ_prev| / δ_max)` - curvature factor from previous steering angle

**Curvature-adaptive behaviour:**

| Situation          | Curve factor `c` | Effective `L_f` at 1.4 m/s |
|--------------------|------------------|-----------------------------|
| Straight           | 0.0              | 0.72 m                      |
| Gentle curve (10°) | 0.33             | 0.57 m                      |
| Medium curve (20°) | 0.67             | 0.43 m                      |
| Tight curve (30°)  | 1.0              | 0.30 m → floor 0.15 m      |

#### 2. Steering Angle Computation

Using the Ackermann steering kinematic model:

```
δ = arctan(2L sin α / (L_f + ε)),   δ ∈ [−30°, +30°]
```

Where:
- `L = 0.30` m - vehicle wheelbase
- `α` - angle from car heading to lookahead point
- `ε = 1×10⁻⁵` - numerical stability guard

#### 3. Steering Smoothing (Adaptive EMA)

EMA weight adapts to driving mode to balance smoothness vs. avoidance response:

```
smoothed_delta = α × prev + (1 − α) × raw

α = 0.75   (normal driving - suppresses micro-corrections on straights)
α = 0.50   (avoidance active - faster response to steer around obstacle)
```

#### 4. Avoidance Offset Ramp

Two separate rates prevent snapping back to path before the person has fully passed:

```
ramp_up   = 2.0 / 30   m per tick   → full offset in ~0.3 s  (urgent)
ramp_down = 0.15 / 30  m per tick   → back to path in ~4 s   (gradual)
```

`ramp_down` only applies when `avoidance_target = 0`; direction changes use `ramp_up`.

#### 5. Cross-Track Error (CTE)

Computed every control tick as the signed lateral distance from the car to the nearest path segment:

```
segment  = path[ni+1] − path[ni]                    (tangent vector at nearest point ni)
left_n   = (−seg_y, seg_x) / |segment|              (unit left normal, 90° CCW from tangent)
CTE      = (car_pos − path[ni]) · left_n

CTE > 0 → car is LEFT  of path centreline
CTE < 0 → car is RIGHT of path centreline
```

CTE is used in the avoidance effective-offset clamp (see Avoidance section) to prevent commanding
a larger lateral shift than the remaining available window when the car has already moved partway
toward the avoidance target.

#### 6. Goal Stopping

```
d_goal = sqrt((x_goal − x)² + (y_goal − y)²)
```

Car stops when `d_goal ≤ 0.02 m`. Speed ramps linearly to zero over the last 0.5 m.

---

## Speed Control

Four independent speed limiters applied as `min(v1, v2, v3, v4)`:

### 1. Curve speed (current steering angle)

```
v_curve = v_max − (v_max − v_min) × (|δ| / δ_max)^0.5
```

The sqrt exponent drops speed more steeply even at moderate steer angles, helping tighten curves.

### 2. Ahead curvature speed

Scans next 30 path points (~1.5 m ahead) and pre-brakes before curves are reached.

### 3. Safe speed (obstacle distance)

Bypassed entirely when `_avoidance_active = True` - car runs at `command_speed` while passing alongside an obstacle. Only active during approach (before avoidance engages) and after avoidance is fully cleared.

```
v_safe = 0                                              if d ≤ d_stop
       = v_max × (d − d_stop) / (d_slow − d_stop)      if d_stop < d ≤ d_slow  (avoidance inactive only)
       = v_max                                          if d > d_slow  OR  avoidance active
```

### 4. Goal approach ramp

```
v_goal = v_max × d_goal / 0.5,   when d_goal < 0.5 m
```

### Asymmetric EMA (anti-jerk)

```python
if raw_speed < smoothed_speed:
    smoothed_speed = raw_speed                               # instant brake
else:
    smoothed_speed = 0.75 × smoothed_speed + 0.25 × raw    # ramp-up (~0.5 s)
```

---

## Obstacle Avoidance

### Why Geometric Avoidance

The car uses a geometric lookahead-point offset rather than learning-based or
model-predictive approaches. The rationale:

| Approach | Why not used |
|---|---|
| **Deep learning (RL / imitation)** | Requires large training data and a calibrated simulator. Behaviour is a black box - unsafe on physical hardware without extensive validation. |
| **Model Predictive Control (MPC)** | Requires an accurate dynamic model and solves an optimisation at every 33 ms tick. Too computationally expensive for Jetson; vehicle dynamics are not well characterised. |
| **Dynamic Window Approach / VFH** | Requires a 2-D occupancy grid costmap with moving-obstacle inflation. We have a single 3-D bounding box from the KF tracker, not a full costmap. |
| **RRT / A\* replanning** | Replanning at 30 Hz is expensive and requires a map with inflation around moving obstacles. |
| **Pure geometric (chosen)** | Deterministic, explainable, no training data needed. Works directly with KF tracker output. Runs in < 1 ms per tick. Safe to deploy on physical hardware with an auditable decision rule. |

### How It Works

```
Detection → Predict position → Probe map → Choose roomier side → Cap offset to clearance → Shift lookahead → Pure pursuit steers
```

1. KF tracker outputs filtered position `(cx, cy)` and velocity `(vx_map, vy_map)` for each person.
2. **Constant Velocity Model (CVM)** predicts future position over `avoidance_predict_t = 1.5 s`:
   `pred = (x_map + vx_map × predict_t, y_map + vy_map × predict_t)`
   The KF tracker smooths the velocity estimate; this node assumes constant velocity over the horizon.
   No acceleration model is used - at typical walking speeds (≤ 1.5 m/s) over 1.5 s the error is small.
3. Closest path point to predicted position found; path tangent computed there.
4. Inflated OccupancyGrid probed in left and right perpendicular directions (up to 1.0 m).
5. Direction chosen by map clearance (roomier side preferred); person position used only as tiebreaker.
6. Commanded offset: `_usable(clear) = min(ideal_offset, clear − 0.05 m)` - uses all available corridor width.
7. Lookahead point shifted perpendicularly; pure pursuit steers toward it.

### Avoidance Offset Formula

```
base_offset    = max(avoidance_clearance, car_width/2 + 0.40 m)           ≥ 0.6 m
speed_bonus    = person_approach × 0.30 + |vlat| × 0.20                   adaptive
ideal_offset   = min(base_offset + speed_bonus, 1.2 m)                    capped
map_offset     = min(ideal_offset, map_clearance − 0.05 m)                map-limited

CTE clamp (applied in control loop before steering):
  cte_in_dir   = CTE × sign(map_offset)                                   displacement already achieved
  window       = |avoidance_target| + 0.1 m                               total allowed window
  available    = max(0, window − max(0, cte_in_dir))                      remaining shift needed
  effective    = clamp(map_offset, −available, +available)                final applied offset
```

The CTE clamp prevents the lookahead point from being shifted further than necessary when the car
has already moved partway toward the avoidance target, avoiding over-correction.

| Person state | `ideal_offset` | Effect |
|---|---|---|
| Stationary | 0.60 m | Baseline |
| Approaching at 0.5 m/s | 0.75 m | More room needed - they'll be central at arrival |
| Approaching at 1.0 m/s | 0.90 m | Higher urgency |
| Crossing at 0.5 m/s | 0.70 m | Lateral bonus |
| Moving away | 0.60 m | No bonus - they're clearing |

### Direction Selection

```
left_clear, right_clear  ← map probe at predicted obstacle position

if left_clear > right_clear + 0.30 m:
    go LEFT   (map dominates - ignore person side)
elif right_clear > left_clear + 0.30 m:
    go RIGHT  (map dominates - ignore person side)
else:
    go AWAY from person's predicted lateral position  (tiebreaker)
    if person centered (|lat_predicted| < 0.15 m): go to roomier side
```

Once a direction is committed while `_avoidance_active`, the side is **locked**. It only changes if
the person's predicted lateral position crosses > 0.25 m to the other side AND that side has map
clearance - prevents noisy `lat_predicted` from flipping direction every frame.

### State Machine

```
NORMAL   → person in corridor, cx ≤ trigger_dist                   → AVOIDING
AVOIDING → person leaves corridor (cx < 0 or |cy| > corridor_half) → HOLD
HOLD     → 6.0 s elapsed with no person in corridor                → RETURNING
RETURNING→ avoidance_offset ramped back to 0 (≈ 4 s)              → NORMAL

Hard stop: cx ≤ person_stop_dist AND |cy| < car_half + 0.10 m AND NOT _avoidance_active
```

Avoidance trigger distance extends dynamically with person approach speed (CVM, same horizon):

```
trigger_dist = avoidance_start_dist + person_approach × avoidance_predict_t
```

> **Note on timeouts:**
> - *Obstacle timeout* (1.0 s): clears stale detection when tracker loses person entirely - resets all avoidance state
> - *Clear hold timer* (`avoidance_clear_hold`, 6.0 s): keeps lateral offset after person leaves corridor - prevents snap-back while passing alongside
> - These are independent mechanisms serving different failure modes

---

## ROS 2 Interface

### Node Information

| Field | Value |
|---|---|
| Node name | `pure_pursuit_node` |
| Launch command | `ros2 launch control_car pure_persuit.launch.py` |
| Control loop | 30 Hz |

> **Note:** The filename `pure_persuit.launch.py` is intentional - it matches the file on disk.

### Publishers

| Topic | Message Type | QoS | Description |
|---|---|---|---|
| `/ackermann_drive` | `AckermannDrive` | RELIABLE | Steering angle & speed commands |
| `/obstacle_information` | `Bool` | RELIABLE | True when hard-stopped for obstacle |
| `/pure_pursuit/lookahead_marker` | `Marker` | 10 | Lookahead sphere + nearest point + heading arrow |
| `/pure_pursuit/avoidance_path` | `Path` | 10 | Shifted avoidance trajectory (yellow when active, cyan when returning) |
| `/pure_pursuit/original_path_ahead` | `Path` | 10 | Original unshifted path ahead |
| `/pure_pursuit/avoidance_viz` | `MarkerArray` | 10 | Coloured line strips + offset arrow |

### Subscribers

| Topic | Message Type | QoS | Description |
|---|---|---|---|
| `/path` | `Path` | TRANSIENT_LOCAL | Reference waypoints from planner |
| `/kinematic_state` | `KinematicState` | BEST_EFFORT | Robot pose, orientation, velocity |
| `/ackermann_drive_feedback` | `AckermannDrive` | BEST_EFFORT | Actual steering/speed feedback |
| `/allowed_to_move` | `Bool` | RELIABLE | Safety signal (False = STOP) |
| `/tracked_objects` | `Detection3DArray` | BEST_EFFORT | KF-tracked persons with velocity |
| `/map` | `OccupancyGrid` | TRANSIENT_LOCAL | Inflated static map for avoidance direction probing |

### KinematicState Message Fields

`KinematicState` is defined in `curobot_msgs`. Fields used by this node:

| Field | Type | Description |
|---|---|---|
| `pose_with_covariance.pose.position.x/y` | float64 | Position in map frame (m) |
| `pose_with_covariance.pose.orientation` | Quaternion | Heading (converted to yaw) |
| `twist_with_covariance.twist.linear.x/y` | float64 | Velocity vector (speed = hypot) |

### Detection3DArray Message Convention

Published by the KF tracker. Fields used by this node:

| Field | Meaning |
|---|---|
| `bbox.center.position.x/y` | Map-frame position of person |
| `results[0].hypothesis.class_id` | Label - must be `"person"` or `"1"` |
| `results[0].pose.pose.position.x/y` | Map-frame velocity (vx, vy) from KF state |

### Expected TF Frames

| Frame | Role |
|---|---|
| `map` | Global reference frame for path, localization, and `/map` |
| `base_link` | Robot body frame |

The node does not perform TF lookups directly - pose is received via `/kinematic_state`.

---

## Control Parameters

Set in `launch/pure_persuit.launch.py`. Override at runtime with:
```bash
ros2 param set /pure_pursuit_node <param> <value>
```

### Path Following

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `command_speed` | 1.4 | m/s | Maximum forward speed |
| `min_curve_speed` | 0.9 | m/s | Minimum speed in tight curves |
| `L_d` | 0.30 | m | Base look-ahead distance |
| `k` | 0.3 | - | Velocity gain for adaptive lookahead |
| `invert_steering` | False | - | Negate steering output (mirror installation) |

### Obstacle Safety

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `person_stop_dist` | 0.4 | m | Hard stop distance (only when not avoiding) |
| `person_slowdown_dist` | 1.5 | m | Begin speed reduction (only when not avoiding) |
| `avoidance_start_dist` | 1.5 | m | Base lateral avoidance trigger distance |
| `avoidance_clearance` | 0.3 | m | Minimum extra lateral gap beyond car + 0.40 m |
| `avoidance_predict_t` | 1.5 | s | KF velocity prediction horizon |
| `avoidance_clear_hold` | 6.0 | s | Hold offset after person leaves corridor before returning to path |
| `map_topic` | `/map` | - | OccupancyGrid topic for free-space probing |
| `map_free_threshold` | 50 | 0–100 | Occupancy value below which a cell is considered free |

### Internal Constants (not ROS parameters)

| Constant | Value | Description |
|---|---|---|
| `STEER_LIMIT` | 0.5236 rad (30°) | Max steering angle |
| Control loop | 30 Hz | `Ts = 1/30` s |
| Wheelbase `L` | 0.30 m | Bicycle model |
| Car width | 0.40 m | Corridor filter + avoidance formula |
| Goal tolerance | 0.02 m | Stop within 2 cm of goal |
| Goal slowdown dist | 0.5 m | Begin speed ramp to zero |
| Obstacle timeout | 1.0 s | Clear stale detection after this long |
| Kinematic timeout | 3.0 s | Stop if no localization for this long |
| Path smoothing | window=7, iter=3 | Moving average passes |
| Path spacing | 0.05 m | Densification step |
| Ramp-up rate | 2.0/30 m/tick | Avoidance offset build-up (~0.3 s to full) |
| Ramp-down rate | 0.15/30 m/tick | Avoidance offset return (~4 s back to path) |
| Map probe distance | max(clearance+0.4, 1.0) m | How far to scan for walls on each side |
| Clearance margin | 0.30 m | Min difference for map clearance to override person-side preference |
| Direction flip threshold | 0.25 m | Min `lat_predicted` crossing needed to change locked avoidance side |
| Steer alpha (normal) | 0.75 | EMA weight on previous steering - smooth on straights |
| Steer alpha (avoidance) | 0.50 | EMA weight on previous steering - responsive during avoidance |

---

## Class Architecture

### `HagenRobot`
Robot state container with kinematic properties.
```python
HagenRobot(x=0, y=0, theta=0, v=0, L=0.3)
```
Fields: `x`, `y` (position, m), `theta` (heading, rad), `v` (velocity, m/s), `L` (wheelbase, m).

### `PurePursuitController`
Implements pure pursuit algorithm with curvature-adaptive lookahead.
- `pure_pursuit_control()` - compute steering angle and target index
- `look_ahead_point_index()` - find look-ahead point on path

### `PurePursuitNode` (ROS 2 Node)
Main control node.

| Method | Description |
|---|---|
| `control_loop()` | 30 Hz control execution - steering, speed, avoidance ramp |
| `path_callback()` | Receives, smooths, and densifies reference path |
| `kinematic_state_callback()` | Updates robot pose, heading, velocity |
| `objects_in_map_frame_callback()` | Person detection, map probing, avoidance decision |
| `_decide_avoidance_action()` | Computes direction + magnitude from map clearance and person kinematics |
| `_map_free_clearance()` | Probes inflated OccupancyGrid for free space in a direction |
| `_clear_obstacle()` | Time-based hold logic - only releases avoidance after 6 s confirmed clear |
| `_compute_safe_speed()` | Obstacle-based speed limit (bypassed when `_avoidance_active`) |
| `_compute_adaptive_L_d()` | Curvature-adaptive lookahead distance |
| `allowed_to_move_callback()` | Safety interlock |
| `_check_obstacle_timeout()` | 0.5 Hz watchdog - clears stale detections after 1 s |
| `_publish_avoidance_viz()` | RViz trajectory visualisation |

---

## Performance Results

### Test Run: 7 m Path with One Obstacle Encounter

#### Per-Metre Segment Performance

| Segment | Time (s) | Avg Speed (m/s) | Cmd Speed (m/s) | Speed Ratio | Phase |
|---|---|---|---|---|---|
| 0 → 1 m | 6.1 | 0.16 | ~1.20 | 13% | Slow start / init |
| 1 → 2 m | 3.2 | 0.32 | ~1.35 | 24% | Straight section |
| 2 → 3 m | 0.8 | 1.30 | ~1.30 | 100% | ✅ Fastest - near commanded |
| 3 → 4 m | 2.4 | 0.41 | ~1.15 | 36% | Curve entry |
| 4 → 5 m | 3.3 | 0.30 | ~0.95 | 32% | Curve mid |
| 5 → 6 m | 3.4 | 0.29 | ~0.90 | 32% | Straight / goal approach |
| **Total** | **19.2 s** | **0.36 avg** | **1.13 avg** | **32% avg** | **7.0 m covered** |

#### CTE (Path Deviation) Summary

| Phase | Distance | CTE Range | Side | Status |
|---|---|---|---|---|
| Start | 0 – 1 m | 0.8 – 2.2 cm | R | ✅ GOOD |
| Straight 1 | 1 – 2 m | 0.8 – 1.8 cm | R | ✅ GOOD |
| Fast section | 2 – 3 m | 4.3 – 6.3 cm | L | ⚠ OK (speed spike) |
| Curve recovery | 3 – 4.3 m | 0.7 – 3.4 cm | L | ✅ GOOD |
| Straight 2 | 4.3 – 5.5 m | 0.3 – 2.9 cm | R | ✅ GOOD (best: 0.3 cm) |
| Goal failure | 5.5 – 7 m | 9.8 – 23.3 cm | L | ❌ BAD (stuck at max steer) |

CTE statistics (0 – 6 m valid range):

| Metric | Value |
|---|---|
| Best CTE | 0.3 cm |
| Worst CTE (valid section) | 6.3 cm |
| Typical CTE | 0.7 – 2.2 cm |
| % time within 5 cm | ~85% |
| % time within 10 cm | ~95% |
| Lane budget (half-width) | 20 cm |

---

## Known Issues and Limitations

### 1. Speed severely under-commanded
The motor controller runs open-loop - no wheel speed encoder or velocity feedback to the ESC.
At 1.4 m/s commanded the car typically achieves only 0.3 – 0.5 m/s (32% average). This will
not be resolved without closed-loop speed control (encoder + PID on the Arduino).

### 2. Wide turns on curves
Localization noise causes the heading estimate to oscillate, making the car track a wider arc
than the path. Tuning direction: increase `L_d` (0.30 → 0.40 m) or decrease `k` (0.3 → 0.2).

### 3. Steering oscillation at low speed
At low actual speeds the control loop runs fast relative to vehicle motion, amplifying small
heading errors. The 0.75/0.25 EMA damps this but does not eliminate it with noisy localization.
Increasing `L_d` reduces sensitivity.

### 4. Goal not reached (7 m test run)
At 6.1 m the car hit maximum left steer (−30°) and stopped with CTE = 22.4 cm. The path had
ended but localization placed the car 0.9 m from the goal. Pure pursuit was chasing a goal
point the steering geometry could not reach - a localization drift issue, not a controller bug.

### 5. Obstacle avoidance requires forward motion
Lateral avoidance shifts the lookahead waypoint; pure pursuit then steers toward it. If the
car is stationary, steering has no effect. The car must be moving to execute a lateral maneuver.

### 6. Map probe depends on inflated map quality
`_map_free_clearance()` reads the static inflated OccupancyGrid. Dynamic obstacles (other
people, chairs) are not in this map. In a corridor where both static sides appear clear but
one side is physically blocked by another person, the car may choose that side. The KF
tracker only provides the closest person for direction, not all obstacles.

---

## Safety Features

1. **External stop signal** - `/allowed_to_move = False` stops vehicle immediately
2. **Person detection** - hard stop at 0.4 m (when not in avoidance); lateral avoidance from 1.5 m
3. **Avoidance active suppression** - hard stop and slowdown disabled while `_avoidance_active`; car passes, not stops
4. **Goal tolerance** - stops within 2 cm of target waypoint
5. **Goal speed ramp** - linearly decelerates to zero over last 0.5 m
6. **Path validation** - requires ≥ 2 waypoints before moving
7. **State validation** - waits for `/kinematic_state`; stops if lost for > 3 s
8. **Obstacle timeout** - resumes and resets avoidance state 1.0 s after last detection
9. **Avoidance hold** - keeps lateral offset for 6.0 s + 4 s ramp after person leaves corridor
10. **Direction lock** - committed avoidance side does not oscillate due to noisy tracking

---

## Installation

```bash
# Install ROS 2 dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build the package
colcon build --packages-select control_car
source install/setup.bash
```

**Python dependencies:** `numpy`, `rclpy`, `math` (all standard with ROS 2 Humble).

---

## Launch Sequence

```bash
# Terminal 1 - System (localization, nav2, arduino, camera TF)
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch ~/ros2_ws/system_startup.launch.py

# Terminal 2 - Camera pipeline
pkill -9 -f "detectnet"; pkill -9 -f "object_localizer"; pkill -9 -f "realsense2_camera"; sleep 2
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch object_localizer camera.launch.py

# Terminal 3 - KF object tracker
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 run kf_object_tracking kf_object_tracker

# Terminal 4 - ros2arduino (start BEFORE pure pursuit - give Arduino 2 s to boot)
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 run ros2arduino ros2arduino_node &
sleep 2

# Terminal 5 - Pure pursuit controller
export ROS_DOMAIN_ID=11
source ~/ros2_ws/install/setup.bash
ros2 launch control_car pure_persuit.launch.py
```

### Monitor Topics

```bash
ros2 topic echo /ackermann_drive        # steering commands
ros2 topic echo /obstacle_information   # obstacle status
ros2 topic echo /kinematic_state        # robot state
ros2 topic hz /tracked_objects          # expect ~10–30 Hz
ros2 topic hz /map                      # expect 1 Hz (latched, arrives once)
```

### First-Run Checklist

On a healthy start you should see the following in the pure pursuit terminal:

1. `[pure_pursuit_node] Waiting for /kinematic_state...` - until localization is up
2. `[pure_pursuit_node] Waiting for /path...` - until planner sends waypoints
3. `[pure_pursuit_node] Path received: N waypoints` - controller activates
4. `/ackermann_drive` topic begins publishing at 30 Hz

If the node stays silent after step 3, check `/allowed_to_move` - it must be `True`.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Vehicle not moving | Check `/allowed_to_move = False`; verify `/path` received; confirm `/kinematic_state` publishing |
| Steering oscillates left–right | Increase `L_d` (try 0.40 m) or decrease `k` (try 0.2); check localization noise |
| Car goes wide on curves | Reduce `k` or slightly increase `L_d` |
| Car stops short of goal | Check localization accuracy; goal tolerance is 2 cm |
| Unexpected obstacle stop | Check `/tracked_objects`; obstacle clears after 1.0 s timeout |
| Car avoids wrong side (toward wall) | Check `/map` is being received; inflated map must be available for probe to work |
| Car returns to path too quickly | Increase `avoidance_clear_hold` in launch file (default 6.0 s) |
| Car stays offset too long | Decrease `avoidance_clear_hold` |
| No `/map` received | Check nav2 or map_server is publishing on the correct topic; default topic is `/map` |
| ros2arduino crashes | Power brownout: Arduino + ESC share power rail. Start ros2arduino before pure pursuit. Sustained speed > 1.0 m/s may cause USB disconnect - power Arduino via USB separately from motor ESC |
| No `/kinematic_state` after 3 s | Node stops the car automatically; verify localization stack is running |

---

## Dependencies

**ROS 2 packages:**
- `ackermann_msgs` - drive commands
- `curobot_msgs` - KinematicState
- `nav_msgs` - Path, OccupancyGrid
- `vision_msgs` - Detection3DArray
- `visualization_msgs` - Marker, MarkerArray
- `geometry_msgs` - PoseStamped, Point
- `std_msgs` - Bool

**Python:** `numpy`, `rclpy`, `math`

---

## References

1. Coulter, R. C. (1992). *Implementation of the Pure Pursuit Path Tracking Algorithm*. CMU-RI-TR-92-01.
2. ROS 2 Humble Documentation - https://docs.ros.org/en/humble/
3. Snider, J. M. (2009). *Automatic Steering Methods for Autonomous Automobile Path Tracking*. CMU-RI-TR-09-08.
