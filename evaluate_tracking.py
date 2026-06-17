#!/usr/bin/env python3
"""
Path tracking evaluation script.

Computes cross-track error (CTE) between planned path and actual driven path.

Usage:
    python3 evaluate_tracking.py <rosbag_dir>

Example:
    python3 evaluate_tracking.py tracking_run/
"""

import sys
import math
import numpy as np

try:
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
except ImportError:
    print("Install rosbags:  pip install rosbags")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not found — skipping plots (pip install matplotlib)")


# ── Read bag ─────────────────────────────────────────────────────────────────

def read_bag(bag_path):
    planned_x, planned_y = [], []
    actual_x,  actual_y  = [], []

    with Reader(bag_path) as reader:
        connections = {c.topic: c for c in reader.connections}

        if '/path' not in connections:
            print("[error] /path topic not found in bag"); sys.exit(1)
        if '/kinematic_state' not in connections:
            print("[error] /kinematic_state topic not found in bag"); sys.exit(1)

        path_saved = False
        for conn, ts, raw in reader.messages():
            if conn.topic == '/path' and not path_saved:
                msg = deserialize_cdr(raw, conn.msgtype)
                for pose in msg.poses:
                    planned_x.append(pose.pose.position.x)
                    planned_y.append(pose.pose.position.y)
                path_saved = True

            elif conn.topic == '/kinematic_state':
                msg = deserialize_cdr(raw, conn.msgtype)
                actual_x.append(msg.pose.position.x)
                actual_y.append(msg.pose.position.y)

    return (np.array(planned_x), np.array(planned_y),
            np.array(actual_x),  np.array(actual_y))


# ── Cross-track error ─────────────────────────────────────────────────────────

def nearest_distance(px, py, ref_x, ref_y):
    """Perpendicular distance from point (px,py) to nearest segment on ref path."""
    best = float('inf')
    for i in range(len(ref_x) - 1):
        ax, ay = ref_x[i],   ref_y[i]
        bx, by = ref_x[i+1], ref_y[i+1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx*dx + dy*dy
        if seg_len2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / seg_len2))
        cx, cy = ax + t*dx, ay + t*dy
        d = math.hypot(px - cx, py - cy)
        if d < best:
            best = d
    return best


def compute_cte(actual_x, actual_y, planned_x, planned_y):
    return np.array([
        nearest_distance(ax, ay, planned_x, planned_y)
        for ax, ay in zip(actual_x, actual_y)
    ])


# ── Report ────────────────────────────────────────────────────────────────────

def report(cte, track_width=0.40):
    mae  = np.mean(cte)
    rmse = np.sqrt(np.mean(cte**2))
    p95  = np.percentile(cte, 95)
    peak = np.max(cte)

    half_width = track_width / 2.0   # max allowable CTE before leaving lane

    print()
    print("══════════════════════════════════════════════")
    print("  PATH TRACKING EVALUATION")
    print("══════════════════════════════════════════════")
    print(f"  Actual positions recorded : {len(cte)}")
    print(f"  Mean CTE  (MAE)           : {mae*100:.1f} cm")
    print(f"  RMS CTE                   : {rmse*100:.1f} cm")
    print(f"  95th percentile CTE       : {p95*100:.1f} cm")
    print(f"  Peak CTE                  : {peak*100:.1f} cm")
    print()
    print(f"  Car half-width (budget)   : {half_width*100:.0f} cm")
    print(f"  MAE as % of lane budget   : {mae/half_width*100:.1f} %")
    print(f"  Peak as % of lane budget  : {peak/half_width*100:.1f} %")
    print()
    within_2cm  = np.sum(cte <= 0.02) / len(cte) * 100
    within_5cm  = np.sum(cte <= 0.05) / len(cte) * 100
    within_10cm = np.sum(cte <= 0.10) / len(cte) * 100
    print(f"  Within  2 cm : {within_2cm:.1f} %  of time")
    print(f"  Within  5 cm : {within_5cm:.1f} %  of time")
    print(f"  Within 10 cm : {within_10cm:.1f} %  of time")
    print("══════════════════════════════════════════════")
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(planned_x, planned_y, actual_x, actual_y, cte):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Path overlay
    ax1.plot(planned_x, planned_y, 'g-',  lw=2,   label='Planned path')
    sc = ax1.scatter(actual_x, actual_y, c=cte*100, cmap='hot_r',
                     s=4, label='Actual (coloured by CTE)')
    plt.colorbar(sc, ax=ax1, label='CTE (cm)')
    ax1.set_title('Planned vs Actual Path')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    ax1.legend(); ax1.set_aspect('equal')

    # CTE over samples
    ax2.plot(cte * 100, color='tomato', lw=1)
    ax2.axhline(np.mean(cte) * 100, color='red',    lw=2, linestyle='--', label=f'Mean {np.mean(cte)*100:.1f} cm')
    ax2.axhline(np.percentile(cte, 95) * 100, color='orange', lw=1.5, linestyle=':', label='95th %ile')
    ax2.set_title('Cross-Track Error over Time')
    ax2.set_xlabel('Sample'); ax2.set_ylabel('CTE (cm)')
    ax2.legend()

    plt.tight_layout()
    out = 'tracking_eval.png'
    plt.savefig(out, dpi=150)
    print(f"  Plot saved → {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    bag_path = sys.argv[1]
    print(f"Reading bag: {bag_path}")

    planned_x, planned_y, actual_x, actual_y = read_bag(bag_path)

    if len(planned_x) < 2:
        print("[error] No /path data found in bag"); sys.exit(1)
    if len(actual_x) < 2:
        print("[error] No /kinematic_state data found in bag"); sys.exit(1)

    print(f"Planned path: {len(planned_x)} waypoints")
    print(f"Actual path : {len(actual_x)} positions")

    cte = compute_cte(actual_x, actual_y, planned_x, planned_y)
    report(cte)

    if HAS_PLOT:
        plot(planned_x, planned_y, actual_x, actual_y, cte)
