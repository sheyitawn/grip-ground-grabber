import sys
import time
import math
from collections import deque

import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ================= CONFIG =================
DEFAULT_PORT = "COM17"
BAUD = 115200
SERIAL_TIMEOUT = 0.02
UI_HZ = 60.0

# Serial mapping:
# A0 = pinkie
# A1 = ring
# A2 = middle
# A3 = index
# D6 = thumb (digital)

# Adjust these to your real hall sensor range
ANALOG_OPEN = 1800
ANALOG_CLOSED = 2200

# Smoothing
SMOOTH_WINDOW = 5

# Bend limits
FINGER_MAX_BEND_DEG = 70
THUMB_OPEN_DEG = 15
THUMB_CLOSED_DEG = 55

# Drawing style
BG_COLOR = "black"
LINE_COLOR = "white"
JOINT_COLOR = "white"
TEXT_COLOR = "white"
LINE_WIDTH = 2.2
JOINT_SIZE = 55

# Tiny z offsets so the 3D plot still renders,
# but visually it looks like your flat sketch.
Z_BASE = 0.0
# ==========================================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_ints_csv(line: str):
    parts = line.strip().split(",")
    if len(parts) < 5:
        return None
    out = []
    for p in parts[:5]:
        try:
            out.append(int(float(p.strip())))
        except ValueError:
            return None
    return out


def normalize_analog(v, open_val=ANALOG_OPEN, closed_val=ANALOG_CLOSED):
    if closed_val == open_val:
        return 0.0
    t = (v - open_val) / float(closed_val - open_val)
    return clamp(t, 0.0, 1.0)


class SmoothedSensor:
    def __init__(self, window=5, initial=0.0):
        self.buf = deque([initial] * window, maxlen=window)

    def update(self, value):
        self.buf.append(value)
        return sum(self.buf) / len(self.buf)


def rotate_2d(vx, vy, angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return vx * c - vy * s, vx * s + vy * c


def make_chain(base_xy, lengths, base_angle_deg, bends_deg):
    """
    Builds a 2D finger chain from a base point.

    base_angle_deg controls the overall finger direction.
    bends_deg is a list of joint bends added progressively.
    """
    x, y = base_xy
    pts = [(x, y)]

    angle = math.radians(base_angle_deg)

    for i, L in enumerate(lengths):
        if i > 0:
            angle -= math.radians(bends_deg[i - 1])
        dx = L * math.cos(angle)
        dy = L * math.sin(angle)
        x += dx
        y += dy
        pts.append((x, y))

    return pts


def finger_bends(sensor_norm):
    total = sensor_norm * FINGER_MAX_BEND_DEG
    # first bend most obvious, then smaller down chain
    return [
        total * 0.50,
        total * 0.32,
        total * 0.18,
    ]


def thumb_bends(digital_value):
    total = THUMB_CLOSED_DEG if digital_value else THUMB_OPEN_DEG
    return [
        total * 0.65,
        total * 0.35,
    ]


def lift_to_3d(points_2d, z=0.0):
    return [(x, y, z) for x, y in points_2d]


def draw_polyline(ax, pts3d, color=LINE_COLOR, lw=LINE_WIDTH):
    xs = [p[0] for p in pts3d]
    ys = [p[1] for p in pts3d]
    zs = [p[2] for p in pts3d]
    ax.plot(xs, ys, zs, color=color, linewidth=lw)


def draw_joints(ax, pts3d, color=JOINT_COLOR, size=JOINT_SIZE):
    xs = [p[0] for p in pts3d]
    ys = [p[1] for p in pts3d]
    zs = [p[2] for p in pts3d]
    ax.scatter(xs, ys, zs, color=color, s=size)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT

    try:
        ser = serial.Serial(port, BAUD, timeout=SERIAL_TIMEOUT)
    except Exception as e:
        print(f"Could not open {port}: {e}")
        sys.exit(1)

    time.sleep(2.0)
    ser.reset_input_buffer()

    pinkie_s = SmoothedSensor(SMOOTH_WINDOW, 0.0)
    ring_s = SmoothedSensor(SMOOTH_WINDOW, 0.0)
    middle_s = SmoothedSensor(SMOOTH_WINDOW, 0.0)
    index_s = SmoothedSensor(SMOOTH_WINDOW, 0.0)
    thumb_s = SmoothedSensor(SMOOTH_WINDOW, 0.0)

    latest_raw = [0, 0, 0, 0, 0]
    latest = {
        "pinkie": 0.0,
        "ring": 0.0,
        "middle": 0.0,
        "index": 0.0,
        "thumb": 0.0,
    }

    plt.ion()
    fig = plt.figure(figsize=(6, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    last_ui = time.time()

    try:
        while True:
            while ser.in_waiting:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue

                if raw.startswith("m1") or raw.lower().startswith("a0"):
                    continue

                vals = parse_ints_csv(raw)
                if vals is None:
                    continue

                latest_raw = vals[:]

                # Mapping:
                # vals[0] = A0 = pinkie
                # vals[1] = A1 = ring
                # vals[2] = A2 = middle
                # vals[3] = A3 = index
                # vals[4] = D6 = thumb
                latest["pinkie"] = pinkie_s.update(normalize_analog(vals[0]))
                latest["ring"] = ring_s.update(normalize_analog(vals[1]))
                latest["middle"] = middle_s.update(normalize_analog(vals[2]))
                latest["index"] = index_s.update(normalize_analog(vals[3]))
                latest["thumb"] = thumb_s.update(1.0 if vals[4] else 0.0)

            now = time.time()
            if now - last_ui < 1.0 / UI_HZ:
                continue
            last_ui = now

            ax.cla()
            ax.set_facecolor(BG_COLOR)

            # ===== Layout chosen to match your sketch =====
            palm = (0.0, 0.0)

            # finger base points all start from the same palm area,
            # but with slightly different initial directions
            thumb_base = palm
            index_base = palm
            middle_base = palm
            ring_base = palm
            pinkie_base = palm

            # lengths chosen to look like the sketch
            thumb_lengths = [2.6, 1.4, 1.0]   # 3 segments visually
            index_lengths = [2.0, 2.3, 1.9, 1.6]
            middle_lengths = [2.5, 2.7, 2.2, 1.8]
            ring_lengths = [1.8, 2.0, 1.9, 1.5]
            pinkie_lengths = [1.6, 1.8, 1.5, 1.2]

            # Base directions to mirror the sketch
            # thumb goes left/down then up slightly
            thumb_angle = 195

            # fingers mostly vertical with slight spread
            index_angle = 118
            middle_angle = 92
            ring_angle = 72
            pinkie_angle = 52

            # bends
            thumb_total = thumb_bends(1 if latest["thumb"] > 0.5 else 0)
            thumb_chain = make_chain(
                thumb_base,
                thumb_lengths,
                thumb_angle,
                [thumb_total[0], thumb_total[1], 0],
            )

            index_chain = make_chain(
                index_base,
                index_lengths,
                index_angle,
                finger_bends(latest["index"]),
            )
            middle_chain = make_chain(
                middle_base,
                middle_lengths,
                middle_angle,
                finger_bends(latest["middle"]),
            )
            ring_chain = make_chain(
                ring_base,
                ring_lengths,
                ring_angle,
                finger_bends(latest["ring"]),
            )
            pinkie_chain = make_chain(
                pinkie_base,
                pinkie_lengths,
                pinkie_angle,
                finger_bends(latest["pinkie"]),
            )

            # lift into 3D with tiny z differences only
            thumb_3d = lift_to_3d(thumb_chain, Z_BASE - 0.02)
            index_3d = lift_to_3d(index_chain, Z_BASE + 0.02)
            middle_3d = lift_to_3d(middle_chain, Z_BASE + 0.03)
            ring_3d = lift_to_3d(ring_chain, Z_BASE + 0.01)
            pinkie_3d = lift_to_3d(pinkie_chain, Z_BASE)

            # Palm connectors like the sketch
            palm_to_index = lift_to_3d([palm, index_chain[1]], Z_BASE)
            palm_to_middle = lift_to_3d([palm, middle_chain[1]], Z_BASE)
            palm_to_ring = lift_to_3d([palm, ring_chain[1]], Z_BASE)
            palm_to_pinkie = lift_to_3d([palm, pinkie_chain[1]], Z_BASE)
            palm_to_thumb = lift_to_3d([palm, thumb_chain[1]], Z_BASE)

            # draw palm center
            ax.scatter([palm[0]], [palm[1]], [Z_BASE], color=JOINT_COLOR, s=90)

            # draw connector lines from palm
            for seg in [palm_to_thumb, palm_to_index, palm_to_middle, palm_to_ring, palm_to_pinkie]:
                draw_polyline(ax, seg)

            # draw each finger
            for chain in [thumb_3d, index_3d, middle_3d, ring_3d, pinkie_3d]:
                draw_polyline(ax, chain)
                draw_joints(ax, chain)

            # Make it visually match your picture:
            # flat front-on view, no axes, dark background
            ax.set_xlim(-4.5, 4.2)
            ax.set_ylim(-2.8, 9.5)
            ax.set_zlim(-0.3, 0.3)

            ax.view_init(elev=90, azim=-90)
            ax.set_axis_off()

            title = (
                f"Hand Skeleton  |  "
                f"A0 Pinkie={latest_raw[0]}  "
                f"A1 Ring={latest_raw[1]}  "
                f"A2 Middle={latest_raw[2]}  "
                f"A3 Index={latest_raw[3]}  "
                f"D6 Thumb={latest_raw[4]}"
            )
            ax.set_title(title, color=TEXT_COLOR, pad=18)

            plt.draw()
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()


if __name__ == "__main__":
    main()