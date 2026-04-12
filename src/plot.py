import sys
import time
from collections import deque

import serial
import matplotlib.pyplot as plt

# ================= CONFIG =================
DEFAULT_PORT = "COM17"      # change if needed
BAUD = 115200
MAX_POINTS = 800
UI_HZ = 60.0

# Set this to True if you want the digital D6 channel to be easier to see
# It will display 0 or 4095 instead of 0 or 1.
SCALE_DIGITAL_FOR_VIEW = True

ANALOG_MIN = 1800
ANALOG_MAX = 2200
# =========================================


def parse_ints_csv(line: str):
    parts = line.strip().split(",")
    if len(parts) < 5:
        return None

    out = []
    for p in parts[:5]:
        p = p.strip()
        try:
            out.append(int(float(p)))
        except ValueError:
            return None
    return out


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT

    try:
        ser = serial.Serial(port, BAUD, timeout=0.02)
    except Exception as e:
        print(f"Could not open {port}: {e}")
        sys.exit(1)

    time.sleep(2.0)
    ser.reset_input_buffer()

    data = [deque([0] * MAX_POINTS, maxlen=MAX_POINTS) for _ in range(5)]
    x = list(range(MAX_POINTS))

    plt.ion()
    fig, ax = plt.subplots()

    lines = []
    labels = [
        "Hall 1 (A0)",
        "Hall 2 (A1)",
        "Hall 3 (A2)",
        "Hall 4 (A3)",
        "Hall 5 (D6 digital)"
    ]

    for i in range(5):
        line, = ax.plot(x, list(data[i]), label=labels[i])
        lines.append(line)

    ax.set_title("Hall Sensor Values")
    ax.set_xlabel("Samples (scrolling)")
    ax.set_ylabel("Value")
    ax.set_xlim(0, MAX_POINTS - 1)
    ax.set_ylim(ANALOG_MIN, ANALOG_MAX)
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    last_ui = time.time()

    try:
        while True:
            while ser.in_waiting:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue

                # Skip CSV header
                if raw.startswith("m1"):
                    continue

                vals = parse_ints_csv(raw)
                if vals is None:
                    continue

                # vals = [a0, a1, a2, a3, d6]
                if SCALE_DIGITAL_FOR_VIEW:
                    vals[4] = ANALOG_MAX if vals[4] else 0

                for i in range(5):
                    data[i].append(vals[i])

            now = time.time()
            if now - last_ui >= (1.0 / UI_HZ):
                last_ui = now

                fig.canvas.restore_region(background)
                for i in range(5):
                    lines[i].set_ydata(list(data[i]))
                    ax.draw_artist(lines[i])

                fig.canvas.blit(ax.bbox)
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()


if __name__ == "__main__":
    main()