import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
import serial

# ================= CONFIG =================
DEFAULT_PORT = "COM17"
BAUD = 115200
SERIAL_TIMEOUT = 0.02
UI_HZ = 60.0

CALIBRATION_FILE = "glove_calibration.json"

# Serial mapping:
# vals[0] = A0 = pinkie
# vals[1] = A1 = ring
# vals[2] = A2 = middle
# vals[3] = A3 = index
# vals[4] = D6 = thumb (digital)
FINGER_NAMES = ["pinkie", "ring", "middle", "index"]
ALL_NAMES = ["pinkie", "ring", "middle", "index", "thumb"]

DEFAULT_ANALOG_OPEN = 1800
DEFAULT_ANALOG_CLOSED = 2200

SMOOTH_WINDOW = 6
BAR_HISTORY = 140

FINGER_MAX_BEND_DEG = {
    "pinkie": 72,
    "ring": 78,
    "middle": 82,
    "index": 80,
}
THUMB_OPEN_DEG = 12
THUMB_CLOSED_DEG = 52

BG_COLOR = "#0f1116"
PANEL_COLOR = "#171a21"
GRID_COLOR = "#2b303b"
TEXT_COLOR = "#e7edf8"
MUTED_TEXT = "#8c96a8"
HAND_LINE = "#f4f7fb"
PALM_FILL = "#242936"
PALM_EDGE = "#4e5563"
BAR_BG = "#252a34"
BAR_FILL = "#77bdfb"
BAR_FILL_2 = "#7df0c5"
WARN_COLOR = "#ffcc66"
ACCENT = "#8b9dff"
JOINT_SIZE = 180
LINE_WIDTH = 3.0
# =========================================


@dataclass
class FingerCalibration:
    open_raw: float = DEFAULT_ANALOG_OPEN
    closed_raw: float = DEFAULT_ANALOG_CLOSED
    invert: bool = False
    dip_ratio: float = 0.38
    pip_ratio: float = 0.62

    def to_dict(self):
        return {
            "open_raw": self.open_raw,
            "closed_raw": self.closed_raw,
            "invert": self.invert,
            "dip_ratio": self.dip_ratio,
            "pip_ratio": self.pip_ratio,
        }

    @staticmethod
    def from_dict(data):
        return FingerCalibration(
            open_raw=float(data.get("open_raw", DEFAULT_ANALOG_OPEN)),
            closed_raw=float(data.get("closed_raw", DEFAULT_ANALOG_CLOSED)),
            invert=bool(data.get("invert", False)),
            dip_ratio=float(data.get("dip_ratio", 0.38)),
            pip_ratio=float(data.get("pip_ratio", 0.62)),
        )


@dataclass
class ThumbCalibration:
    threshold: float = 0.5
    open_deg: float = THUMB_OPEN_DEG
    closed_deg: float = THUMB_CLOSED_DEG

    def to_dict(self):
        return {
            "threshold": self.threshold,
            "open_deg": self.open_deg,
            "closed_deg": self.closed_deg,
        }

    @staticmethod
    def from_dict(data):
        return ThumbCalibration(
            threshold=float(data.get("threshold", 0.5)),
            open_deg=float(data.get("open_deg", THUMB_OPEN_DEG)),
            closed_deg=float(data.get("closed_deg", THUMB_CLOSED_DEG)),
        )


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


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


class SmoothedSensor:
    def __init__(self, window=6, initial=0.0):
        self.buf = deque([initial] * window, maxlen=window)

    def update(self, value):
        self.buf.append(value)
        return sum(self.buf) / len(self.buf)


class CalibrationManager:
    def __init__(self, path=CALIBRATION_FILE):
        self.path = path
        self.fingers: Dict[str, FingerCalibration] = {
            name: FingerCalibration() for name in FINGER_NAMES
        }
        self.thumb = ThumbCalibration()
        self.calibrating = False
        self.phase_index = 0
        self.capture_start = 0.0
        self.capture_duration = 2.0
        self.samples: List[List[int]] = []
        self.messages = [
            "Hold your hand OPEN and relaxed",
            "Make a comfortable CLOSED fist",
        ]
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name in FINGER_NAMES:
                if name in data.get("fingers", {}):
                    self.fingers[name] = FingerCalibration.from_dict(data["fingers"][name])
            if "thumb" in data:
                self.thumb = ThumbCalibration.from_dict(data["thumb"])
        except Exception:
            pass

    def save(self):
        payload = {
            "fingers": {name: cal.to_dict() for name, cal in self.fingers.items()},
            "thumb": self.thumb.to_dict(),
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def begin(self):
        self.calibrating = True
        self.phase_index = 0
        self.samples = []
        self.capture_start = time.time()

    def cancel(self):
        self.calibrating = False
        self.samples = []

    def status_text(self):
        if not self.calibrating:
            return "Press C to calibrate | R to reset smoothing | S to save calibration"
        elapsed = time.time() - self.capture_start
        remain = max(0.0, self.capture_duration - elapsed)
        msg = self.messages[self.phase_index]
        return f"Calibration: {msg}  |  capturing in {remain:0.1f}s"

    def update(self, latest_raw: List[int]):
        if not self.calibrating:
            return False

        self.samples.append(latest_raw[:])
        elapsed = time.time() - self.capture_start
        if elapsed < self.capture_duration:
            return False

        averaged = [sum(col) / len(self.samples) for col in zip(*self.samples)]
        if self.phase_index == 0:
            self._open_values = averaged
            self.phase_index = 1
            self.samples = []
            self.capture_start = time.time()
            return False

        self._closed_values = averaged
        self._finish()
        return True

    def _finish(self):
        for idx, name in enumerate(FINGER_NAMES):
            open_raw = float(self._open_values[idx])
            closed_raw = float(self._closed_values[idx])
            invert = closed_raw < open_raw
            self.fingers[name] = FingerCalibration(
                open_raw=open_raw,
                closed_raw=closed_raw,
                invert=invert,
                dip_ratio=0.38,
                pip_ratio=0.62,
            )

        thumb_open = self._open_values[4]
        thumb_closed = self._closed_values[4]
        self.thumb.threshold = 0.5 * (thumb_open + thumb_closed)
        self.save()
        self.calibrating = False
        self.samples = []

    def normalize(self, name: str, raw_value: int) -> float:
        cal = self.fingers[name]
        a = cal.open_raw
        b = cal.closed_raw
        if abs(b - a) < 1e-6:
            return 0.0
        t = (raw_value - a) / (b - a)
        return clamp(t, 0.0, 1.0)

    def thumb_state(self, raw_value: int) -> float:
        return 1.0 if raw_value >= self.thumb.threshold else 0.0


class HandModel:
    def __init__(self, calibration: CalibrationManager):
        self.cal = calibration
        # Finger roots are spread along the top of the palm so each chain fans out
        # from a realistic MCP position instead of starting too close together.
        self.base_points = {
            "thumb": (-1.95, -1.05),
            "index": (1.15, 2.02),
            "middle": (0.38, 2.17),
            "ring": (-0.34, 2.05),
            "pinkie": (-1.02, 1.82),
        }
        # Wider base angles stop fingers crossing through each other.
        self.base_angles = {
            "thumb": 208,
            "index": 58,
            "middle": 84,
            "ring": 99,
            "pinkie": 114,
        }
        self.lengths = {
            "thumb": [1.55, 1.20, 0.95],
            "index": [1.90, 1.55, 1.10, 0.92],
            "middle": [2.10, 1.72, 1.18, 0.95],
            "ring": [1.95, 1.60, 1.12, 0.92],
            "pinkie": [1.60, 1.28, 0.95, 0.76],
        }

    def finger_bends(self, name: str, norm: float):
        max_total = FINGER_MAX_BEND_DEG[name]
        total = max_total * norm
        cal = self.cal.fingers[name]
        pip = total * cal.pip_ratio
        dip = total * cal.dip_ratio
        mcp = total * (1.0 - cal.pip_ratio * 0.55 - cal.dip_ratio * 0.35)
        mcp = max(0.0, mcp)
        return [mcp, pip, dip]

    def thumb_bends(self, state: float):
        total = lerp(self.cal.thumb.open_deg, self.cal.thumb.closed_deg, state)
        return [total * 0.58, total * 0.42, 0.0]

    def make_chain(self, base_xy, lengths, base_angle_deg, bends_deg):
        x, y = base_xy
        pts = [(x, y)]
        angle = math.radians(base_angle_deg)
        for i, L in enumerate(lengths):
            if i > 0:
                angle -= math.radians(bends_deg[i - 1])
            x += L * math.cos(angle)
            y += L * math.sin(angle)
            pts.append((x, y))
        return pts

    def build(self, state: Dict[str, float]):
        chains = {}
        for name in FINGER_NAMES:
            chains[name] = self.make_chain(
                self.base_points[name],
                self.lengths[name],
                self.base_angles[name],
                self.finger_bends(name, state[name]),
            )
        chains["thumb"] = self.make_chain(
            self.base_points["thumb"],
            self.lengths["thumb"],
            self.base_angles["thumb"],
            self.thumb_bends(state["thumb"]),
        )
        return chains


class VisualizerApp:
    def __init__(self, port: str):
        self.port = port
        self.ser = serial.Serial(port, BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

        self.calibration = CalibrationManager()
        self.model = HandModel(self.calibration)

        self.smoothers = {name: SmoothedSensor(SMOOTH_WINDOW, 0.0) for name in ALL_NAMES}
        self.latest_raw = [0, 0, 0, 0, 0]
        self.latest_norm = {name: 0.0 for name in ALL_NAMES}
        self.histories = {name: deque([0.0] * BAR_HISTORY, maxlen=BAR_HISTORY) for name in ALL_NAMES}
        self.last_ui = 0.0
        self.info_flash = ""
        self.info_flash_until = 0.0

        self._build_figure()

    def _build_figure(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(13, 8), facecolor=BG_COLOR)
        gs = GridSpec(1, 2, width_ratios=[3.2, 1.35], wspace=0.05, figure=self.fig)
        self.ax_hand = self.fig.add_subplot(gs[0, 0])
        self.ax_side = self.fig.add_subplot(gs[0, 1])

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        try:
            self.fig.canvas.manager.set_window_title("Glove Hand Visualiser")
        except Exception:
            pass

    def flash(self, text, duration=2.0):
        self.info_flash = text
        self.info_flash_until = time.time() + duration

    def on_key(self, event):
        if event.key in ("c", "C"):
            self.calibration.begin()
            self.flash("Calibration started")
        elif event.key in ("s", "S"):
            self.calibration.save()
            self.flash(f"Calibration saved to {self.calibration.path}")
        elif event.key in ("r", "R"):
            self.smoothers = {name: SmoothedSensor(SMOOTH_WINDOW, self.latest_norm[name]) for name in ALL_NAMES}
            self.flash("Smoothing reset")
        elif event.key in ("escape",):
            if self.calibration.calibrating:
                self.calibration.cancel()
                self.flash("Calibration cancelled")

    def read_serial(self):
        updated = False
        while self.ser.in_waiting:
            raw = self.ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue
            if raw.startswith("m1") or raw.lower().startswith("a0"):
                continue
            vals = parse_ints_csv(raw)
            if vals is None:
                continue

            updated = True
            self.latest_raw = vals[:]
            self.calibration.update(self.latest_raw)

            for idx, name in enumerate(FINGER_NAMES):
                norm = self.calibration.normalize(name, vals[idx])
                self.latest_norm[name] = self.smoothers[name].update(norm)
                self.histories[name].append(self.latest_norm[name])

            thumb_state = self.calibration.thumb_state(vals[4])
            self.latest_norm["thumb"] = self.smoothers["thumb"].update(thumb_state)
            self.histories["thumb"].append(self.latest_norm["thumb"])
        return updated

    def draw_hand(self):
        ax = self.ax_hand
        ax.cla()
        ax.set_facecolor(BG_COLOR)
        ax.set_aspect("equal")
        ax.set_xlim(-4.8, 4.5)
        ax.set_ylim(-3.2, 6.2)
        ax.axis("off")

        chains = self.model.build(self.latest_norm)

        palm_points = [
            (-1.7, -1.4),
            (-2.0, 1.1),
            (-1.1, 2.0),
            (0.2, 2.15),
            (1.0, 1.95),
            (1.55, 1.15),
            (1.7, -0.6),
            (0.4, -1.55),
        ]
        palm = Polygon(
            palm_points,
            closed=True,
            facecolor=PALM_FILL,
            edgecolor=PALM_EDGE,
            linewidth=2.0,
            alpha=0.95,
            joinstyle="round",
        )
        ax.add_patch(palm)

        # Draw subtle palm-to-knuckle connectors first so the fingers feel anchored
        for name in FINGER_NAMES:
            root_x, root_y = chains[name][0]
            ax.plot([0.0, root_x], [1.2, root_y], color=PALM_EDGE, linewidth=1.8, alpha=0.65, solid_capstyle="round")

        thumb_root_x, thumb_root_y = chains["thumb"][0]
        ax.plot([-1.25, thumb_root_x], [-0.55, thumb_root_y], color=PALM_EDGE, linewidth=1.8, alpha=0.65, solid_capstyle="round")

        # Draw palm-side fingers first, then the front fingers, so overlaps look cleaner.
        draw_order = ["pinkie", "ring", "middle", "index", "thumb"]
        for name in draw_order:
            chain = chains[name]
            xs = [p[0] for p in chain]
            ys = [p[1] for p in chain]
            ax.plot(xs, ys, color=HAND_LINE, linewidth=LINE_WIDTH, solid_capstyle="round")
            for i, (x, y) in enumerate(chain):
                radius = 0.12 if i < len(chain) - 1 else 0.10
                circ = Circle((x, y), radius, facecolor=HAND_LINE, edgecolor="none", alpha=0.98)
                ax.add_patch(circ)

        ax.text(-4.45, 5.75, "Hand Visualiser", color=TEXT_COLOR, fontsize=20, fontweight="bold")
        ax.text(-4.45, 5.35, self.calibration.status_text(), color=ACCENT if self.calibration.calibrating else MUTED_TEXT, fontsize=10)

        if time.time() < self.info_flash_until:
            ax.text(-4.45, 4.95, self.info_flash, color=WARN_COLOR, fontsize=10)

        # finger labels
        labels = {
            "thumb": (-3.15, -0.25),
            "index": (2.55, 4.15),
            "middle": (1.10, 5.15),
            "ring": (-0.55, 4.90),
            "pinkie": (-2.20, 4.10),
        }
        for name, (x, y) in labels.items():
            ax.text(x, y, name.capitalize(), color=MUTED_TEXT, fontsize=10)

    def draw_side_panel(self):
        ax = self.ax_side
        ax.cla()
        ax.set_facecolor(PANEL_COLOR)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        panel = FancyBboxPatch(
            (0.03, 0.03), 0.94, 0.94,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=1.2,
            edgecolor=GRID_COLOR,
            facecolor=PANEL_COLOR,
        )
        ax.add_patch(panel)

        ax.text(0.08, 0.94, "Sensor State", color=TEXT_COLOR, fontsize=16, fontweight="bold")
        ax.text(0.08, 0.91, "Calibrated live values", color=MUTED_TEXT, fontsize=9)

        y = 0.84
        row_h = 0.14
        for idx, name in enumerate(ALL_NAMES):
            value = self.latest_norm[name]
            raw = self.latest_raw[idx] if idx < len(self.latest_raw) else 0
            fill = BAR_FILL_2 if name == "thumb" else BAR_FILL

            ax.text(0.08, y + 0.045, name.capitalize(), color=TEXT_COLOR, fontsize=11, fontweight="bold")
            ax.text(0.90, y + 0.045, f"{value:0.2f}", color=TEXT_COLOR, fontsize=10, ha="right")
            ax.text(0.08, y + 0.012, f"raw {raw}", color=MUTED_TEXT, fontsize=8)

            ax.add_patch(FancyBboxPatch(
                (0.08, y - 0.01), 0.82, 0.028,
                boxstyle="round,pad=0.003,rounding_size=0.01",
                linewidth=0,
                facecolor=BAR_BG,
            ))
            ax.add_patch(FancyBboxPatch(
                (0.08, y - 0.01), 0.82 * value, 0.028,
                boxstyle="round,pad=0.003,rounding_size=0.01",
                linewidth=0,
                facecolor=fill,
            ))

            hist = list(self.histories[name])
            if hist:
                xs = [0.08 + 0.82 * i / max(1, len(hist) - 1) for i in range(len(hist))]
                ys = [y - 0.055 + 0.035 * v for v in hist]
                ax.plot(xs, ys, color=fill, linewidth=1.0, alpha=0.95)
            y -= row_h

        ax.text(0.08, 0.16, "Controls", color=TEXT_COLOR, fontsize=13, fontweight="bold")
        controls = [
            "C  start guided calibration",
            "S  save calibration",
            "R  reset smoothing",
            "Esc cancel calibration",
        ]
        yy = 0.13
        for line in controls:
            ax.text(0.08, yy, line, color=MUTED_TEXT, fontsize=9)
            yy -= 0.03

        cal = self.calibration
        ax.text(0.08, 0.03, f"Calibration file: {os.path.basename(cal.path)}", color=MUTED_TEXT, fontsize=8)

    def update(self):
        self.read_serial()
        now = time.time()
        if now - self.last_ui < 1.0 / UI_HZ:
            return
        self.last_ui = now
        self.draw_hand()
        self.draw_side_panel()
        plt.draw()
        plt.pause(0.001)

    def run(self):
        try:
            while plt.fignum_exists(self.fig.number):
                self.update()
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.ser.close()


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    try:
        app = VisualizerApp(port)
    except Exception as e:
        print(f"Could not open {port}: {e}")
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
