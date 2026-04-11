# ============================================================
# modules/shared_state.py — Thread-Safe Shared Application State
# 3-camera architecture: vehicle | driver | cargo
# Toggle controls which detector (driver/cargo) is ACTIVE.
# ============================================================

import threading
import queue
import collections
import time
import cv2
import numpy as np


class SharedState:
    """
    Central store. Fine-grained per-domain locks.

    Camera modes:
        'driver' → DrowsinessDetector active,  Q3 annotated, Q2 shows raw/paused
        'cargo'  → CargoDetector active,        Q2 annotated, Q3 shows raw/paused
    """

    def __init__(self):

        # ── Three independent frame buffers (JPEG bytes) ──────
        self._frame_lock = threading.Lock()
        
        # Initialize with placeholder to avoid 'No Signal' during slow startup
        self._frames = {
            "vehicle": self._make_placeholder("INITIALIZING..."),
            "driver":  self._make_placeholder("INITIALIZING..."),
            "cargo":   self._make_placeholder("INITIALIZING..."),
        }

        # ── Mode toggle ───────────────────────────────────────
        self._mode_lock   = threading.Lock()
        self._camera_mode = "driver"    # "driver" | "cargo"

        # ── GPS ───────────────────────────────────────────────
        self._gps_lock = threading.Lock()
        self.gps = {
            "lat": 0.0, "lon": 0.0,
            "speed_kmh": 0.0, "heading": 0.0,
            "altitude": 0.0, "fix": False,
            "satellites": 0, "timestamp": None,
        }

        # ── Alert queue (consumed by ExcelLogger) ─────────────
        self.alert_queue = queue.Queue()

        # ── Recent log lines for dashboard real-time panel ────
        self._log_lock  = threading.Lock()
        self.recent_logs = collections.deque(maxlen=300)

        # ── Benchmark snapshot ────────────────────────────────
        self._bench_lock = threading.Lock()
        self.benchmark = {
            "fps_vehicle": 0.0,
            "fps_driver":  0.0,
            "fps_cargo":   0.0,
            "lat_vehicle_ms": 0.0,
            "lat_driver_ms":  0.0,
            "lat_cargo_ms":   0.0,
            "cpu_percent":    0.0,
            "ram_percent":    0.0,
            "ram_used_mb":    0.0,
            "system_temp_c":  0.0,   # Jetson SoC (benchmark only)
            "sensor_temp_c":  0.0,   # DS18B20 (alert logic)
            "timestamp":      None,
        }

        # ── Per-detector stats (written by detector threads) ──
        self._stats_lock = threading.Lock()
        self.stats = {
            "vehicle": {"fps": 0.0, "latency_ms": 0.0},
            "driver":  {"fps": 0.0, "latency_ms": 0.0},
            "cargo":   {"fps": 0.0, "latency_ms": 0.0},
        }

        # ── Camera index map (populated by CameraManager) ─────
        self.camera_indices = {"vehicle": None, "driver": None, "cargo": None}

        # ── Running flag ──────────────────────────────────────
        self.running = True

    # ─────────────────────────────────────────────────────────
    # Frames
    # ─────────────────────────────────────────────────────────

    def set_frame(self, key: str, jpeg: bytes):
        with self._frame_lock:
            self._frames[key] = jpeg

    def get_frame(self, key: str):
        with self._frame_lock:
            return self._frames.get(key)

    # ─────────────────────────────────────────────────────────
    # Mode
    # ─────────────────────────────────────────────────────────

    @property
    def camera_mode(self) -> str:
        with self._mode_lock:
            return self._camera_mode

    @camera_mode.setter
    def camera_mode(self, value: str):
        with self._mode_lock:
            self._camera_mode = value

    def toggle_camera_mode(self) -> str:
        with self._mode_lock:
            self._camera_mode = (
                "cargo" if self._camera_mode == "driver" else "driver"
            )
            return self._camera_mode

    # ─────────────────────────────────────────────────────────
    # GPS
    # ─────────────────────────────────────────────────────────

    def update_gps(self, **kwargs):
        with self._gps_lock:
            self.gps.update(kwargs)

    def get_gps(self) -> dict:
        with self._gps_lock:
            return dict(self.gps)

    # ─────────────────────────────────────────────────────────
    # Logs
    # ─────────────────────────────────────────────────────────

    def add_log(self, message: str, level: str = "INFO", source: str = "SYS"):
        entry = {
            "time":    time.strftime("%H:%M:%S"),
            "level":   level,
            "source":  source,
            "message": message,
        }
        with self._log_lock:
            self.recent_logs.append(entry)

    def get_logs(self) -> list:
        with self._log_lock:
            return list(self.recent_logs)

    # ─────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────

    def update_stats(self, key: str, fps: float, latency_ms: float):
        with self._stats_lock:
            self.stats[key]["fps"]        = round(fps, 1)
            self.stats[key]["latency_ms"] = round(latency_ms, 2)

    def get_stats(self, key: str) -> dict:
        with self._stats_lock:
            return dict(self.stats.get(key, {"fps": 0.0, "latency_ms": 0.0}))

    # ─────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────

    def update_benchmark(self, **kwargs):
        with self._bench_lock:
            self.benchmark.update(kwargs)

    def get_benchmark(self) -> dict:
        with self._bench_lock:
            return dict(self.benchmark)
    def _make_placeholder(self, text: str) -> bytes:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (25, 25, 35) # Dark slate
        
        ts = time.strftime("%H:%M:%S")
        full_text = f"{text} [{ts}]"
        
        (tw, th), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(img, full_text, ((640-tw)//2, (480+th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 140), 2)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes()
