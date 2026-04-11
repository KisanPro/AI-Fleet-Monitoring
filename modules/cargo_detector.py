# ============================================================
# modules/cargo_detector.py — Cargo Object Security
#
# Logic:
#   1. Rolling 15-second baseline: we learn what objects are normally
#      present in the cargo area.
#   2. Alert if an object from the baseline GOES MISSING for more
#      than CARGO_MISS_TOLERANCE checks.
#   3. Also detects PEOPLE (person class) — alerts immediately.
#
# Mode gate: only runs when shared_state.camera_mode == 'cargo'
# ============================================================

import cv2
import time
import threading
import logging
import os
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

log = logging.getLogger(__name__)

# Alert keys
ALERT_CARGO_MISSING = "CARGO_MISSING"
ALERT_INTRUSION     = "CARGO_INTRUSION"


class _FPS:
    def __init__(self, n=30):
        self._ts = deque(maxlen=n)
    def tick(self):
        self._ts.append(time.perf_counter())
    @property
    def fps(self):
        if len(self._ts) < 2: return 0.0
        e = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / e if e > 0 else 0.0


class CargoDetector(threading.Thread):

    def __init__(self, config, shared_state, camera_manager, excel_logger, buzzer):
        super().__init__(name="CargoDetector", daemon=True)
        self.cfg     = config
        self.state   = shared_state
        self.cam_mgr = camera_manager
        self.alerts  = excel_logger
        self.buzzer  = buzzer

        self.model = None
        self.initialized = False

    def _lazy_init(self):
        """Heavy initialization of YOLO engine on first activation."""
        if self.initialized:
            return
        
        log.info("CargoDetector: Starting lazy initialization (GPU Engine)...")
        self.state.set_frame("cargo", self.state._make_placeholder("LOADING AI ENGINE..."))

        # ── Model selection ───────────────────────────────────
        model_path = self.cfg.CARGO_MODEL
        if not os.path.exists(model_path):
            log.warning(f"Cargo model {model_path} not found, "
                        f"falling back to {self.cfg.YOLO_MODEL_FALLBACK}")
            model_path = self.cfg.YOLO_MODEL_FALLBACK

        # TensorRT (.engine) or PyTorch (.pt) models use CUDA/GPU natively in ultralytics
        if model_path.endswith(".onnx"):
            log.info(f"CargoDetector: Loading ONNX model {model_path}...")
            self.model = YOLO(model_path, task="detect")
            log.info(f"CargoDetector: ONNX loaded.")
        else:
            log.info(f"CargoDetector: Loading engine {model_path}...")
            # For engine/onnx, device is handled internally or auto-detected.
            self.model = YOLO(model_path)
            log.info(f"CargoDetector: {model_path} ready.")

        # Warmup inference
        log.info("CargoDetector: Running warmup inference...")
        dummy = np.zeros((self.cfg.CAMERA_HEIGHT, self.cfg.CAMERA_WIDTH, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        log.info("CargoDetector: Warmup complete.")
        
        self.names = self.model.names   # id → class name
        self.initialized = True

    # ─────────────────────────────────────────────────────────
    def run(self):
        log.info("CargoDetector starting (Main Loop) …")

        # ── State ─────────────────────────────────────────────
        baseline: dict[int, int]     = {}
        miss_strikes: dict[int, int] = defaultdict(int)
        last_baseline_time           = 0.0
        last_check_time              = 0.0
        fps_ctr                      = _FPS()
        W, H = self.cfg.CAMERA_WIDTH, self.cfg.CAMERA_HEIGHT

        log.info("CargoDetector ready — waiting for cargo mode")

        while self.state.running:

            # ── Mode gate ─────────────────────────────────────
            if self.state.camera_mode != "cargo":
                self._push_paused(W, H, "cargo", "CARGO CAM — CLOSED")
                time.sleep(0.1)
                baseline.clear()
                miss_strikes.clear()
                last_baseline_time = 0
                continue

            # Lazy init on first activation
            if not self.initialized:
                self._lazy_init()
                names = self.names # sync local ref

            frame = self.cam_mgr.read("cargo")
            if frame is None:
                time.sleep(0.033)
                continue

            t0   = time.perf_counter()
            disp = frame.copy()
            now  = time.time()

            # ── Inference ─────────────────────────────────────
            results = self.model(frame, conf=self.cfg.CARGO_CONF, verbose=False)
            current: dict[int, int] = defaultdict(int)

            for det in results[0].boxes:
                cls_id = int(det.cls[0])
                conf   = float(det.conf[0])
                x1,y1,x2,y2 = map(int, det.xyxy[0].tolist())
                label  = names.get(cls_id, str(cls_id))

                if cls_id == self.cfg.PERSON_CLASS_ID:
                    # Intruders alert immediately
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 3)
                    cv2.putText(disp, "INTRUDER!", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    self.alerts.fire(ALERT_INTRUSION, "cargo_camera", "PERSON DETECTED")
                    self.buzzer.beep(0.5)
                else:
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,100), 2)
                    cv2.putText(disp, f"{label} {conf:.2f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,100), 1)
                    current[cls_id] += 1

            # ── Baseline Logic ────────────────────────────────
            # If no baseline or baseline expired after 15 s, refresh it
            if not baseline or (now - last_baseline_time > self.cfg.CARGO_BASELINE_WINDOW):
                baseline = dict(current)
                miss_strikes.clear()
                last_baseline_time = now
                log.info(f"Cargo baseline updated: {baseline}")
                self.state.add_log(f"Cargo baseline calibrated: {len(baseline)} objs",
                                   source="CARGO")

            # ── Comparison Logic ──────────────────────────────
            if now - last_check_time > self.cfg.CARGO_CHECK_INTERVAL:
                last_check_time = now
                for cls_id, count in baseline.items():
                    if current.get(cls_id, 0) < count:
                        miss_strikes[cls_id] += 1
                        if miss_strikes[cls_id] >= self.cfg.CARGO_MISS_TOLERANCE:
                            lbl = names.get(cls_id, f"ID:{cls_id}")
                            self.alerts.fire(ALERT_CARGO_MISSING, "cargo_camera",
                                             f"{lbl} missing (Expected {count})")
                            self.buzzer.beep(1.0)
                    else:
                        miss_strikes[cls_id] = 0

            # ── HUD ───────────────────────────────────────────
            cv2.putText(disp, f"Baseline age: {int(now - last_baseline_time)}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

            cv2.putText(disp, "CARGO CAM — ACTIVE",
                        (W-200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,180,0), 1)

            # ── Encode → shared state ─────────────────────────
            _, buf = cv2.imencode(".jpg", disp,
                                  [cv2.IMWRITE_JPEG_QUALITY,
                                   self.cfg.STREAM_QUALITY])
            self.state.set_frame("cargo", buf.tobytes())

            fps_ctr.tick()
            lat = (time.perf_counter() - t0) * 1000
            self.state.update_stats("cargo", fps_ctr.fps, lat)

            # ── Pace Loop ─────────────────────────────────────
            elapsed = time.perf_counter() - t0
            target  = 1.0 / self.cfg.CAMERA_FPS
            if elapsed < target:
                time.sleep(target - elapsed)

        log.info("CargoDetector stopped")

    def _push_paused(self, W, H, role, text):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (30, 20, 20)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(img, text,
                    ((W-tw)//2, (H+th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 80, 80), 1)
        _, buf = cv2.imencode(".jpg", img,
                              [cv2.IMWRITE_JPEG_QUALITY, 50])
        self.state.set_frame(role, buf.tobytes())
