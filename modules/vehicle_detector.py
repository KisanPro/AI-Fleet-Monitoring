# ============================================================
# modules/vehicle_detector.py — Vehicle Ahead Detection (Thread-2)
# YOLOv8n with CUDA — all COCO vehicle classes
# ============================================================

import cv2
import time
import threading
import logging
import numpy as np
from ultralytics import YOLO
from collections import deque

log = logging.getLogger(__name__)

# Class colours (BGR)
CLASS_COLORS = {
    "bicycle":    (255, 165,   0),
    "car":        (  0, 255,   0),
    "motorcycle": (  0, 200, 255),
    "bus":        (128,   0, 255),
    "truck":      (255,   0, 128),
}


class FPSCounter:
    def __init__(self, window=30):
        self._ts = deque(maxlen=window)

    def tick(self):
        self._ts.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0


class VehicleDetector(threading.Thread):
    """
    Thread-2: Detect vehicles on road camera (constant feed).
    Draws annotated frame and updates shared_state.
    """

    def __init__(self, config, shared_state, camera_manager, alert_logger):
        super().__init__(name="VehicleDetector", daemon=True)
        self.cfg     = config
        self.state   = shared_state
        self.cam_mgr = camera_manager
        self.alerts  = alert_logger

    def run(self):
        log.info("VehicleDetector starting …")

        # ONNX models in ultralytics don't use .to() or .half() in the same way as .pt
        # We explicitly set task="detect" to suppress warnings
        # TensorRT (.engine) or PyTorch (.pt) models use CUDA/GPU natively in ultralytics
        if self.cfg.YOLO_MODEL.endswith(".onnx"):
            log.info(f"VehicleDetector: Loading ONNX model {self.cfg.YOLO_MODEL}...")
            model = YOLO(self.cfg.YOLO_MODEL, task="detect")
            log.info(f"VehicleDetector: ONNX loaded.")
        else:
            log.info(f"VehicleDetector: Loading engine {self.cfg.YOLO_MODEL}...")
            # For engine/onnx, device is handled internally or auto-detected.
            model = YOLO(self.cfg.YOLO_MODEL)
            log.info(f"VehicleDetector: {self.cfg.YOLO_MODEL} ready.")

        # Warmup inference (critical for TensorRT)
        log.info("VehicleDetector: Running warmup inference...")
        dummy = np.zeros((self.cfg.CAMERA_HEIGHT, self.cfg.CAMERA_WIDTH, 3), dtype=np.uint8)
        model(dummy, verbose=False)
        log.info("VehicleDetector: Warmup complete. Entering main loop.")

        vehicle_class_ids = list(self.cfg.VEHICLE_CLASSES.keys())
        fps_ctr = FPSCounter()

        log.info("VehicleDetector ready")

        while self.state.running:
            frame = self.cam_mgr.read("vehicle")
            if frame is None:
                time.sleep(0.033)
                continue

            t0   = time.perf_counter()
            h, w = frame.shape[:2]
            disp = frame.copy()

            # ── Inference ─────────────────────────────────────
            results = model(
                frame,
                classes=vehicle_class_ids,
                conf=self.cfg.VEHICLE_CONF,
                verbose=False,
            )

            detected = {}
            for det in results[0].boxes:
                cls_id = int(det.cls[0])
                label  = self.cfg.VEHICLE_CLASSES.get(cls_id, "vehicle")
                conf   = float(det.conf[0])
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                color  = CLASS_COLORS.get(label, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                txt = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(disp, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(disp, txt, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

                detected[label] = detected.get(label, 0) + 1

            # ── Count overlay ─────────────────────────────────
            y_off = 25
            for label, cnt in detected.items():
                cv2.putText(disp, f"{label}: {cnt}", (8, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            CLASS_COLORS.get(label, (255, 255, 255)), 2)
                y_off += 22

            # ── FPS overlay ───────────────────────────────────
            fps_ctr.tick()
            lat_ms = (time.perf_counter() - t0) * 1000
            self.state.update_stats("vehicle", fps_ctr.fps, lat_ms)

            cv2.putText(disp, f"FPS:{fps_ctr.fps:.1f}", (w - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
            cv2.putText(disp, "VEHICLE AHEAD CAM", (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # ── Encode and store ──────────────────────────────
            _, buf = cv2.imencode(
                ".jpg", disp,
                [cv2.IMWRITE_JPEG_QUALITY, self.cfg.STREAM_QUALITY],
            )
            self.state.set_frame("vehicle", buf.tobytes())

            # ── Pace Loop ─────────────────────────────────────
            elapsed = time.perf_counter() - t0
            target  = 1.0 / self.cfg.CAMERA_FPS
            if elapsed < target:
                time.sleep(target - elapsed)

        log.info("VehicleDetector stopped")
