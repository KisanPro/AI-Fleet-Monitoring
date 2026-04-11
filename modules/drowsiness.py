# ============================================================
# modules/drowsiness.py — Driver Drowsiness & Distraction
#
# Pipeline per frame:
#   1. MediaPipe FaceMesh  → EAR (eyes) + MAR (mouth) + Head pose
#   2. YOLOv8n             → Phone-in-hand detection
#
# Calibration phase (first CALIBRATION_SECONDS):
#   Collects open-eye EAR and closed-mouth MAR baselines.
#   Derives personalised thresholds for this driver.
#
# Alerts:
#   SLEEPY         — EAR below threshold for > 2 s
#   YAWNING        — MAR above threshold for > 3 s
#   DISTRACTION    — head fully turned left OR right for > 8 s
#   PHONE_DETECTED — YOLOv8n detects class 67 (cell phone)
#
# Mode gate: only runs when shared_state.camera_mode == 'driver'
# When inactive → pushes a "PAUSED" overlay frame so dashboard
# still has a valid JPEG to show.
# ============================================================

import cv2
import time
import threading
import logging
import numpy as np
import mediapipe as mp
from collections import deque
from ultralytics import YOLO

log = logging.getLogger(__name__)

# ── MediaPipe landmark indices ───────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
UPPER_LIP  = [13]
LOWER_LIP  = [14]
MOUTH_L    = [78]
MOUTH_R    = [308]

# 6-point PnP model (mm)
HEAD_2D_IDS = [1, 152, 263, 33, 287, 57]
FACE_3D = np.array([
    [ 0.0,    0.0,   0.0],
    [ 0.0, -330.0, -65.0],
    [-225.0, 170.0,-135.0],
    [ 225.0, 170.0,-135.0],
    [-150.0,-150.0,-125.0],
    [ 150.0,-150.0,-125.0],
], dtype=np.float64)

# Alert keys
ALERT_SLEEPY      = "SLEEPY"
ALERT_YAWNING     = "YAWNING"
ALERT_DISTRACTION = "DISTRACTION"
ALERT_PHONE       = "PHONE_DETECTED"


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def _ear(lms, ids, w, h) -> float:
    """Eye Aspect Ratio — 6-point formula."""
    pts = np.array([(lms[i].x * w, lms[i].y * h) for i in ids], dtype=np.float64)
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(lms, w, h) -> float:
    """Mouth Aspect Ratio — vertical opening / horizontal width."""
    uy = lms[UPPER_LIP[0]].y * h
    ly = lms[LOWER_LIP[0]].y * h
    lx = lms[MOUTH_L[0]].x  * w
    rx = lms[MOUTH_R[0]].x  * w
    vertical   = abs(ly - uy)
    horizontal = abs(rx - lx) + 1e-6
    return vertical / horizontal


def _head_pose(lms, w, h, cam_mat, dist):
    """Return (pitch°, yaw°, roll°) or None."""
    pts2d = np.array(
        [(lms[i].x * w, lms[i].y * h) for i in HEAD_2D_IDS],
        dtype=np.float64,
    )
    ok, rvec, _ = cv2.solvePnP(
        FACE_3D, pts2d, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360, angles[1] * 360, angles[2] * 360


# ─────────────────────────────────────────────────────────────
# FPS counter
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Detector thread
# ─────────────────────────────────────────────────────────────

class DrowsinessDetector(threading.Thread):

    def __init__(self, config, shared_state, camera_manager, excel_logger, buzzer):
        super().__init__(name="DrowsinessDetector", daemon=True)
        self.cfg     = config
        self.state   = shared_state
        self.cam_mgr = camera_manager
        self.alerts  = excel_logger
        self.buzzer  = buzzer

    # ─────────────────────────────────────────────────────────
        self.yolo      = None
        self.face_mesh = None
        self.initialized = False

    def _lazy_init(self, W, H):
        """Heavy initialization of MediaPipe and YOLO on first activation."""
        if self.initialized:
            return
        
        log.info("DrowsinessDetector: Starting lazy initialization (GPU/MediaPipe)...")
        self.state.set_frame("driver", self.state._make_placeholder("LOADING AI ENGINE..."))

        # MediaPipe
        mp_fm = mp.solutions.face_mesh
        self.face_mesh = mp_fm.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # YOLO
        if self.cfg.YOLO_MODEL.endswith(".onnx"):
            log.info(f"DrowsinessDetector: Loading ONNX model {self.cfg.YOLO_MODEL}...")
            self.yolo = YOLO(self.cfg.YOLO_MODEL, task="detect")
            log.info(f"DrowsinessDetector: ONNX loaded.")
        else:
            log.info(f"DrowsinessDetector: Loading engine {self.cfg.YOLO_MODEL}...")
            self.yolo = YOLO(self.cfg.YOLO_MODEL)
            log.info(f"DrowsinessDetector: {self.cfg.YOLO_MODEL} ready.")

        # Warmup inference
        log.info("DrowsinessDetector: Running YOLO warmup inference...")
        dummy = np.zeros((H, W, 3), dtype=np.uint8)
        self.yolo(dummy, verbose=False)
        log.info("DrowsinessDetector: Warmup complete.")
        
        self.initialized = True

    # ─────────────────────────────────────────────────────────
    def run(self):
        log.info("DrowsinessDetector starting (Main Loop) …")

        # Camera intrinsics
        W, H   = self.cfg.CAMERA_WIDTH, self.cfg.CAMERA_HEIGHT
        fl     = W
        cam_mat = np.array([[fl, 0, W/2],[0, fl, H/2],[0,0,1]], dtype=np.float64)
        dist    = np.zeros((4, 1), dtype=np.float64)

        # ── Thresholds (updated after calibration) ───────────
        ear_thresh = self.cfg.EAR_DEFAULT
        mar_thresh = self.cfg.MAR_DEFAULT
        calibrated = False

        # ── Calibration buffers ───────────────────────────────
        cal_ears   = []
        cal_mars   = []
        cal_start  = None

        # ── State timers ──────────────────────────────────────
        eye_closed_since    = None   # epoch when eyes first closed
        mouth_open_since    = None   # epoch when mouth first opened
        head_left_since     = None
        head_right_since    = None

        fps_ctr = _FPS()
        log.info("DrowsinessDetector ready — waiting for driver mode")

        while self.state.running:

            # ── Mode gate ─────────────────────────────────────
            if self.state.camera_mode != "driver":
                # Push paused overlay so dashboard Q3 isn't blank
                self._push_paused(W, H, "driver", "DRIVER CAM — CLOSED")
                time.sleep(0.1)
                # Reset timers when coming back
                eye_closed_since = mouth_open_since = None
                head_left_since  = head_right_since = None
                calibrated       = False
                cal_ears.clear();  cal_mars.clear()
                cal_start        = None
                continue

            # Lazy init on first activation
            if not self.initialized:
                self._lazy_init(W, H)

            frame = self.cam_mgr.read("driver")
            if frame is None:
                time.sleep(0.033)
                continue

            t0   = time.perf_counter()
            disp = frame.copy()
            now  = time.time()
            alerts_this_frame = []

            # ── MediaPipe ─────────────────────────────────────
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark

                ear_l = _ear(lms, LEFT_EYE,  W, H)
                ear_r = _ear(lms, RIGHT_EYE, W, H)
                ear   = (ear_l + ear_r) / 2.0
                mar   = _mar(lms, W, H)

                # ── Calibration phase ─────────────────────────
                if not calibrated:
                    if cal_start is None:
                        cal_start = now
                        self.state.add_log(
                            "Calibrating driver baseline (4 s)…", source="DROWSY"
                        )
                    cal_ears.append(ear)
                    cal_mars.append(mar)

                    remaining = self.cfg.CALIBRATION_SECONDS - (now - cal_start)
                    pct = min((now - cal_start) / self.cfg.CALIBRATION_SECONDS, 1.0)
                    bar = int(pct * (W - 20))
                    cv2.rectangle(disp, (10, H-30), (10+bar, H-15), (0,220,120), -1)
                    cv2.putText(disp, f"CALIBRATING… {max(0,remaining):.1f}s",
                                (10, H-34), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0,220,120), 2)

                    if now - cal_start >= self.cfg.CALIBRATION_SECONDS:
                        if cal_ears:
                            ear_thresh = float(np.mean(cal_ears)) * 0.75
                            mar_thresh = float(np.mean(cal_mars)) * 1.6
                        calibrated = True
                        self.state.add_log(
                            f"Calibration done — EAR_thresh={ear_thresh:.3f}  "
                            f"MAR_thresh={mar_thresh:.3f}",
                            source="DROWSY",
                        )
                        log.info(f"Calibrated: EAR={ear_thresh:.3f} MAR={mar_thresh:.3f}")

                # ── Detection (post-calibration) ─────────────
                else:
                    # ── Eyes closed → SLEEPY ──────────────────
                    if ear < ear_thresh:
                        if eye_closed_since is None:
                            eye_closed_since = now
                        elif now - eye_closed_since > self.cfg.DROWSY_SECONDS:
                            alerts_this_frame.append(("sleepy", ALERT_SLEEPY))
                            self.alerts.fire(ALERT_SLEEPY, "driver_camera",
                                             f"EAR={ear:.3f}")
                            self.buzzer.beep(1.0)
                    else:
                        eye_closed_since = None

                    # ── Mouth open → YAWNING ──────────────────
                    if mar > mar_thresh:
                        if mouth_open_since is None:
                            mouth_open_since = now
                        elif now - mouth_open_since > self.cfg.YAWN_SECONDS:
                            alerts_this_frame.append(("yawning", ALERT_YAWNING))
                            self.alerts.fire(ALERT_YAWNING, "driver_camera",
                                             f"MAR={mar:.3f}")
                            self.buzzer.beep(0.8)
                    else:
                        mouth_open_since = None

                    # ── Head pose → DISTRACTION ───────────────
                    pose = _head_pose(lms, W, H, cam_mat, dist)
                    if pose:
                        pitch, yaw, roll = pose
                        if yaw > self.cfg.HEAD_YAW_THRESHOLD:
                            # head turned LEFT (positive yaw)
                            head_right_since = None
                            if head_left_since is None:
                                head_left_since = now
                            elif now - head_left_since > self.cfg.HEAD_TURN_SECONDS:
                                alerts_this_frame.append(
                                    ("distraction", ALERT_DISTRACTION)
                                )
                                self.alerts.fire(ALERT_DISTRACTION, "driver_camera",
                                                 f"HeadLeft yaw={yaw:.1f}°")
                                self.buzzer.beep(0.7)
                        elif yaw < -self.cfg.HEAD_YAW_THRESHOLD:
                            # head turned RIGHT (negative yaw)
                            head_left_since = None
                            if head_right_since is None:
                                head_right_since = now
                            elif now - head_right_since > self.cfg.HEAD_TURN_SECONDS:
                                alerts_this_frame.append(
                                    ("distraction", ALERT_DISTRACTION)
                                )
                                self.alerts.fire(ALERT_DISTRACTION, "driver_camera",
                                                 f"HeadRight yaw={yaw:.1f}°")
                                self.buzzer.beep(0.7)
                        else:
                            head_left_since  = None
                            head_right_since = None

                        # ── HUD ───────────────────────────────
                        cv2.putText(disp,
                            f"EAR:{ear:.2f}  MAR:{mar:.2f}  Yaw:{yaw:.1f}",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

                        # Eye-close progress bar
                        if eye_closed_since:
                            prog = min(
                                (now - eye_closed_since) / self.cfg.DROWSY_SECONDS, 1.0
                            )
                            bw = int(prog * 200)
                            col = (0,0,255) if prog > 0.8 else (0,140,255)
                            cv2.rectangle(disp, (8,38), (8+bw,52), col, -1)
                            cv2.putText(disp, f"EYE:{prog*100:.0f}%",
                                        (10,50), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4, (255,255,255), 1)

                        # Yawn progress bar
                        if mouth_open_since:
                            prog = min(
                                (now - mouth_open_since) / self.cfg.YAWN_SECONDS, 1.0
                            )
                            bw = int(prog * 200)
                            cv2.rectangle(disp, (8,58), (8+bw,72), (0,200,255), -1)
                            cv2.putText(disp, f"YAWN:{prog*100:.0f}%",
                                        (10,70), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4, (255,255,255), 1)

            # ── YOLO Phone detection ──────────────────────────
            yolo_res = self.yolo(
                frame, classes=[self.cfg.PHONE_CLASS_ID],
                conf=self.cfg.PHONE_CONF, verbose=False
            )
            for det in yolo_res[0].boxes:
                x1,y1,x2,y2 = map(int, det.xyxy[0].tolist())
                cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(disp, f"PHONE {det.conf[0]:.0%}",
                            (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
                alerts_this_frame.append(("phone detected", ALERT_PHONE))
                self.alerts.fire(ALERT_PHONE, "driver_camera",
                                 f"conf={float(det.conf[0]):.2f}")
                self.buzzer.beep(0.5)

            # ── Alert overlay banners ─────────────────────────
            y_off = H // 2 - len(alerts_this_frame) * 20
            for label, _ in alerts_this_frame:
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )
                cx = (W - tw) // 2
                cv2.rectangle(disp, (cx-8, y_off-th-4),
                              (cx+tw+8, y_off+4), (0,0,200), -1)
                cv2.putText(disp, label, (cx, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                y_off += th + 12

            cv2.putText(disp, "DRIVER CAM — ACTIVE",
                        (W-200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,180), 1)

            # ── Encode → shared state ─────────────────────────
            _, buf = cv2.imencode(".jpg", disp,
                                  [cv2.IMWRITE_JPEG_QUALITY,
                                   self.cfg.STREAM_QUALITY])
            self.state.set_frame("driver", buf.tobytes())

            fps_ctr.tick()
            lat = (time.perf_counter() - t0) * 1000
            self.state.update_stats("driver", fps_ctr.fps, lat)

            # ── Pace Loop ─────────────────────────────────────
            elapsed = time.perf_counter() - t0
            target  = 1.0 / self.cfg.CAMERA_FPS
            if elapsed < target:
                time.sleep(target - elapsed)

        if self.face_mesh:
            self.face_mesh.close()
        log.info("DrowsinessDetector stopped")

    def _push_paused(self, W, H, role, text):
        """Render a dark 'paused' frame so the stream stays alive."""
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (20, 20, 30)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        cv2.putText(img, text,
                    ((W-tw)//2, (H+th)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 100), 1)
        _, buf = cv2.imencode(".jpg", img,
                              [cv2.IMWRITE_JPEG_QUALITY, 50])
        self.state.set_frame(role, buf.tobytes())
