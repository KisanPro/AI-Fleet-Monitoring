# ============================================================
# config.py — Fleet Monitor Configuration
# Jetson Orin Nano | 3-Camera AI Safety System
# ============================================================

import os
import torch

def _best_model(base_name: str) -> str:
    # 1. TensorRT (Fastest on Jetson GPU)
    path_engine = f"models/{base_name}.engine"
    if os.path.exists(path_engine): return path_engine
    
    # 2. ONNX (Portable)
    path_onnx   = f"models/{base_name}.onnx"
    if os.path.exists(path_onnx):   return path_onnx
    
    # 3. PyTorch (Standard)
    return f"models/{base_name}.pt"

class Config:

    # ── Camera Hardware ───────────────────────────────────────
    CAMERA_WIDTH        = 640
    CAMERA_HEIGHT       = 480
    CAMERA_FPS          = 15
    USE_GSTREAMER       = False     # True for Jetson CSI/MIPI cameras

    # Camera role assignment (auto-mapped by index order detected)
    # Index 0 → road/vehicle ahead   (constant, never toggled)
    # Index 1 → driver drowsiness    (Q3, active when mode='driver')
    # Index 2 → cargo                (Q2, active when mode='cargo')
    # Override manually if auto-detect order is wrong:
    CAM_VEHICLE_OVERRIDE = None     # e.g. 0  — set None for auto
    CAM_DRIVER_OVERRIDE  = None     # e.g. 1
    CAM_CARGO_OVERRIDE   = None     # e.g. 2

    # ── Driver Drowsiness / Distraction ──────────────────────
    CALIBRATION_SECONDS  = 4.0     # seconds to collect baseline EAR/MAR
    # After calibration, thresholds are derived:
    #   ear_thresh = calibrated_mean_ear * 0.75
    #   mar_thresh = calibrated_mean_mar * 1.6
    EAR_DEFAULT          = 0.25    # fallback if calibration fails
    MAR_DEFAULT          = 0.55    # fallback mouth-open ratio
    DROWSY_SECONDS       = 2.0     # eyes closed > 2 s → SLEEPY alert
    YAWN_SECONDS         = 3.0     # mouth open  > 3 s → YAWNING alert
    HEAD_TURN_SECONDS    = 8.0     # head fully turned > 8 s → DISTRACTION
    HEAD_YAW_THRESHOLD   = 30.0    # degrees yaw = "fully turned"
    PHONE_CONF           = 0.45    # YOLO confidence for phone
    PHONE_CLASS_ID       = 67      # COCO class: cell phone

    # ── Vehicle Detection ─────────────────────────────────────
    VEHICLE_CLASSES = {
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }
    VEHICLE_CONF        = 0.45

    # ── Cargo Monitoring ──────────────────────────────────────
    # Rolling 15-second baseline: every 15 s the baseline is
    # re-evaluated from the last observation window.
    CARGO_BASELINE_WINDOW   = 15    # seconds — rolling window
    CARGO_CHECK_INTERVAL    = 3     # seconds between object checks
    CARGO_MISS_TOLERANCE    = 2     # consecutive misses before alert
    CARGO_CONF              = 0.25  # low threshold → catch all objects for top notch baseline
    PERSON_CLASS_ID         = 0     # COCO: person
    CARGO_MODEL             = _best_model("yolov8s")   # larger model for accuracy
    # fallback if yolov8s not present:
    YOLO_MODEL_FALLBACK     = _best_model("yolov8n")

    # ── GPS (SIM7600G-H) ──────────────────────────────────────
    GPS_CANDIDATE_PORTS = [
        "/dev/ttyUSB1", "/dev/ttyUSB2", "/dev/ttyUSB3",
        "/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyACM1",
    ]
    GPS_BAUDRATE        = 115200
    GPS_TIMEOUT         = 2.0
    GPS_RETRY_INTERVAL  = 5.0      # seconds between fix-retry attempts

    # ── GPIO Pins (Jetson BOARD numbering) ────────────────────
    BUZZER_PIN          = 15
    ULTRASONIC_TRIG     = 11
    ULTRASONIC_ECHO     = 13
    DOOR_OPEN_CM        = 35       # cm → cargo door is open
    BUZZER_FREQ         = 1000
    BUZZER_DUTY         = 50

    # ── Temperature Sensor (DS18B20 via 1-Wire) ───────────────
    # Alert logic uses DS18B20 exclusively.
    # Benchmark log uses Jetson SoC thermal (system health info).
    TEMP_ALERT_THRESHOLD = 18.0    # °C — alert if cargo > this
    TEMP_POLL_INTERVAL   = 2.0
    W1_BASE_PATH         = "/sys/bus/w1/devices/"
    # Jetson SoC thermal paths (for benchmark metrics only)
    JETSON_THERMAL_PATHS = [
        "/sys/class/thermal/thermal_zone1/temp",   # CPU cluster
        "/sys/class/thermal/thermal_zone0/temp",   # fallback
    ]

    # ── Alert Deduplication ───────────────────────────────────
    ALERT_COOLDOWN      = 5.0      # seconds between identical alerts

    # ── Unified Excel Log (two sheets) ───────────────────────
    # Sheet "Alerts"      → alert rows
    # Sheet "Performance" → benchmark rows
    EXCEL_LOG_PATH      = "logs/fleet_log.xlsx"

    # ── YOLO Inference ────────────────────────────────────────
    YOLO_MODEL          = _best_model("yolov8n")
    YOLO_DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
    YOLO_HALF           = True if torch.cuda.is_available() else False

    # ── Dashboard ─────────────────────────────────────────────
    DASHBOARD_HOST      = "0.0.0.0"
    DASHBOARD_PORT      = 5000
    STREAM_QUALITY      = 75       # JPEG quality %
    LOG_BUFFER_SIZE     = 300      # recent log entries kept in memory
