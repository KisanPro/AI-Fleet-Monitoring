#!/usr/bin/env python3
# ============================================================
# main.py — Fleet Monitor Entry Point
# Jetson Orin Nano | 3-Camera AI Safety System
#
# Thread map:
#   CameraReader × 3   vehicle / driver / cargo
#   VehicleDetector    always active (road cam)
#   DrowsinessDetector gates on mode == 'driver'
#   CargoDetector      gates on mode == 'cargo'
#   GPSManager         SIM7600G-H NMEA reader
#   SensorManager      DS18B20 + HC-SR04
#   ExcelLogger        fleet_log.xlsx writer (alerts + perf)
#   Flask+SocketIO     dashboard (main thread)
# ============================================================

import os
import sys
import signal
import logging
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

from config                      import Config
from modules.shared_state        import SharedState
from modules.camera_manager      import CameraManager
from modules.drowsiness          import DrowsinessDetector
from modules.vehicle_detector    import VehicleDetector
from modules.cargo_detector      import CargoDetector
from modules.gps_manager         import GPSManager
# from modules.sensor_manager      import SensorManager, Buzzer  # if you decide to enable them
from modules.sensor_manager      import SensorManager, Buzzer
from modules.excel_logger        import ExcelLogger
import dashboard.server          as dash

# ── Logging ──────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-22s] %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/runtime.log"),
    ],
)
log = logging.getLogger("main")


def main():
    cfg   = Config()
    state = SharedState()

    log.info("=" * 64)
    log.info("  FLEET MONITOR v2.0 — Jetson Orin Nano")
    log.info("=" * 64)

    # ── Hardware devices ──────────────────────────────────────
    buzzer = Buzzer(pin=cfg.BUZZER_PIN, freq=cfg.BUZZER_FREQ,
                    duty=cfg.BUZZER_DUTY)

    # ── Excel logger (start first — everything else uses it) ──
    excel = ExcelLogger(cfg, state)
    excel.start()
    state.add_log("ExcelLogger ready — logs/fleet_log.xlsx", source="MAIN")

    # ── Camera manager (auto-detects 3 cameras) ───────────────
    cam_mgr = CameraManager(cfg, state)
    try:
        cam_mgr.start()
    except RuntimeError as e:
        log.critical(f"Camera init failed: {e}")
        state.running = False
        sys.exit(1)

    # ── Detector threads ──────────────────────────────────────
    vehicle_det    = VehicleDetector(cfg, state, cam_mgr, excel)
    drowsiness_det = DrowsinessDetector(cfg, state, cam_mgr, excel, buzzer)
    cargo_det      = CargoDetector(cfg, state, cam_mgr, excel, buzzer)

    log.info("Starting detector threads...")
    vehicle_det.start()
    drowsiness_det.start()
    cargo_det.start()
    state.add_log("All detector threads started", source="MAIN")

    # ── GPS ───────────────────────────────────────────────────
    gps = GPSManager(cfg, state, excel)
    gps.start()

    # ── Sensors ───────────────────────────────────────────────
    sensors = SensorManager(cfg, state, excel, buzzer)
    sensors.start()
    state.add_log("GPS + Sensor managers started", source="MAIN")

    # ── Shutdown handler ──────────────────────────────────────
    def shutdown(sig, _frame):
        log.info("Shutdown requested …")
        state.running = False
        cam_mgr.stop()
        buzzer.cleanup()
        log.info("Fleet Monitor stopped cleanly.")
        os._exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Ready ─────────────────────────────────────────────────
    log.info(f"Dashboard → http://localhost:{cfg.DASHBOARD_PORT}")
    state.add_log(
        f"System ready | Default mode: DRIVER | "
        f"Dashboard: http://localhost:{cfg.DASHBOARD_PORT}",
        source="MAIN",
    )

    # ── Flask dashboard (blocking) ────────────────────────────
    dash.serve(cfg, state, cam_mgr)


if __name__ == "__main__":
    main()
