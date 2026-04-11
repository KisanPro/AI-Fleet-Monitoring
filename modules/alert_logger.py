# ============================================================
# modules/excel_logger.py — Unified Excel Logger
#
# Single workbook: logs/fleet_log.xlsx
#   Sheet "Alerts"      — every fired alert (deduplicated)
#   Sheet "Performance" — sampled every BENCHMARK_INTERVAL s
#
# Thread-safe writes via a single write lock.
# Uses openpyxl; creates/opens the file on startup.
# ============================================================

import os
import time
import queue
import threading
import logging
import psutil
from datetime import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

log = logging.getLogger(__name__)

BENCHMARK_INTERVAL = 5.0    # seconds between performance rows

# ── Column definitions ────────────────────────────────────────────────────────
ALERT_COLS   = ["Alert_Name", "Timestamp", "GPS_Lat", "GPS_Lon",
                "Camera_Source", "Details"]
PERF_COLS    = ["Timestamp", "FPS_Vehicle", "FPS_Driver", "FPS_Cargo",
                "Lat_Vehicle_ms", "Lat_Driver_ms", "Lat_Cargo_ms",
                "CPU_Percent", "RAM_Percent", "RAM_Used_MB",
                "System_Temp_C", "Sensor_Temp_C"]

# ── Style constants ───────────────────────────────────────────────────────────
_HDR_FILL_ALERT = PatternFill("solid", fgColor="C62828")   # dark red
_HDR_FILL_PERF  = PatternFill("solid", fgColor="1565C0")   # dark blue
_HDR_FONT       = Font(bold=True, color="FFFFFF")


def _write_header(ws, cols, fill):
    for col, title in enumerate(cols, 1):
        cell = ws.cell(row=1, column=col, value=title)
        cell.font      = _HDR_FONT
        cell.fill      = fill
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"


def _init_workbook(path: str) -> Workbook:
    """Create a new workbook with two formatted sheets."""
    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    ws_a = wb.create_sheet("Alerts")
    _write_header(ws_a, ALERT_COLS, _HDR_FILL_ALERT)
    ws_a.column_dimensions["A"].width = 26
    ws_a.column_dimensions["B"].width = 20
    ws_a.column_dimensions["C"].width = 14
    ws_a.column_dimensions["D"].width = 14
    ws_a.column_dimensions["E"].width = 22
    ws_a.column_dimensions["F"].width = 40

    ws_p = wb.create_sheet("Performance")
    _write_header(ws_p, PERF_COLS, _HDR_FILL_PERF)
    ws_p.column_dimensions["A"].width = 20
    for col in "BCDEFGHIJKL":
        ws_p.column_dimensions[col].width = 16

    wb.save(path)
    log.info(f"Created Excel log: {path}")
    return wb


# ─────────────────────────────────────────────────────────────
# ExcelLogger
# ─────────────────────────────────────────────────────────────

class ExcelLogger(threading.Thread):
    """
    Dual-purpose logger thread:
      • Consumes alert_queue → writes to 'Alerts' sheet
      • Samples system metrics every 5 s → writes to 'Performance' sheet
    """

    def __init__(self, config, shared_state):
        super().__init__(name="ExcelLogger", daemon=True)
        self.cfg    = config
        self.state  = shared_state

        os.makedirs(os.path.dirname(config.EXCEL_LOG_PATH), exist_ok=True)
        self._path  = config.EXCEL_LOG_PATH
        self._lock  = threading.Lock()      # serialize ALL workbook writes

        # Deduplication: key → last fired epoch
        self._cooldowns: dict[str, float] = {}
        self._cd_lock = threading.Lock()

        # Create workbook if it doesn't exist yet
        if not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
            _init_workbook(self._path)

    # ─────────────────────────────────────────────────────────
    # Public API (called from any thread)
    # ─────────────────────────────────────────────────────────

    def fire(self, alert_name: str, camera_source: str = "system",
             details: str = ""):
        """Enqueue an alert for logging. Dedup applied in thread."""
        self.state.alert_queue.put({
            "name":   alert_name,
            "camera": camera_source,
            "detail": details,
            "epoch":  time.time(),
        })

    # ─────────────────────────────────────────────────────────
    # Thread body
    # ─────────────────────────────────────────────────────────

    def run(self):
        log.info("ExcelLogger started")
        last_perf = time.time()

        while self.state.running or not self.state.alert_queue.empty():
            now = time.time()

            # ── Drain alert queue ─────────────────────────────
            while True:
                try:
                    alert = self.state.alert_queue.get_nowait()
                    self._write_alert(alert)
                except queue.Empty:
                    break

            # ── Performance sample ────────────────────────────
            if now - last_perf >= BENCHMARK_INTERVAL:
                last_perf = now
                self._write_perf()

            time.sleep(0.2)

        log.info("ExcelLogger stopped")

    # ─────────────────────────────────────────────────────────
    # Internal writers
    # ─────────────────────────────────────────────────────────

    def _write_alert(self, alert: dict):
        name   = alert["name"]
        camera = alert["camera"]
        detail = alert["detail"]
        epoch  = alert["epoch"]

        # ── Deduplication ─────────────────────────────────────
        dup_key = f"{name}::{camera}"
        now = time.time()
        with self._cd_lock:
            if now - self._cooldowns.get(dup_key, 0) < self.cfg.ALERT_COOLDOWN:
                return
            self._cooldowns[dup_key] = now

        ts  = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
        gps = self.state.get_gps()
        lat = f"{gps['lat']:.6f}" if gps["fix"] else "NO_FIX"
        lon = f"{gps['lon']:.6f}" if gps["fix"] else "NO_FIX"

        row = [name, ts, lat, lon, camera, detail]
        self._append_row("Alerts", row)

        # Mirror to dashboard log panel
        msg = f"[ALERT] {name} | {camera} | {detail} | GPS({lat},{lon})"
        self.state.add_log(msg, level="ALERT", source=camera.upper()[:12])
        log.warning(msg)

    def _write_perf(self):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        v    = self.state.get_stats("vehicle")
        d    = self.state.get_stats("driver")
        c    = self.state.get_stats("cargo")
        cpu  = psutil.cpu_percent(interval=None)
        ram  = psutil.virtual_memory()
        b    = self.state.get_benchmark()

        row = [
            ts,
            round(v["fps"],        1),
            round(d["fps"],        1),
            round(c["fps"],        1),
            round(v["latency_ms"], 2),
            round(d["latency_ms"], 2),
            round(c["latency_ms"], 2),
            round(cpu,             1),
            round(ram.percent,     1),
            round(ram.used/1_048_576, 1),
            round(b.get("system_temp_c", 0), 1),
            round(b.get("sensor_temp_c", 0), 1),
        ]
        self._append_row("Performance", row)

        # Update benchmark snapshot for dashboard
        self.state.update_benchmark(
            fps_vehicle  = v["fps"],
            fps_driver   = d["fps"],
            fps_cargo    = c["fps"],
            lat_vehicle_ms = v["latency_ms"],
            lat_driver_ms  = d["latency_ms"],
            lat_cargo_ms   = c["latency_ms"],
            cpu_percent  = cpu,
            ram_percent  = ram.percent,
            ram_used_mb  = round(ram.used/1_048_576, 1),
            timestamp    = ts,
        )

    def _append_row(self, sheet_name: str, row: list):
        with self._lock:
            try:
                wb = load_workbook(self._path)
                ws = wb[sheet_name]
                ws.append(row)
                wb.save(self._path)
            except Exception as e:
                log.error(f"Excel write error ({sheet_name}): {e}")
                # Attempt to recreate if file is corrupt
                try:
                    _init_workbook(self._path)
                except Exception:
                    pass
