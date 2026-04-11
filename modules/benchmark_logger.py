# ============================================================
# modules/benchmark_logger.py — System Benchmark Logger (Thread-5)
# Logs FPS, Latency, CPU, RAM, Temperature to benchmark_log.csv
# ============================================================

import csv
import os
import time
import psutil
import threading
import logging
from datetime import datetime

log = logging.getLogger(__name__)


class BenchmarkLogger(threading.Thread):
    """
    Thread-5: Periodically samples system + detector metrics and
    appends a row to benchmark_log.csv.
    """

    def __init__(self, config, shared_state):
        super().__init__(name="BenchmarkLogger", daemon=True)
        self.cfg   = config
        self.state = shared_state
        os.makedirs(os.path.dirname(config.BENCHMARK_CSV), exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        path = self.cfg.BENCHMARK_CSV
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.cfg.BENCHMARK_HEADER)
            log.info(f"Created benchmark log: {path}")

    def run(self):
        log.info("BenchmarkLogger started")

        while self.state.running:
            time.sleep(self.cfg.BENCHMARK_INTERVAL)
            self._sample()

        log.info("BenchmarkLogger stopped")

    def _sample(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Detector stats
        v_stats  = self.state.get_stats("vehicle")
        dc_stats = self.state.get_stats("driver_cargo")

        # System metrics
        cpu     = psutil.cpu_percent(interval=0.5)
        ram     = psutil.virtual_memory()
        temp    = self.state.get_benchmark().get("temperature_c", 0.0)

        row = [
            now,
            round(v_stats["fps"],        1),
            round(dc_stats["fps"],       1),
            round(v_stats["latency_ms"], 2),
            round(dc_stats["latency_ms"],2),
            round(cpu,                   1),
            round(ram.percent,           1),
            round(ram.used / 1_048_576,  1),  # bytes → MB
            round(temp,                  1),
        ]

        try:
            with open(self.cfg.BENCHMARK_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except OSError as e:
            log.error(f"Benchmark CSV write failed: {e}")

        # Update shared state for dashboard
        self.state.update_benchmark(
            fps_vehicle          = v_stats["fps"],
            fps_driver_cargo     = dc_stats["fps"],
            latency_vehicle_ms   = v_stats["latency_ms"],
            latency_dc_ms        = dc_stats["latency_ms"],
            cpu_percent          = cpu,
            ram_percent          = ram.percent,
            ram_used_mb          = round(ram.used / 1_048_576, 1),
            timestamp            = now,
        )
