# ============================================================
# modules/sensor_manager.py — Temperature & Ultrasonic Sensors
#
# Temperature dual-source design:
#   DS18B20 (1-Wire)  → cargo temp ALERT logic  (TEMP_ALERT_THRESHOLD)
#   Jetson SoC thermal → benchmark log           (system health)
#
# Ultrasonic HC-SR04 → CARGO_DOOR_OPEN alert
# Buzzer             → shared PWM driver
# ============================================================

import os
import glob
import time
import threading
import logging

log = logging.getLogger(__name__)

ALERT_HIGH_TEMP = "HIGH_TEMPERATURE"
ALERT_DOOR_OPEN = "CARGO_DOOR_OPEN"


# ─────────────────────────────────────────────────────────────
# GPIO abstraction
# ─────────────────────────────────────────────────────────────

try:
    import Jetson.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO_AVAILABLE = True
    log.info("GPIO: Jetson.GPIO loaded")
except ImportError:
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO_AVAILABLE = True
        log.info("GPIO: RPi.GPIO loaded")
    except ImportError:
        GPIO = None
        GPIO_AVAILABLE = False
        log.warning("GPIO: No library found — sensors simulated")


# ─────────────────────────────────────────────────────────────
# Buzzer
# ─────────────────────────────────────────────────────────────

class Buzzer:
    """Thread-safe PWM buzzer. Non-blocking beep()."""

    def __init__(self, pin: int, freq=1000, duty=50):
        self.pin  = pin
        self._lock = threading.Lock()
        self._pwm  = None
        if GPIO_AVAILABLE:
            try:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                self._pwm = GPIO.PWM(pin, freq)
                log.info(f"Buzzer: GPIO pin {pin} ready")
            except Exception as e:
                log.warning(f"Buzzer init failed: {e}")

    def beep(self, duration=0.5):
        threading.Thread(target=self._beep, args=(duration,), daemon=True).start()

    def _beep(self, duration):
        with self._lock:
            if self._pwm:
                try:
                    self._pwm.start(50)
                    time.sleep(duration)
                    self._pwm.stop()
                except Exception:
                    pass
            else:
                log.debug(f"[BUZZER-SIM] {duration:.2f}s")
                time.sleep(duration)

    def cleanup(self):
        if self._pwm:
            try: self._pwm.stop()
            except Exception: pass


# ─────────────────────────────────────────────────────────────
# Temperature reading
# ─────────────────────────────────────────────────────────────

def read_ds18b20(base_path: str) -> float | None:
    """
    Read temperature from DS18B20 1-Wire sensor via sysfs.
    Returns °C or None if sensor not present.
    """
    devices = glob.glob(os.path.join(base_path, "28-*", "w1_slave"))
    if not devices:
        return None
    try:
        with open(devices[0]) as f:
            raw = f.read()
        if "YES" not in raw:
            return None
        idx = raw.find("t=")
        return float(raw[idx+2:]) / 1000.0
    except Exception:
        return None


def read_jetson_thermal(paths: list) -> float:
    """Read Jetson SoC temperature (for benchmark metrics only)."""
    for path in paths:
        try:
            with open(path) as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            continue
    return 0.0


# ─────────────────────────────────────────────────────────────
# Ultrasonic HC-SR04
# ─────────────────────────────────────────────────────────────

def measure_distance_cm(trig: int, echo: int) -> float:
    """Returns distance in cm, -1 on timeout."""
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig, GPIO.LOW)

    deadline = time.time() + 0.04
    while GPIO.input(echo) == 0:
        if time.time() > deadline:
            return -1.0
    t_start = time.time()

    deadline = t_start + 0.04
    while GPIO.input(echo) == 1:
        if time.time() > deadline:
            return -1.0
    t_end = time.time()

    return (t_end - t_start) * 17150.0


# ─────────────────────────────────────────────────────────────
# SensorManager thread
# ─────────────────────────────────────────────────────────────

class SensorManager(threading.Thread):

    def __init__(self, config, shared_state, excel_logger, buzzer: Buzzer):
        super().__init__(name="SensorManager", daemon=True)
        self.cfg    = config
        self.state  = shared_state
        self.alerts = excel_logger
        self.buzzer = buzzer
        self._gpio_ready = False
        self._ds18_present = False

    def run(self):
        log.info("SensorManager starting …")
        self._setup_gpio()
        self._probe_ds18b20()

        t_last  = 0.0
        u_last  = 0.0

        while self.state.running:
            now = time.time()

            # ── DS18B20 → cargo temperature alert ─────────────
            if now - t_last >= self.cfg.TEMP_POLL_INTERVAL:
                t_last = now

                # Always read system temp for benchmarks
                sys_temp = read_jetson_thermal(self.cfg.JETSON_THERMAL_PATHS)
                self.state.update_benchmark(system_temp_c=round(sys_temp, 1))

                # DS18B20 for alert threshold logic
                sensor_temp = read_ds18b20(self.cfg.W1_BASE_PATH)
                if sensor_temp is not None:
                    self._ds18_present = True
                    self.state.update_benchmark(sensor_temp_c=round(sensor_temp, 1))

                    if sensor_temp > self.cfg.TEMP_ALERT_THRESHOLD:
                        self.alerts.fire(
                            ALERT_HIGH_TEMP, "temperature_sensor",
                            f"DS18B20={sensor_temp:.1f}°C > "
                            f"threshold {self.cfg.TEMP_ALERT_THRESHOLD}°C",
                        )
                        self.buzzer.beep(1.0)
                        self.state.add_log(
                            f"HIGH TEMP: {sensor_temp:.1f}°C",
                            level="ALERT", source="TEMP_SENSOR",
                        )
                else:
                    # Fallback: use system temp if DS18B20 absent (for metrics only, NO ALERT)
                    if not self._ds18_present:
                        self.state.update_benchmark(sensor_temp_c=round(sys_temp, 1))

            # ── HC-SR04 → door open alert ─────────────────────
            if self._gpio_ready and now - u_last >= 1.0:
                u_last = now
                dist = measure_distance_cm(
                    self.cfg.ULTRASONIC_TRIG, self.cfg.ULTRASONIC_ECHO
                )
                if 0 < dist < self.cfg.DOOR_OPEN_CM:
                    self.alerts.fire(
                        ALERT_DOOR_OPEN, "ultrasonic_sensor",
                        f"distance={dist:.1f}cm < {self.cfg.DOOR_OPEN_CM}cm",
                    )
                    self.buzzer.beep(0.8)
                    self.state.add_log(
                        f"CARGO DOOR OPEN: {dist:.1f}cm",
                        level="ALERT", source="ULTRASONIC",
                    )

            time.sleep(0.2)

        if GPIO_AVAILABLE:
            GPIO.cleanup()
        log.info("SensorManager stopped")

    def _setup_gpio(self):
        if not GPIO_AVAILABLE:
            return
        try:
            GPIO.setup(self.cfg.ULTRASONIC_TRIG, GPIO.OUT)
            GPIO.setup(self.cfg.ULTRASONIC_ECHO, GPIO.IN)
            GPIO.output(self.cfg.ULTRASONIC_TRIG, GPIO.LOW)
            time.sleep(0.1)
            self._gpio_ready = True
            log.info("SensorManager: ultrasonic GPIO ready")
        except Exception as e:
            log.warning(f"SensorManager GPIO setup: {e}")

    def _probe_ds18b20(self):
        t = read_ds18b20(self.cfg.W1_BASE_PATH)
        if t is not None:
            self._ds18_present = True
            log.info(f"SensorManager: DS18B20 found — {t:.1f}°C")
            self.state.add_log(f"DS18B20 sensor: {t:.1f}°C", source="SENSOR")
        else:
            log.warning("SensorManager: DS18B20 not found — using SoC thermal fallback")
            self.state.add_log(
                "DS18B20 not found. Using Jetson SoC temp for alert logic.",
                level="WARN", source="SENSOR",
            )
