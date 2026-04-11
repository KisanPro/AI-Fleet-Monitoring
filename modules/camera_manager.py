# ============================================================
# modules/camera_manager.py — Multi-Camera I/O
# Jetson Orin Nano
# ============================================================

import cv2
import time
import threading
import logging
import numpy as np

log = logging.getLogger(__name__)


class CameraReader(threading.Thread):
    """
    Background thread that keeps a local cv2.VideoCapture open.
    Updates self.frame as fast as possible.
    """

    def __init__(self, index, name, cfg):
        super().__init__(name=f"CamReader-{name}", daemon=True)
        self.index   = index
        self.name    = name
        self.cfg     = cfg
        self._cap    = None
        self.frame   = None
        self.running = True
        self.active  = False  # if False, we release the HW

    def run(self):
        log.info(f"Reader [{self.name}] thread starting …")
        while self.running:
            if not self.active:
                if self._cap:
                    log.info(f"Reader [{self.name}] releasing index {self.index}")
                    self._cap.release()
                    self._cap = None
                time.sleep(0.3)
                continue

            # Active!
            if self._cap is None:
                if not self._open():
                    # Retry logic handled inside _open or wait here
                    time.sleep(1.0)
                    continue

            ret, frame = self._cap.read()
            if not ret:
                log.warning(f"Reader [{self.name}] read fail. Releasing.")
                self._cap.release()
                self._cap = None
                continue

            self.frame = frame

        if self._cap:
            self._cap.release()

    def _open(self):
        """Try to open the camera index. Retries few times."""
        # Jetson GStreamer strings often preferred for CSI, but standard V4L2 for USB.
        # Since we use USB cams:
        for attempt in range(3):
            log.info(f"Reader [{self.name}] opening {self.index} (Attempt {attempt+1})…")
            # Explicitly use V4L2 backend for stability on Jetson/Linux
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            if cap.isOpened():
                # Set resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS,          self.cfg.CAMERA_FPS)
                
                # Check if it really works
                ok, _ = cap.read()
                if ok:
                    log.info(f"Reader [{self.name}] confirmed open at {self.index}")
                    self._cap = cap
                    return True
                cap.release()
            
            if attempt == 0:
                time.sleep(0.5)  # give it a moment
            else:
                time.sleep(1.0)
        
        log.error(f"Reader [{self.name}] failed to open {self.index}")
        return False

    def stop(self):
        self.running = False


class CameraManager:
    """
    Coordinates 3 cameras. Auto-probes if indexes aren't specified.
    - vehicle (always active)
    - driver  (active if mode='driver')
    - cargo   (active if mode='cargo')
    """

    def __init__(self, cfg, state):
        self.cfg   = cfg
        self.state = state
        self.readers = {
            "vehicle": None,
            "driver":  None,
            "cargo":   None,
        }
        self._active_aux = None  # tracker for driver vs cargo

    def start(self):
        log.info("CameraManager: probing cameras …")
        available = self._probe_indexes()
        
        # Mapping logic
        v_idx = self.cfg.CAM_VEHICLE_OVERRIDE if self.cfg.CAM_VEHICLE_OVERRIDE is not None else (available[0] if len(available) > 0 else 0)
        d_idx = self.cfg.CAM_DRIVER_OVERRIDE  if self.cfg.CAM_DRIVER_OVERRIDE  is not None else (available[1] if len(available) > 1 else 1)
        c_idx = self.cfg.CAM_CARGO_OVERRIDE   if self.cfg.CAM_CARGO_OVERRIDE   is not None else (available[2] if len(available) > 2 else 4)

        log.info(f"Camera Map: vehicle={v_idx}, driver={d_idx}, cargo={c_idx}")

        self.readers["vehicle"] = CameraReader(v_idx, "vehicle", self.cfg)
        self.readers["driver"]  = CameraReader(d_idx, "driver",  self.cfg)
        self.readers["cargo"]   = CameraReader(c_idx, "cargo",   self.cfg)

        # Start threads staggered to avoid USB bus pressure/race conditions
        for role, r in self.readers.items():
            log.info(f"CameraManager: starting {role} reader...")
            r.start()
            time.sleep(0.5)

        # Initial state
        self.readers["vehicle"].active = True
        self._sync_aux_mode()

    def _sync_aux_mode(self):
        """Toggle driver/cargo HW based on state.camera_mode."""
        mode = self.state.camera_mode
        if mode == "driver":
            self.readers["cargo"].active  = False
            self.readers["driver"].active = True
        elif mode == "cargo":
            self.readers["driver"].active = False
            self.readers["cargo"].active  = True

    def _probe_indexes(self, max_idx=10):
        found = []
        for i in range(max_idx):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                found.append(i)
                cap.release()
                log.info(f"  Camera index {i} → OK")
        return found

    def read(self, role):
        """Thread-safe pull of latest frame."""
        # When switching modes, we handle activation/deactivation here
        self._sync_aux_mode()
        reader = self.readers.get(role)
        if reader:
            return reader.frame
        return None

    def suspend(self, role: str):
        """Release HW camera for the given role."""
        reader = self.readers.get(role)
        if reader:
            log.info(f"CameraManager: suspending {role} reader")
            reader.active = False

    def resume(self, role: str):
        """Acquire HW camera for the given role."""
        reader = self.readers.get(role)
        if reader:
            log.info(f"CameraManager: resuming {role} reader")
            reader.active = True

    def stop(self):
        for r in self.readers.values():
            r.stop()
            r.join(timeout=1.0)
