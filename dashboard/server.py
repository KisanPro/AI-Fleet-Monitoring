# ============================================================
# dashboard/server.py — Flask + SocketIO Dashboard Server
# 3 MJPEG streams: /video/vehicle  /video/driver  /video/cargo
# Toggle: switches shared_state.camera_mode driver ↔ cargo
# ============================================================

import time
import threading
import logging
import numpy as np
import cv2
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit

log = logging.getLogger(__name__)

app      = Flask(__name__, template_folder="templates",
                static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_state   = None
_cfg     = None
_cam_mgr = None

def init(config, shared_state, camera_manager=None):
    global _state, _cfg, _cam_mgr
    _cfg     = config
    _state   = shared_state
    _cam_mgr = camera_manager


# ─────────────────────────────────────────────────────────────
# MJPEG helpers
# ─────────────────────────────────────────────────────────────

def _blank(w=640, h=480, msg="NO SIGNAL") -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(img, msg, ((w-tw)//2, (h+th)//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70,70,70), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _stream(cam_key: str):
    blank = _blank()
    while True:
        frame = _state.get_frame(cam_key) if _state else None
        payload = frame if frame else blank
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + payload + b"\r\n")
        time.sleep(0.033)


# ─────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video/vehicle")
def vid_vehicle():
    return Response(_stream("vehicle"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/driver")
def vid_driver():
    return Response(_stream("driver"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/cargo")
def vid_cargo():
    return Response(_stream("cargo"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    if not _state:
        return jsonify({})
    return jsonify({
        "mode":      _state.camera_mode,
        "gps":       _state.get_gps(),
        "benchmark": _state.get_benchmark(),
        "cameras":   _state.camera_indices,
    })


# ─────────────────────────────────────────────────────────────
# SocketIO events
# ─────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    if _state:
        emit("mode_changed", {"mode": _state.camera_mode})
        emit("gps_update",   _state.get_gps())
        emit("bench_update", _state.get_benchmark())


@socketio.on("toggle_camera")
def on_toggle():
    if _state:
        new_mode = _state.toggle_camera_mode()
        
        # Suspend inactive, resume active
        if _cam_mgr:
            if new_mode == "cargo":
                _cam_mgr.suspend("driver")
                _cam_mgr.resume("cargo")
            else:
                _cam_mgr.suspend("cargo")
                _cam_mgr.resume("driver")

        _state.add_log(
            f"Camera mode toggled → {new_mode.upper()}",
            source="DASHBOARD",
        )
        socketio.emit("mode_changed", {"mode": new_mode})
        log.info(f"Camera mode → {new_mode}")


# ─────────────────────────────────────────────────────────────
# Background push loop (1 Hz)
# ─────────────────────────────────────────────────────────────

def _push_loop():
    last_log_idx = 0
    while True:
        time.sleep(1.0)
        if not _state:
            continue

        socketio.emit("gps_update",   _state.get_gps())
        socketio.emit("bench_update", _state.get_benchmark())

        logs = _state.get_logs()
        if len(logs) > last_log_idx:
            socketio.emit("log_batch", logs[last_log_idx:])
            last_log_idx = len(logs)


# ─────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────

def serve(config, shared_state, camera_manager=None):
    init(config, shared_state, camera_manager)
    threading.Thread(target=_push_loop, name="SocketPusher", daemon=True).start()
    log.info(f"Dashboard → http://localhost:{config.DASHBOARD_PORT}")
    socketio.run(app,
                 host=config.DASHBOARD_HOST,
                 port=config.DASHBOARD_PORT,
                 debug=False,
                 use_reloader=False,
                 log_output=False)
