# ============================================================
# modules/gps_manager.py — SIM7600G-H GPS Manager
#
# Behaviour:
#   1. Auto-scan /dev/ttyUSB* ports for SIM7600.
#   2. Send AT+CGPS=1 to enable the GPS engine.
#   3. Poll AT+CGPSINFO every 3 seconds for coordinates.
#   4. Fallback to mmcli if serial ports are busy.
#   5. Map shows live lat/lon on dashboard (Leaflet/OSM).
# ============================================================

import serial
import serial.tools.list_ports
import time
import threading
import logging
import subprocess
import glob

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────

def _parse_cgpsinfo(response: str) -> dict | None:
    """
    Parses response: '+CGPSINFO: 1317.221851,N,07735.691476,E,080426,115032.0,932.2,0.0,0.0'
    """
    if "+CGPSINFO:" not in response or "," not in response:
        return None
    
    try:
        data = response.split(":")[1].strip()
        fields = data.split(",")
        
        if not fields[0]: # No fix yet
            return None
            
        # Latitude: DDMM.MMMMMM
        raw_lat = fields[0]
        lat = float(raw_lat[:2]) + float(raw_lat[2:]) / 60.0
        if fields[1] == "S":
            lat = -lat
            
        # Longitude: DDDMM.MMMMMM
        raw_lon = fields[2]
        lon = float(raw_lon[:3]) + float(raw_lon[3:]) / 60.0
        if fields[3] == "W":
            lon = -lon
            
        return {
            "lat": lat,
            "lon": lon,
            "altitude": float(fields[6] or 0),
            "speed_kmh": float(fields[7] or 0) * 1.852, # knots to kmh
            "fix": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        log.debug(f"GPS: Parse error on +CGPSINFO: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Port detection & AT commands
# ─────────────────────────────────────────────────────────────

def _find_sim7600_port(candidates: list, baud: int) -> serial.Serial | None:
    """Find a port that responds to AT."""
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    # Prioritise candidates from config
    for c in candidates:
        if c in ports:
            ports.remove(c)
            ports.insert(0, c)

    for port in ports:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            ser.write(b"AT\r\n")
            time.sleep(0.5)
            resp = ser.read_all().decode(errors="ignore")
            if "OK" in resp:
                log.info(f"GPS: SIM7600 AT port detected on {port}")
                return ser
            ser.close()
        except Exception:
            pass
    return None


def _init_gps_engine(ser: serial.Serial):
    """Start the GPS engine if not already running."""
    # Check status first
    ser.write(b"AT+CGPS?\r\n")
    time.sleep(0.5)
    resp = ser.read_all().decode(errors="ignore")
    
    if "+CGPS: 1" in resp:
        log.info("GPS Engine is already active.")
        return

    # Try to start it
    ser.write(b"AT+CGPS=1\r\n")
    time.sleep(1)
    resp = ser.read_all().decode(errors="ignore")
    if "OK" in resp:
        log.info("GPS Engine started successfully.")
    else:
        log.warning(f"GPS Engine enable response: {resp.strip()}")


# ─────────────────────────────────────────────────────────────
# GPSManager thread
# ─────────────────────────────────────────────────────────────

class GPSManager(threading.Thread):

    def __init__(self, config, shared_state, excel_logger):
        super().__init__(name="GPSManager", daemon=True)
        self.cfg    = config
        self.state  = shared_state
        self.alerts = excel_logger

    def run(self):
        log.info("GPSManager starting …")

        ser = None
        while self.state.running and ser is None:
            ser = _find_sim7600_port(self.cfg.GPS_CANDIDATE_PORTS, self.cfg.GPS_BAUDRATE)
            if ser is None:
                log.warning("GPS module not found — retrying mmcli enable + wait...")
                # Try to enable via mmcli as fallback if port is locked
                try:
                    subprocess.run(["mmcli", "-m", "0", "--location-enable-gps-nmea"], capture_output=True, timeout=5)
                except: pass
                time.sleep(10)

        if ser is None:
            return

        _init_gps_engine(ser)
        self.state.add_log("GPS module ready — polling for coordinates…", source="GPS")

        while self.state.running:
            try:
                # Poll for coordinates
                ser.write(b"AT+CGPSINFO\r\n")
                time.sleep(0.5)
                resp = ser.read_all().decode(errors="ignore")
                
                data = _parse_cgpsinfo(resp)
                if data:
                    # Update shared state
                    prev_fix = self.state.get_gps().get("fix", False)
                    if not prev_fix:
                        self.state.add_log(f"GPS fix acquired: {data['lat']:.5f}, {data['lon']:.5f}", source="GPS")
                    
                    self.state.update_gps(**data)
                else:
                    # If we don't have a fix, ensure state reflects that
                    self.state.update_gps(fix=False)
                
                time.sleep(2.5) # Total ~3 seconds interval

            except serial.SerialException as e:
                log.error(f"GPS serial error: {e} — re-probing...")
                ser.close()
                ser = None
                while self.state.running and ser is None:
                    ser = _find_sim7600_port(self.cfg.GPS_CANDIDATE_PORTS, self.cfg.GPS_BAUDRATE)
                    time.sleep(5)
                if ser:
                    _init_gps_engine(ser)

        if ser:
            ser.close()
        log.info("GPSManager stopped")
