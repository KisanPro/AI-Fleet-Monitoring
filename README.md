# Fleet Monitor — Jetson Orin Nano
### Senior Embedded AI System | Production Deployment Guide

---

## 🏗 System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                       FLEET MONITOR                           │
│                     Jetson Orin Nano                          │
├────────────────────┬────────────────────┬───────────────────┤
│  Camera 0 (Road)   │  Camera 1 (Driver) │ Camera 2 (Cargo)    │
│  ALWAYS ACTIVE     │  SWITCHABLE        │ SWITCHABLE          │
└────────────────────┴────────────────────┴───────────────────┘
         │                     │                    │
   ┌─────▼──────┐        ┌─────▼───────┐      ┌─────▼───────┐
   │VehicleDet  │        │Drowsiness   │      │CargoDetector│
   │Thread      │        │Detector T   │      │Thread       │
   └────────────┘        └──────┬──────┘      └──────┬──────┘
         │                      │                    │
   ┌─────▼──────────────────────▼────────────────────▼────────┐
   │                    ExcelLogger (T)                         │
   │      fleet_log.xlsx — (Sheet: Alerts, Sheet: Performance)   │
   └──────────────────────────────────────────────────────────┘
   ┌──────────────────────────────────────────────────────────┐
   │  GPSManager T (SIM7600G-H) | Automatic Port/AT Detection │
   │  SensorManager T (DS18B20 Temp, HC-SR04 Ultrasonic)      │
   │  Flask+SocketIO Dashboard (Main)                         │
   └──────────────────────────────────────────────────────────┘
```

---

## 🔌 Hardware Connections

### Cameras
| Role | Interface | Status |
|------|-----------|--------|
| Vehicle/Road | USB or CSI | **Always On**, Auto-detected index 0 |
| Driver Drowsiness | USB or CSI | **Toggleable**, Auto-detected index 1 |
| Cargo Monitoring | USB or CSI | **Toggleable**, Auto-detected index 2 |

*Note: Toggling physically suspends unused cameras to conserve memory and processing power.*

### GPIO Sensors (BOARD numbering, Jetson Orin Nano)
| Pin | Component / Purpose |
|-----|---------------------|
| 11 | Ultrasonic TRIG (Cargo door mechanism) |
| 13 | Ultrasonic ECHO (Cargo door mechanism) |
| 15 | Buzzer (PWM) |

### Additional Modules
- **GPS (SIM7600G-H)**: Connect via USB (appears as `/dev/ttyUSB*`). The script automatically loops, detects the correct COM port, auto-sends AT engine initialization commands, and retries until coordinates drop.
- **Temperature (DS18B20)**: Connect to 1-Wire GPIO (sysfs: `/sys/bus/w1/devices/28-*/w1_slave`). This standalone temperature sensor governs the High Temperature Alert (independent of Jetson SoC thermals).

---

## 🚀 Execution & Setup Guide

```bash
# 1. Clone / copy the complete project to your Jetson
cd /home/pathvision/Downloads
# (If downloading ZIP, unzip it as JRF_V1)
cd JRF_V1

# 2. Add YOLOv8 engine fallbacks (Optional, script checks anyway)
mkdir -p models
# If `yolov8s.engine` isn't found, it defaults to `yolov8n.engine`. 
# To convert models for Jetson, use on your system:
# yolo export model=yolov8n.pt format=engine
# yolo export model=yolov8s.pt format=engine

# 3. Ensure OS-level permissions (For GPIO, I2C, and USB serial)
sudo usermod -aG dialout $USER
sudo usermod -aG gpio $USER

# 4. Install required packages
pip install -r requirements.txt --break-system-packages

# 5. Run the System!
# Simply execute the main file. Do NOT start individual modules.
python3 main.py
```

*Go to **http://localhost:5000** on your machine or `http://<jetson-ip>:5000` to see the live system.*

---

## 🎛 Dashboard Design Layout (2x2 Grid)

The interface boasts a fully dynamic, real-time 2x2 grid containing three camera streams and one live tracking map:

```text
┌─────────────────────────┬─────────────────────────┐
│           Q1            │           Q2            │
│  🛣️ Road Camera         │  📦 Cargo Camera        │
│  (Always Active, FPS)   │  (Suspending/Active)    │
├─────────────────────────┼─────────────────────────┤
│           Q3            │           Q4            │
│  🚗 Driver Camera       │  🛰️ GPS Map API Tracker │
│  (Suspending/Active)    │  (Leaflet/OSM Realtime) │
└─────────────────────────┴─────────────────────────┘
```
**Switch Button Behavior**: Central switch actively dictates the active camera mode. Activating Q3 Driver automatically physically closes the hardware resources linked to Q2 Cargo—triggering a clear "CLOSED" state label overlay on inactive grid quadrants.

---

## 📊 Single Log File (Excel Multi-Sheet Output)

As requested, all tabular logs flow into **a single file separated uniquely by Sheets/Tabs**. Since standard CSV formats physically cannot handle multiple sheets within a single file natively, the built-in `ExcelLogger` library handles this via `logs/fleet_log.xlsx` structured exactly as a CSV spreadsheet would be. 

File Path: `logs/fleet_log.xlsx`
1. **Sheet: `Alerts`** -> Columns: `[Alert_Name | Timestamp | GPS_Lat | GPS_Lon | Camera_Source | Details]`
2. **Sheet: `Performance`** -> Columns: `[Timestamp | FPS (all cameras) | Inference latency | CPU % | RAM % | Sensor_Temp_C | System_Temp_C (Jetson)]`

## 🧠 Core Systems & Alert Rules

- **Drowsiness Calibrations (4s)**: Auto-calibrates individual EAR/MAR constraints dynamically over 4 seconds before beginning to monitor.
- **Sleepy Alert**: EAR drops below strictly > `2.0s`. Sounds buzzer, logs "sleepy".
- **Yawning Alert**: MAR crosses threshold strictly > `3.0s`. Sounds buzzer, logs "yawning".
- **Distraction Alert**: Head fully turned horizontally > `8.0s`. Sounds buzzer, logs "distraction".
- **Phone Detection**: Mobile in hand recognized immediately via YOLOv8 model. Logs "phone detected".
- **Cargo Rolling Baseline Accuracy**: Re-evaluates baseline items fully automatically every `15s` against the `yolov8s` tracker. Explicitly drops `missing` logs tracking missing item/quantity gaps. CONF index tightened to 0.25 logic for maximum object recall.
