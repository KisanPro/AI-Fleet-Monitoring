# 🚛 AI Fleet Monitoring System  
## Jetson Orin Nano + 3 USB Cameras + SIM7600 USB

---

# 📌 Project Overview

This project builds a real-time AI fleet monitoring system using:

- NVIDIA Jetson Orin Nano
- 3 USB Cameras
- SIM7600 4G LTE + GPS module
- YOLOv8 AI model
- MediaPipe AI models
- Flask Web Dashboard

The system can:

- 🛣 Detect objects on the road (always active)
- 👤 Detect driver drowsiness
- 📦 Detect unauthorized cargo access
- 📍 Log GPS location
- 📲 Send SMS alerts
- 💬 Send Telegram alerts
- 🌐 Show live camera streams in browser
- 🔁 Switch between Driver and Cargo modes

---

# 🧰 Hardware Requirements

## 1️⃣ Processing Unit
- NVIDIA Jetson Orin Nano (8GB recommended)
- Official power adapter
- Cooling fan (recommended)

## 2️⃣ Cameras (3x USB Webcams)
- Any UVC compatible USB cameras

Connect as:

| USB Port | Camera Role |
|----------|------------|
| /dev/video0 | Driver Camera |
| /dev/video1 | Road Camera |
| /dev/video2 | Cargo Camera |
Adjust the port Number after connections

## 3️⃣ SIM7600 USB LTE + GPS Module
- SIM7600 USB version
- Active SIM card (SMS enabled)
- GPS antenna connected

Plug SIM7600 into USB.

## 4️⃣ Internet
- WiFi or Ethernet for setup
- Optional: SIM7600 data

---

# 🖥 Operating System Setup

1. Flash Jetson using NVIDIA SDK Manager.
2. Install JetPack 5.x or 6.x.
3. Boot into Ubuntu desktop.
4. Connect to internet.

Update system:

```bash
sudo apt update
sudo apt upgrade -y
```

---

# 📷 Verify Cameras

Check camera devices:

```bash
ls /dev/video*
```

Expected:

```
/dev/video0
/dev/video1
/dev/video2
```

Test visually:

```bash
sudo apt install cheese
cheese
```

Confirm:
- video0 = Driver
- video1 = Road
- video2 = Cargo

If order is wrong, swap USB ports.

---

# 📡 Verify SIM7600

Check ports:

```bash
ls /dev/ttyUSB*
```

Expected:

```
/dev/ttyUSB0
/dev/ttyUSB1
/dev/ttyUSB2
/dev/ttyUSB3
```

Install minicom:

```bash
sudo apt install minicom
```

Test connection:

```bash
minicom -D /dev/ttyUSB2 -b 115200
```

Inside minicom:

```
AT
```

Response should be:

```
OK
```

Enable GPS:

```
AT+CGPS=1
```

Exit: `CTRL + A`, then `X`

---

# 📦 Install Python Dependencies

```bash
pip install ultralytics mediapipe flask opencv-python pyserial requests numpy
```

Optional (performance improvement):

```bash
pip install PyTurboJPEG
```

---

# 🤖 YOLO Model Setup

Option 1 (simple):
Script auto-downloads YOLOv8n.

Option 2 (recommended - faster):

```bash
yolo export model=yolov8n.pt format=engine
```

Place `yolov8n.engine` in project folder.

---

# 💬 Telegram Bot Setup

## Create Bot

1. Open Telegram
2. Search `@BotFather`
3. Type:
   ```
   /newbot
   ```
4. Copy the Bot Token.

## Get Chat ID

Send a message to your bot.

Open:

```
https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
```

Find:

```
"chat":{"id":123456789}
```

Copy the ID.

---

# 📁 Project Structure

```
ai_fleet/
│
├── ai_fleet.py
├── yolov8n.pt (optional)
├── ai_fleet_log.csv
└── README.md
```

---

# ▶️ Run the System

Navigate to project folder:

```bash
cd ai_fleet
```

Run:

```bash
python3 ai_fleet.py
```

---

# 🌐 Open Dashboard

Find Jetson IP:

```bash
hostname -I
```

Open in browser:

```
http://<jetson_ip>:5000
```

You will see:

- Road Stream (always active)
- Secondary Stream (Driver or Cargo)
- Mode switch buttons

---

# 🔁 Mode Switching

Default mode: Driver

Driver Mode:
- Uses Camera 0 + Camera 1

Cargo Mode:
- Uses Camera 2 + Camera 1

When switching:
- Secondary camera is released
- New camera starts
- Road camera continues running

---

# 🚨 Alert System

When triggered:

System sends:
- Telegram message
- SMS message
- Logs event to CSV

Alerts include:
- DRIVER DROWSY
- UNAUTHORIZED CARGO ACCESS
- HIGH VIBRATION
- DOOR OPEN

---

# 📍 GPS Logging

Every 5 seconds:
- Reads GPS from SIM7600
- Saves to `ai_fleet_log.csv`

---

# 📊 Log File Format

```
timestamp,module,submodule,message
```

Example:

```
2026-02-12T10:22:15,GPS,DATA,+CGPSINFO:...
```

---

# ⚙️ Auto Start on Boot (Optional)

Create service file:

```bash
sudo nano /etc/systemd/system/ai_fleet.service
```

Paste:

```
[Unit]
Description=AI Fleet System
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/<username>/ai_fleet/ai_fleet.py
WorkingDirectory=/home/<username>/ai_fleet
Restart=always
User=<username>

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai_fleet
sudo systemctl start ai_fleet
```

---

# 🛠 Troubleshooting

## Camera Not Detected
```
ls /dev/video*
```
Replug USB.

## SIM Not Detected
```
ls /dev/ttyUSB*
```

## GPS Not Working
Inside minicom:
```
AT+CGPS=1
```

## Low FPS
Reduce resolution in script:
```
self.cap.set(3,640)
self.cap.set(4,480)
```

---

# 📌 Final System Behavior

| Feature | Status |
|----------|--------|
| Road Detection | Always Active |
| Driver Detection | Switchable |
| Cargo Detection | Switchable |
| GPS Logging | Active |
| SMS Alerts | Active |
| Telegram Alerts | Active |
| Dashboard | Active |

---

# 🎓 Project Result

This system creates a real-time AI fleet monitoring node capable of:

- Driver safety monitoring
- Cargo security monitoring
- Road object detection
- GPS tracking
- Remote alerting
- Live video streaming
- Dynamic mode switching
