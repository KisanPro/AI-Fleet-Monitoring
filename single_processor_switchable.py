# AI FLEET – 3 Cameras Connected (Dynamic Mode Switching)
# Road always active - Driver & Cargo switchable from dashboard

import cv2, time, threading, numpy as np, serial, requests, csv
from queue import Queue
from datetime import datetime
from flask import Flask, Response
from ultralytics import YOLO
import mediapipe as mp

# ---------------- CONFIG ----------------

DRIVER_CAM = 0
ROAD_CAM   = 1
CARGO_CAM  = 2

MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.4

LOG_FILE = "ai_fleet_log.csv"

TELEGRAM_URL = "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/sendMessage"
CHAT_ID = "<CHAT_ID>"
SMS_NUMBER = "<PHONE>"

MODE = "driver"   # default mode

# ---------------- LOGGING ----------------

def log_event(module, sub, msg):
    try:
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(), module, sub, msg
            ])
    except:
        pass

# ---------------- ALERTS ----------------

def send_telegram(msg):
    try:
        requests.post(TELEGRAM_URL,
                      data={"chat_id": CHAT_ID, "text": msg},
                      timeout=1)
    except:
        pass

# ---------------- SIM7600X ----------------

def init_sim7600():
    for p in ["/dev/ttyUSB2","/dev/ttyUSB3","/dev/ttyUSB1","/dev/ttyUSB0"]:
        try:
            return serial.Serial(p,115200,timeout=1)
        except:
            continue
    send_telegram("⚠️ SIM7600X NOT FOUND")
    return None

SIM = init_sim7600()

def send_sms(msg):
    if not SIM: return
    try:
        SIM.write(b'AT+CMGF=1\r')
        time.sleep(0.5)
        SIM.write(f'AT+CMGS="{SMS_NUMBER}"\r'.encode())
        time.sleep(0.5)
        SIM.write(msg.encode() + b'\x1A')
    except:
        pass

def send_alert(msg):
    send_telegram(msg)
    send_sms(msg)
    log_event("ALERT","SYSTEM",msg)

def read_gps():
    if not SIM: return None
    try:
        SIM.write(b'AT+CGPSINFO\r')
        time.sleep(0.5)
        for l in SIM.readlines():
            l=l.decode(errors="ignore")
            if "," in l:
                return l.strip()
    except:
        pass
    return None

# ---------------- JPEG ENCODING ----------------

try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    TURBO = True
except:
    jpeg = None
    TURBO = False

def encode(frame):
    if TURBO:
        try: return jpeg.encode(frame)
        except: pass
    return cv2.imencode(".jpg",frame)[1].tobytes()

# ---------------- CAMERA THREAD ----------------

class Cam(threading.Thread):
    def __init__(self, idx, q, name):
        super().__init__(daemon=True)
        self.idx = idx
        self.q = q
        self.name = name
        self.cap = None
        self.running = True
        self.init()

    def init(self):
        self.cap = cv2.VideoCapture(self.idx)
        if not self.cap.isOpened():
            self.cap = None
            send_alert(f"{self.name} CAMERA FAIL")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def run(self):
        while self.running:
            if not self.cap:
                time.sleep(1)
                self.init()
                continue

            ret, frm = self.cap.read()
            if not ret:
                self.cap.release()
                self.cap = None
                continue

            if self.q.full():
                self.q.get_nowait()
            self.q.put(frm)

            time.sleep(0.01)

# ---------------- AI MODELS ----------------

yolo = YOLO(MODEL_PATH)
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_hands = mp.solutions.hands.Hands()

# ---------------- QUEUES ----------------

road_q   = Queue(1)
second_q = Queue(1)

road_d   = Queue(1)
second_d = Queue(1)

secondary_thread = None

# ---------------- DRIVER EAR ----------------

def ear(lm, idx):
    p=[lm[i] for i in idx]
    v=np.linalg.norm([p[1].x-p[5].x,p[1].y-p[5].y])
    h=np.linalg.norm([p[0].x-p[3].x,p[0].y-p[3].y])
    return v/h if h else 0

# ---------------- ROAD LOOP ----------------

def road_loop():
    while True:
        f = road_q.get()
        r = yolo(f,conf=CONF_THRES,verbose=False)[0]
        for b in r.boxes.xyxy:
            x1,y1,x2,y2=map(int,b)
            cv2.rectangle(f,(x1,y1),(x2,y2),(0,255,0),2)

        if road_d.full(): road_d.get()
        road_d.put(f)

# ---------------- SECOND LOOP (Driver/Cargo) ----------------

def second_loop():
    global MODE
    drowsy_counter = 0

    while True:
        f = second_q.get()

        if MODE == "driver":
            r = mp_face.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            if r.multi_face_landmarks:
                lm=r.multi_face_landmarks[0].landmark
                e=(ear(lm,[33,160,158,133,153,144]) +
                   ear(lm,[362,385,387,263,373,380]))/2
                drowsy_counter = drowsy_counter+1 if e<0.18 else 0
                if drowsy_counter >= 12:
                    send_alert("⚠️ DRIVER DROWSY")
                    drowsy_counter = 0

        elif MODE == "cargo":
            r=yolo(f,conf=CONF_THRES,verbose=False)[0]
            for b in r.boxes.xyxy:
                x1,y1,x2,y2=map(int,b)
                cv2.rectangle(f,(x1,y1),(x2,y2),(255,0,0),2)

            h=mp_hands.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
            if h.multi_hand_landmarks:
                send_alert("⚠️ UNAUTHORIZED CARGO ACCESS")

        if second_d.full(): second_d.get()
        second_d.put(f)

# ---------------- ENV + GPS ----------------

def env_loop():
    while True:
        vib=np.random.randint(0,8)
        door=np.random.choice(["OPEN","CLOSED"])
        if vib>5: send_alert("⚠️ HIGH VIBRATION")
        if door=="OPEN": send_alert("⚠️ DOOR OPEN")
        time.sleep(2)

def gps_loop():
    while True:
        g=read_gps()
        if g: log_event("GPS","DATA",g)
        time.sleep(5)

# ---------------- SWITCH CAMERA ----------------

def switch_secondary_camera(mode):
    global secondary_thread

    if secondary_thread:
        secondary_thread.stop()
        time.sleep(1)

    if mode == "driver":
        secondary_thread = Cam(DRIVER_CAM, second_q, "DRIVER")
    elif mode == "cargo":
        secondary_thread = Cam(CARGO_CAM, second_q, "CARGO")

    secondary_thread.start()
    log_event("SYSTEM","MODE_CHANGE",mode)

# ---------------- FLASK ----------------

app=Flask(__name__)

def gen(q):
    while True:
        f=q.get()
        yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+encode(f)+b"\r\n"

@app.route("/")
def ui():
    return f"""
    <h2>Current Mode: {MODE.upper()}</h2>
    <a href='/set/driver'>Driver Mode</a> |
    <a href='/set/cargo'>Cargo Mode</a>
    <br><br>
    <img src=/road width=48%>
    <img src=/second width=48%>
    """

@app.route("/set/<mode>")
def set_mode(mode):
    global MODE
    if mode in ["driver","cargo"]:
        MODE = mode
        switch_secondary_camera(mode)
    return "Mode changed to " + MODE

@app.route("/road")
def road_stream():
    return Response(gen(road_d),
        mimetype="multipart/x-mixed-replace;boundary=frame")

@app.route("/second")
def second_stream():
    return Response(gen(second_d),
        mimetype="multipart/x-mixed-replace;boundary=frame")

# ---------------- MAIN ----------------

if __name__=="__main__":

    # Road always active
    Cam(ROAD_CAM, road_q, "ROAD").start()

    # Start secondary cam
    switch_secondary_camera(MODE)

    threading.Thread(target=road_loop,daemon=True).start()
    threading.Thread(target=second_loop,daemon=True).start()
    threading.Thread(target=env_loop,daemon=True).start()
    threading.Thread(target=gps_loop,daemon=True).start()

    app.run("0.0.0.0",5000,threaded=True)
