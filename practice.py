import cv2
import numpy as np
from pygame import mixer
import time

# ---- Config ----
EAR_THRESHOLD = 0.25  # Adjusted for rectangle EAR
DRAW = True
SLEEP_TIME = 1        # seconds eyes must stay closed to trigger alarm
ALARM_FILE = "alarm.mp3"

# ---- Initialize mixer ----
mixer.init(frequency=44100)

# ---- Functions ----
def rect_ear(rect):
    """Approximate EAR using rectangle: height / width"""
    _, _, w, h = rect
    return h / w

def draw_eye_debug(img, eye_rects, color=(0, 255, 0)):
    for (x, y, w, h) in eye_rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)

# ---- Video Stream ----
cap = cv2.VideoCapture(0)
sleep_start = None
alarm_playing = False

# ---- Haar cascades ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30,30))

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])  # Sort by x to get left/right eye
            left_ear = rect_ear(eyes[0])
            right_ear = rect_ear(eyes[1])
            mean_ear = (left_ear + right_ear) / 2.0
        else:
            mean_ear = 1.0  # assume eyes open if detection fails

        # ---- Sleep check ----
        if mean_ear <= EAR_THRESHOLD:
            if sleep_start is None:
                sleep_start = time.time()
            elif time.time() - sleep_start >= SLEEP_TIME:
                status = "Sleeping"
                if not alarm_playing:
                    mixer.music.load(ALARM_FILE)
                    mixer.music.play(-1)
                    alarm_playing = True
        else:
            sleep_start = None
            status = "Not Sleeping"
            if alarm_playing:
                mixer.music.stop()
                alarm_playing = False

        # ---- Draw debug ----
        if DRAW:
            draw_eye_debug(roi_color, eyes, (0,255,0))
            cv2.putText(frame, f"EAR:{mean_ear:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(frame, status, (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.8,
                        (0,255,0) if status=="Not Sleeping" else (0,0,255),2)

    cv2.imshow("Sleep Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
