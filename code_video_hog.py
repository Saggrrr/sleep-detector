import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from pygame import mixer
from skimage.feature import hog

# ---- Config ----
EAR_THRESHOLD = 0.25      # EAR threshold for eye closed
SLEEP_TIME = 5            # seconds eyes must stay closed
ALARM_FILE = "alarm.mp3"

# Init pygame mixer
mixer.init()

# MediaPipe Face Mesh setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye landmark indices
LEFT_EYE =  [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Euclidean distance
def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Compute EAR from landmarks
def ear_from_landmarks(lms, w, h, idxs):
    pts = [(lms[i].x * w, lms[i].y * h) for i in idxs]
    p1, p2, p3, p4, p5, p6 = pts
    vert = euclid(p2, p6) + euclid(p3, p5)
    horiz = 2.0 * euclid(p1, p4)
    if horiz == 0:
        return 0.0, pts
    return vert / horiz, pts

# HOG-based eye closure
def compute_hog_closed(eye_pts, frame_gray):
    x, y, w, h = cv2.boundingRect(np.array(eye_pts))
    eye_patch = frame_gray[y:y+h, x:x+w]
    if eye_patch.size == 0:
        return False
    features = hog(eye_patch, pixels_per_cell=(8,8), cells_per_block=(1,1))
    return np.mean(features) < 0.1

# Alarm thread control
alarm_active = False
def alarm_loop():
    global alarm_active
    mixer.music.load(ALARM_FILE)
    mixer.music.play(-1)  # loop forever
    while alarm_active:
        time.sleep(0.1)
    mixer.music.stop()

# Video input (0 = webcam)
cap = cv2.VideoCapture(0)

eye_closed_start = None
sleeping = False

print("Press 'Q' in video window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = face_mesh.process(rgb)

    eyes_closed = False

    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0].landmark
        left_ear, left_pts = ear_from_landmarks(lms, w, h, LEFT_EYE)
        right_ear, right_pts = ear_from_landmarks(lms, w, h, RIGHT_EYE)
        mean_ear = (left_ear + right_ear) / 2.0

        left_closed_hog = compute_hog_closed(left_pts, gray)
        right_closed_hog = compute_hog_closed(right_pts, gray)
        eyes_closed = (mean_ear < EAR_THRESHOLD) or (left_closed_hog and right_closed_hog)

        # Draw eyes
        for eye_pts in [left_pts, right_pts]:
            pts = np.array(eye_pts, dtype=np.int32)
            color = (0, 0, 255) if sleeping else (0, 255, 0)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    else:
        mean_ear = 0.0  # No face detected

    # Eye closed logic
    if eyes_closed:
        if eye_closed_start is None:
            eye_closed_start = time.time()
        elapsed = time.time() - eye_closed_start
        if elapsed >= SLEEP_TIME and not sleeping:
            sleeping = True
            alarm_active = True
            threading.Thread(target=alarm_loop, daemon=True).start()
    else:
        eye_closed_start = None
        if sleeping:
            sleeping = False
            alarm_active = False  # stop alarm immediately

    # Display info
    status = "Sleeping" if sleeping else "Not Sleeping"
    cv2.putText(frame, f"EAR: {mean_ear:.3f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status, (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255) if sleeping else (0, 255, 0), 3)

    cv2.imshow("Sleep Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
alarm_active = False
mixer.music.stop()
