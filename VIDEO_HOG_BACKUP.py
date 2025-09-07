import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# ---- Config ----
EAR_THRESHOLD = 0.20
DRAW = True

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def draw_eye_debug(img, pts, color=(0, 255, 0)):
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    cv2.line(img, tuple(pts[0]), tuple(pts[3]), color, 1)
    cv2.line(img, tuple(pts[1]), tuple(pts[5]), color, 1)
    cv2.line(img, tuple(pts[2]), tuple(pts[4]), color, 1)

# ---- Video Stream ----
cap = cv2.VideoCapture(0)  # 0 = default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        mean_ear = (left_ear + right_ear) / 2.0

        status = "Not Sleeping" if mean_ear > EAR_THRESHOLD else "Sleeping"

        if DRAW:
            draw_eye_debug(frame, left_eye, (0, 255, 0))
            draw_eye_debug(frame, right_eye, (0, 255, 255))
            cv2.putText(frame, f"EAR:{mean_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if status == "Not Sleeping" else (0, 0, 255), 2)

    cv2.imshow("Sleep Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
