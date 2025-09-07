import cv2
import os
import numpy as np
import dlib
from scipy.spatial import distance as dist

# ---- Config ----
FOLDER = "images"
EAR_THRESHOLD = 0.25
DRAW = True

# Dlib setup
# Download the shape_predictor_68_face_landmarks.dat file and place it in the same directory
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indices for Dlib's 68-point model
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def draw_eye_debug(img, pts, color=(0, 255, 0)):
    # draw small circles
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    # draw lines
    cv2.line(img, (int(pts[0][0]), int(pts[0][1])), (int(pts[3][0]), int(pts[3][1])), color, 1)
    cv2.line(img, (int(pts[1][0]), int(pts[1][1])), (int(pts[5][0]), int(pts[5][1])), color, 1)
    cv2.line(img, (int(pts[2][0]), int(pts[2][1])), (int(pts[4][0]), int(pts[4][1])), color, 1)

# Process all images
for fname in sorted(os.listdir(FOLDER)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(FOLDER, fname)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"{fname}: could not read image")
        continue

    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(img_gray, 0)
    if len(faces) == 0:
        print(f"{fname}: no face detected")
        continue

    # Assuming only one face for simplicity
    face = faces[0]
    shape = predictor(img_gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # Get eye points from landmarks
    left_eye_points = landmarks[LEFT_EYE]
    right_eye_points = landmarks[RIGHT_EYE]

    # Compute EAR
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    mean_ear = (left_ear + right_ear) / 2.0

    status = "Not Sleeping" if mean_ear > EAR_THRESHOLD else "Sleeping"

    # Draw
    out = img_bgr.copy()
    if DRAW:
        draw_eye_debug(out, left_eye_points, (0, 255, 0))
        draw_eye_debug(out, right_eye_points, (0, 255, 255))
        cv2.putText(out, f"EAR L:{left_ear:.3f} R:{right_ear:.3f} M:{mean_ear:.3f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if status == "Not Sleeping" else (0, 0, 255), 2)

    print(f"{fname}: {status}  (EAR={mean_ear:.3f})")
    cv2.imshow(f"{fname} - {status}", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()