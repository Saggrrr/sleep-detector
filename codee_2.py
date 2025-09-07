import cv2
import os
import numpy as np
import dlib

# ---- Config ----
FOLDER = "images"
EAR_THRESHOLD = 0.25
DRAW = True

# Load dlib's face detector (HOG-based) and 68-point landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Landmark indices for eyes (from 68-point model)
LEFT_EYE  = [36, 37, 38, 39, 40, 41]  # p1..p6
RIGHT_EYE = [42, 43, 44, 45, 46, 47]  # p1..p6

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def points_from_landmarks(shape, idxs):
    return [(shape.part(i).x, shape.part(i).y) for i in idxs]

def ear_from_points(pts):
    p1, p2, p3, p4, p5, p6 = pts
    vert = euclid(p2, p6) + euclid(p3, p5)
    horiz = 2.0 * euclid(p1, p4)
    if horiz == 0:
        return 0.0
    return vert / horiz

def draw_eye_debug(img, pts, color=(0,255,0)):
    for (x, y) in pts:
        cv2.circle(img, (x, y), 2, color, -1)
    p1, p2, p3, p4, p5, p6 = pts
    cv2.line(img, p1, p4, color, 1)
    cv2.line(img, p2, p6, color, 1)
    cv2.line(img, p3, p5, color, 1)

# Process all images
for fname in sorted(os.listdir(FOLDER)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(FOLDER, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"{fname}: could not read image")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    if len(faces) == 0:
        print(f"{fname}: no face detected")
        continue

    for face in faces:
        shape = predictor(gray, face)

        left_pts  = points_from_landmarks(shape, LEFT_EYE)
        right_pts = points_from_landmarks(shape, RIGHT_EYE)

        left_ear  = ear_from_points(left_pts)
        right_ear = ear_from_points(right_pts)
        mean_ear  = (left_ear + right_ear) / 2.0

        status = "Not Sleeping" if mean_ear > EAR_THRESHOLD else "Sleeping"

        out = img.copy()
        if DRAW:
            draw_eye_debug(out, left_pts,  (0, 255, 0))
            draw_eye_debug(out, right_pts, (0, 255, 255))
            cv2.putText(out, f"EAR L:{left_ear:.3f} R:{right_ear:.3f} M:{mean_ear:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(out, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if status == "Not Sleeping" else (0, 0, 255), 2)

        print(f"{fname}: {status}  (EAR={mean_ear:.3f})")
        cv2.imshow(f"{fname} - {status}", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
