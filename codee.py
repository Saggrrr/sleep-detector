import cv2
import os
import numpy as np
import mediapipe as mp

# ---- Config ----
FOLDER = "images"
EAR_THRESHOLD = 0.25  # tweak: 0.22–0.28 usually works
DRAW = True           # set False if you don't want drawings

# MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Eye landmark indices for MediaPipe Face Mesh
# EAR formula uses: (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
LEFT_EYE =  [33, 160, 158, 133, 153, 144]  # p1, p2, p3, p4, p5, p6
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # p1, p2, p3, p4, p5, p6

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def points_from_landmarks(landmarks, img_w, img_h, idxs):
    pts = []
    for i in idxs:
        lm = landmarks[i]
        pts.append((lm.x * img_w, lm.y * img_h))
    return pts  # [(x1,y1), ..., (x6,y6)]

def ear_from_points(pts):
    # pts order: [p1, p2, p3, p4, p5, p6]
    p1, p2, p3, p4, p5, p6 = pts
    vert = euclid(p2, p6) + euclid(p3, p5)
    horiz = 2.0 * euclid(p1, p4)
    if horiz == 0:
        return 0.0
    return vert / horiz

def draw_eye_debug(img, pts, color=(0, 255, 0)):
    # draw small circles
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    # draw lines p1-p4 (horiz), p2-p6 and p3-p5 (vert)
    p1, p2, p3, p4, p5, p6 = pts
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p4[0]), int(p4[1])), color, 1)
    cv2.line(img, (int(p2[0]), int(p2[1])), (int(p6[0]), int(p6[1])), color, 1)
    cv2.line(img, (int(p3[0]), int(p3[1])), (int(p5[0]), int(p5[1])), color, 1)

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
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(img_rgb)
    if not result.multi_face_landmarks:
        print(f"{fname}: no face detected")
        continue

    lms = result.multi_face_landmarks[0].landmark

    # Get eye points
    left_pts  = points_from_landmarks(lms, w, h, LEFT_EYE)
    right_pts = points_from_landmarks(lms, w, h, RIGHT_EYE)

    # Compute EAR
    left_ear  = ear_from_points(left_pts)
    right_ear = ear_from_points(right_pts)
    mean_ear  = (left_ear + right_ear) / 2.0

    status = "Not Sleeping" if mean_ear > EAR_THRESHOLD else "Sleeping"

    # Draw
    out = img_bgr.copy()
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

face_mesh.close()
