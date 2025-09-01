import cv2
import numpy as np
import mediapipe as mp

# ---- Config ----
EAR_THRESHOLD = 0.25      # threshold for eye closed
CONSEC_FRAMES = 15        # number of consecutive frames to mark "Sleeping"

# MediaPipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye landmark indices
LEFT_EYE =  [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def ear_from_landmarks(lms, w, h, idxs):
    pts = [(lms[i].x * w, lms[i].y * h) for i in idxs]
    p1, p2, p3, p4, p5, p6 = pts
    vert = euclid(p2, p6) + euclid(p3, p5)
    horiz = 2.0 * euclid(p1, p4)
    if horiz == 0:
        return 0.0
    return vert / horiz, pts  # return both EAR and eye points

# Video input (0 = webcam, or give filename "video.mp4")
cap = cv2.VideoCapture(0)

frame_count = 0
sleeping = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0].landmark
        left_ear, left_pts = ear_from_landmarks(lms, w, h, LEFT_EYE)
        right_ear, right_pts = ear_from_landmarks(lms, w, h, RIGHT_EYE)
        mean_ear = (left_ear + right_ear) / 2.0

        # Check EAR threshold
        if mean_ear < EAR_THRESHOLD:
            frame_count += 1
            if frame_count >= CONSEC_FRAMES:
                sleeping = True
        else:
            frame_count = 0
            sleeping = False

        # Draw green outline around eyes
        for eye_pts in [left_pts, right_pts]:
            pts = np.array(eye_pts, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

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
