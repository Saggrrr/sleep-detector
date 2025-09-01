
import cv2
import os

# Step 1: Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Step 2: Path to images folder
folder_path = "images"

# Step 3: Loop through all images in the folder
for file_name in os.listdir(folder_path):
    if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
        continue  # skip non-images

    image_path = os.path.join(folder_path, file_name)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 4: Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    status = "Sleeping"

    # Step 5: If eyes detected, calculate eye aspect ratio (height/width)
    for (x, y, w, h) in eyes:
        ear_like = h / float(w)   # simple eye "aspect ratio"
        # Step 6: Classifier (threshold)
        # If eye box is tall enough → open eye → not sleeping
        if ear_like > 0.25:   # threshold chosen experimentally
            status = "Not Sleeping"

        # Draw rectangle on eye
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Step 7: Show result
    print(f"{file_name}: {status}")
    cv2.imshow(f"{file_name} - {status}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
