import cv2
import os
import time

name = input("Enter name: ")

folder = f"dataset/{name}"
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)

# Set camera resolution (important fix)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

print("Capturing 5 images automatically...")

while count < 5:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Capture", frame)

    # 🔥 Save image in BEST format
    img_path = f"{folder}/img{count}.jpg"
    cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print("Saved:", img_path)

    count += 1
    time.sleep(1)  # 1 sec gap

cap.release()
cv2.destroyAllWindows()

print("Done capturing images!")
