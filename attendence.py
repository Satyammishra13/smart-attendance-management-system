import cv2
import numpy as np
import csv
from datetime import datetime
from collections import deque

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

label_map = np.load("labels.npy", allow_pickle=True).item()

marked = set()
total_students = len(label_map)

# Buffer for smoothing predictions
prediction_buffer = deque(maxlen=10)

def mark_attendance(name):
    if name in marked:
        return

    marked.add(name)

    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), "Present"])

    print("Marked:", name)


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (600, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve lighting
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    name = "Unknown"

    for (x, y, w, h) in faces:
        # Larger region (helps during motion)
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = gray[y1:y2, x1:x2]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        # Lower confidence threshold for motion tolerance
        if confidence < 75:
            current_name = label_map[label]
        else:
            current_name = "Unknown"

        # Add to buffer
        prediction_buffer.append(current_name)

        # Majority voting (stabilizes recognition)
        if len(prediction_buffer) == prediction_buffer.maxlen:
            name = max(set(prediction_buffer), key=prediction_buffer.count)

        if name != "Unknown":
            mark_attendance(name)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(frame, name, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # -------- DASHBOARD --------
    dashboard = np.zeros((400, 400, 3), dtype=np.uint8)

    present = len(marked)
    absent = total_students - present

    cv2.putText(dashboard, "ATTENDANCE", (90, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.putText(dashboard, f"Total: {total_students}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(dashboard, f"Present: {present}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(dashboard, f"Absent: {absent}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(dashboard, "Students:", (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    y_offset = 280
    for n in list(marked)[-6:]:
        cv2.putText(dashboard, n, (60, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        y_offset += 25

    combined = np.hstack((frame, dashboard))

    cv2.imshow("Smart Attendance System", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()