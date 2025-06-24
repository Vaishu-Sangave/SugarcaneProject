from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(r"C:\Users\VAISHNAVI\Desktop\SugarcaneProject\best (2).pt")

# Open two webcams (adjust device IDs if needed)
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both webcams.")
    exit()

# Confidence threshold for clean detections
CONFIDENCE_THRESHOLD = 0.6  # You can increase to 0.7 for stricter filtering

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Could not read from one or both cameras.")
        break

    # Run detection on both frames
    results1 = model(frame1)
    results2 = model(frame2)

    # Draw results for camera 1
    detections1 = results1[0].boxes
    for box in detections1:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0]

        if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame1, f"Bud {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw results for camera 2
    detections2 = results2[0].boxes
    for box in detections2:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0]

        if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame2, f"Bud {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show both frames
    cv2.imshow("Camera 1 - Bud Detection", frame1)
    cv2.imshow("Camera 2 - Bud Detection", frame2)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap1.release()
cap2.release()
cv2.destroyAllWindows()
