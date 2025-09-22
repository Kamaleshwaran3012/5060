from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load input video
cap = cv2.VideoCapture("people.mp4")

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection (only persons)
    results = model(frame, classes=[0])

    # Annotate detections
    annotated_frame = results[0].plot()

    # Write frame to output file
    out.write(annotated_frame)

cap.release()
out.release()
print("✅ Processing complete! Check output.avi")
