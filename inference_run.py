import cv2
from ultralytics import YOLO

# model = YOLO("yolo11s-seg.pt")
# model = YOLO("yolo11m.pt")
model = YOLO("runs/detect/train6/weights/best.pt")  # Load your trained model

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # set height

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame,
                            conf=0.25,    # confidence threshold
                            iou=0.45,     # NMS IoU threshold
                            stream=True)  # enable streaming for lower memory usage :contentReference[oaicite:0]{index=0}
    for res in results:
        annotated_frame = res.plot()
        cv2.imshow("YOLO11 Webcam", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # exit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()
