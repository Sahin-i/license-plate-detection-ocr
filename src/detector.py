"""Ultralytics YOLOv8 plate detection."""

from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path="models/best.pt", confidence=0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image):
        result = self.model.predict(image, conf=self.confidence, classes=[0], verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
            })
        return detections
