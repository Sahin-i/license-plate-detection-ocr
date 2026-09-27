"""One callable detector -> padded crop -> EasyOCR -> cleaned text pipeline."""

import os
import cv2

from src.detector import PlateDetector
from src.ocr import PlateOCR
from src.utils import crop_plate, process_plate_text


class LicensePlatePipeline:
    def __init__(self, detector_path="models/best.pt", detector=None, ocr=None):
        self.detector = detector if detector is not None else PlateDetector(detector_path)
        self.ocr = ocr if ocr is not None else PlateOCR()

    def process(self, image_path):
        """Return all detections; accepts a path or an OpenCV BGR image array."""
        image = cv2.imread(os.fspath(image_path)) if isinstance(image_path, (str, os.PathLike)) else image_path
        if image is None:
            raise ValueError(f"Could not load image: {image_path if isinstance(image_path, (str, os.PathLike)) else 'array'}")
        results = []
        for detection in self.detector.detect(image):
            bbox = detection["bbox"]  # the model supplies this box, never the ground truth
            try:
                crop = crop_plate(image, bbox)
            except ValueError:
                continue
            if crop.size == 0:
                continue
            raw_text, ocr_details = self.ocr.recognize(crop)
            results.append({
                "bbox": bbox,
                "detection_confidence": detection["confidence"],
                "raw_text": raw_text,
                "cleaned_text": process_plate_text(raw_text),
                "ocr_details": ocr_details,
            })
        return results

    def recognize_plate(self, image_path):
        """One-call image -> highest-confidence plate text (or empty string)."""
        results = self.process(image_path)
        return max(results, key=lambda row: row["detection_confidence"])["cleaned_text"] if results else ""
