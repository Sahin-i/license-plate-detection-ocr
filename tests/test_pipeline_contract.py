"""Contract checks for matching, padded crops, and model-to-OCR handoff."""

import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

from src.evaluation import match_boxes, score_plate
from src.utils import crop_plate, process_plate_text


class EvaluationTests(unittest.TestCase):
    def test_zero_overlap_does_not_match(self):
        matches, unmatched = match_boxes([[0, 0, 10, 10]], [{"bbox": [20, 20, 30, 30]}])
        self.assertEqual(matches, {})
        self.assertEqual(unmatched, {0})

    def test_two_plates_match_distinct_predictions(self):
        preds = [{"bbox": [30, 0, 40, 10]}, {"bbox": [0, 0, 10, 10]}]
        matches, unmatched = match_boxes([[0, 0, 10, 10], [30, 0, 40, 10]], preds)
        self.assertEqual(matches, {0: (1, 1.0), 1: (0, 1.0)})
        self.assertFalse(unmatched)

    def test_missed_text_counts_as_failure(self):
        label = {"image": "test.png", "group": "difficult", "plate_index": "0",
                 "text_gt": "AB12", "ocr_evaluable": "true"}
        scored = score_plate(label, [0, 0, 10, 10])
        self.assertFalse(scored["correct"])
        self.assertFalse(scored["exact_match"])
        self.assertEqual(scored["cer"], 1.0)

    def test_format_agnostic_cleaning(self):
        self.assertEqual(process_plate_text("a-b 12O3!"), "AB1203")
        self.assertEqual(process_plate_text("A0B"), "AOB")


class PipelineTests(unittest.TestCase):
    def test_multi_plate_handoff_and_zero_detections(self):
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.imread = lambda path: np.zeros((100, 200, 3), dtype=np.uint8)
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = object
        fake_easyocr = types.ModuleType("easyocr")
        fake_easyocr.Reader = object
        with patch.dict(sys.modules, {"cv2": fake_cv2, "ultralytics": fake_ultralytics,
                                      "easyocr": fake_easyocr}):
            from src.pipeline import LicensePlatePipeline

            class Detector:
                boxes = [{"bbox": [10, 10, 30, 20], "confidence": 0.8},
                         {"bbox": [180, 90, 200, 100], "confidence": 0.9}]

                def detect(self, image):
                    return self.boxes

            class OCR:
                def __init__(self):
                    self.shapes = []

                def recognize(self, crop):
                    self.shapes.append(crop.shape)
                    return ("AB-12" if len(self.shapes) == 1 else "1O3", [])

            detector, ocr = Detector(), OCR()
            pipeline = LicensePlatePipeline(detector=detector, ocr=ocr)
            results = pipeline.process(np.zeros((100, 200, 3), dtype=np.uint8))
            self.assertEqual([row["cleaned_text"] for row in results], ["AB12", "103"])
            self.assertEqual(ocr.shapes, [(12, 22, 3), (11, 21, 3)])
            self.assertEqual(pipeline.recognize_plate(np.zeros((100, 200, 3), dtype=np.uint8)), "103")
            detector.boxes = []
            self.assertEqual(pipeline.process(np.zeros((100, 200, 3), dtype=np.uint8)), [])
            self.assertEqual(pipeline.recognize_plate(np.zeros((100, 200, 3), dtype=np.uint8)), "")

    def test_bad_box_cannot_be_cropped(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            crop_plate(image, [10, 10, 5, 15])


if __name__ == "__main__":
    unittest.main()
