# Requirements audit

This audit refers to the rebuilt project split, the supplied saved model outputs, and the attached image data. It distinguishes checks performed here from a fresh model rerun.

| Requirement | Evidence | Status |
| --- | --- | --- |
| License plate detector | Fine-tuned YOLOv8n checkpoint `models/best.pt`; `src/detector.py` reads its class-0 boxes. Training history in `runs/detect/train-5/`. | Present; fresh inference pending runtime |
| At least 20 labeled held-out images, mean IoU | `metadata/plates.csv`: 25 test images / 38 plate boxes; final CSV mean IoU 0.419, recall 21/38 at IoU >= 0.50. All 25 test GT box overlays were visually reviewed in `results/label_review/`; YOLO and pixel labels agree within 1 px. The original annotation source is public; manual origin of every upstream box is not independently documented. | Recorded evaluation verified |
| Padded crops on at least three multi-vehicle images | `verify_recorded_crops.py` extracted four nonempty model-box crops from three images; evidence is in `results/multivehicle_crops/`. | Verified from saved model predictions and source pixels |
| OCR raw output and cleaning | `src/ocr.py` uses EasyOCR; `src/pipeline.py` retains `raw_text`, OCR details, and `cleaned_text`; `src/utils.py` handles noise and neighbor-only ambiguous characters without country-format assumptions. | Implemented; saved OCR outputs present |
| End-to-end table and accuracy | `results/recorded_e2e_table.csv`: every test plate with predicted box, raw/clean OCR, GT and correctness where a complete text label exists. Final 3/10 exact and 3/10 end-to-end correct, mean CER 0.171. | Recorded evaluation verified |
| At least three shown and explained failures | Four actual test-image figures in `results/failure_examples/`, explanations in `results/recorded_test_report.md`. | Complete |
| No manual cropping for inference | `LicensePlatePipeline.process` crops only `PlateDetector.detect` boxes; GT enters only scoring and visual analysis. | Code verified |
| OpenCV, detector framework, OCR, labeled data | `cv2`, Ultralytics YOLOv8, EasyOCR, 85 annotated images; dependencies in `requirements.txt`. | Present |
| Blurry, angled, occluded, multiple or zero detections | Figures and test groups show difficult, angled, masked, low-light, and multi-vehicle cases. Pipeline handles zero or multiple model boxes. These failure modes are measured, not guaranteed to be solved. | Addressed with explicit limits |
| Optional video and parking log | `video_demo.py`, `src/parking_log.py`. | Implemented; webcam run not verified here |
| Optional Tesseract comparison | `compare_ocr.py` uses identical held-out model boxes for both OCR engines. | Script ready; Tesseract comparison has not been executed |
| Optional local-format regex | Intentionally omitted for mixed plate formats, as requested. | Skipped |

## What remains to confirm on your machine

Run `pip install -r requirements.txt`, `python verify_dataset.py`, and `python evaluate.py`. This inspection environment did not have Torch, Ultralytics, OpenCV, or EasyOCR, and its package source could not supply them. `python evaluate.py` will create fresh results, including separate false positives and an accuracy summary; the saved baseline/final CSVs remain unchanged for comparison. Optional webcam and Tesseract functions need their respective runtime or system executable.
