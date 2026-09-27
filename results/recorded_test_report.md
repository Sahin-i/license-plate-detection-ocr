# Recorded held-out test results

Computed from the supplied CSV snapshots, which have been kept unchanged. The matching test images were restored in the original project split. These are **recorded results**, not a fresh model rerun.
The detector metrics use every labeled plate. Text metrics use only plates with a complete transcription.
The 28 other plates have partial or unavailable transcriptions; their text is not counted as correct or incorrect.

| Measure | Baseline | Final |
| --- | ---: | ---: |
| Labeled test images | 25 | 25 |
| Ground-truth plates | 38 | 38 |
| Mean IoU (including misses as 0) | 0.419 | 0.419 |
| Recall at IoU >= 0.50 | 21/38 (55.26%) | 21/38 (55.26%) |
| Text exact match, all evaluable | 2/10 (20.00%) | 3/10 (30.00%) |
| Mean CER, all evaluable (missed detection = empty string) | 0.363 | 0.171 |
| End-to-end correct (IoU >= 0.50 AND exact text) | 2/10 (20.00%) | 3/10 (30.00%) |

Per-plate image -> detected box -> raw/cleaned OCR -> correct/incorrect results are in `recorded_e2e_table.csv`.
OCR exact match here includes missed detections as empty output, and is not conditional OCR accuracy. The final code also reports conditional OCR accuracy separately.

## Four test failures, with images

1. **[latin_difficult_06.png](failure_examples/latin_difficult_06_failure.jpg), plate 0:** IoU 0.739; ground truth `OMGBCKY`, OCR `OHGBCKY`. The detector found the plate, but OCR confused `M` with `H` on a small, soft-looking plate (one substitution).
2. **[latin_medium_08.png](failure_examples/latin_medium_08_failure.jpg), plate 0:** IoU 0.597; ground truth `MMN306`, OCR `GO0`. The angled plate sits inside a model box that extends well beyond the label into surrounding car trim; the recognition stage lost most characters. Box `[44, 143, 214, 239]` versus GT `[71, 124, 198, 231]`.
3. **[dataset_a_multiple_vehicles_04.jpg](failure_examples/dataset_a_multiple_vehicles_04_failure.jpg), plates 0 and 1:** neither has a predicted box; plate 2 has IoU 0.634. Two dark, low-contrast plates were missed among multiple vehicles, so OCR never received their crops. Text labels are incomplete and excluded from OCR accuracy.

4. **[dataset_a_partial_occlusion_01.jpg](failure_examples/dataset_a_partial_occlusion_01_failure.jpg), plate 0:** no detection (IoU 0). The image contains masked/obscured characters and difficult lighting; this is a detector miss. The complete text is unavailable and excluded from OCR scoring.

Red rectangles show GT **for analysis only**; green rectangles show saved model predictions. OCR never uses GT crops. Recreate these images with `python make_failure_examples.py`.
The fresh YOLO/EasyOCR rerun was not performed here because those dependencies were unavailable in the inspection environment; use `python evaluate.py` in the project environment.
