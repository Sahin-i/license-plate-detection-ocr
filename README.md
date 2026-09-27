# License Plate Detection and Recognition Pipeline

This project chains two models into one pipeline: a fine-tuned **Ultralytics YOLOv8n** detector finds license plates, and **EasyOCR** reads each padded plate crop. OpenCV handles image loading and drawing. The trained detector is included at `models/best.pt`.

Ground-truth boxes are used for evaluation and failure analysis only. OCR crops always come from the detector's own predictions.

## Setup and usage

Run commands from the project root so `dataset.yaml`, `data/`, and `models/` resolve correctly.

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python verify_dataset.py
python demo.py data/test/images/latin_easy_11.png
python evaluate.py
python -m unittest discover -s tests -v
```

On Windows PowerShell, you can use `.\.venv\Scripts\python.exe` instead of activating the environment. EasyOCR is the primary OCR engine, so the main pipeline does not require a Tesseract system installation. EasyOCR may download model weights on first use. CPU inference works; a compatible PyTorch installation is needed to use a GPU. See the [Ultralytics quickstart](https://docs.ultralytics.com/quickstart/) for the YOLO API.

The pipeline can also be called from Python:

```python
from src.pipeline import LicensePlatePipeline

pipeline = LicensePlatePipeline("models/best.pt")

# Returns all detected plates, including boxes, confidence, and raw/cleaned OCR.
plates = pipeline.process("data/test/images/latin_easy_11.png")

# Returns the text of the highest-confidence plate, or "" if none is detected.
text = pipeline.recognize_plate("data/test/images/latin_easy_11.png")
```

`process()` accepts an image path or an OpenCV BGR image array. It returns an empty list when no plate is detected. For each detection, the pipeline adds 5% crop padding, clips the crop to image bounds, runs EasyOCR, and retains both the raw and cleaned text.

Text cleaning uppercases letters, removes non-alphanumeric OCR noise, and applies neighbor-based O/0 and I/1 corrections. The dataset contains different plate layouts, so **no country-specific plate-format regex is used**. This heuristic can still make errors on mixed formats; raw OCR remains available for inspection.

## Dataset and split

The project contains **85 annotated vehicle images**: 50 train, 10 validation, and 25 held-out test images. There are 110 labeled plate boxes in total: 61 train, 11 validation, and 38 test. Eleven test images contain multiple labeled plates.

Only **10 of the 38 test plates** have complete text transcriptions suitable for exact-match and character-error-rate (CER) scoring. The other 28 still contribute to detection metrics but are excluded from text accuracy. An unknown or partially obscured plate string is not treated as an empty ground-truth string.

`metadata/plates.csv` contains pixel boxes, splits, image groups, transcription status, and source information. `data/*/labels/` contains normalized YOLO class-0 boxes. The 25 test box overlays are available in `results/label_review/` for visual inspection. The boxes come from public annotated datasets; the manual origin of every upstream annotation is not independently documented.

The separately supplied dataset archive used a different split. Three images assigned to this project's test set appeared in that archive's training folder, and three project training images appeared in its test folder. This project reconstructs the split used for the trained model and its recorded evaluation, preserving the original image pixels and labels. Do not replace its `data/` folders with those from the other archive. `python verify_dataset.py` checks for duplicate image pixels and confirms that the YOLO labels agree with the pixel boxes. Source attribution is preserved in `metadata/dataset_source_README.md`.

## Latest local test run

The following results came from a fresh `python evaluate.py` run on the held-out test set. **These are the current results to use when describing the runnable project.**

| Metric | Latest local run |
| --- | ---: |
| Test images | 25 |
| Ground-truth plates | 38 |
| Mean IoU, including missed plates as 0 | 0.419 |
| Detection recall at IoU ≥ 0.50 | 21/38 (55.26%) |
| Unmatched predictions | 7 |
| False positives at IoU ≥ 0.50 | 8 |
| Plates with complete OCR ground truth | 10 |
| Exact text match on evaluable plates | 2/10 (20.00%) |
| Mean CER on evaluable plates | 0.181 |
| End-to-end correct: IoU ≥ 0.50 **and** exact text | 2/10 (20.00%) |
| OCR accuracy conditional on IoU ≥ 0.50 | 20.00% |

The complete values from `results/evaluation_summary.json` are:

```json
{
  "test_images": 25,
  "gt_plates": 38,
  "mean_iou": 0.41947516420407,
  "recall_at_iou_50": 0.5526315789473685,
  "unmatched_predictions": 7,
  "false_positives_at_iou_50": 8,
  "ocr_evaluable_plates": 10,
  "exact_match_all_evaluable": 0.2,
  "mean_cer_all_evaluable": 0.18095238095238095,
  "end_to_end_accuracy_at_iou_50": 0.2,
  "ocr_accuracy_conditional_on_iou_50": 0.2
}
```

The fresh run writes `results/evaluation.csv` with per-plate boxes, IoU, raw OCR, cleaned OCR, text ground truth, and correctness. It also writes `results/false_positives.csv` and `results/evaluation_summary.json`. Ground-truth boxes are used only to match and score predictions, never to choose OCR crops. A missed plate receives IoU 0 and empty predicted text when text evaluation is possible.

### Earlier saved comparison

The project also retains two earlier evaluation snapshots. They use the same 25-image, 38-plate test set. The detector metrics stayed the same while the saved OCR outputs changed:

| Metric | Saved baseline | Saved final snapshot | Latest local run |
| --- | ---: | ---: | ---: |
| Mean IoU | 0.419 | 0.419 | 0.419 |
| Recall at IoU ≥ 0.50 | 21/38 (55.26%) | 21/38 (55.26%) | 21/38