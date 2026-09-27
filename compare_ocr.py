"""Optional EasyOCR vs Tesseract comparison on identical held-out model crops."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import pytesseract
from src.evaluation import match_boxes, score_plate
from src.pipeline import LicensePlatePipeline
from src.utils import crop_plate, process_plate_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tesseract-cmd", help="Path to tesseract.exe if it is not on PATH")
    parser.add_argument("--output", type=Path, default=Path("results/ocr_comparison.csv"))
    args = parser.parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    try:
        print("Tesseract version:", pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError as exc:
        parser.error(f"Install Tesseract OCR or pass --tesseract-cmd: {exc}")

    with Path("metadata/plates.csv").open(newline="", encoding="utf-8-sig") as fp:
        truth = [r for r in csv.DictReader(fp) if r["split"] == "test"]
    grouped = defaultdict(list)
    for row in truth:
        grouped[row["image"]].append(row)
    missing = [path for path in grouped if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Test images missing: {missing[:3]}")
    pipeline = LicensePlatePipeline("models/best.pt")
    rows = []
    for image, labels in sorted(grouped.items()):
        predictions = pipeline.process(image)  # once per image; same detector for both OCR methods
        frame = cv2.imread(image)
        boxes = [[int(row[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")] for row in labels]
        matches, _ = match_boxes(boxes, predictions)
        for gi, (label, gt_box) in enumerate(zip(labels, boxes)):
            if label["ocr_evaluable"].lower() != "true":
                continue
            pi, iou = matches[gi] if gi in matches else (None, 0.0)
            easy = predictions[pi] if pi is not None else None
            tess = None
            if easy:
                crop = crop_plate(frame, easy["bbox"])
                preprocessed = pipeline.ocr.preprocess(crop)
                raw = pytesseract.image_to_string(
                    preprocessed, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                ).strip()
                tess = {"bbox": easy["bbox"], "raw_text": raw, "cleaned_text": process_plate_text(raw)}
            for engine, pred in (("EasyOCR", easy), ("Tesseract", tess)):
                scored = score_plate(label, gt_box, pred, iou)
                rows.append({"engine": engine, **scored})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for engine in ("EasyOCR", "Tesseract"):
        group = [r for r in rows if r["engine"] == engine]
        print(f"{engine}: {sum(r['exact_match'] for r in group)}/{len(group)} exact; "
              f"mean CER={sum(r['cer'] for r in group)/len(group):.3f}; "
              f"E2E={sum(r['correct'] for r in group)}/{len(group)}")
    print(f"Identical test images and detector boxes; saved {args.output}")


if __name__ == "__main__":
    main()
