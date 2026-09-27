"""Run the full held-out test on restored data/test/images and save plate-level results."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.evaluation import match_boxes, score_plate
from src.pipeline import LicensePlatePipeline

FIELDNAMES = [
    "image", "group", "plate_index", "gt_box", "pred_box", "iou",
    "detected_iou_50", "raw_ocr", "cleaned_ocr", "gt_text",
    "ocr_evaluable", "exact_match", "cer", "correct",
]


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("metadata/plates.csv"))
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.csv"))
    args = parser.parse_args()

    with args.metadata.open(newline="", encoding="utf-8-sig") as fp:
        truth = [row for row in csv.DictReader(fp) if row["split"] == "test"]
    grouped = defaultdict(list)
    for row in truth:
        grouped[row["image"]].append(row)
    if len(grouped) < 20:
        raise ValueError(f"Expected at least 20 independent test images, got {len(grouped)}")
    missing = [path for path in grouped if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test images are absent. Restore the original data/ tree first; "
            f"examples: {missing[:3]}"
        )
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)

    pipeline = LicensePlatePipeline(args.model)
    rows, false_positives = [], []
    unmatched_total = 0
    for image_path, labels in sorted(grouped.items()):
        if len({r["plate_index"] for r in labels}) != len(labels):
            raise ValueError(f"Duplicate plate_index in {image_path}")
        predictions = pipeline.process(image_path)
        gt_boxes = [
            [int(row[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
            for row in labels
        ]
        matches, unmatched = match_boxes(gt_boxes, predictions)
        unmatched_total += len(unmatched)
        for gi, (label, gt_box) in enumerate(zip(labels, gt_boxes)):
            pi, iou = matches[gi] if gi in matches else (None, 0.0)
            rows.append(score_plate(label, gt_box, predictions[pi] if pi is not None else None, iou))
        for pi in sorted(unmatched):
            false_positives.append({
                "image": image_path,
                "pred_box": predictions[pi]["bbox"],
                "raw_ocr": predictions[pi]["raw_text"],
                "cleaned_ocr": predictions[pi]["cleaned_text"],
                "reason": "no overlapping GT",
            })
        for pi, overlap in matches.values():
            if overlap < 0.5:
                false_positives.append({
                    "image": image_path,
                    "pred_box": predictions[pi]["bbox"],
                    "raw_ocr": predictions[pi]["raw_text"],
                    "cleaned_ocr": predictions[pi]["cleaned_text"],
                    "reason": "matched GT, IoU below 0.50",
                })
        print(f"{image_path}: {len(labels)} GT, {len(predictions)} predictions")

    save_csv(args.output, rows, FIELDNAMES)
    save_csv(args.output.with_name("false_positives.csv"), false_positives,
             ["image", "pred_box", "raw_ocr", "cleaned_ocr", "reason"])
    evaluable = [row for row in rows if row["ocr_evaluable"]]
    detected = [row for row in evaluable if row["detected_iou_50"]]
    summary = {
        "test_images": len(grouped),
        "gt_plates": len(rows),
        "mean_iou": sum(r["iou"] for r in rows) / len(rows),
        "recall_at_iou_50": sum(r["detected_iou_50"] for r in rows) / len(rows),
        "unmatched_predictions": unmatched_total,
        "false_positives_at_iou_50": len(false_positives),
        "ocr_evaluable_plates": len(evaluable),
        "exact_match_all_evaluable": sum(r["exact_match"] for r in evaluable) / len(evaluable) if evaluable else None,
        "mean_cer_all_evaluable": sum(r["cer"] for r in evaluable) / len(evaluable) if evaluable else None,
        "end_to_end_accuracy_at_iou_50": sum(r["correct"] for r in evaluable) / len(evaluable) if evaluable else None,
        "ocr_accuracy_conditional_on_iou_50": sum(r["exact_match"] for r in detected) / len(detected) if detected else None,
    }
    summary_path = args.output.with_name("evaluation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {args.output}; summary: {summary_path}")


if __name__ == "__main__":
    main()
