"""Summarize supplied evaluation snapshots; does not rerun or modify the model."""

import csv
from pathlib import Path
from src.utils import clean_plate_text

RESULTS = Path("results")


def load(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def metrics(rows):
    eval_rows = [r for r in rows if r["ocr_evaluable"].lower() == "true"]
    return {
        "plates": len(rows),
        "images": len({r["image"] for r in rows}),
        "mean_iou": sum(float(r["iou"]) for r in rows) / len(rows),
        "detected": sum(r["detected_iou_50"].lower() == "true" for r in rows),
        "evaluable": len(eval_rows),
        "exact": sum(r["exact_match"].lower() == "true" for r in eval_rows),
        "cer": sum(float(r["cer"]) for r in eval_rows) / len(eval_rows),
        "e2e": sum(
            r["exact_match"].lower() == "true" and float(r["iou"]) >= 0.5
            for r in eval_rows
        ),
    }


def main():
    baseline = load("baseline-evaluation.csv")
    final = load("evaluation_final.csv")
    if [(r["image"], r["plate_index"]) for r in baseline] != [
        (r["image"], r["plate_index"]) for r in final
    ]:
        raise ValueError("Baseline and final rows differ; cannot claim identical test sets")
    with Path("metadata/plates.csv").open(newline="", encoding="utf-8-sig") as fp:
        metadata = {(r["image"], r["plate_index"]): r for r in csv.DictReader(fp) if r["split"] == "test"}
    if len(final) != len(metadata):
        raise ValueError("Snapshot and test metadata have different numbers of plates")

    b, f = metrics(baseline), metrics(final)
    table = []
    for row in final:
        key = row["image"], row["plate_index"]
        if key not in metadata:
            raise ValueError(f"Unknown test label: {key}")
        evaluable = row["ocr_evaluable"].lower() == "true"
        if evaluable and not clean_plate_text(row["gt_text"]):
            raise ValueError(f"Text label missing: {key}")
        table.append({
            "image": row["image"], "plate_index": row["plate_index"],
            "group": metadata[key]["group"], "gt_box": row["gt_box"],
            "detected_box": row["pred_box"], "iou": row["iou"],
            "raw_ocr": row["raw_ocr"], "cleaned_ocr": row["cleaned_ocr"],
            "ground_truth_text": row["gt_text"], "text_evaluable": evaluable,
            "correct_at_iou_50": (
                str(row["exact_match"].lower() == "true" and float(row["iou"]) >= 0.5)
                if evaluable else "N/A (no complete text label)"
            ),
        })
    with (RESULTS / "recorded_e2e_table.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    lines = [
        "# Recorded held-out test results",
        "",
        "Computed from the supplied CSV snapshots, which have been kept unchanged. The matching test images were restored in the original project split. These are **recorded results**, not a fresh model rerun.",
        "The detector metrics use every labeled plate. Text metrics use only plates with a complete transcription.",
        "The 28 other plates have partial or unavailable transcriptions; their text is not counted as correct or incorrect.",
        "",
        "| Measure | Baseline | Final |",
        "| --- | ---: | ---: |",
        f"| Labeled test images | {b['images']} | {f['images']} |",
        f"| Ground-truth plates | {b['plates']} | {f['plates']} |",
        f"| Mean IoU (including misses as 0) | {b['mean_iou']:.3f} | {f['mean_iou']:.3f} |",
        f"| Recall at IoU >= 0.50 | {b['detected']}/{b['plates']} ({b['detected']/b['plates']:.2%}) | {f['detected']}/{f['plates']} ({f['detected']/f['plates']:.2%}) |",
        f"| Text exact match, all evaluable | {b['exact']}/{b['evaluable']} ({b['exact']/b['evaluable']:.2%}) | {f['exact']}/{f['evaluable']} ({f['exact']/f['evaluable']:.2%}) |",
        f"| Mean CER, all evaluable (missed detection = empty string) | {b['cer']:.3f} | {f['cer']:.3f} |",
        f"| End-to-end correct (IoU >= 0.50 AND exact text) | {b['e2e']}/{b['evaluable']} ({b['e2e']/b['evaluable']:.2%}) | {f['e2e']}/{f['evaluable']} ({f['e2e']/f['evaluable']:.2%}) |",
        "",
        "Per-plate image -> detected box -> raw/cleaned OCR -> correct/incorrect results are in `recorded_e2e_table.csv`.",
        "OCR exact match here includes missed detections as empty output, and is not conditional OCR accuracy. The final code also reports conditional OCR accuracy separately.",
        "",
        "## Four test failures, with images",
        "",
        "1. **[latin_difficult_06.png](failure_examples/latin_difficult_06_failure.jpg), plate 0:** IoU 0.739; ground truth `OMGBCKY`, OCR `OHGBCKY`. The detector found the plate, but OCR confused `M` with `H` on a small, soft-looking plate (one substitution).",
        "2. **[latin_medium_08.png](failure_examples/latin_medium_08_failure.jpg), plate 0:** IoU 0.597; ground truth `MMN306`, OCR `GO0`. The angled plate sits inside a model box that extends well beyond the label into surrounding car trim; the recognition stage lost most characters. Box `[44, 143, 214, 239]` versus GT `[71, 124, 198, 231]`.",
        "3. **[dataset_a_multiple_vehicles_04.jpg](failure_examples/dataset_a_multiple_vehicles_04_failure.jpg), plates 0 and 1:** neither has a predicted box; plate 2 has IoU 0.634. Two dark, low-contrast plates were missed among multiple vehicles, so OCR never received their crops. Text labels are incomplete and excluded from OCR accuracy.",
        "",
        "4. **[dataset_a_partial_occlusion_01.jpg](failure_examples/dataset_a_partial_occlusion_01_failure.jpg), plate 0:** no detection (IoU 0). The image contains masked/obscured characters and difficult lighting; this is a detector miss. The complete text is unavailable and excluded from OCR scoring.",
        "",
        "Red rectangles show GT **for analysis only**; green rectangles show saved model predictions. OCR never uses GT crops. Recreate these images with `python make_failure_examples.py`.",
        "The fresh YOLO/EasyOCR rerun was not performed here because those dependencies were unavailable in the inspection environment; use `python evaluate.py` in the project environment.",
        "",
    ]
    (RESULTS / "recorded_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote recorded_e2e_table.csv and recorded_test_report.md")


if __name__ == "__main__":
    main()
