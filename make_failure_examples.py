"""Visualize four failures recorded by the saved model run on the held-out test."""

import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from src.recorded_boxes import parse_box

SELECTED = (
    ("latin_difficult_06.png", "OCR substitution: M read as H"),
    ("latin_medium_08.png", "Loose detection box and severe OCR error"),
    ("dataset_a_multiple_vehicles_04.jpg", "Two missed plates in a multi-vehicle image"),
    ("dataset_a_partial_occlusion_01.jpg", "Missed plate from partial-occlusion group"),
)


def main():
    with Path("results/evaluation_final.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    out = Path("results/failure_examples")
    out.mkdir(parents=True, exist_ok=True)
    for filename, explanation in SELECTED:
        entries = [r for r in rows if Path(r["image"]).name == filename]
        if not entries:
            raise ValueError(f"No recorded evaluation for {filename}")
        original = Image.open(entries[0]["image"]).convert("RGB")
        canvas = original.copy()
        draw = ImageDraw.Draw(canvas)
        line_width = max(3, original.width // 600)
        for row in entries:
            gt_box = parse_box(row["gt_box"])
            pred_box = parse_box(row["pred_box"])
            draw.rectangle(gt_box, outline="red", width=line_width)
            if pred_box:
                draw.rectangle(pred_box, outline="lime", width=line_width)
        canvas.thumbnail((1200, 850))
        # Focus strips are analysis aids only: the live OCR pipeline crops model boxes.
        focus = []
        for row in entries:
            pred_box = parse_box(row["pred_box"])
            area = original.crop(pred_box or parse_box(row["gt_box"]))
            area.thumbnail((440, 180))
            focus.append((area, "MODEL BOX" if pred_box else "GT BOX - missed by model"))
        width = max(canvas.width, 600)
        header = 100
        strip_h = 220 * len(focus)
        visual = Image.new("RGB", (width, header + canvas.height + strip_h), "#161616")
        visual.paste(canvas, (0, header))
        ink = ImageDraw.Draw(visual)
        ink.text((12, 10), filename + " | " + explanation, fill="white")
        ink.text((12, 38), "RED = ground truth for evaluation; GREEN = model prediction", fill="white")
        ink.text((12, 65), "Boxes from saved test CSV; analysis crops do not feed OCR.", fill="white")
        for index, (row, (area, label)) in enumerate(zip(entries, focus)):
            y = header + canvas.height + 220 * index
            visual.paste(area, (8, y + 24))
            gt = row["gt_text"] or "(complete GT unavailable)"
            pred = row["cleaned_ocr"] or "(none)"
            caption = f"plate {row['plate_index']}: {label} | GT: {gt} | OCR: {pred} | IoU {float(row['iou']):.3f}"
            ink.text((8, y + 4), caption, fill="white")
        destination = out / (Path(filename).stem + "_failure.jpg")
        visual.save(destination, quality=93)
        print(destination)


if __name__ == "__main__":
    main()
