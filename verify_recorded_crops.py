"""Verify padded model-box crops on three multi-vehicle test images.

These coordinates came from the saved detector outputs. A fresh model rerun is
also available through `python evaluate.py` on a machine with ML dependencies.
"""

import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from src.recorded_boxes import parse_box
from src.utils import crop_plate

SELECTED = (
    "dataset_a_multiple_vehicles_02.jpg",
    "dataset_a_multiple_vehicles_04.jpg",
    "dataset_a_multiple_vehicles_05.jpg",
)


def main():
    with Path("results/evaluation_final.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    with Path("metadata/plates.csv").open(newline="", encoding="utf-8-sig") as fp:
        labels = list(csv.DictReader(fp))
    output = Path("results/multivehicle_crops")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for filename in SELECTED:
        matching = [row for row in rows if Path(row["image"]).name == filename]
        gt = [row for row in labels if row["split"] == "test" and Path(row["image"]).name == filename]
        if len(gt) < 2 or not matching:
            raise AssertionError(f"Not a labeled multi-vehicle test image: {filename}")
        path = Path(matching[0]["image"])
        original = Image.open(path).convert("RGB")
        rgb_array = np.array(original)
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        seen = set()
        valid_crops = 0
        for row in matching:
            box = parse_box(row["pred_box"])
            if box is None or tuple(box) in seen:
                continue
            seen.add(tuple(box))
            crop = crop_plate(rgb_array, box, padding=0.05)
            if not crop.size or crop.shape[0] <= 0 or crop.shape[1] <= 0:
                raise AssertionError(f"Empty crop in {filename} from {box}")
            valid_crops += 1
            Image.fromarray(crop).save(output / f"{path.stem}_model_crop_{valid_crops}.png")
            draw.rectangle(box, outline="lime", width=max(3, original.width // 1000))
        if valid_crops == 0:
            raise AssertionError(f"No saved model box to verify for {filename}")
        annotated.thumbnail((1200, 850))
        annotated.save(output / f"{path.stem}_detections.jpg", quality=90)
        records.append({"image": str(path), "gt_plates": len(gt), "verified_model_crops": valid_crops})
    with (output / "crop_check.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Verified {sum(r['verified_model_crops'] for r in records)} crops on {len(records)} multi-vehicle images: {output}")


if __name__ == "__main__":
    main()
