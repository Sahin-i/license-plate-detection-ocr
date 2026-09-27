"""Check split integrity and YOLO normalized boxes against pixel GT metadata."""

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


def main():
    with Path("metadata/plates.csv").open(newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    images = defaultdict(list)
    for row in rows:
        images[row["image"]].append(row)
    if len(images) < 30 or len({p for p in images if p.startswith("data/test/")}) < 20:
        raise AssertionError("Dataset/test split is too small")
    digests = {}
    maximum_coordinate_error = 0.0
    for relpath, labels in images.items():
        path = Path(relpath)
        if not path.is_file():
            raise FileNotFoundError(path)
        if len({r["split"] for r in labels}) != 1:
            raise AssertionError(f"Mixed split in {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in digests:
            raise AssertionError(f"Duplicate image pixels: {path}, {digests[digest]}")
        digests[digest] = path
        label_path = path.parent.parent / "labels" / (path.stem + ".txt")
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        width, height = Image.open(path).size
        yolo_rows = [list(map(float, line.split())) for line in label_path.read_text().splitlines() if line.strip()]
        if len(yolo_rows) != len(labels):
            raise AssertionError(f"YOLO/metadata count mismatch: {path}")
        expected_boxes = [tuple(int(r[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")) for r in labels]
        actual_boxes = []
        for values in yolo_rows:
            if len(values) != 5 or values[0] != 0 or any(not 0 <= v <= 1 for v in values[1:]):
                raise AssertionError(f"Invalid YOLO class/normalized coordinates: {label_path}")
            _, cx, cy, bw, bh = values
            actual_boxes.append(((cx-bw/2)*width, (cy-bh/2)*height,
                                 (cx+bw/2)*width, (cy+bh/2)*height))
        # YOLO label row order should correspond to metadata plate_index, but
        # compare boxes by nearest GT to tolerate an annotation tool's row order.
        unassigned = list(actual_boxes)
        for expected in expected_boxes:
            closest = min(unassigned, key=lambda box: max(abs(a-b) for a,b in zip(box,expected)))
            error = max(abs(a-b) for a,b in zip(closest,expected))
            maximum_coordinate_error = max(maximum_coordinate_error, error)
            if error > 1.0:
                raise AssertionError(f"YOLO/metadata boxes differ by {error:.1f}px: {path}")
            unassigned.remove(closest)
    splits = Counter(r["split"] for r in rows)
    image_splits = Counter(Path(path).parts[1] for path in images)
    print(f"OK: {len(images)} distinct images, {len(rows)} boxes; image splits {dict(image_splits)}; "
          f"plate splits {dict(splits)}; YOLO/metadata max difference {maximum_coordinate_error:.5f}px")


if __name__ == "__main__":
    main()
