"""Run the end-to-end model on one image and print every detected plate."""

import argparse
from pathlib import Path
from src.pipeline import LicensePlatePipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Vehicle image to process")
    parser.add_argument("--model", default="models/best.pt")
    args = parser.parse_args()
    pipeline = LicensePlatePipeline(args.model)
    results = pipeline.process(args.image)
    if not results:
        print("No plate detected. Single-text output: ''")
    for item in results:
        print(f"box={item['bbox']} confidence={item['detection_confidence']:.3f} "
              f"raw={item['raw_text']!r} cleaned={item['cleaned_text']!r}")
    best = max(results, key=lambda row: row["detection_confidence"])["cleaned_text"] if results else ""
    print("Highest-confidence plate text:", repr(best))


if __name__ == "__main__":
    main()
