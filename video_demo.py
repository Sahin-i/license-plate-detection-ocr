"""Optional webcam/video preview and local timestamped plate log; press q to quit."""

import argparse
import time
import cv2
from src.pipeline import LicensePlatePipeline
from src.parking_log import open_log, log_detection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Webcam index (0) or video path")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--every", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--log", help="Optional SQLite file, e.g. results/parking.db")
    parser.add_argument("--cooldown", type=float, default=30, help="Seconds before logging same text again")
    args = parser.parse_args()
    if args.every < 1 or args.cooldown < 0:
        parser.error("--every must be >= 1 and --cooldown >= 0")
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"Cannot open video source: {args.source}")
    log = open_log(args.log) if args.log else None
    pipeline = LicensePlatePipeline(args.model)
    last_seen = {}
    results = []
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % args.every == 0:
                results = pipeline.process(frame)
                now = time.monotonic()
                for row in results:
                    plate = row["cleaned_text"]
                    if log and plate and now - last_seen.get(plate, float("-inf")) >= args.cooldown:
                        log_detection(log, plate, row["detection_confidence"])
                        last_seen[plate] = now
            for row in results:
                x1, y1, x2, y2 = row["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, row["cleaned_text"] or "plate", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("License plates (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log:
            log.close()


if __name__ == "__main__":
    main()
