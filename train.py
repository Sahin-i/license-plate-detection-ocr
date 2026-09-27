from ultralytics import YOLO


def main():

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=80,
        imgsz=640,
        batch=4,
        patience=15,
        device="cpu"
    )


if __name__ == "__main__":
    main()