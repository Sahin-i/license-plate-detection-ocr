import cv2
import easyocr


class PlateOCR:

    def __init__(self):
        self.reader = easyocr.Reader(["en"])

    def preprocess(self, image):

        height, width = image.shape[:2]

        if height < 100:

            scale = 100 / height

            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return gray

    def recognize(self, image):

        processed_image = self.preprocess(
            image
        )

        result = self.reader.readtext(
            processed_image,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            decoder="beamsearch",
            beamWidth=5,
            mag_ratio=1.5,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            add_margin=0.05,
            paragraph=False,
            detail=1
        )

        if not result:
            return "", []

        result = sorted(
            result,
            key=lambda item: min(
                point[0]
                for point in item[0]
            )
        )

        useful_results = [
            item
            for item in result
            if item[2] >= 0.30
        ]

        if not useful_results:
            useful_results = result

        joined_text = "".join(
            item[1]
            for item in useful_results
        )

        best_result = max(
            useful_results,
            key=lambda item: item[2]
        )

        best_text = best_result[1]

        if len(joined_text) > 10:
            raw_text = best_text
        else:
            raw_text = joined_text

        return raw_text, useful_results