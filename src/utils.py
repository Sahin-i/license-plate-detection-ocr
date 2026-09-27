"""Format agnostic OCR cleaning and padded cropping."""


def clean_plate_text(text):
    """Keep ASCII letters and digits; preserve their order without a plate template."""
    return "".join(ch for ch in str(text).upper() if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def correct_ambiguous_characters(text):
    """Use immediate neighbors to correct O/0 and I/1, with no fixed format."""
    chars = list(text)
    for i in range(1, len(chars) - 1):
        previous, current, following = chars[i - 1], chars[i], chars[i + 1]
        if previous.isdigit() and following.isdigit() and current in {"O", "I"}:
            chars[i] = {"O": "0", "I": "1"}[current]
        elif previous.isalpha() and following.isalpha() and current in {"0", "1"}:
            chars[i] = {"0": "O", "1": "I"}[current]
    return "".join(chars)


def process_plate_text(text):
    return correct_ambiguous_characters(clean_plate_text(text))


def crop_plate(image, bbox, padding=0.05):
    """Crop a predicted xyxy box; clip proportional padding to image bounds."""
    if not 0 <= padding <= 1:
        raise ValueError("padding must be between 0 and 1")
    x1, y1, x2, y2 = (int(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: {bbox}")
    image_height, image_width = image.shape[:2]
    pad_x = max(1, round((x2 - x1) * padding)) if padding else 0
    pad_y = max(1, round((y2 - y1) * padding)) if padding else 0
    x1 = min(image_width, max(0, x1 - pad_x))
    y1 = min(image_height, max(0, y1 - pad_y))
    x2 = min(image_width, max(0, x2 + pad_x))
    y2 = min(image_height, max(0, y2 + pad_y))
    return image[y1:y2, x1:x2]
