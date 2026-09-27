"""Pure evaluation helpers, independent of model and OCR installations."""

from src.metrics import calculate_iou, character_error_rate
from src.utils import clean_plate_text


def match_boxes(gt_boxes, predictions):
    """Greedily match overlapping model detections to GT, one-to-one by IoU.

    Ground-truth boxes are used only here for scoring, never for OCR cropping.
    Zero-overlap boxes cannot be a match. Unmatched predictions are false positives.
    """
    candidates = []
    for gi, gt_box in enumerate(gt_boxes):
        for pi, prediction in enumerate(predictions):
            overlap = calculate_iou(gt_box, prediction["bbox"])
            if overlap > 0:
                candidates.append((overlap, gi, pi))
    candidates.sort(reverse=True)
    matches, used_predictions = {}, set()
    for overlap, gi, pi in candidates:
        if gi in matches or pi in used_predictions:
            continue
        matches[gi] = (pi, overlap)
        used_predictions.add(pi)
    return matches, set(range(len(predictions))) - used_predictions


def score_plate(gt, gt_box, prediction=None, iou=0.0, min_iou=0.5):
    """Score one labeled plate; a missed plate has IoU 0 and empty OCR."""
    text_gt = gt["text_gt"].strip()
    evaluable = str(gt["ocr_evaluable"]).strip().lower() == "true"
    if evaluable and not clean_plate_text(text_gt):
        raise ValueError(f"OCR-evaluable plate has no text: {gt['image']}")
    pred_box = prediction["bbox"] if prediction else None
    raw = prediction["raw_text"] if prediction else ""
    cleaned = prediction["cleaned_text"] if prediction else ""
    expected = clean_plate_text(text_gt)
    exact = cleaned == expected if evaluable else None
    correct = (iou >= min_iou and exact) if evaluable else None
    return {
        "image": gt["image"],
        "group": gt["group"],
        "plate_index": int(gt["plate_index"]),
        "gt_box": gt_box,
        "pred_box": pred_box,
        "iou": float(iou),
        "detected_iou_50": iou >= min_iou,
        "raw_ocr": raw,
        "cleaned_ocr": cleaned,
        "gt_text": text_gt,
        "ocr_evaluable": evaluable,
        "exact_match": exact,
        "cer": character_error_rate(expected, cleaned) if evaluable else None,
        "correct": correct,  # correct detection AND correct text
    }
