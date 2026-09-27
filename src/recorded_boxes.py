"""Read coordinate strings from the supplied saved evaluation CSVs."""

import ast


def parse_box(value):
    if not value:
        return None
    # Older pandas exports formatted elements as np.int64(123).
    cleaned = str(value).replace("np.int64(", "").replace(")", "")
    box = ast.literal_eval(cleaned)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"Expected four xyxy coordinates: {value}")
    return [int(v) for v in box]
