def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )

    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union

def levenshtein_distance(
    ground_truth,
    prediction
):

    rows = len(
        ground_truth
    ) + 1

    cols = len(
        prediction
    ) + 1

    matrix = [
        [0] * cols
        for _ in range(rows)
    ]

    for i in range(rows):
        matrix[i][0] = i

    for j in range(cols):
        matrix[0][j] = j

    for i in range(
        1,
        rows
    ):

        for j in range(
            1,
            cols
        ):

            if (
                ground_truth[i - 1]
                == prediction[j - 1]
            ):
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(

                matrix[i - 1][j] + 1,

                matrix[i][j - 1] + 1,

                matrix[i - 1][j - 1]
                + cost
            )

    return matrix[-1][-1]

def character_error_rate(
    ground_truth,
    prediction
):

    if len(ground_truth) == 0:
        return 0.0

    distance = (
        levenshtein_distance(
            ground_truth,
            prediction
        )
    )

    return (
        distance
        / len(ground_truth)
    )
