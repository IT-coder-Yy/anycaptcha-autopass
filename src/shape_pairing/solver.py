"""在内置拼板的离散剪影间选择一对一配对。"""
import math

import cv2
import numpy as np

from common.vision import best_assignment
from .lab import masks


def _views(mask, angles):
    moments = cv2.moments(mask, binaryImage=True)
    center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    side = math.ceil(math.hypot(*mask.shape))
    offset = (side - 1) / 2 - np.asarray(center)
    for angle in angles:
        transform = cv2.getRotationMatrix2D(center, angle, 1)
        transform[:, 2] += offset
        view = cv2.warpAffine(mask, transform, (side, side), flags=cv2.INTER_NEAREST)
        yield view, cv2.countNonZero(view)


def match(pieces, slots):
    targets = [next(_views(slot, (0,))) for slot in slots]
    scores = np.zeros((len(pieces), len(slots)))
    for i, piece in enumerate(pieces):
        for view, area in _views(piece, range(0, 360, 5)):
            for j, (slot, slot_area) in enumerate(targets):
                overlap = cv2.countNonZero(cv2.bitwise_and(view, slot))
                scores[i, j] = max(scores[i, j], overlap / (area + slot_area - overlap))
    indices, _, _ = best_assignment(scores, keep_ties=True)
    return [j + 1 if j is not None else None for j in indices]


def evaluate(cases):
    samples = []
    for case, rows in cases.items():
        selected = match(*masks(rows))
        expected = [int(row["slot"]) for row in rows]
        samples.append({"id": case, "title": rows[0]["title"], "expected": expected, "selected": selected,
                        "correct": selected == expected,
                        "pieces_correct": sum(a == b for a, b in zip(selected, expected, strict=True))})
    return {"kind": "shape-pairing", "engine": "rotated-mask-iou", "total": len(samples),
            "correct": sum(s["correct"] for s in samples), "pieces_total": sum(len(s["expected"]) for s in samples),
            "pieces_correct": sum(s["pieces_correct"] for s in samples),
            "unanswered_pieces": sum(s["selected"].count(None) for s in samples), "samples": samples}
