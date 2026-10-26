"""Small geometry helpers the gaze cues share.

Just plain coordinate maths for pulling the face/head out of a detected person (from the YOLO-pose
keypoints) and cropping it. Nothing heavy is imported here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# where the gaze model weights get cached / looked for
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"

# COCO-17 keypoint indices for the face points
NOSE_INDEX = 0
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 2
LEFT_EAR_INDEX = 3
RIGHT_EAR_INDEX = 4
HEAD_INDICES = (NOSE_INDEX, LEFT_EYE_INDEX, RIGHT_EYE_INDEX, LEFT_EAR_INDEX, RIGHT_EAR_INDEX)


def keypoint(person: dict[str, Any], index: int) -> tuple[float, float, float]:
    # (x, y, confidence) for a keypoint, or zeros if it is missing
    keypoints = person.get("keypoints") or []
    if index >= len(keypoints):
        return 0.0, 0.0, 0.0
    point = keypoints[index]
    try:
        x = float(point.get("x") or 0.0)
        y = float(point.get("y") or 0.0)
        conf = float(point.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0, 0.0, 0.0
    return x, y, conf


def confident_point(person: dict[str, Any], index: int, min_conf: float = 0.2):
    # (x, y) only if the model was confident enough about this point, otherwise None
    x, y, conf = keypoint(person, index)
    if conf >= min_conf:
        return x, y
    return None


def average_point(points):
    # mean (x, y) of a list of points
    count = len(points)
    mean_x = sum(p[0] for p in points) / count
    mean_y = sum(p[1] for p in points) / count
    return mean_x, mean_y


def visible_face_points(person: dict[str, Any]):
    # the face keypoints (nose, eyes, ears) the model actually saw
    points = []
    for index in HEAD_INDICES:
        x, y, conf = keypoint(person, index)
        if conf > 0.0:
            points.append((x, y))
    return points


def person_box(person: dict[str, Any]):
    # the person's box as four floats, or None if it looks wrong
    box = person.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    if len(box) != 4:
        return None
    return [float(v) for v in box]


def box_height(person: dict[str, Any]) -> float:
    box = person_box(person)
    if box is None:
        return 0.0
    return max(0.0, box[3] - box[1])


def head_center(person: dict[str, Any]):
    # best guess at the middle of the head, trying three things in order

    # 1. a value someone already worked out
    center = person.get("estimated_head_center")
    if center:
        return float(center[0]), float(center[1])

    # 2. the average of the visible face points
    face_points = visible_face_points(person)
    if face_points:
        return average_point(face_points)

    # 3. last resort: the top-middle of the bounding box
    box = person_box(person)
    if box is not None and box[2] > box[0] and box[3] > box[1]:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, y1 + (y2 - y1) * 0.125
    return None


def crop_half_width(person: dict[str, Any]) -> float:
    # how far the face crop reaches from its centre: based on the gap between the eyes,
    # with sensible minimums
    left_eye = confident_point(person, LEFT_EYE_INDEX)
    right_eye = confident_point(person, RIGHT_EYE_INDEX)

    eye_gap = 0.0
    if left_eye and right_eye:
        eye_gap = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])

    box = person_box(person)
    box_part = (box[3] - box[1]) * 0.16 if box else 0.0
    return max(eye_gap * 3.0, box_part, 40.0)


def keep_box_in_image(cx: float, cy: float, half: float, img_w: int, img_h: int):
    # turn a centre + reach into a box that stays inside the image
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(img_w, int(cx + half))
    y2 = min(img_h, int(cy + half))
    return x1, y1, x2, y2


def face_crop_box(person: dict[str, Any], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    # a square crop around the face

    # gather whichever face points we have
    face_points = []
    for index in HEAD_INDICES:
        point = confident_point(person, index)
        if point:
            face_points.append(point)

    if len(face_points) >= 2:
        # enough points: centre on them and size from the eyes
        cx, cy = average_point(face_points)
        half = crop_half_width(person)
    else:
        # not enough points: fall back to the top of the bounding box
        box = person_box(person) or [0.0, 0.0, float(img_w), float(img_h)]
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = y1 + (y2 - y1) * 0.12
        half = (x2 - x1) * 0.55

    return keep_box_in_image(cx, cy, half, img_w, img_h)
