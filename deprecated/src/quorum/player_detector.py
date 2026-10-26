from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def detect_people(image_path: Path, model_name: str = "yolov8n-pose.pt", confidence: float = 0.25) -> list[dict]:
    # YOLO pose model: find each person and the points on their body (eyes, shoulders, etc.)
    from ultralytics import YOLO  # imported here so the scraper runs without it

    model = YOLO(model_name)
    results = model(str(image_path), conf=confidence, verbose=False)
    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return []

    keypoints = _keypoints_per_person(result)
    people = []
    for i, (box, score, cls) in enumerate(
        zip(
            boxes.xyxy.cpu().numpy().tolist(),
            boxes.conf.cpu().numpy().tolist(),
            boxes.cls.cpu().numpy().tolist(),
        )
    ):
        if int(cls) != 0:  # class 0 is "person"
            continue
        people.append(
            {
                "player": len(people) + 1,
                "confidence": float(score),
                "box": [float(v) for v in box],
                "keypoints": keypoints[i] if i < len(keypoints) else [],
            }
        )
    return people


def _keypoints_per_person(result) -> list[list[dict]]:
    # pull the 17 body points (x, y, confidence) out for each person
    points = getattr(result, "keypoints", None)
    data = getattr(points, "data", None) if points is not None else None
    if data is None:
        return []
    out = []
    for person in data.cpu().numpy().tolist():
        out.append(
            [
                {"x": float(p[0]), "y": float(p[1]), "confidence": float(p[2]) if len(p) > 2 else 1.0}
                for p in person
            ]
        )
    return out


def draw_people(image_path: Path, people: list[dict], output_path: Path) -> None:
    # draw a red box round each person and a green dot on each body point, to check it by eye
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for person in people:
        x1, y1, x2, y2 = person["box"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text((x1 + 4, y1 + 4), f"player {person['player']}", fill="red")
        for point in person.get("keypoints", []):
            if point["confidence"] < 0.3:  # skip points the model wasn't sure about
                continue
            x, y = point["x"], point["y"]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="lime")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
