from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def detect_people(image_path: Path, model_name: str = "yolov8n.pt", confidence: float = 0.25) -> list[dict]:
    # first YOLO step: just find the people in the picture
    from ultralytics import YOLO  # imported here so the scraper runs without it

    model = YOLO(model_name)
    results = model(str(image_path), conf=confidence, verbose=False)
    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None:
        return []

    people = []
    for box, score, cls in zip(
        boxes.xyxy.cpu().numpy().tolist(),
        boxes.conf.cpu().numpy().tolist(),
        boxes.cls.cpu().numpy().tolist(),
    ):
        if int(cls) != 0:  # class 0 is "person"
            continue
        people.append({"player": len(people) + 1, "confidence": float(score), "box": [float(v) for v in box]})
    return people


def draw_people(image_path: Path, people: list[dict], output_path: Path) -> None:
    # draw a red box round each person so I can check the detection by eye
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for person in people:
        x1, y1, x2, y2 = person["box"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text((x1 + 4, y1 + 4), f"player {person['player']}", fill="red")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
