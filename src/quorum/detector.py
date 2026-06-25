from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parents[2]


def model_path(model_name: str) -> str:
    # Prefer a model in the local models folder. If it is not there, Ultralytics can download it.
    local_model = PROJECT_DIR / "models" / model_name
    if local_model.exists():
        return str(local_model)
    return model_name


def detect_people(image_path: Path, model_name: str = "yolov8n.pt", confidence: float = 0.25) -> list[dict[str, Any]]:
    # YOLO class 0 is "person", so this function ignores every other object class.
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -e .[yolo]") from exc

    model = YOLO(model_path(model_name))
    results = model(str(image_path), conf=confidence, verbose=False)
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy().tolist()
    scores = boxes.conf.cpu().numpy().tolist()
    classes = boxes.cls.cpu().numpy().tolist()

    people = []
    for index, box in enumerate(xyxy):
        class_id = int(classes[index])
        if class_id != 0:
            continue
        people.append(
            {
                "person_number": len(people) + 1,
                "confidence": float(scores[index]),
                "box": [float(value) for value in box],
            }
        )
    return people


def draw_people(image_path: Path, people: list[dict[str, Any]], output_path: Path) -> None:
    # Draw simple red boxes so the result is easy to inspect in the GUI.
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for person in people:
        x1, y1, x2, y2 = person["box"]
        label = f"person {person['person_number']} {person['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.rectangle((x1, max(0, y1 - 18), x1 + 120, y1), fill="red")
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_detection_json(image_path: Path, people: list[dict[str, Any]], output_path: Path, model_name: str) -> None:
    # Save plain JSON so it is easy to inspect or reuse later.
    payload = {
        "image": str(image_path),
        "model": model_name,
        "people_count": len(people),
        "people": people,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
