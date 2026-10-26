# Read the images, choose training and evaluation matches, and remove the labelled balls

import ast
import csv
import hashlib
import math
from pathlib import Path
from zipfile import ZipFile

from file_helpers import save_json
from study_settings import BALL_REMOVAL

# Pass image details to prediction without ball coordinates
TARGET_FREE_FIELDS = (
    "sample", "game", "partition",
    "original61", "image_width", "image_height", "prepared_path",
)


def get_match_folder_name(row):
    parts = [row[name] for name in ("league", "season", "match")]
    return "/".join(parts)


def get_frame_file_name(row):
    if row["action"]:
        frame = f"{row['main_action']}.png"
    else:
        frame = f"{row['main_action']}_{int(float(row['replay']))}.png"
    return frame


def read_study_records(source_csv):
    # Keep test matches out of training
    with Path(source_csv).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row["is_ball"].lower() == "true" and row["ball_bbox"]]
    grouped = {}
    for row in rows:
        game = get_match_folder_name(row)
        if game not in grouped:
            grouped[game] = []
        grouped[game].append(row)

    # Keep source order and convert each ball box to its centre, divided by image width and height
    records = []
    for game, game_rows in grouped.items():
        for row in game_rows:
            width, height = int(float(row["img_w"])), int(float(row["img_h"]))
            box = [float(value) for value in ast.literal_eval(row["ball_bbox"])]
            x, y, box_width, box_height = box
            target = [(x + box_width / 2) / width, (y + box_height / 2) / height]
            records.append({
                "sample": f"snv3d_{int(row['']):05d}", "game": game, "frame": get_frame_file_name(row),
                "source_set": row["set"], "source_order": len(records),
                "image_width": width, "image_height": height,
                "target_xy": target, "ball_bbox": box,
            })

    # Ball positions and prediction errors do not enter the ranking
    candidates = {}
    for record in records:
        if record["source_set"] == "test":
            game = record["game"]
            if game not in candidates:
                candidates[game] = []
            candidates[game].append(record)

    sample_61_records = []
    for game_records in candidates.values():
        chosen_record = None
        chosen_rank = None
        for record in game_records:
            frame_name = record["game"] + "/" + record["frame"]
            frame_hash = hashlib.sha256(frame_name.encode("utf-8")).hexdigest()
            rank = (frame_hash, record["sample"])
            if chosen_record is None or rank < chosen_rank:
                chosen_record = record
                chosen_rank = rank
        sample_61_records.append(chosen_record)
    sample_61_records.sort(key=lambda row: (row["game"], row["sample"]))
    sample_61_ids = {row["sample"] for row in sample_61_records}
    evaluation_games = {row["game"] for row in sample_61_records}

    # Apply the split to whole matches
    for record in records:
        record["partition"] = "evaluation" if record["game"] in evaluation_games else "training"
        record["original61"] = record["sample"] in sample_61_ids
        if record["partition"] == "training":
            record["training_target_xy"] = [float(f"{value:.8f}") for value in record["target_xy"]]
    training_records = [row for row in records if row["partition"] == "training"]
    additional_evaluation_records = [row for row in records if row["partition"] == "evaluation" and not row["original61"]]
    return training_records + sample_61_records + additional_evaluation_records


def read_frame_from_match_archive(record, images_root):
    archive_path = Path(images_root) / record["game"] / "Frames-v3.zip"
    with ZipFile(archive_path) as archive:
        return archive.read(record["frame"])


def get_ball_removal_bounds(box, width, height):
    # Add padding to the ball box
    x, y, box_width, box_height = box
    padding = BALL_REMOVAL["padding"]
    return [max(0, math.floor(x) - padding), max(0, math.floor(y) - padding),
            min(width, math.ceil(x + box_width) + padding), min(height, math.ceil(y + box_height) + padding)]


def inpaint_rectangle(image_bgr, bounds):
    # Fill the rectangle with Telea inpainting
    import cv2
    import numpy as np

    x0, y0, x1, y1 = bounds
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return cv2.inpaint(image_bgr, mask, BALL_REMOVAL["radius"], cv2.INPAINT_TELEA), mask


def copy_records_without_ball_labels(records):
    prediction_records = []
    for record in records:
        image_details = {}
        for name in TARGET_FREE_FIELDS:
            image_details[name] = record[name]
        prediction_records.append(image_details)
    return prediction_records


def prepare_ball_removed_images(source_csv, images_root, run_dir):
    import cv2
    import numpy as np

    source_csv = Path(source_csv).resolve()
    images_root = Path(images_root).resolve()
    run_dir = Path(run_dir).resolve()
    records = read_study_records(source_csv)

    data_dir = run_dir / "data"

    completed = []
    for number, base in enumerate(records, 1):
        image_path = data_dir / "images" / (base["sample"] + ".png")
        bounds = get_ball_removal_bounds(base["ball_bbox"], base["image_width"], base["image_height"])

        if not image_path.exists():
            image_bytes = read_frame_from_match_archive(base, images_root)
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            removed, _ = inpaint_rectangle(image, bounds)
            _, encoded = cv2.imencode(".png", removed)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(encoded.tobytes())

        completed.append({**base, "prepared_path": str(image_path), "mask_bounds": bounds})
        if number % 100 == 0 or number == len(records):
            print(f"Prepared {number}/{len(records)} images")
    save_json(data_dir / "records.json", completed)
    return completed
