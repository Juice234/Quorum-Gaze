# Build head boxes from the facial landmarks already found with the players

from pathlib import Path

import numpy as np

from file_helpers import read_json, save_json
from study_settings import HEAD_DETECTION_SETTINGS

HEAD_KEYPOINT_INDICES = (0, 1, 2, 3, 4)

def make_head_box_from_landmarks(player, image_width, image_height):
    settings = HEAD_DETECTION_SETTINGS
    face_points = []
    for index in HEAD_KEYPOINT_INDICES:
        point = player["keypoints"][index]
        if point["confidence"] >= settings["minimum_head_keypoint_confidence"]:
            face_points.append((float(point["x"]), float(point["y"])))
    if len(face_points) < settings["minimum_head_keypoints"]:
        return None

    # Estimate the head size from the spread of facial landmarks
    points = np.array(face_points)
    centre_x, centre_y = float(points[:, 0].mean()), float(points[:, 1].mean())
    distances = []
    for index, first in enumerate(points):
        for second in points[index + 1:]:
            distances.append(np.linalg.norm(first - second))
    size = float(max(distances)) * settings["head_keypoint_size_scale"]
    player_height = player["bbox"][3] - player["bbox"][1]
    size = max(size, player_height * settings["minimum_head_player_height_ratio"])

    half_crop_size = size * settings["head_crop_expansion"] / 2
    bounds = [max(0, int(centre_x - half_crop_size)), max(0, int(centre_y - half_crop_size)),
              min(image_width, int(centre_x + half_crop_size)), min(image_height, int(centre_y + half_crop_size))]
    return [bounds[0] / image_width, bounds[1] / image_height,
            bounds[2] / image_width, bounds[3] / image_height]


def find_player_head_boxes(players, image_width, image_height):
    boxes = []
    for player in players:
        box = make_head_box_from_landmarks(player, image_width, image_height)
        if box is not None:
            boxes.append(box)
    return boxes


def prepare_head_boxes_for_comparison(original_evaluation_images, run_dir):
    run_dir = Path(run_dir)
    records = list(original_evaluation_images)
    for number, row in enumerate(records, 1):
        destination = run_dir / "heads" / (row["sample"] + ".json")
        if destination.exists():
            continue
        players = read_json(run_dir / "detections" / (row["sample"] + ".json"))["filtered_players"]
        boxes = find_player_head_boxes(players, row["image_width"], row["image_height"])
        save_json(destination, {"sample": row["sample"], "normalized_head_boxes": boxes})
        print(f"Head boxes {number}/{len(records)}")
