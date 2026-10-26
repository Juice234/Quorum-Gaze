# Find players with one full-image detection pass, then check grass

import math

import numpy as np

from study_settings import PLAYER_DETECTION_SETTINGS, GRASS_CHECK_SETTINGS


def read_players_from_detection_result(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    players = []

    for box, confidence, points in zip(boxes, confidences, keypoints):
        body_points = []
        for point in points:
            body_points.append({"x": float(point[0]), "y": float(point[1]),
                                "confidence": float(point[2])})
        players.append({"bbox": [float(value) for value in box],
                        "detector_conf": float(confidence), "keypoints": body_points})
    return players


def detect_players_in_whole_image(model, image):
    settings = PLAYER_DETECTION_SETTINGS
    result = model.predict(image, imgsz=settings["whole_size"], conf=settings["confidence"],
                           device=settings["device"], verbose=False, max_det=settings["max_det"])[0]
    return read_players_from_detection_result(result)



def calculate_point_to_box_distance(x, y, box, width, height):
    # Measure the shortest distance to the box, divided by the image diagonal
    point_x, point_y = x * width, y * height
    horizontal_distance = max(float(box[0]) - point_x, 0.0, point_x - float(box[2]))
    vertical_distance = max(float(box[1]) - point_y, 0.0, point_y - float(box[3]))
    return float(math.hypot(horizontal_distance, vertical_distance) / math.hypot(width, height))


def measure_grass_under_player(image, box):
    # Count green pixels in an RGB patch around the bottom of the player's box
    height, width = image.shape[:2]
    box_left, box_top, box_right, box_bottom = map(float, box)
    box_width = max(1.0, box_right - box_left)
    box_height = max(1.0, box_bottom - box_top)
    patch_x1 = max(0, int(box_left + GRASS_CHECK_SETTINGS["side_crop"] * box_width))
    patch_x2 = min(width, int(box_right - GRASS_CHECK_SETTINGS["side_crop"] * box_width))
    patch_y1 = max(0, int(box_bottom - GRASS_CHECK_SETTINGS["above_feet"] * box_height))
    patch_y2 = min(height, int(box_bottom + GRASS_CHECK_SETTINGS["below_feet"] * box_height))
    patch = image[patch_y1:patch_y2, patch_x1:patch_x2]
    if patch.size == 0:
        return 0.0
    red = patch[..., 0].astype(np.float32)
    green = patch[..., 1].astype(np.float32)
    blue = patch[..., 2].astype(np.float32)
    grass_pixels = ((green > red * GRASS_CHECK_SETTINGS["green_over_red"]) &
                    (green > blue * GRASS_CHECK_SETTINGS["green_over_blue"]) &
                    (green > GRASS_CHECK_SETTINGS["minimum_green"]))
    return float(grass_pixels.mean())


def filter_players_by_grass_support(raw_players, image_bgr):
    import cv2

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    grass_scores = [measure_grass_under_player(image_rgb, player["bbox"]) for player in raw_players]
    kept_indices = [index for index, score in enumerate(grass_scores)
                    if score >= PLAYER_DETECTION_SETTINGS["field_minimum"]]
    kept_indices.sort(key=lambda index: float(raw_players[index]["detector_conf"]), reverse=True)

    players = []
    for player_index, old_index in enumerate(kept_indices):
        player = raw_players[old_index].copy()
        player["player_index"] = player_index
        player["field_support"] = float(grass_scores[old_index])
        players.append(player)
    return players
