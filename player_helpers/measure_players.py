# Measure players, find the player nearest the ball and shuffle measurements
import hashlib

import numpy as np

from .detect_players import calculate_point_to_box_distance
from study_settings import PLAYER_SHUFFLE_PREFIX, PLAYER_SHUFFLE_SEED, KEYPOINT_VISIBILITY_THRESHOLD


POSITION_FEATURES = ["box_cx", "box_cy", "box_w", "box_h", "box_area",
                     "detector_conf", "field_support"]
BODY_FEATURES = ["box_aspect", "visible_keypoint_fraction"]
for keypoint in range(17):
    BODY_FEATURES.extend([f"kp{keypoint}_x", f"kp{keypoint}_y", f"kp{keypoint}_visible"])
FULL_FEATURES = POSITION_FEATURES + BODY_FEATURES


def calculate_player_features(player, image_width, image_height):
    box_left, box_top, box_right, box_bottom = map(float, player["bbox"])
    box_width = max(1.0, box_right - box_left)
    box_height = max(1.0, box_bottom - box_top)
    features = {
        "box_cx": ((box_left + box_right) / 2) / image_width,
        "box_cy": ((box_top + box_bottom) / 2) / image_height,
        "box_w": box_width / image_width,
        "box_h": box_height / image_height,
        "box_area": box_width * box_height / (image_width * image_height),
        "box_aspect": box_width / box_height,
        "detector_conf": float(player["detector_conf"]),
        "field_support": float(player["field_support"]),
    }
    confident_keypoint_count = 0

    for keypoint_index, keypoint in enumerate(player["keypoints"]):
        confidence = float(keypoint["confidence"])
        keypoint_passes_confidence_check = confidence >= KEYPOINT_VISIBILITY_THRESHOLD
        confident_keypoint_count += int(keypoint_passes_confidence_check)
        features[f"kp{keypoint_index}_x"] = (
            (float(keypoint["x"]) - box_left) / box_width if keypoint_passes_confidence_check else 0.0
        )
        features[f"kp{keypoint_index}_y"] = (
            (float(keypoint["y"]) - box_top) / box_height if keypoint_passes_confidence_check else 0.0
        )
        features[f"kp{keypoint_index}_visible"] = float(keypoint_passes_confidence_check)
    features["visible_keypoint_fraction"] = confident_keypoint_count / 17
    return features


def find_nearest_player_to_ball(players, target_x, target_y, image_width, image_height):
    # Return the nearest player's index and distances, breaking ties by centre distance then index
    if not players:
        return None
    ranked_players = []
    target_pixel = np.array([target_x * image_width, target_y * image_height])
    image_diagonal = np.hypot(image_width, image_height)

    for player in players:
        box = player["bbox"]
        box_distance = calculate_point_to_box_distance(target_x, target_y, box, image_width, image_height)
        player_centre_pixels = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        centre_distance = float(np.linalg.norm(player_centre_pixels - target_pixel) / image_diagonal)
        ranked_players.append((box_distance, centre_distance, int(player["player_index"])))
    ranked_players.sort()
    box_distance, centre_distance, player_index = ranked_players[0]
    return {"player_index": player_index, "box_distance": box_distance,
            "centre_distance": centre_distance}


def make_repeatable_player_shuffle(sample, length):
    # Seed each image shuffle from its name and the fixed settings
    sample_digest = hashlib.sha256(
        f"{PLAYER_SHUFFLE_PREFIX}:{sample}:{PLAYER_SHUFFLE_SEED}".encode()
    ).digest()
    shuffle_generator = np.random.default_rng(int.from_bytes(sample_digest[:8], "little"))
    return shuffle_generator.permutation(length)
