# Detect players, fit the player-selection and offset models, and predict ball positions

from pathlib import Path

from file_helpers import read_json, save_json
from player_helpers.detect_players import detect_players_in_whole_image, filter_players_by_grass_support
from player_helpers.measure_players import (
    FULL_FEATURES, POSITION_FEATURES, calculate_player_features, find_nearest_player_to_ball,
)
from player_helpers.predict_from_players import load_player_models, predict_ball_from_player_features
from study_settings import PLAYER_DETECTION_SETTINGS, FITTING_SETTINGS


def read_player_detections(row, run_dir):
    path = Path(run_dir) / "detections" / (row["sample"] + ".json")
    return read_json(path)


def detect_players_in_study_images(inference_records, detector_weights, run_dir):
    # Detect on the whole image, then filter by grass support
    import cv2
    import torch
    from threadpoolctl import threadpool_limits
    from ultralytics import YOLO

    run_dir = Path(run_dir).resolve()
    rows = list(inference_records)
    detector_weights = Path(detector_weights).resolve()
    model = None
    with threadpool_limits(limits=PLAYER_DETECTION_SETTINGS["threads"]):
        for number, row in enumerate(rows, 1):
            path = run_dir / "detections" / (row["sample"] + ".json")
            if path.exists():
                continue

            if model is None:
                torch.set_num_threads(PLAYER_DETECTION_SETTINGS["threads"])
                torch.manual_seed(PLAYER_DETECTION_SETTINGS["seed"])
                torch.use_deterministic_algorithms(True, warn_only=True)
                model = YOLO(str(detector_weights))
            image = cv2.imread(str(row["prepared_path"]), cv2.IMREAD_COLOR)
            raw = detect_players_in_whole_image(model, image)
            players = filter_players_by_grass_support(raw, image)
            record = {"sample": row["sample"],
                      "filtered_players": players}
            save_json(path, record)
            if number % 100 == 0 or number == len(rows):
                print(f"Detected players {number}/{len(rows)}")


def fit_player_selection_and_offset_models(training_records, run_dir):
    # Label the nearest player as 1 and the others as 0 in training images
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    run_dir = Path(run_dir).resolve()
    rows = sorted(training_records, key=lambda row: row["source_order"])
    folder = run_dir / "fit"
    model_path = folder / "models.joblib"
    record_path = folder / "model_record.json"
    if model_path.exists():
        print("Reusing saved fitted models")
        return model_path

    # Use training images only when a player box is close enough to the ball
    eligible_count = 0
    player_rows = []
    for row in rows:
        detection = read_player_detections(row, run_dir)
        players = detection["filtered_players"]
        target_x, target_y = row["training_target_xy"]
        nearest = find_nearest_player_to_ball(players, target_x, target_y, row["image_width"], row["image_height"])
        if nearest is None or not np.isfinite(nearest["box_distance"]) or nearest["box_distance"] > FITTING_SETTINGS["near_ball_distance"]:
            continue
        nearest_player_index = nearest["player_index"]
        eligible_count += 1
        for player in players:
            features = calculate_player_features(player, row["image_width"], row["image_height"])
            player_rows.append({"sample": row["sample"], "game": row["game"], "player_index": player["player_index"],
                                "is_nearest_to_ball": int(player["player_index"] == nearest_player_index),
                                "target_x": target_x, "target_y": target_y, **features})
    table = pd.DataFrame(player_rows)
    training_average = np.array([row["training_target_xy"] for row in rows]).mean(axis=0)
    models = {"training_average": training_average}
    feature_columns = {}

    with threadpool_limits(limits=FITTING_SETTINGS["threads"]):
        for name, seed in (("position", FITTING_SETTINGS["position_x_seed"]), ("full", FITTING_SETTINGS["full_x_seed"])):
            columns = POSITION_FEATURES if name == "position" else FULL_FEATURES

            selector = make_pipeline(StandardScaler(), LogisticRegression(
                **FITTING_SETTINGS["selector"], random_state=FITTING_SETTINGS["selector_seed"]
            ))
            selector.fit(table[columns].to_numpy(dtype=float), table["is_nearest_to_ball"].to_numpy(dtype=int))
            nearest_players = table.loc[table["is_nearest_to_ball"].eq(1)]
            offset_models = []
            for axis, seed_offset in (("x", 0), ("y", FITTING_SETTINGS["y_seed_increment"])):
                estimator = HistGradientBoostingRegressor(
                    **FITTING_SETTINGS["offset"], random_state=seed + seed_offset
                )
                offset = nearest_players["target_" + axis] - nearest_players["box_c" + axis]
                estimator.fit(nearest_players[columns].to_numpy(dtype=float), offset.to_numpy(dtype=float))
                offset_models.append(estimator)
            models[name + "_selector"] = selector
            models[name + "_offsets"] = tuple(offset_models)
            feature_columns[name] = {"columns": columns}
            print(f"Fitted {name} selector and x/y offsets")

    folder.mkdir(parents=True, exist_ok=True)
    with model_path.open("xb") as stream:
        joblib.dump(models, stream)
    save_json(record_path, {"settings": FITTING_SETTINGS, "resolved_parameters": feature_columns,
                            "training_frames": len(rows), "training_matches": len({row["game"] for row in rows}),
                            "eligible_training_frames": eligible_count, "training_player_rows": len(table),
                            "training_average": training_average.tolist(),
                            "target_precision": "Ball centres rounded to eight decimal places"})
    return model_path


def print_ball_prediction_summary(predictions, shown=5):
    no_players = sum(row["no_players"] for row in predictions)
    print(f"Predicted the ball in {len(predictions)} images, {no_players} had no players "
          "and use the training average.")
    print("First predictions (x, y as fractions of image width and height):")
    for row in predictions[:shown]:
        raw, blended = row["points"]["raw"], row["points"]["blended"]
        print(f"  {row['sample']}: {len(row['players'])} players, chosen player {row['selected_players']['full']}, "
              f"raw ({raw[0]:.3f}, {raw[1]:.3f}), blended ({blended[0]:.3f}, {blended[1]:.3f})")


def predict_ball_positions(inference_records, model_path, run_dir):
    # Predict from saved detections and fitted models
    from threadpoolctl import threadpool_limits

    run_dir = Path(run_dir).resolve()
    rows = list(inference_records)
    model_path = Path(model_path).resolve()
    output_path = run_dir / "predictions/player_points.json"
    if output_path.exists():
        print("Reusing saved predictions:", output_path)
        return output_path

    models = load_player_models(model_path)
    predictions = []
    with threadpool_limits(limits=FITTING_SETTINGS["threads"]):
        for row in rows:
            detection = read_player_detections(row, run_dir)
            players = detection["filtered_players"]
            features = [{"player_index": player["player_index"],
                         **calculate_player_features(player, row["image_width"], row["image_height"])}
                        for player in players]
            result = predict_ball_from_player_features(models, features, row["sample"])
            predictions.append({"sample": row["sample"], "game": row["game"], "original61": row["original61"],
                                "image_width": row["image_width"], "image_height": row["image_height"],
                                "players": players,
                                "features": features, **result})
    save_json(output_path, predictions)
    print_ball_prediction_summary(predictions)
    return output_path
