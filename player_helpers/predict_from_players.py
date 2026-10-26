# Choose a player, predict the ball's offset from them and blend with the training average

import numpy as np

from study_settings import FITTING_SETTINGS, PLAYER_METHODS
from .measure_players import (
    POSITION_FEATURES, FULL_FEATURES, BODY_FEATURES, make_repeatable_player_shuffle,
)


def load_player_models(path):
    import joblib

    models = joblib.load(path)
    if "training_average" not in models:
        models["training_average"] = models.pop("prior")
        for name in ("full", "position"):
            models[name + "_offsets"] = models.pop(name + "_xy")
    return models


def blend_with_training_average(training_average, raw_point):
    blend_divisor = 1 / FITTING_SETTINGS["blend_weight"]
    return np.asarray(training_average, dtype=float) + (
        np.asarray(raw_point, dtype=float) - np.asarray(training_average, dtype=float)
    ) / blend_divisor


def predict_ball_from_player_features(models, features, sample):
    import pandas as pd

    training_average = np.asarray(models["training_average"], dtype=float)
    player_table = pd.DataFrame(features)

    # With no detected players, return the training average for every player
    if player_table.empty:
        return {
            "points": {method: training_average.tolist() for method in PLAYER_METHODS},
            "raw_points": {name: training_average.tolist() for name in ("full", "position", "shuffled")},
            "selected_players": {"full": None, "position": None, "shuffled": None},
            "no_players": True,
        }
    player_table = player_table.sort_values("player_index").reset_index(drop=True)

    # Shuffle body measurements only
    shuffled_player_table = player_table.copy()
    permutation = make_repeatable_player_shuffle(sample, len(player_table))
    shuffled_player_table.loc[:, BODY_FEATURES] = player_table.iloc[permutation][BODY_FEATURES].to_numpy()
    selected_players = {}
    raw_points = {}

    for selector_name in ("full", "position", "shuffled"):
        model_name = "full"
        feature_table = player_table
        feature_columns = FULL_FEATURES
        if selector_name == "position":
            model_name = "position"
            feature_columns = POSITION_FEATURES
        if selector_name == "shuffled":
            feature_table = shuffled_player_table

        selector = models[model_name + "_selector"]
        offset_models = models[model_name + "_offsets"]
        player_probabilities = selector.predict_proba(feature_table[feature_columns].to_numpy(dtype=float))[:, 1]
        selected_player_row = int(np.argmax(np.asarray(player_probabilities, dtype=float)))
        selected_player_features = feature_table.iloc[selected_player_row]
        selected_players[selector_name] = int(selected_player_features["player_index"])
        selected_feature_values = selected_player_features[feature_columns].to_numpy(dtype=float)[None, :]
        predicted_x = float(selected_player_features["box_cx"]) + float(
            offset_models[0].predict(selected_feature_values)[0]
        )
        predicted_y = float(selected_player_features["box_cy"]) + float(
            offset_models[1].predict(selected_feature_values)[0]
        )
        raw_points[selector_name] = np.clip([predicted_x, predicted_y], 0, 1)
    return {
        "points": {
            "blended": blend_with_training_average(training_average, raw_points["full"]).tolist(),
            "raw": raw_points["full"].tolist(),
            "position_only_blended": blend_with_training_average(training_average, raw_points["position"]).tolist(),
            "shuffled_body_blended": blend_with_training_average(training_average, raw_points["shuffled"]).tolist(),
            "training_average": training_average.tolist(),
        },
        "raw_points": {name: point.tolist() for name, point in raw_points.items()},
        "selected_players": selected_players, "no_players": False,
    }
