# Compare player selection and predicted offsets with known selection and player centres

import numpy as np

from calculate_results import (
    convert_point_to_list, calculate_point_error, index_records_by_sample, calculate_error_metrics,
    make_bootstrap_sample_indices, calculate_bootstrap_interval,
    calculate_results_for_each_match, compare_two_methods_with_bootstrap,
)
from player_helpers.measure_players import FULL_FEATURES, find_nearest_player_to_ball
from player_helpers.predict_from_players import blend_with_training_average
from study_settings import BOOTSTRAP, METHODS, CLOSE_ERROR_THRESHOLD, NEAR_BALL_DISTANCE


def find_player_features(features, player_index):
    for player in features:
        if int(player["player_index"]) == player_index:
            return player


def predict_ball_with_answer_selected_player(models, features, index):
    training_average = np.asarray(models["training_average"], dtype=float)
    if index is None:
        return training_average.tolist(), training_average.tolist()
    selected = find_player_features(features, index)
    vector = np.asarray([[selected[column] for column in FULL_FEATURES]], dtype=float)
    offset = [float(model.predict(vector)[0]) for model in models["full_offsets"]]
    raw = np.clip([selected["box_cx"] + offset[0], selected["box_cy"] + offset[1]], 0, 1)
    return raw.tolist(), blend_with_training_average(training_average, raw).tolist()


def make_answer_assisted_player_rows(original, models):
    # Select the nearest player using the answer
    rows = []
    for source in original:
        width, height = source["image_width"], source["image_height"]
        nearest = find_nearest_player_to_ball(source["players"], *source["target_xy"], width, height)
        index = None if nearest is None else nearest["player_index"]
        raw, blend = predict_ball_with_answer_selected_player(models, source["features"], index)
        points = {method: source["points"][method] for method in METHODS}
        points.update(answer_assisted_raw=raw, answer_assisted_blended=blend)
        chosen = source["selected_players"]["full"]
        rows.append({"sample": source["sample"], "game": source["game"],
                     "target_xy": source["target_xy"], "image_width": width, "image_height": height,
                     "player_count": len(source["players"]), "nearest_player_index": index,
                     # Distance from the ball to the nearest box, as a fraction of the image diagonal
                     "nearest_box_distance": None if nearest is None else nearest["box_distance"],
                     "eligible": nearest is not None and nearest["box_distance"] <= NEAR_BALL_DISTANCE,
                     "chosen_player_index": chosen,
                     "chose_nearest_player": None if nearest is None else chosen == index,
                     "no_players": source["no_players"], "points": points,
                     "errors": {prediction_method: calculate_point_error(predicted_point, source["target_xy"]) for prediction_method, predicted_point in points.items()}})

    return rows


def summarize_answer_assisted_player_rows(rows):
    summary = {"subgroups": {}, "selector_agreement": {},
               "bootstrap": {**BOOTSTRAP, "unit": "whole match, one image each"}}

    # Compare errors with and without a player near the ball
    groups = {"all_61": rows,
              "eligible": [row for row in rows if row["eligible"]],
              "ineligible": [row for row in rows if not row["eligible"]]}
    for name, subset in groups.items():
        summary["subgroups"][name] = {
            "images": len(subset), "no_player_count": sum(image_result["no_players"] for image_result in subset),
            "methods": calculate_error_metrics(subset, (*METHODS, "answer_assisted_raw", "answer_assisted_blended")),
            "paired": {"answer_assisted_raw minus raw": compare_two_methods_with_bootstrap(subset, "answer_assisted_raw", "raw"),
                       "answer_assisted_blended minus blended": compare_two_methods_with_bootstrap(subset, "answer_assisted_blended", "blended")}}

    # Count agreement with nearest-player selection for all player-present and eligible images
    agreement_groups = {"player_present": [row for row in rows if row["player_count"]],
                        "eligible": groups["eligible"]}
    for name, subset in agreement_groups.items():
        count = sum(bool(image_result["chose_nearest_player"]) for image_result in subset)
        summary["selector_agreement"][name] = {"denominator": len(subset), "count": count,
                                               "rate": count / len(subset) if subset else None}
    return summary


def evaluate_answer_assisted_player_selection(original, models):
    rows = make_answer_assisted_player_rows(original, models)
    return summarize_answer_assisted_player_rows(rows), rows


def get_selected_player_centre(row):
    # The chosen player's box centre, or the training average when no player was found
    if row["no_players"]:
        return row["points"]["training_average"]
    index = row["selected_players"]["full"]
    feature = find_player_features(row["features"], index)
    return convert_point_to_list([feature["box_cx"], feature["box_cy"]])


def make_player_centre_comparison_rows(evaluation, player_selection_rows):
    # Score each predicted point and the centre of the player it was based on
    lookup = index_records_by_sample(player_selection_rows)
    rows = []
    extra_checks = {"near_close_threshold": [], "no_player_samples": [], "different_full_position_selection": 0}

    for row in evaluation:
        points = {"raw": row["points"]["raw"], "position_only_raw": row["raw_points"]["position"],
                  "chosen_player_centre": get_selected_player_centre(row)}
        if row["original61"]:
            assisted = lookup[row["sample"]]
            index = assisted["nearest_player_index"]
            if index is None:
                centre = row["points"]["training_average"]
            else:
                feature = find_player_features(row["features"], index)
                centre = [feature["box_cx"], feature["box_cy"]]
            points.update(answer_assisted_raw=assisted["points"]["answer_assisted_raw"], answer_selected_centre=centre)
        errors = {}
        for method, point in points.items():
            difference = np.asarray(convert_point_to_list(point)) - row["target_xy"]
            errors[method] = float(np.linalg.norm(difference))

        if row["no_players"]:
            extra_checks["no_player_samples"].append(row["sample"])
        else:
            extra_checks["different_full_position_selection"] += int(
                row["selected_players"]["full"] != row["selected_players"]["position"])
        for method, error in errors.items():
            if abs(error - CLOSE_ERROR_THRESHOLD) <= 1e-12:
                extra_checks["near_close_threshold"].append({"sample": row["sample"], "method": method, "error": error})
        rows.append({"sample": row["sample"], "game": row["game"], "original61": row["original61"],
                     "no_players": row["no_players"], "points": points, "target_xy": row["target_xy"], "errors": errors})

    return rows, extra_checks


def summarize_player_centre_sample(rows, order, methods):
    matches = calculate_results_for_each_match(rows, order, methods)
    image_scores = calculate_error_metrics(rows, methods)
    summary = {"images": len(rows), "matches": len(order),
               "no_player_count": sum(row["no_players"] for row in rows), "methods": {}}
    for method in methods:
        values = image_scores[method].copy()
        values["match_balanced_mean_error"] = float(np.mean([row["methods"][method]["mean_error"] for row in matches]))
        values["match_balanced_close_rate"] = float(np.mean([row["methods"][method]["close_rate"] for row in matches]))
        summary["methods"][method] = values
    return summary, matches


def summarize_player_centre_comparisons(rows, order, extra_checks):
    original = [row for row in rows if row["original61"]]
    ordinary_methods = ("raw", "position_only_raw", "chosen_player_centre")
    all_methods = (*ordinary_methods, "answer_assisted_raw", "answer_selected_centre")
    groups = {"original61": (original, all_methods), "all813": (rows, ordinary_methods)}
    summaries, match_results, contrasts = {}, [], []
    bootstrap_indices = make_bootstrap_sample_indices(len(order))

    for name, (sample_rows, methods) in groups.items():
        summaries[name], matches = summarize_player_centre_sample(sample_rows, order, methods)

        # Compare the same methods using the same matches
        pairs = [("raw", "position_only_raw"), ("raw", "chosen_player_centre")]
        if name == "original61":
            pairs.append(("answer_assisted_raw", "answer_selected_centre"))
        for first, second in pairs:
            mean_error_change = np.asarray([result_row["methods"][first]["mean_error"] - result_row["methods"][second]["mean_error"] for result_row in matches])
            close_rate_change = 100 * np.asarray([result_row["methods"][first]["close_rate"] - result_row["methods"][second]["close_rate"] for result_row in matches])
            comparison = {"sample_set": name, "first": first, "second": second,
                          "mean_error": calculate_bootstrap_interval(mean_error_change[bootstrap_indices].mean(axis=1), mean_error_change.mean()),
                          "close_rate": calculate_bootstrap_interval(close_rate_change[bootstrap_indices].mean(axis=1), close_rate_change.mean(), True)}
            if first == "answer_assisted_raw":
                answer_assisted_comparison = comparison
            else:
                contrasts.append(comparison)
        for row in matches:
            match_result = {"sample_set": name}
            match_result.update(row)
            match_results.append(match_result)

    # Show comparisons for both samples before the known check
    contrasts.append(answer_assisted_comparison)
    return {"samples": summaries, "contrasts": contrasts, "edge_cases": extra_checks,
            "bootstrap": {**BOOTSTRAP, "unit": "whole match, equally weighted",
                          "shared_across_all_five_contrasts": True}}, match_results


def compare_predictions_with_player_centres(evaluation, player_selection_rows, order):
    rows, extra_checks = make_player_centre_comparison_rows(evaluation, player_selection_rows)
    summary, match_results = summarize_player_centre_comparisons(rows, order, extra_checks)
    return summary, rows, match_results
