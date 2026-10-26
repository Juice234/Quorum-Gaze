# Calculate point errors, close predictions and paired comparisons between methods
# Point error is measured using x/image width and y/image height

from pathlib import Path

import numpy as np

from file_helpers import save_json
from study_settings import BOOTSTRAP, CLOSE_ERROR_THRESHOLD, METHODS, PLAYER_METHODS

LABELS = {
    "blended": "Blended prediction", "raw": "Raw prediction",
    "position_only_blended": "Position-only blend",
    "shuffled_body_blended": "Shuffled body measurements",
    "training_average": "Training average", "image_centre": "Image centre",
    "gaze_lle": "Gaze-LLE", "gaze_lle_with_heads": "Gaze-LLE with detected heads",
    "answer_assisted_raw": "Answer-assisted selector, raw",
    "answer_assisted_blended": "Answer-assisted selector, blended",
    "position_only_raw": "Position-only raw",
    "chosen_player_centre": "Chosen player's centre",
    "answer_selected_centre": "Answer-selected player's centre",
}


def convert_point_to_list(value):
    return np.asarray(value, dtype=float).tolist()


def calculate_point_error(prediction, target):
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    horizontal_error, vertical_error = prediction - target
    return float(np.hypot(horizontal_error, vertical_error))


def index_records_by_sample(rows):
    return {row["sample"]: row for row in rows}


def calculate_error_metrics(rows, methods):
    result = {}
    for method in methods:
        values = np.asarray([row["errors"][method] for row in rows], dtype=float)
        result[method] = {
            "images": len(rows),
            "mean_error": float(values.mean()) if len(rows) else None,
            "median_error": float(np.median(values)) if len(rows) else None,
            "close_count": int((values <= CLOSE_ERROR_THRESHOLD).sum()),
            "close_rate": float((values <= CLOSE_ERROR_THRESHOLD).mean()) if len(rows) else None,
        }
    return result


def make_bootstrap_sample_indices(count):
    return np.random.RandomState(BOOTSTRAP["seed"]).randint(
        0, count, size=(BOOTSTRAP["draws"], count))


def calculate_bootstrap_interval(values, estimate, percentage_points=False):
    lower_bound, upper_bound = np.percentile(values, BOOTSTRAP["percentiles"], method=BOOTSTRAP["method"])
    suffix = "_pp" if percentage_points else ""
    return {"estimate" + suffix: float(estimate), "lower" + suffix: float(lower_bound),
            "upper" + suffix: float(upper_bound)}


def compare_methods_with_paired_bootstrap(rows, methods):
    indices = make_bootstrap_sample_indices(len(rows))
    values, draws = {}, {}
    for method in methods:
        errors = np.asarray([row["errors"][method] for row in rows])
        values[method] = errors
        draws[method] = {"mean": errors[indices].mean(axis=1),
                         "close": (errors[indices] <= CLOSE_ERROR_THRESHOLD).mean(axis=1)}
    pairs = {}
    for index, first in enumerate(methods):
        for second in methods[index + 1:]:
            mean_changes = draws[first]["mean"] - draws[second]["mean"]
            mean_estimate = values[first].mean() - values[second].mean()
            close_changes = 100 * (draws[first]["close"] - draws[second]["close"])
            close_estimate = 100 * ((values[first] <= CLOSE_ERROR_THRESHOLD).mean() -
                                    (values[second] <= CLOSE_ERROR_THRESHOLD).mean())
            pairs[first + " minus " + second] = {
                "mean_error": calculate_bootstrap_interval(mean_changes, mean_estimate),
                "close_rate": calculate_bootstrap_interval(close_changes, close_estimate, True),
            }
    return {**BOOTSTRAP, "unit": "one image per whole match", "pairs": pairs}


def calculate_results_for_each_match(rows, order, methods):
    result = []
    for game in order:
        block = [row for row in rows if row["game"] == game]
        match_scores = {}
        for method in methods:
            match_scores[method] = {
                "mean_error": float(np.mean([image_result["errors"][method] for image_result in block])),
                "close_rate": float(np.mean([image_result["errors"][method] <= CLOSE_ERROR_THRESHOLD for image_result in block])),
            }
        result.append({"game": game, "images": len(block), "methods": match_scores})
    return result


def calculate_match_balanced_results(rows, order):
    # Average within each match first, matches with more frames do not dominate
    matches = calculate_results_for_each_match(rows, order, METHODS)
    indices = make_bootstrap_sample_indices(len(matches))
    means, rates = {}, {}
    for method in METHODS:
        means[method] = np.asarray([result_row["methods"][method]["mean_error"] for result_row in matches])
        rates[method] = np.asarray([result_row["methods"][method]["close_rate"] for result_row in matches])
    pairs = {}
    for index, first in enumerate(METHODS):
        for second in METHODS[index + 1:]:
            difference = means[first] - means[second]
            close_difference = rates[first] - rates[second]
            mean_interval = calculate_bootstrap_interval(difference[indices].mean(axis=1), difference.mean())
            close_interval = calculate_bootstrap_interval(100 * close_difference[indices].mean(axis=1),
                                                          100 * close_difference.mean(), True)
            pairs[first + " minus " + second] = {
                "mean_error": mean_interval,
                "close_rate": close_interval,
            }
    output = {"images": len(rows), "matches": len(matches),
              "new_images": sum(not row["original61"] for row in rows),
              "reused_images": sum(row["original61"] for row in rows),
              "primary_weighting": "mean within each match, then equal mean across matches",
              "secondary_weighting": "all images equally weighted",
              "no_player_count": sum(row["no_players"] for row in rows), "methods": {}}
    for method in METHODS:
        values = np.asarray([result_row["errors"][method] for result_row in rows])
        output["methods"][method] = {
            "match_balanced_mean_error": float(means[method].mean()),
            "match_balanced_close_rate": float(rates[method].mean()),
            "image_weighted_mean_error": float(values.mean()),
            "image_weighted_median_error": float(np.median(values)),
            "image_close_count": int((values <= CLOSE_ERROR_THRESHOLD).sum()),
            "image_close_rate": float((values <= CLOSE_ERROR_THRESHOLD).mean()),
        }
    output["bootstrap"] = {**BOOTSTRAP, "pairs": pairs,
                           "unit": "whole match, equally weighted match means/rates"}
    return output, matches


def compare_two_methods_with_bootstrap(rows, first, second):
    if not rows:
        return {"images": 0, "mean_error": None, "close_rate": None}
    first_errors = np.asarray([image_result["errors"][first] for image_result in rows])
    second_errors = np.asarray([image_result["errors"][second] for image_result in rows])
    indices = make_bootstrap_sample_indices(len(rows))
    difference = first_errors - second_errors
    close_change = 100 * ((first_errors <= CLOSE_ERROR_THRESHOLD).astype(int) -
                          (second_errors <= CLOSE_ERROR_THRESHOLD).astype(int))
    return {"images": len(rows),
            "mean_error": calculate_bootstrap_interval(difference[indices].mean(axis=1), difference.mean()),
            "close_rate": calculate_bootstrap_interval(close_change[indices].mean(axis=1), close_change.mean(), True)}


def collect_evaluation_predictions(records, player_predictions, gaze_predictions):
    player_lookup = index_records_by_sample(player_predictions)
    gaze_lookup = index_records_by_sample(gaze_predictions)
    training, selected = [], []
    for record in records:
        if record["partition"] == "training":
            training.append(record)
        if record["partition"] == "evaluation":
            selected.append(record)
    original_records = [image_record for image_record in selected if image_record["original61"]]
    order = [image_record["game"] for image_record in original_records]
    original_ids = {image_record["sample"] for image_record in original_records}
    evaluation = []

    for record in selected:
        sample = record["sample"]
        prediction, gaze = player_lookup[sample], gaze_lookup[sample]
        target = convert_point_to_list(record["target_xy"])
        points = {}
        for method in PLAYER_METHODS:
            points[method] = convert_point_to_list(prediction["points"][method])
        points["image_centre"] = [.5, .5]
        points["gaze_lle"] = convert_point_to_list(gaze["no_head_point"])
        points["gaze_lle_with_heads"] = (None if gaze["with_heads_point"] is None
                                         else convert_point_to_list(gaze["with_heads_point"]))
        width, height = int(record["image_width"]), int(record["image_height"])
        errors = {}
        for method, point in points.items():
            errors[method] = None if point is None else calculate_point_error(point, target)
        evaluation.append({"sample": sample, "game": record["game"], "original61": sample in original_ids,
                           "target_xy": target, "image_width": width, "image_height": height,
                           "players": prediction["players"], "features": prediction["features"],
                           "selected_players": prediction["selected_players"],
                           "no_players": bool(prediction["no_players"]),
                           "points": points, "raw_points": prediction["raw_points"],
                           "errors": errors})
    by_sample = index_records_by_sample(evaluation)
    original_rows = [by_sample[image_record["sample"]] for image_record in original_records]
    return {"training_records": training, "evaluation_records": selected,
            "match_order": order, "gaze_lookup": gaze_lookup,
            "evaluation": evaluation, "original_61": original_rows}


def summarize_evaluation_group(rows, methods):
    return {"image_count": len(rows), "methods": calculate_error_metrics(rows, methods),
            "bootstrap": compare_methods_with_paired_bootstrap(rows, methods),
            "no_player_count": sum(image_result["no_players"] for image_result in rows)}


def summarize_head_marker_comparison(original_rows):
    # Add Gaze-LLE with detected heads, scored on the original images where heads were found
    with_heads = [row for row in original_rows if row["points"]["gaze_lle_with_heads"] is not None]
    summary = {"all_61": summarize_evaluation_group(original_rows, METHODS)}
    if with_heads:
        summary["images_with_heads"] = summarize_evaluation_group(with_heads, (*METHODS, "gaze_lle_with_heads"))
    summary["images_without_heads"] = len(original_rows) - len(with_heads)
    return summary


def print_method_scores(name, methods):
    print(name)
    for method, values in methods.items():
        mean = values["mean_error"] if "mean_error" in values else values["match_balanced_mean_error"]
        close = values["close_rate"] if "close_rate" in values else values["match_balanced_close_rate"]
        print(f"  {LABELS[method]:<28} mean error {mean:.4f}, close {100 * close:.1f}%")


def calculate_main_results(records, player_predictions, gaze_predictions, run_dir):
    run_dir = Path(run_dir)
    data = collect_evaluation_predictions(records, player_predictions, gaze_predictions)
    original = {"all_61": summarize_evaluation_group(data["original_61"], METHODS)}
    all_813, matches = calculate_match_balanced_results(data["evaluation"], data["match_order"])
    results = run_dir / "results"
    for folder, summary, rows in (("original_61_images", original, data["original_61"]),
                                  ("all_813_images", all_813, data["evaluation"])):
        save_json(results / folder / "results_summary.json", summary)
        save_json(results / folder / "per_image_points.json", rows)
    save_json(results / "all_813_images/per_match_metrics.json", matches)

    data_flow = {
        "training_images": len(data["training_records"]),
        "training_matches": len({image_record["game"] for image_record in data["training_records"]}),
        "original_images": len(data["original_61"]), "evaluation_matches": len(data["match_order"]),
        "additional_images": len(data["evaluation_records"]) - len(data["original_61"]),
        "expanded_images": len(data["evaluation_records"])}
    save_json(results / "data_flow.json", data_flow)

    from graphs import draw_result_graphs
    draw_result_graphs(data["original_61"], data["evaluation"], run_dir)

    print_method_scores(f"{len(data['original_61'])} original images (one per match)", original["all_61"]["methods"])
    print_method_scores(f"{len(data['evaluation'])} evaluation images (matches weighted equally)", all_813["methods"])
