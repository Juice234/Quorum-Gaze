# Add a second ball-sized patch and measure changes in Gaze-LLE's predictions
# Compare no-head predictions before and after adding another filled-in patch

from pathlib import Path
import numpy as np

from file_helpers import read_json, save_json
from calculate_results import convert_point_to_list, make_bootstrap_sample_indices, calculate_bootstrap_interval
from prepare_images import inpaint_rectangle
from study_settings import BOOTSTRAP, CLOSE_ERROR_THRESHOLD, EXTRA_PATCH_SEED, EXTRA_PATCH_SETTINGS


def calculate_patch_overlap_area(first, second):
    return max(0., min(first[2], second[2]) - max(first[0], second[0])) * max(0., min(first[3], second[3]) - max(first[1], second[1]))


def choose_extra_patch_bounds(random_generator, ball_patch, width, height, ball_point):
    # Pick a random ball-sized rectangle that does not touch the ball patch
    patch_width, patch_height = ball_patch[2] - ball_patch[0], ball_patch[3] - ball_patch[1]
    max_attempts = EXTRA_PATCH_SETTINGS["max_attempts"]
    for _ in range(max_attempts):
        left = int(random_generator.randint(0, width - patch_width + 1))
        top = int(random_generator.randint(0, height - patch_height + 1))
        bounds = [left, top, left + patch_width, top + patch_height]
        centre = np.array([(left + patch_width / 2) / width, (top + patch_height / 2) / height])
        if (calculate_patch_overlap_area(bounds, ball_patch) == 0
                and np.linalg.norm(centre - np.asarray(ball_point)) >= EXTRA_PATCH_SETTINGS["minimum_ball_distance"]):
            return bounds
    return None


def prepare_extra_patches(records, run_dir):
    import cv2
    rows = [row for row in records if row["original61"]]
    random_generator = np.random.RandomState(EXTRA_PATCH_SEED)
    results = []

    for row in rows:
        width, height = row["image_width"], row["image_height"]
        bounds = choose_extra_patch_bounds(random_generator, row["mask_bounds"], width, height, row["target_xy"])
        output = Path(run_dir) / "data/extra_patch_images" / (row["sample"] + ".png")

        before = cv2.imread(row["prepared_path"], cv2.IMREAD_COLOR)
        after, mask = inpaint_rectangle(before, bounds)

        # Record pixel changes inside and outside the patch mask
        delta = np.abs(after.astype(np.int16) - before.astype(np.int16))
        changed = np.any(delta > 0, axis=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(cv2.imencode(".png", after)[1].tobytes())
        left, top, right, bottom = bounds
        result = {"sample": row["sample"], "game": row["game"],
                  "prepared_path": str(output.resolve()),
                  "image_width": width, "image_height": height,
                  "extra_patch_bounds": bounds,
                  "extra_patch_centre": [(left + right) / (2 * width), (top + bottom) / (2 * height)],
                  "changed_pixels": int(changed.sum()), "mask_pixels": int((mask > 0).sum()),
                  "maximum_absolute_channel_change": int(delta.max()),
                  "mean_absolute_channel_change_inside_patch": float(delta[mask > 0].mean()),
                  "outside_mask_unchanged": bool(not changed[mask == 0].any())}
        results.append(result)
    destination = Path(run_dir) / "data/extra_patches.json"
    save_json(destination, results)
    return destination


def read_gaze_predictions_for_patch_check(records, run_dir):
    # Read the original, repeated and extra-patch predictions without any head-box results
    run_dir = Path(run_dir)
    gaze = run_dir / "gaze_lle"
    no_head_points = {row["sample"]: row["point"] for row in read_json(gaze / "no_head_points.json")}
    repeat_points = {row["sample"]: row["point"] for row in read_json(gaze / "no_head_repeat_points.json")}
    extra_patch_points = {row["sample"]: row["point"] for row in read_json(gaze / "extra_patch_points.json")}
    extra_patch_records = {row["sample"]: row for row in read_json(run_dir / "data/extra_patches.json")}
    results = []

    for row in records:
        if not row["original61"]:
            continue
        sample = row["sample"]
        edit = extra_patch_records[sample]
        players = read_json(run_dir / "detections" / (sample + ".json"))["filtered_players"]

        # Compare the repeated maps to measure differences
        first = np.load(gaze / "no_head" / sample / "maps.npz")["native"]
        repeated = np.load(gaze / "no_head_repeat" / sample / "maps.npz")["native"]
        difference = np.abs(first.astype(float) - repeated.astype(float))
        results.append({"sample": sample, "game": row["game"], "no_head_point": no_head_points[sample],
                        "extra_patch_point": extra_patch_points[sample],
                        "extra_patch_centre": edit["extra_patch_centre"],
                        "repeat_point": repeat_points[sample],
                        "repeat_identical": bool(np.array_equal(first, repeated)),
                        "repeat_maximum_difference": float(difference.max()),
                        "patch_changes": edit,
                        "detected_player_overlap_count": sum(
                            calculate_patch_overlap_area(edit["extra_patch_bounds"], player["bbox"]) > 0
                            for player in players)})
    return results


def measure_extra_patch_changes(original, gaze_lookup):
    # Measure how far each prediction moved, and whether it moved towards the extra patch
    rows = []
    for image in original:
        gaze = gaze_lookup[image["sample"]]
        before = convert_point_to_list(gaze["repeat_point"])
        after = convert_point_to_list(gaze["extra_patch_point"])
        centre = convert_point_to_list(gaze["extra_patch_centre"])
        target = image["target_xy"]
        shift = np.asarray(after) - before
        old_distance = float(np.linalg.norm(np.asarray(before) - centre))
        new_distance = float(np.linalg.norm(np.asarray(after) - centre))
        old_error = float(np.linalg.norm(np.asarray(before) - target))
        new_error = float(np.linalg.norm(np.asarray(after) - target))
        # Shift towards the patch centre, positive values mean movement towards it
        towards = None if old_distance == 0 else float(np.dot(shift, (np.asarray(centre) - before) / old_distance))
        rows.append({"sample": image["sample"], "game": image["game"], "repeat_point": before,
                     "extra_patch_point": after, "target_xy": target, "extra_patch_centre": centre,
                     "target_error_before": old_error, "target_error_after": new_error,
                     "target_error_change": new_error - old_error,
                     "patch_distance_before": old_distance, "patch_distance_after": new_distance,
                     "patch_distance_change": new_distance - old_distance,
                     "movement_towards_patch": towards, "point_shift": shift.tolist(),
                     "shift_length": float(np.linalg.norm(shift)),
                     "target_close_before": old_error <= CLOSE_ERROR_THRESHOLD,
                     "target_close_after": new_error <= CLOSE_ERROR_THRESHOLD,
                     "patch_near_before": old_distance <= CLOSE_ERROR_THRESHOLD,
                     "patch_near_after": new_distance <= CLOSE_ERROR_THRESHOLD,
                     "patch_changes": gaze["patch_changes"],
                     "detected_player_overlap_count": gaze["detected_player_overlap_count"]})
    return rows


def calculate_extra_patch_intervals(rows):
    bootstrap_indices = make_bootstrap_sample_indices(len(rows))
    intervals = {}

    # Omit undefined movements from each resampled set
    for name in ("target_error_change", "patch_distance_change", "movement_towards_patch", "shift_length",
                 "target_close_change_pp", "patch_near_change_pp"):
        changes = []
        for row in rows:
            if name == "target_close_change_pp":
                change = 100 * (int(row["target_close_after"]) - int(row["target_close_before"]))
            elif name == "patch_near_change_pp":
                change = 100 * (int(row["patch_near_after"]) - int(row["patch_near_before"]))
            else:
                change = np.nan if row[name] is None else row[name]
            changes.append(change)

        values = np.asarray(changes, dtype=float)
        valid_values = np.isfinite(values)
        sampled_changes = values[bootstrap_indices]
        valid_counts = np.isfinite(sampled_changes).sum(axis=1)
        bootstrap_means = np.nansum(sampled_changes, axis=1)[valid_counts > 0] / valid_counts[valid_counts > 0]
        if valid_values.any():
            interval = calculate_bootstrap_interval(bootstrap_means, values[valid_values].mean())
        else:
            interval = {"estimate": None, "lower": None, "upper": None}
        intervals[name] = {**interval, "denominator": int(valid_values.sum()),
                           "valid_bootstrap_draws": len(bootstrap_means)}
    return intervals


def summarize_extra_patch_changes(original, gaze_lookup, rows, intervals):
    repeats = [gaze_lookup[image_result["sample"]] for image_result in original]
    identical = [image_result["repeat_identical"] for image_result in repeats]
    differences = [image_result["repeat_maximum_difference"] for image_result in repeats]
    means = {}
    for name in ("target_error_before", "target_error_after", "patch_distance_before", "patch_distance_after", "shift_length"):
        means[name] = float(np.mean([row[name] for row in rows]))
    counts = {}
    for name in ("target_close_before", "target_close_after", "patch_near_before", "patch_near_after"):
        counts[name] = sum(bool(row[name]) for row in rows)

    return {
        "requested_images": len(original), "constructed_images": len(rows),
        "repeat_agreement": {"identical_count": sum(identical),
                             "maximum_difference": max(differences)},
        "means": means, "counts": counts,
        "undefined_movement_count": sum(image_result["movement_towards_patch"] is None for image_result in rows),
        "towards_patch_count": sum(image_result["movement_towards_patch"] is not None and image_result["movement_towards_patch"] > 0 for image_result in rows),
        "away_from_patch_count": sum(image_result["movement_towards_patch"] is not None and image_result["movement_towards_patch"] < 0 for image_result in rows),
        "unchanged_point_count": sum(image_result["shift_length"] == 0 for image_result in rows),
        "zero_pixel_change_count": sum(image_result["patch_changes"]["changed_pixels"] == 0 for image_result in rows),
        "pixel_change_record_count": len(rows),
        "detected_player_overlap_count": sum(bool(image_result["detected_player_overlap_count"]) for image_result in rows),
        "paired_intervals": intervals, "bootstrap": {**BOOTSTRAP, "unit": "whole match, one image per match"}}


def compare_predictions_after_extra_patch(original, gaze_lookup):
    rows = measure_extra_patch_changes(original, gaze_lookup)
    intervals = calculate_extra_patch_intervals(rows)
    return summarize_extra_patch_changes(original, gaze_lookup, rows, intervals), rows
