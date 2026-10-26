# Run the player, head-box, extra-patch comparison separately

import argparse
from pathlib import Path

from calculate_results import collect_evaluation_predictions, summarize_head_marker_comparison
from file_helpers import read_json, save_json
from gaze_comparison import read_main_gaze_predictions, predict_with_gaze_lle
from head_detection.find_head_boxes import prepare_head_boxes_for_comparison
from main import prepare_run_folder
from player_helpers.predict_from_players import load_player_models
from prepare_images import copy_records_without_ball_labels
from supporting_checks.extra_patch import (
    prepare_extra_patches, read_gaze_predictions_for_patch_check, compare_predictions_after_extra_patch,
)
from supporting_checks.player_checks import (
    evaluate_answer_assisted_player_selection, compare_predictions_with_player_centres,
)


CHECKS = ("players", "heads", "extra_patch")


def read_main_comparison_data(records, run_dir, head_points=None):
    player_points = read_json(run_dir / "predictions/player_points.json")
    gaze_points = read_main_gaze_predictions(records, run_dir)
    if head_points is not None:
        for row in gaze_points:
            if row["sample"] in head_points:
                row["with_heads_point"] = head_points[row["sample"]]
    return collect_evaluation_predictions(records, player_points, gaze_points)


def run_player_checks(records, run_dir):
    from threadpoolctl import threadpool_limits

    data = read_main_comparison_data(records, run_dir)
    models = load_player_models(run_dir / "fit/models.joblib")

    # Use the known ball position to check player selection, then compare offsets with centres
    with threadpool_limits(limits=1):
        selection, selection_rows = evaluate_answer_assisted_player_selection(data["original_61"], models)
    centres, centre_rows, match_rows = compare_predictions_with_player_centres(
        data["evaluation"], selection_rows, data["match_order"])

    for name, summary, rows in (("player_selection_check", selection, selection_rows),
                                ("player_centre_comparison", centres, centre_rows)):
        folder = run_dir / "results" / name
        save_json(folder / "results_summary.json", summary)
        save_json(folder / "per_image.json", rows)
    save_json(run_dir / "results/player_centre_comparison/per_match_metrics.json", match_rows)


def run_head_comparison(records, run_dir, assets_file, score_only):
    if not score_only:
        from download_models import read_model_paths

        images = [row for row in copy_records_without_ball_labels(records) if row["original61"]]
        prepare_head_boxes_for_comparison(images, run_dir)
        heads = {row["sample"]: read_json(run_dir / "heads" / (row["sample"] + ".json")) for row in images}
        predict_with_gaze_lle(images, read_model_paths(assets_file), run_dir, "with_heads", heads)

    # Compare saved head-marked predictions with main-method predictions
    head_points = {row["sample"]: row["point"] for row in read_json(run_dir / "gaze_lle/with_heads_points.json")}
    data = read_main_comparison_data(records, run_dir, head_points)
    summary = summarize_head_marker_comparison(data["original_61"])
    save_json(run_dir / "results/head_marker_comparison/results_summary.json", summary)


def run_extra_patch_check(records, run_dir, assets_file, score_only):
    if not score_only:
        from download_models import read_model_paths

        prepare_extra_patches(records, run_dir)
        assets = read_model_paths(assets_file)
        images = [row for row in copy_records_without_ball_labels(records) if row["original61"]]
        predict_with_gaze_lle(images, assets, run_dir, "no_head_repeat")
        patch_records = read_json(run_dir / "data/extra_patches.json")
        patch_images = [{"sample": row["sample"], "prepared_path": row["prepared_path"]} for row in patch_records]
        predict_with_gaze_lle(patch_images, assets, run_dir, "extra_patch")

    # Measure movement towards the extra patch
    original = [row for row in records if row["original61"]]
    gaze_rows = read_gaze_predictions_for_patch_check(records, run_dir)
    gaze_lookup = {row["sample"]: row for row in gaze_rows}
    summary, rows = compare_predictions_after_extra_patch(original, gaze_lookup)
    folder = run_dir / "results/extra_patch_check"
    save_json(folder / "results_summary.json", summary)
    save_json(folder / "per_image.json", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=(*CHECKS, "all"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--score-only", action="store_true")
    options = parser.parse_args()
    run_dir = prepare_run_folder(options.run_dir)
    records = read_json(run_dir / "data/records.json")

    for check in CHECKS if options.check == "all" else (options.check,):
        print("Starting", check)
        if check == "players":
            run_player_checks(records, run_dir)
        if check == "heads":
            run_head_comparison(records, run_dir, options.assets, options.score_only)
        if check == "extra_patch":
            run_extra_patch_check(records, run_dir, options.assets, options.score_only)
        print("Finished", check)


if __name__ == "__main__":
    main()
