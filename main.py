import argparse
from pathlib import Path

from calculate_results import calculate_main_results
from file_helpers import read_json, save_json
from gaze_comparison import read_main_gaze_predictions, predict_with_gaze_lle
from player_model import (
    detect_players_in_study_images, predict_ball_positions,
    fit_player_selection_and_offset_models,
)
from prepare_images import copy_records_without_ball_labels, prepare_ball_removed_images
import study_settings

STAGES = ("prepare", "detect", "fit", "predict", "gaze", "score")

def prepare_run_folder(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "ball_removal": study_settings.BALL_REMOVAL,
        "player_detection": study_settings.PLAYER_DETECTION_SETTINGS,
        "grass_check": study_settings.GRASS_CHECK_SETTINGS,
        "player_fitting": study_settings.FITTING_SETTINGS,
        "keypoint_confidence": study_settings.KEYPOINT_VISIBILITY_THRESHOLD,
        "player_shuffle_seed": study_settings.PLAYER_SHUFFLE_SEED,
        "player_shuffle_prefix": study_settings.PLAYER_SHUFFLE_PREFIX,
        "gaze_seed": study_settings.GAZE_SEED,
        "gaze_threads": study_settings.GAZE_THREADS,
        "gaze_grid_size": study_settings.GAZE_GRID_SIZE,
        "head_detection": study_settings.HEAD_DETECTION_SETTINGS,
        "extra_patch": study_settings.EXTRA_PATCH_SETTINGS,
        "extra_patch_seed": study_settings.EXTRA_PATCH_SEED,
        "close_error_threshold": study_settings.CLOSE_ERROR_THRESHOLD,
        "bootstrap": study_settings.BOOTSTRAP,
    }
    save_json(run_dir / "settings.json", settings)
    return run_dir


def run_main_stage(stage, options):
    run_dir = options.run_dir
    if stage == "prepare":
        return prepare_ball_removed_images(options.annotations, options.images, run_dir)

    records = read_json(run_dir / "data/records.json")
    images = copy_records_without_ball_labels(records)
    evaluation_images = [row for row in images if row["partition"] == "evaluation"]
    if stage == "fit":
        # Fit using training-match labels only
        training_images = [row for row in records if row["partition"] == "training"]
        return fit_player_selection_and_offset_models(training_images, run_dir)
    if stage == "predict":
        return predict_ball_positions(evaluation_images, run_dir / "fit/models.joblib", run_dir)
    if stage == "score":
        return calculate_main_results(
            records, read_json(run_dir / "predictions/player_points.json"),
            read_main_gaze_predictions(records, run_dir), run_dir,
        )

    from download_models import read_model_paths
    assets = read_model_paths(options.assets)
    if stage == "detect":
        return detect_players_in_study_images(images, assets["pose"], run_dir)
    if stage == "gaze":
        return predict_with_gaze_lle(evaluation_images, assets, run_dir, "no_head")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--assets", type=Path)
    options = parser.parse_args()
    options.run_dir = prepare_run_folder(options.run_dir)
    stages = STAGES if options.stage == "all" else (options.stage,)

    for stage in stages:
        print("Starting", stage)
        run_main_stage(stage, options)
        print("Finished", stage)


if __name__ == "__main__":
    main()
