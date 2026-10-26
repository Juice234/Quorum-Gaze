# Library defaults

# Seeds
PLAYER_SHUFFLE_SEED = 20260703
PLAYER_SHUFFLE_PREFIX = "pivotal"
EXTRA_PATCH_SEED = 2026101503

# Rules for scoring
CLOSE_ERROR_THRESHOLD = .10
KEYPOINT_VISIBILITY_THRESHOLD = .25
NEAR_BALL_DISTANCE = .05

BALL_REMOVAL = {
    "padding": 2,
    "radius": 3,
    "method": "INPAINT_TELEA",
}

# Grass-filter rules
GRASS_CHECK_SETTINGS = {
    "side_crop": .20,
    "above_feet": .08,
    "below_feet": .12,
    "green_over_red": 1.08,
    "green_over_blue": 1.08,
    "minimum_green": 45,
}

PLAYER_THREADS = 8

PLAYER_DETECTION_SETTINGS = {
    "whole_size": 1280,
    "confidence": .15,
    "max_det": 100,

    "field_minimum": .15,

    "device": "cpu",
    "threads": PLAYER_THREADS,
    "seed": 2026101502,
}

FITTING_SETTINGS = {
    "threads": PLAYER_THREADS,
    "selector_seed": 20260703,
    "position_x_seed": 20260703,
    "full_x_seed": 20260803,
    "y_seed_increment": 100,

    "near_ball_distance": NEAR_BALL_DISTANCE,
    "blend_weight": 1 / 3,

    "selector": {
        "class_weight": "balanced",
        "max_iter": 2000,
    },
    "offset": {
        "min_samples_leaf": 20,
        "learning_rate": .05,
        "max_iter": 250,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    },
}

GAZE_THREADS = 8
GAZE_SEED = 20261015

# Changed from Gaze-LLE  map size
GAZE_GRID_SIZE = 128


EXTRA_PATCH_SETTINGS = {
    "minimum_ball_distance": .20,
    "max_attempts": 10000,
}

PLAYER_METHODS = (
    "blended",
    "raw",
    "position_only_blended",
    "shuffled_body_blended",
    "training_average",
)
METHODS = (*PLAYER_METHODS, "image_centre", "gaze_lle")

# NumPY defaults
BOOTSTRAP = {
    "method": "linear",

    # Project resampling choices
    "draws": 5000,
    "seed": 20260721,
    "generator": "RandomState",

    "percentiles": [2.5, 97.5],
    "interval": "exploratory pointwise 95%",
}

HEAD_DETECTION_SETTINGS = {
    "minimum_head_keypoint_confidence": 0.3,
    "minimum_head_keypoints": 2,
    "head_keypoint_size_scale": 2.2,
    "minimum_head_player_height_ratio": 0.06,
    "head_crop_expansion": 1.6,
}
