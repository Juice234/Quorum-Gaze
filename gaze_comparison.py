# Run Gaze-LLE on ball-removed images and read the peak of each heatmap
# no_head: main comparison without head boxes
# no_head_repeat: repeat the original 61 inputs to measure changes
# with_heads: with detected head boxes
# extra_patch: use copies containing a second filled-in patch

import os
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

from file_helpers import read_json, save_json
import prepare_heatmaps
from study_settings import GAZE_SEED, GAZE_THREADS

def find_heatmap_peak_point(values):
    # Return the centre of the highest grid cell as (x, y) fractions of the image
    values = np.asarray(values, dtype=float)
    height, width = values.shape
    row, column = divmod(int(np.argmax(values)), width)
    return [(column + .5) / width, (row + .5) / height]


def load_gaze_lle_model(assets):
    # Build Gaze-LLE and weights
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(GAZE_THREADS)
    torch.manual_seed(GAZE_SEED)
    torch.use_deterministic_algorithms(False)
    dino = torch.hub.load(str(assets["dino_source"]), "dinov2_vitb14", pretrained=False, source="local")
    dino.load_state_dict(torch.load(assets["dino_weights"], map_location="cpu", weights_only=True))
    source = Path(assets["gaze_source"]).resolve()
    sys.path.insert(0, str(source))
    from gazelle.model import get_gazelle_model

    def use_downloaded_backbone(repository, name, *args, **kwargs):
        return dino

    with patch.object(torch.hub, "load", side_effect=use_downloaded_backbone):
        model, transform = get_gazelle_model("gazelle_dinov2_vitb14_inout")

    # Load the gaze weights
    model.load_gazelle_state_dict(torch.load(assets["gaze_weights"], map_location="cpu", weights_only=True))
    model.eval()
    return torch, model, transform


def predict_gaze_lle_heatmaps(torch, model, transform, image_path, boxes):
    import cv2
    from PIL import Image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with torch.inference_mode():
        output = model({"images": transform(rgb).unsqueeze(0), "bboxes": [boxes]})
    maps = output["heatmap"][0].detach().cpu().numpy().astype(np.float32)
    return maps


def predict_with_gaze_lle(records, assets, run_dir, mode="no_head", heads=None):
    mode_folder = Path(run_dir) / "gaze_lle" / mode
    results = []
    torch = model = transform = None

    for number, row in enumerate(records, 1):
        boxes = heads[row["sample"]]["normalized_head_boxes"] if mode == "with_heads" else [None]
        folder = mode_folder / row["sample"]
        destination = folder / "prediction.json"
        if destination.exists():
            results.append(read_json(destination))
            continue

        # Without a detected head, leave the head-marker prediction empty
        if boxes:
            if model is None:
                torch, model, transform = load_gaze_lle_model(assets)
            maps = predict_gaze_lle_heatmaps(torch, model, transform, row["prepared_path"], boxes)
            if mode == "with_heads":
                prepared = prepare_heatmaps.average_head_heatmaps(maps)
            else:
                prepared = prepare_heatmaps.prepare_gaze_heatmap(maps[0])
            point = find_heatmap_peak_point(prepared)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "maps.npz").open("xb") as stream:
                np.savez_compressed(stream, native=maps)
        else:
            point = None
        record = {"sample": row["sample"], "point": point}
        save_json(destination, record)
        results.append(record)
        if number % 25 == 0 or number == len(records):
            print("Gaze-LLE", mode, number, "/", len(records))
    destination = Path(run_dir) / "gaze_lle" / (mode + "_points.json")
    save_json(destination, results)
    return destination


def read_main_gaze_predictions(records, run_dir):
    points = {row["sample"]: row["point"]
              for row in read_json(Path(run_dir) / "gaze_lle/no_head_points.json")}
    return [{"sample": row["sample"], "game": row["game"],
             "no_head_point": points[row["sample"]], "with_heads_point": None}
            for row in records if row["partition"] == "evaluation"]
