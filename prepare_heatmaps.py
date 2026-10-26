# Resize and normalise Gaze-LLE heatmaps for point scoring

import cv2
import numpy as np

from study_settings import GAZE_GRID_SIZE


def resize_heatmap(heatmap, size=GAZE_GRID_SIZE):
    source_map = np.asarray(heatmap, dtype=np.float64)
    return cv2.resize(source_map, (size, size), interpolation=cv2.INTER_LINEAR)


def prepare_gaze_heatmap(heatmap, size=GAZE_GRID_SIZE):
    source_map = np.asarray(heatmap, dtype=np.float64)
    values = np.maximum(resize_heatmap(source_map, size), 0.0)
    maximum = float(values.max())
    if maximum <= 0 or not np.isfinite(source_map).all() or not np.isfinite(maximum):
        return None

    # Scale before summing so large scores do not overflow
    scaled = values / maximum
    return scaled / float(scaled.sum())


def average_head_heatmaps(maps, size=GAZE_GRID_SIZE):
    # Normalise each head before averaging so they contribute equally
    prepared = [prepare_gaze_heatmap(heatmap, size) for heatmap in maps]
    average = np.mean(np.stack(prepared), axis=0)
    return prepare_gaze_heatmap(average, size)
