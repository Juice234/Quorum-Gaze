# Draw error curves and prediction examples from calculated points and player boxes

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from calculate_results import LABELS
from study_settings import CLOSE_ERROR_THRESHOLD, METHODS


COLORS = ["#005b5e", "#139a77", "#7bb9cb", "#a276b1", "#777777", "#bb8158", "#244a91"]
STYLES = ["-", "-", "--", ":", "--", "-.", "-"]


def save_figure(figure, directory, name):
    destination = directory / (name + ".png")
    figure.savefig(destination, dpi=240, facecolor="white")
    plt.close(figure)


def draw_error_curve(rows, directory, name, match_weighted=False):
    games = Counter(row["game"] for row in rows)
    weights = np.asarray([1 / (len(games) * games[row["game"]]) if match_weighted else 1 / len(rows)
                          for row in rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    largest = 0.0

    # Plot the share of predictions below each error, giving matches equal weight
    for method, color, style in zip(METHODS, COLORS, STYLES):
        values = np.asarray([row["errors"][method] for row in rows])
        order = np.argsort(values, kind="stable")
        x = np.r_[0, values[order]]
        y = np.r_[0, np.cumsum(weights[order])]
        axis.step(x, y, where="post", label=LABELS[method], color=color,
                  linestyle=style, linewidth=1.7)
        largest = max(largest, float(values.max()))
    axis.axvline(CLOSE_ERROR_THRESHOLD, color="#999999", linestyle=":", linewidth=1)
    axis.set_xlabel("Point error in normalised image coordinates", fontsize=11)
    axis.set_ylabel("Match-weighted proportion at or below this error" if match_weighted
                    else "Proportion of pictures at or below this error", fontsize=11)
    axis.set_xlim(0, max(.11, largest * 1.04))
    axis.set_ylim(0, 1.015)
    axis.grid(True, color="#e7e7e7", linewidth=.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", fontsize=9, frameon=True,
                facecolor="white", edgecolor="#dddddd")
    figure.subplots_adjust(left=.12, right=.98, bottom=.16, top=.96)
    save_figure(figure, directory, name)


def draw_prediction_example(axis, row, title):
    axis.set_xlim(0, 1)
    axis.set_ylim(1, 0)
    axis.set_aspect("equal")
    axis.set_xlabel("Horizontal image coordinate", fontsize=11)
    axis.set_ylabel("Vertical image coordinate", fontsize=11)
    axis.set_title(title, fontsize=12)
    axis.grid(True, color="#eeeeee", linewidth=.5)
    axis.set_axisbelow(True)
    if row is None:
        axis.text(.5, .5, "No qualifying example", ha="center", va="center",
                  transform=axis.transAxes)
        return
    width, height = row["image_width"], row["image_height"]
    selected = row["selected_players"]["full"]

    for player in row["players"]:
        left, top, right, bottom = player["bbox"]
        chosen = player["player_index"] == selected
        axis.add_patch(Rectangle((left / width, top / height), (right - left) / width,
                                 (bottom - top) / height, fill=False,
                                 edgecolor="#222222" if chosen else "#bbbbbb",
                                 linewidth=1.5 if chosen else .8))

    for label, coordinate, marker, color, size in (
        ("Ball label", row["target_xy"], "*", "#111111", 130),
        ("Raw prediction", row["points"]["raw"], "o", "#139a77", 105),
        ("Blended prediction", row["points"]["blended"], "D", "#005b5e", 42),
        ("Gaze-LLE", row["points"]["gaze_lle"], "^", "#244a91", 65),
    ):
        axis.scatter(*coordinate, marker=marker, s=size, label=label, edgecolors=color,
                     facecolors="none" if marker == "o" else color, zorder=5)


def draw_prediction_examples(rows, directory):
    ordered = sorted(rows, key=lambda row: row["sample"])
    # Select the first sample
    close_example = next((row for row in ordered
                          if row["errors"]["raw"] <= CLOSE_ERROR_THRESHOLD
                          and row["errors"]["blended"] > CLOSE_ERROR_THRESHOLD), None)
    large_error_example = next((row for row in ordered if row["errors"]["raw"] >= .40), None)
    selected = [close_example, large_error_example]

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    for axis, row, title in zip(axes, selected, ("(a) A close raw prediction", "(b) A large raw error")):
        draw_prediction_example(axis, row, title)
    handles, labels = [], []
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False)
    figure.subplots_adjust(left=.085, right=.99, top=.90, bottom=.20, wspace=.28)
    save_figure(figure, directory, "prediction_examples")


def draw_result_graphs(original, expanded, run_dir):
    directory = Path(run_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    draw_error_curve(original, directory, "errors_61_images")
    draw_error_curve(expanded, directory, "errors_813_images", match_weighted=True)
    draw_prediction_examples(original, directory)
