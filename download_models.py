# Download the pretrained models and source files

import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

WEIGHTS = {
    "pose": {
        "name": "yolov8m-pose.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-pose.pt",
    },
    "gaze_weights": {
        "name": "gazelle_dinov2_vitb14_inout.pt",
        "url": "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt",
    },
    "dino_weights": {
        "name": "dinov2_vitb14_pretrain.pth",
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    },
}
SOURCES = {
    "gaze_source": {
        "repository": "fkryan/gazelle",
        "commit": "2d78f9f3bd2a5db360354954ec1a4c526ee8ab55",
    },
    "dino_source": {
        "repository": "facebookresearch/dinov2",
        "commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
    },
}


def read_model_paths(config_file):
    config_file = Path(config_file).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8-sig"))
    paths = {}
    for name in (*WEIGHTS, *SOURCES):
        paths[name] = str((config_file.parent / Path(config[name]).expanduser()).resolve())
    return paths


def download_file(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "football-reproduction/1"})

    # Give the file its final name only after the download finishes
    partial = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(destination)
    return destination


def extract_source_archive(archive, sources_folder):
    sources_folder = Path(sources_folder)
    with zipfile.ZipFile(archive) as zipped:
        folder_name = zipped.namelist()[0].split("/")[0]
        zipped.extractall(sources_folder)
    return sources_folder / folder_name


def download_models_and_sources(destination, from_local=None):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if from_local is not None:
        paths = read_model_paths(from_local)
    else:
        paths = {}
        for name, item in WEIGHTS.items():
            paths[name] = str(download_file(item["url"], destination / "weights" / item["name"]))

        for name, item in SOURCES.items():
            repository = item["repository"]
            commit = item["commit"]
            archive = destination / "archives" / f"{repository.replace('/', '-')}-{commit}.zip"
            url = f"https://codeload.github.com/{repository}/zip/{commit}"
            download_file(url, archive)
            paths[name] = str(extract_source_archive(archive, destination / "sources"))

    output = destination / "assets.json"
    output.write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--from-local", type=Path)
    arguments = parser.parse_args()
    download_models_and_sources(arguments.destination, arguments.from_local)
    print(arguments.destination.expanduser().resolve() / "assets.json")


if __name__ == "__main__":
    main()
