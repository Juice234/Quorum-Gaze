# Download SoccerNet-v3D annotations and match-frame

import argparse
import json
from pathlib import Path
import urllib.request

ANNOTATIONS_URL = "https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/SNv3D.csv"

def download_annotations(annotation_folder):
    annotation_folder = Path(annotation_folder).resolve()
    annotation_folder.mkdir(parents=True, exist_ok=True)
    annotation_path = annotation_folder / "SNv3D.csv"

    request = urllib.request.Request(ANNOTATIONS_URL, headers={"User-Agent": "football-study-reproduction"})
    with urllib.request.urlopen(request, timeout=120) as response:
        annotation_bytes = response.read()
    with annotation_path.open("wb") as stream:
        stream.write(annotation_bytes)
    return annotation_path


def download_match_frames(annotation_csv, frames_folder, list_only=False):
    from prepare_images import read_study_records
    annotation_records = read_study_records(annotation_csv)
    required_matches = []
    for row in annotation_records:
        if row["game"] not in required_matches:
            required_matches.append(row["game"])
    if list_only:
        print(json.dumps({"matches": len(required_matches), "games": required_matches}, indent=2))
        return
    from SoccerNet.Downloader import SoccerNetDownloader
    from SoccerNet.utils import getListGames
    frames_folder = Path(frames_folder).resolve()
    frames_folder.mkdir(parents=True, exist_ok=True)
    official_matches = {}

    # Match the selected games to SoccerNet's download names and splits
    for split in ("train", "valid", "test"):
        for name in getListGames(split, "frames"):
            official_matches[name.replace("\\", "/")] = (split, name)
    downloader = SoccerNetDownloader(LocalDirectory=str(frames_folder))

    for match_number, game in enumerate(required_matches, 1):
        split, name = official_matches[game]
        print("Downloading frames", match_number, "/", len(required_matches), game)
        downloader.downloadGame(name, files=["Frames-v3.zip"], spl=split, verbose=True,
                                source="HuggingFace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--list-matches", action="store_true")
    arguments = parser.parse_args()
    annotation_path = download_annotations(arguments.annotations)
    print("Annotations:", annotation_path)
    if arguments.frames is not None or arguments.list_matches:
        download_match_frames(annotation_path, arguments.frames, arguments.list_matches)


if __name__ == "__main__":
    main()
