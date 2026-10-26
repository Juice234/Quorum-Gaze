# Football Gaze

This project estimates where a ball was removed from a football image. YOLO finds the players and their body keypoints. A player is then selected, and a model predicts the ball's position relative to that player. The ball labels are used to remove the ball, train the local models and check the predictions afterwards.

The results are compared with [Gaze-LLE](https://github.com/fkryan/gazelle), the image centre and the average ball position in the training images. Gaze-LLE gets the same edited image without a head box. The images and labels come from [SoccerNet-v3D](https://github.com/mguti97/SoccerNet-v3D).

## Setup

Use Python 3.11. Open PowerShell and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
The frames need [SoccerNet access](https://www.soccer-net.org/data) through Hugging Face. 

Leave the `Frames-v3.zip` files zipped in their match folders. The program reads the images from those archives.

## Running it

```powershell
.\.venv\Scripts\python.exe main.py all --run-dir runs/football --annotations data/annotations/SNv3D.csv --images data/frames --assets external/assets.json
```

This prepares the images, detects the players, trains the player-selection and position models, makes predictions, runs Gaze-LLE and calculates the results. Training and evaluation are split by match.

Its run in this order:

```text
prepare -> detect -> fit -> predict -> gaze -> score
```

Results and figures are saved under `runs/football/results` and `runs/football/figures`. Edited images, detections, fitted models and predictions are also kept in the run folder. Existing outputs are reused, so use a new run folder after changing settings or code.

Error is measured after dividing coordinates by the image width and height. Lower is better. An error of 0.10 or less counts as a close prediction. The program saves both raw predictions and predictions moved towards the training average, called blended predictions.

## Extra checks

These can be run separately after the main steps:

```powershell
.\.venv\Scripts\python.exe supporting_analysis.py players --run-dir runs/football
.\.venv\Scripts\python.exe supporting_analysis.py heads --run-dir runs/football --assets external/assets.json
.\.venv\Scripts\python.exe supporting_analysis.py extra_patch --run-dir runs/football --assets external/assets.json
```

- `players` uses the known answer to select a player, then checks how much that changes the result. It also compares the predicted ball position with the player's centre. 
- `heads` checks whether giving Gaze-LLE the detected head boxes helps.
- `extra_patch` adds another edited area away from the ball and checks whether Gaze-LLE moves towards it.



## Files

- `main.py` runs the main steps
- `prepare_images.py` reads the labels, splits matches and removes the balls
- `player_model.py` and `player_helpers/` handle player detection, measurements, fitting and prediction
- `gaze_comparison.py` runs Gaze-LLE, and `prepare_heatmaps.py` resizes its heatmaps and prepares them for scoring
- `calculate_results.py` calculates errors and comparisons, while `graphs.py` draws the figures
- `supporting_analysis.py` and `supporting_checks/` handle the extra checks
- `head_detection/` builds head boxes from the saved player landmarks
- `study_settings.py` contains the settings, and `file_helpers.py` reads and saves JSON files
- `download_data.py` and `download_models.py` download the files needed to run the project

The dataset and model weights are not included in the repository. 
