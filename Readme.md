# SORT Tracker with Kalman Filter

An implementation of the SORT tracking algorithm using a custom Kalman Filter for prediction of objects' positions. Detections are provided by YOLOv11 and tracked objects are updated by associating predictions with new observations.

- The multi-object tracking system monitors multiple objects (e.g., vehicles) within video sequences.
- A Kalman Filter is employed to estimate an object's position, even when the object is not detected in a given frame.
- The Hungarian Algorithm constructs a cost matrix (often based on IoU) to optimally match new detections with predictions. If no satisfactory match is found, a new tracker is initiated.
- Bounding boxes are annotated with identification numbers for verification of the tracking process.

The Kalman Filter code can be found on: https://github.com/ManuelZ/Kalman-Filter

Some rules to show the tracking boxes differ from the original paper code.


https://github.com/user-attachments/assets/e3cb5306-74a7-4d91-9893-121f2385e023


## Installation

The project is an installable package. Environments and dependencies are managed with [uv](https://docs.astral.sh/uv/):

```
uv venv
uv pip install -e .
```

Optional dependency groups:
- `uv pip install -e ".[deep]"` — PyTorch + torchvision + rerun-sdk (DeepSORT re-ID)
- `uv pip install -e ".[eval]"` — trackeval (benchmark scoring)
- `uv pip install -e ".[deep,eval]"` — both

## Scripts

The [scripts/](scripts/) directory contains entry points for running, benchmarking, and tuning the tracker:

- [scripts/mot16_sort.py](scripts/mot16_sort.py) — Runs SORT or DeepSORT on a single MOT16 sequence and writes tracking results in MOT CSV format. Detections come either from the dataset's `det.txt` (`--det`) or from a YOLO model (`--yolo`); the tracker is selected with `--tracker sort|deepsort`.
- [scripts/mot16_benchmark.py](scripts/mot16_benchmark.py) — Runs the tracker over every sequence under `--mot16-root`, arranges the outputs in the layout expected by [TrackEval](https://github.com/JonathonLuiten/TrackEval), and invokes it to report the **HOTA** metric.
- [scripts/measure_R.py](scripts/measure_R.py) — Empirically estimates the observation-noise covariance matrix **R** for the Kalman filter by matching detections to ground truth and measuring residuals in the SORT measurement space `[u, v, s, r]`. Prints an `np.diag([...])` line to paste into `measurement_noise_covariance` in [tracking/sort.py](tracking/sort.py).

Usage examples for each script are shown in the sections below.

## The MOT16 dataset

[MOT16](https://motchallenge.net/data/MOT16/) is a multi-object tracking benchmark focused on pedestrian tracking in crowded scenes, released as part of the MOTChallenge. It contains 14 video sequences (7 for training with public ground truth, 7 for testing) captured from static and moving cameras, at various resolutions and frame rates, under different lighting and viewpoint conditions.

Layout on disk:
```
MOT16/
├── train/                 # ground truth is provided
│   ├── MOT16-02/
│   │   ├── img1/          # frames as 000001.jpg, 000002.jpg, ...
│   │   ├── det/det.txt    # public detections (DPM)
│   │   ├── gt/gt.txt      # ground-truth trajectories
│   │   └── seqinfo.ini    # name, imDir, frameRate, seqLength, imWidth, imHeight, imExt
│   ├── MOT16-04/...
│   ├── MOT16-05/...
│   ├── MOT16-09/...
│   ├── MOT16-10/...
│   ├── MOT16-11/...
│   └── MOT16-13/...
└── test/                  # no gt/ folder; submit results to MOTChallenge
    ├── MOT16-01/
    ├── MOT16-03/...
    ├── MOT16-06/...
    ├── MOT16-07/...
    ├── MOT16-08/...
    ├── MOT16-12/...
    └── MOT16-14/...
```

Both `det.txt` and `gt.txt` follow the MOT CSV convention: `frame, id, x, y, w, h, conf/valid, class, visibility` (in `det.txt`, `id` is always `-1` and only `conf` is meaningful; in `gt.txt`, `class` identifies the object category and `visibility ∈ [0, 1]`).

Sequences vary in resolution (e.g. 1920×1080 for sequences 01-05,07-14, 640×480 for 05,06) and frame rate (14, 25, or 30 fps).

## Evaluation on MOT16

### Benchmarking all sequences (TrackEval / HOTA)

`scripts/mot16_benchmark.py` runs the tracker on every sequence under `--mot16-root`, writes MOT-format results in TrackEval's expected layout, and invokes [TrackEval](https://github.com/JonathonLuiten/TrackEval) to report the **HOTA** metric.

Throughout this section, **"dataset detections"** means the public DPM detections shipped with MOT16 (`det/det.txt` inside each sequence), while **"YOLO detections"** means detections produced on-the-fly by a YOLO model passed via `--yolo`.

**SORT with YOLO detections:**
```
python scripts/mot16_benchmark.py ^
  --mot16-root /path/to/MOT16/train ^
  --tracker sort ^
  --yolo yolo11x.pt
```

**DeepSORT with YOLO detections:**
```
python scripts/mot16_benchmark.py ^
  --mot16-root /path/to/MOT16/train ^
  --tracker deepsort ^
  --yolo yolo11x.pt
```

**Ultralytics ByteTrack (YOLO detections + tracking):**
```
python scripts/mot16_benchmark.py ^
  --mot16-root /path/to/MOT16/train ^
  --tracker ul_bytetrack ^
  --yolo yolo11x.pt
```

**Ultralytics BoT-SORT (YOLO detections + tracking):**
```
python scripts/mot16_benchmark.py ^
  --mot16-root /path/to/MOT16/train ^
  --tracker ul_botsort ^
  --yolo yolo11x.pt
```

**SORT with dataset detections:**
```
python scripts/mot16_benchmark.py ^
  --mot16-root /path/to/MOT16/train ^
  --tracker sort
```

**DeepSORT with dataset detections:**
```
python scripts/mot16_benchmark.py ^
  --mot16-root /path/to/MOT16/train ^
  --tracker deepsort
```

The `ul_bytetrack` and `ul_botsort` options delegate detection and tracking to Ultralytics' built-in `model.track()` pipeline (`bytetrack.yaml` / `botsort.yaml`) and require `--yolo`. They're included for side-by-side comparison against this repo's SORT/DeepSORT.

Per-sequence results and TrackEval's CSV summaries land under `trackeval_results/` by default (override with `--results-dir`).

### Running a single sequence

`scripts/mot16_sort.py` runs SORT or DeepSORT on a single MOT16 sequence and writes results in MOT CSV format.
Detections can come from the dataset's own detection files (`--det`) or from a YOLO model (`--yolo`).
The tracker is selected with `--tracker sort` (default) or `--tracker deepsort`.

**SORT with YOLO detections:**
```
python scripts/mot16_sort.py ^
  --frames /path/to/MOT16/train/MOT16-13/img1 ^
  --yolo yolo11x.pt ^
  --output results/seq_13_sort_yolo.txt
```

**DeepSORT with YOLO detections:**
```
python scripts/mot16_sort.py ^
  --frames /path/to/MOT16/train/MOT16-13/img1 ^
  --yolo yolo11x.pt ^
  --tracker deepsort ^
  --output results/seq_13_deepsort_yolo.txt
```

**SORT with MOT16 detections:**
```
python scripts/mot16_sort.py ^
  --frames /path/to/MOT16/train/MOT16-13/img1 ^
  --det /path/to/MOT16/train/MOT16-13/det/det.txt ^
  --output results/seq_13_sort.txt
```

**DeepSORT with MOT16 detections:**
```
python scripts/mot16_sort.py ^
  --frames /path/to/MOT16/train/MOT16-13/img1 ^
  --det /path/to/MOT16/train/MOT16-13/det/det.txt ^
  --tracker deepsort ^
  --output results/seq_13_deepsort.txt
```

## References

This implementation reflects my own learning journey, drawing on insights from the structure, code, and techniques found in:
  - https://github.com/abewley/sort/blob/master/sort.py
  - https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition/blob/master/chapter10/sort.py
  - https://github.com/Jeremy26/tracking_course/tree/master
