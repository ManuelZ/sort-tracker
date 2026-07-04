"""
Run SORT or DeepSORT on all MOT16 sequences and evaluate with TrackEval
(HOTA, CLEAR, Identity).

Usage with dataset detections:
python mot16_benchmark.py ^
    --mot16-root F:/DATASETS/MOT16/train ^
    --tracker    sort

Usage with YOLO detections:
python mot16_benchmark.py ^
    --mot16-root F:/DATASETS/MOT16/train ^
    --tracker    sort ^
    --yolo       yolo11l.pt
"""

# Standard Library imports
import argparse
import shutil
from datetime import datetime
from pathlib import Path

# External imports
import cv2
import trackeval

# Local imports
from scripts.mot16_sort import (
    ULTRALYTICS_TRACKER_YAMLS,
    iter_mot16,
    iter_yolo,
    make_reid_model,
    track_sequence_ultralytics,
)
from tracking.sort import DEFAULT_CONFIG_PATH as SORT_CONFIG_PATH
from tracking.sort import Sort
from tracking.utils import write_mot_results


def track_sequence(
    det_iter,
    tracker_name: str,
    out_file,
    reid_model=None,
    reid_transform=None,
) -> None:
    """Run tracker on a sequence and stream MOT-format results to out_file."""
    if tracker_name == "sort":
        tracker = Sort(max_cycles_without_update=3)
        for frame_number, boxes, _ in det_iter:
            tracked, _ = tracker.update_tracks(list(boxes))
            write_mot_results(out_file, frame_number, tracked)
    else:
        from tracking.deepsort import DeepSort, extract_descriptors

        tracker = DeepSort(max_cycles_without_update=30, motion_weight=0)
        for frame_number, boxes, frame_path in det_iter:
            frame = cv2.imread(str(frame_path))
            descriptors = (
                extract_descriptors(frame, list(boxes), reid_model, reid_transform)
                if len(boxes)
                else None
            )
            tracked = tracker.update(list(boxes), descriptors)
            write_mot_results(out_file, frame_number, tracked)


def run_trackeval(
    gt_folder: Path,
    trackers_folder: Path,
    output_folder: Path,
    tracker_name: str,
    seqmap_file: Path,
) -> None:
    # Evaluator configuration
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False
    evaluator = trackeval.Evaluator(eval_config)

    # Dataset configuration
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_folder),
            "OUTPUT_FOLDER": str(output_folder),
            "TRACKERS_TO_EVAL": [tracker_name],
            "BENCHMARK": "MOT16",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQMAP_FILE": str(seqmap_file),
            "PRINT_CONFIG": False,
        }
    )
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    # Metrics configuration
    metrics_config = {"METRICS": ["HOTA"], "THRESHOLD": 0.5}
    metrics_list = [trackeval.metrics.HOTA(metrics_config)]

    evaluator.evaluate(dataset_list, metrics_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mot16-root",
        required=True,
        help="MOT16 sequences directory (e.g. MOT16/train)",
    )
    parser.add_argument(
        "--tracker",
        choices=["sort", "deepsort", "ul_bytetrack", "ul_botsort"],
        default="sort",
    )
    parser.add_argument(
        "--yolo", default=None, help="YOLO model path — omit to use dataset detections"
    )

    parser.add_argument(
        "--results-dir",
        default="trackeval_results",
        help="Where to write per-sequence MOT-format results and TrackEval output",
    )
    args = parser.parse_args()

    is_ultralytics = args.tracker.startswith("ul_")
    if is_ultralytics and not args.yolo:
        parser.error(f"--yolo is required for {args.tracker}")

    mot16_root = Path(args.mot16_root).resolve()
    sequences = sorted(p for p in mot16_root.iterdir() if p.is_dir())

    # Prepare output dir
    yolo_suffix = "_yolo" if args.yolo else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracker_name = f"{args.tracker}{yolo_suffix}_{timestamp}"
    results_dir = Path(args.results_dir).resolve()
    tracker_run_dir = results_dir / tracker_name
    tracker_out_dir = tracker_run_dir / "data"
    tracker_out_dir.mkdir(parents=True, exist_ok=True)

    if args.tracker == "sort":
        shutil.copy(SORT_CONFIG_PATH, tracker_run_dir / SORT_CONFIG_PATH.name)

    reid_model = reid_transform = None
    if args.tracker == "deepsort":
        reid_model, reid_transform = make_reid_model()

    tracker_yaml = ULTRALYTICS_TRACKER_YAMLS.get(args.tracker, "")

    seq_names = []
    for seq_dir in sequences:
        seq_name = seq_dir.name
        print(f"Processing {seq_name}...")

        frames_dir = seq_dir / "img1"
        out_path = tracker_out_dir / f"{seq_name}.txt"

        with out_path.open("w") as f:
            if is_ultralytics:
                track_sequence_ultralytics(args.yolo, frames_dir, tracker_yaml, f)
            else:
                frame_paths = sorted(frames_dir.glob("*.jpg"))
                det_iter = (
                    iter_yolo(args.yolo, str(frames_dir))
                    if args.yolo
                    else iter_mot16(str(seq_dir / "det" / "det.txt"), frame_paths)
                )
                track_sequence(det_iter, args.tracker, f, reid_model, reid_transform)
        seq_names.append(seq_name)

    seqmap_path = results_dir / "seqmap.txt"
    with seqmap_path.open("w") as f:
        f.write("name\n")
        for name in seq_names:
            f.write(f"{name}\n")

    print(f"\nWrote results to {tracker_out_dir}")
    print("Running TrackEval...\n")
    run_trackeval(
        gt_folder=mot16_root,
        trackers_folder=results_dir,
        output_folder=results_dir,
        tracker_name=tracker_name,
        seqmap_file=seqmap_path,
    )


if __name__ == "__main__":
    main()
