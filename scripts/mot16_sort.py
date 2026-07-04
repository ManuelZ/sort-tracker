"""
Run SORT or DeepSORT tracker on a MOT16 sequence.

Usage with MOT16 detections:
python mot16_sort.py ^
    --frames  F:/DATASETS/MOT16/train/MOT16-13/img1 ^
    --det     F:/DATASETS/MOT16/train/MOT16-13/det/det.txt ^
    --output  sort_results.txt

Usage with YOLO detector:
python mot16_sort.py ^
    --frames  F:/DATASETS/MOT16/train/MOT16-13/img1 ^
    --yolo    yolo11l.pt ^
    --tracker deepsort ^
    --output  deepsort_results_yolo.txt
"""

# Standard Library imports
import argparse
from pathlib import Path
from typing import Iterator

# External imports
import cv2
import numpy as np
from ultralytics import YOLO

# Local imports
from tracking.sort import Sort
from tracking.utils import TrackResult, load_detections, write_mot_results


ULTRALYTICS_TRACKER_YAMLS = {
    "ul_bytetrack": "bytetrack.yaml",
    "ul_botsort": "botsort.yaml",
}


def iter_mot16(
    det_path: str, frame_paths: list[Path]
) -> Iterator[tuple[int, np.ndarray, Path]]:
    detections = load_detections(det_path)
    for frame_path in frame_paths:
        frame_number = int(frame_path.stem)
        yield (
            frame_number,
            detections.get(frame_number, np.empty((0, 4), dtype=np.float32)),
            frame_path,
        )


def iter_yolo(
    model_path: str, frames_dir: str
) -> Iterator[tuple[int, np.ndarray, Path]]:
    model = YOLO(model_path)
    selected_indices = [list(model.names.values()).index("person")]
    for r in model.predict(
        frames_dir, classes=selected_indices, stream=True, verbose=False
    ):
        frame_number = int(Path(r.path).stem)
        boxes = np.array(
            [box.xyxy[0].cpu().numpy() for box in r.boxes or []], dtype=np.float32
        ).reshape(-1, 4)
        yield frame_number, boxes, Path(r.path)


def track_sequence(
    det_iter,
    tracker_name: str,
    out_file,
    reid_model=None,
    reid_transform=None,
) -> None:
    """Run SORT or DeepSORT on a sequence and stream MOT-format results to out_file."""
    if tracker_name == "sort":
        tracker = Sort(max_cycles_without_update=3)
        for frame_number, boxes, scores, _ in det_iter:
            tracked, _ = tracker.update_tracks(list(zip(boxes, scores)))
            write_mot_results(out_file, frame_number, tracked)
    else:
        from tracking.deepsort import DeepSort, extract_descriptors

        tracker = DeepSort(max_cycles_without_update=3)
        for frame_number, boxes, _scores, frame_path in det_iter:
            frame = cv2.imread(str(frame_path))
            descriptors = (
                extract_descriptors(frame, list(boxes), reid_model, reid_transform)
                if len(boxes)
                else None
            )
            tracked = tracker.update(list(boxes), descriptors)
            write_mot_results(out_file, frame_number, tracked)


def track_sequence_ultralytics(
    yolo_path: str,
    frames_dir: Path,
    tracker_yaml: str,
    out_file,
) -> None:
    """Run an Ultralytics-integrated tracker (ByteTrack/BoT-SORT) on a sequence."""
    model = YOLO(yolo_path)
    person_idx = [list(model.names.values()).index("person")]
    results = model.track(
        source=str(frames_dir),
        tracker=tracker_yaml,
        persist=True,
        stream=True,
        classes=person_idx,
        verbose=False,
    )
    for r in results:
        frame_number = int(Path(r.path).stem)
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        tracked = [TrackResult(bbox=box, id=int(tid)) for box, tid in zip(xyxy, ids)]
        write_mot_results(out_file, frame_number, tracked)


def make_reid_model():
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms

    model = tv_models.mobilenet_v3_small(
        weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
    )
    model.classifier = torch.nn.Sequential(torch.nn.Identity())
    model.eval()
    transform = tv_transforms.Compose(
        [
            tv_transforms.ToPILImage(),
            tv_transforms.Resize((128, 128)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return model, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frames", required=True, help="Directory of MOT16 frames (img1/)"
    )
    parser.add_argument(
        "--det", default=None, help="MOT16 detections file (det/det.txt)"
    )
    parser.add_argument(
        "--yolo", default=None, help="YOLO model path (e.g. yolo11l.pt)"
    )
    parser.add_argument(
        "--tracker",
        choices=["sort", "deepsort", "ul_bytetrack", "ul_botsort"],
        default="sort",
    )
    parser.add_argument(
        "--output", default="results/sort_results.txt", help="Output MOT CSV file"
    )
    args = parser.parse_args()

    is_ultralytics = args.tracker.startswith("ul_")

    if is_ultralytics:
        if not args.yolo:
            parser.error(f"--yolo is required for {args.tracker}")
        if args.det:
            parser.error(f"--det cannot be used with {args.tracker}")
    else:
        if not args.det and not args.yolo:
            parser.error("Provide either --det or --yolo")
        if args.det and args.yolo:
            parser.error("--det and --yolo are mutually exclusive")

    with open(args.output, "w") as mot_file:
        if is_ultralytics:
            track_sequence_ultralytics(
                args.yolo,
                Path(args.frames),
                ULTRALYTICS_TRACKER_YAMLS[args.tracker],
                mot_file,
            )
            print(f"Saved: {args.output}")
            return

        frame_paths = sorted(Path(args.frames).glob("*.jpg"))
        det_iter = (
            iter_mot16(args.det, frame_paths)
            if args.det
            else iter_yolo(args.yolo, args.frames)
        )

        reid_model = reid_transform = None
        if args.tracker == "deepsort":
            reid_model, reid_transform = make_reid_model()

        track_sequence(det_iter, args.tracker, mot_file, reid_model, reid_transform)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
