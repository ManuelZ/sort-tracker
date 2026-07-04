# Standard Library imports
from dataclasses import dataclass

# External imports
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class TrackResult:
    bbox: npt.NDArray
    id: int


@dataclass
class MaskTrackResult:
    mask: npt.NDArray[np.bool_]  # H×W binary mask
    id: int


def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Computes IoU between two bboxes in the form [x1,y1,x2,y2]
    Modified from: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection area and dividing it by the sum of
    # prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def id_to_color(idx):
    """
    Random function to convert an id to a color. Do what you want here but keep numbers below 255.
    From: https://github.com/Jeremy26/tracking_course
    """
    blue = idx * 5 % 256
    green = idx * 12 % 256
    red = idx * 23 % 256
    return (red, green, blue)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Nx4 xywh (top-left origin)  →  Nx4 xyxy."""
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x1, y1, x1 + w, y1 + h], axis=1)


def xyxy_to_uvsr(boxes: np.ndarray) -> np.ndarray:
    """Nx4 xyxy  →  Nx4 [u, v, s, r]  (SORT measurement space)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    u = x1 + w / 2
    v = y1 + h / 2
    s = w * h
    r = w / h
    return np.stack([u, v, s, r], axis=1)


def xyxy_to_wh(boxes: np.ndarray) -> np.ndarray:
    """Nx4 xyxy  →  Nx2 [w, h]."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([w, h], axis=1)


def load_detections(det_path: str) -> dict[int, npt.NDArray[np.float32]]:
    """
    Load a MOT detections file and return a dict mapping frame number to an Nx4
    array of bounding boxes in xyxy format.

    MOT det format: frame, id, x, y, w, h, conf, -1, -1, -1  (xywh, top-left)
    """
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "c1", "c2", "c3"]
    df = pd.read_csv(det_path, header=None, names=cols)

    detections: dict[int, npt.NDArray[np.float32]] = {}
    for frame, group in df.groupby("frame"):
        x1 = group["x"].to_numpy(dtype=np.float32)
        y1 = group["y"].to_numpy(dtype=np.float32)
        x2 = x1 + group["w"].to_numpy(dtype=np.float32)
        y2 = y1 + group["h"].to_numpy(dtype=np.float32)
        detections[int(frame)] = np.stack([x1, y1, x2, y2], axis=1)

    return detections


def write_mots_results(
    file, frame_number: int, tracked: list["MaskTrackResult"], class_id: int = 1
) -> None:
    """
    Append tracking results for one frame to an open file in MOTS format:
      frame_id track_id class_id height width rle
    Fields are space-separated. track_id is encoded as class_id * 1000 + instance_id
    (MOTS convention). Masks must be binary H×W arrays. Requires pycocotools.
    Frame number must be 1-indexed.
    """
    from pycocotools import mask as mask_utils

    for obs in tracked:
        h, w = obs.mask.shape
        rle = mask_utils.encode(np.asfortranarray(obs.mask.astype(np.uint8)))
        counts = rle["counts"]
        rle_str = counts.decode("utf-8") if isinstance(counts, bytes) else counts
        track_id = class_id * 1000 + obs.id
        file.write(f"{frame_number} {track_id} {class_id} {h} {w} {rle_str}\n")


def write_mot_results(file, frame_number: int, tracked: list["TrackResult"]) -> None:
    """
    Append tracking results for one frame to an open file in MOT CSV format:
      frame, id, x, y, w, h, conf, -1, -1, -1
    Bboxes are converted from xyxy to xywh. Frame number must be 1-indexed.
    """
    for obs in tracked:
        x1, y1, x2, y2 = obs.bbox
        file.write(
            f"{frame_number},{obs.id},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n"
        )


def draw_tracking_box(image, observation, color=None):
    """
    Modified from: https://github.com/Jeremy26/tracking_course
    """
    left, top, right, bottom = map(int, observation.bbox)
    observation_id = observation.id
    box_color = color if color is not None else id_to_color(observation_id * 10)

    image = cv2.rectangle(
        img=image,
        pt1=(left, top),
        pt2=(right, bottom),
        color=box_color,
        thickness=3,
    )
    image = cv2.putText(
        img=image,
        text=str(observation_id),
        org=(left - 10, top - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=id_to_color(observation_id * 10),
        thickness=3,
    )
    return image
