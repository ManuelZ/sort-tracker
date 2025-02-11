# Standard Library imports
from itertools import count
from typing import Tuple
import sys
sys.path.append(r"<<PATH_TO_Kalman-Filter_code>>")

# External imports
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# Local imports
import kalman  # https://github.com/ManuelZ/Kalman-Filter

"""

My implementation of the SORT tracker, using my implementation of a Kalman Filter. Heavily draws from:
  - https://github.com/abewley/sort/blob/master/sort.py
  - https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition/blob/master/chapter10/sort.py
  - https://github.com/Jeremy26/tracking_course/tree/master


Observation model
The state of each target is modeled as:

    x = [u, v, s, r, u_dot, v_dot, s_dot].T 

where:
- u represents the horizontal pixel location of the center of the target
- v represents the vertical pixel location of the centre of the target
- s represents the scale (area)
- r represents the aspect ratio of the target's bounding box. The aspect ratio is considered to be constant. 


The state transition model
The velocities and the aspect ratio stay constant over time (with some process noise). 

"""



class KFTracker:
    def __init__(self, bbox, tracking_id):
        self.id = tracking_id
        self.filter = self.create_filter(bbox)
        self.posterior_bbox = bbox

        # Number of cycles elapsed since the last update.
        # - This counter is incremented during each call to predict().
        # - It resets to 0 whenever update() is called.
        self.cycles_since_update = 0

        # Current consecutive update streak.
        # - This counter increments with each update() call, tracking how many updates have been applied in succession.
        # - It resets to 0 if a cycle passes without an update (i.e., if more than one cycle elapses without an update).
        self.update_streak = 0

        # Added because when a misdetection ocurrs, update_streak resets to zero and the tracked object isn't displayed
        # anymore, although the tracker is still alive. With this variable and an updated rule, the tracker box is 
        # shown even if misdetections happen and only dissapears when the tracker is removed.
        self.updates = 0


    def create_filter(self, bbox):
        """
        State:
            x = [u, v, s, r, u_dot, v_dot, s_dot].T 

        Motion model:
            u[t] = u[t-1] + u_dot * dt
            v[t] = u[t-1] + v_dot * dt
            s[t] = s[t-1] + v_dot * dt
            r[t] = r[t-1]
            u_dot[t] = u_dot[t-1]
            v_dot[t] = v_dot[t-1]
            s_dot[t] = s_dot[t-1]
        """

        # State transition model that predicts the new state
        state_transition_model = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]
            ], dtype=np.float64
        )
        
        process_noise_covariance = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(np.float64)
        
        # No Control model
        control_model = None

        # Measurement matrix, aka Observation matrix. Only u, v, s and r are being measured.
        observation_matrix = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]
            ], dtype=np.float64
        )

        measurement_noise_covariance = np.diag([1, 1, 1, 1]).astype(np.float64)
        
        # Create the initial state with zero velocities
        initial_state = np.vstack((
            self.__bbox_to_state(bbox),
            np.array([0, 0, 0], dtype=np.float64).reshape(-1, 1)
        ))

        initial_state_covariance = np.diag([1, 1, 1, 1, 1e-1, 1e-1, 1e-1]).astype(np.float64)

        return kalman.Kalman(
            F=state_transition_model,
            B=control_model,
            H=observation_matrix,
            Q=process_noise_covariance,
            R=measurement_noise_covariance,
            x_prev=initial_state,
            P_prev=initial_state_covariance
        )

    def __bbox_to_state(self, bbox, dtype=np.float64):
        """
        Convert a bounding box defined by its top-left and bottom-right corners [x1, y1, x2, y2] to an observation 
        vector [x, y, s, r], where:

        - x, y : float
            The center coordinates of the bounding box.
        - s : float
            The scale (area) of the bounding box.
        - r : float
            The aspect ratio (width divided by height) of the bounding box.

        Parameters
        ----------
        bbox : array_like
            A list or array-like with four elements [x1, y1, x2, y2] representing the
            bounding box coordinates.
        dtype : data-type, optional
            Desired data type for the returned array (default is numpy.float64).

        Returns
        -------
        np.ndarray
            A 4x1 column vector (numpy array) containing [x, y, s, r].

        Modified from: https://github.com/abewley/sort/blob/master/sort.py
        """

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        s = w * h
        r = w / h

        return np.array([x, y, s, r], dtype=dtype).reshape(-1, 1)

    def __state_to_bbox(self, x):
        """
        From: https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition/blob/master/chapter10/sort.py#L129
        """
        center_x, center_y, s, r, _, _, _ = x.flatten()
        w = np.sqrt(s * r)
        h = s / w
        center = np.array([center_x, center_y])
        half_size = np.array([w, h]) / 2
        corners = center - half_size, center + half_size
        return np.concatenate(corners).astype(np.float64)

    def predict(self):
        """
        Compute and return the prior bounding box.

        Performs the following steps:
          1. Executes the Kalman filter's prediction step to obtain the prior state estimate.
          2. Converts the predicted state into a bounding box format.
          3. Returns the resulting bounding box.
        """
        prior, prior_P = self.filter.predict()
        if self.cycles_since_update > 0:
            self.update_streak = 0
        self.cycles_since_update += 1
        return self.__state_to_bbox(prior)
    
    def update(self, bbox):
        """ """
        self.cycles_since_update = 0
        self.update_streak += 1
        self.updates += 1
        state = self.__bbox_to_state(bbox)
        posterior, posterior_P = self.filter.update(state)
        self.posterior_bbox = self.__state_to_bbox(posterior)


class SortTracker:
    def __init__(self, iou_threshold=0.3, max_cycles_without_update=2, min_hits=3, starting_id=1):
        """
        """
        
        self.trackers:list[KFTracker] = []
        self._id_counter = count(starting_id)
        self.iou_threshold = iou_threshold
        self.min_updates = min_updates
        self.max_cycles_without_update = max_cycles_without_update
        self.frame_count = 0

    def get_next_id(self):
        """ Return a sequential integer."""
        return next(self._id_counter)

    def associate_detected_and_predicted_boxes(
        self,
        detections: np.ndarray,
        predictions: np.ndarray,
        dtype=np.float64
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Associate detection and tracking boxes based on their IoU values.
        
        - In assigning detections to existing targets, each target's bounding box geometry is estimated by predicting 
          its new location in the current frame. 
        - The assignment cost matrix is then computed as the intersection-over-union (IOU) distance between each 
          detection and all predicted bounding boxes from the existing targets. 
        - The assignment is solved optimally using the Hungarian algorithm. 

        Description from paper: "Simple Online And Realtime Tracking"
        """
        # TODO: improve the cost matrix
        iou_matrix = self._calculate_cost_matrix(detections, predictions)
        
        if len(iou_matrix) > 0:        
            # (row_idx, col_idx) pairs that identify the prediction that corresponds to a detection
            detection_indices, prediction_indices = linear_sum_assignment(-iou_matrix)
        else:
            detection_indices = []
            prediction_indices = []

        """
        - Additionally, a minimum IOU is imposed to reject assignments where the detection to target overlap is less 
          than IOUmin.
        - When objects enter and leave the image, unique identities need to be created or destroyed accordingly. 
          For creating trackers, we consider any detection with an overlap less than IOUmin to signify the existence of 
          an untracked object.

        Description from paper: "Simple Online And Realtime Tracking"
        """

        matches: list[tuple[int,int]] = []
        unmatched_detections: list[int] = []
        #unmatched_predictions: list[int] = []
        for detection_idx, prediction_idx in zip(detection_indices, prediction_indices):
            if iou_matrix[detection_idx, prediction_idx] >= self.iou_threshold:
                matches.append((detection_idx, prediction_idx))
            else:
                unmatched_detections.append(detection_idx)
                #unmatched_predictions.append(prediction_idx)
        
        # List detections without a matching track box. A  new tracker will be created for them.
        for detection_idx, detection in enumerate(detections):
            if detection_idx not in detection_indices:
                unmatched_detections.append(detection_idx)

        # for prediction_idx, prediction in enumerate(detected):
        #     if prediction_idx not in prediction_indices:
        #         unmatched_predictions.append(prediction_idx)

        return matches, unmatched_detections#, unmatched_predictions

    def _get_bbox_priors(self) -> list[list[float]]:
        """
        Compute and return a list of predicted bounding boxes, excluding any predictions with NaN values.
        
        This function iterates over all trackers and performs the prediction step of the Kalman filter for each.
        During the Kalman filter update, numerical issues can cause the state variable representing the bounding box 
        size (e.g., in a state vector [u, v, s, r, ...]) to become negative. This leads to the calculated bounding boxes
        to be formed by NaN values.
        This may occur when the error between the prediction and the measurement (the innovation) is large—such as when
        a tracked object leaves the frame— leading to an invalid bounding box.

        Detailed Explanation:
        During the update step of the Kalman filter, the state variable `size` can become negative.

            x = [u, v, s, r, ...]
                      ^^^
                      size variable

        The size can become negative during the calculation of the posterior, when `y` is negative and `K` is big, 
        `K @ y` can become a big number that makes the `size` variable negative.

            # This is the error between the prediction and measurement
            self.y = z - self.H @ self.x_prior
            
            # Update the estimate with measurements from sensor.
            self.x_posterior = self.x_prior + K @ self.y

        This scenario is typically observed when a bounding box shrinks excessively due to a tracked object leaving the 
        visible frame.

        """
        predictions = []
        for tracker in self.trackers:
            bbox_prior = tracker.predict()
            if np.isnan(bbox_prior).any():
                print(f"NaN detected, skipping tracker with id '{tracker.id}'.")
                continue
            predictions.append(bbox_prior)
        return predictions

    def filter_dead_trackers(self):
        """ """
        alive_trackers = []
        for tracker in self.trackers:
            if tracker.cycles_since_update < self.max_cycles_without_update:
                alive_trackers.append(tracker)
            else:
                print(f"Removing dead tracker with id '{tracker.id}'")
        return alive_trackers
    
    def create_trackers(self, detections, unmatched_detections):
        """ """
        trackers = []
        for i in unmatched_detections:
            new_id = self.get_next_id()
            new_tracker = KFTracker(detections[i], new_id)
            trackers.append(new_tracker)
            print(f"New tracker created with id '{new_id}'")
        return trackers
        
    def should_output_tracker(self, tracker):
        """
        Determine whether a given tracker should be output based on its update status, hit streak, and the current 
        frame count.        
        """
        
        # Check if the tracker was updated in the current cycle.
        # In the original code, this is `tracker.cycles_since_update < 1`
        tracker_updated_this_cycle = tracker.cycles_since_update < self.max_cycles_without_update
        
        # Check if the tracker has accumulated enough hits to be considered reliable.
        # In the original code, this is `update_streak > self.min_updates`
        has_sufficient_updates = tracker.updates >= self.min_updates
        
        # Check if we're in the early frames where lower hit streaks are acceptable.
        is_in_initial_phase = self.frame_count <= self.min_updates
        
        # The tracker should be output if it was updated this cycle and either has a sufficient hit streak or is in the 
        # initial phase.
        return tracker_updated_this_cycle and (has_sufficient_updates or is_in_initial_phase)


    def update(self, detections):
        """
        - The tracker is initialised using the geometry of the bounding box with the velocity set to zero. 
        - Since the velocity is unobserved at this point the covariance of the velocity component is initialised with 
          large values, reflecting this uncertainty. 
        - Additionally, the new tracker then undergoes a probationary period where the target needs to be associated 
          with detections to accumulate enough evidence in order to prevent tracking of false positives.

        - Tracks are terminated if they are not detected for TLost frames. This prevents an unbounded growth in the 
          number of trackers and localisation errors caused by predictions over long durations without corrections from 
          the detector. 
        
        - In all experiments TLost is set to 1 for two reasons:
           - Firstly, the constant velocity model is a poor predictor of the true dynamics
           - Secondly we are primarily concerned with frame-to-frame tracking where object re-identification is beyond 
             the scope of this work. 
          Additionally, early deletion of lost targets aids efficiency. Should an object reappear, tracking will 
          implicitly resume under a new identity.

        Description from paper: "Simple Online And Realtime Tracking"
        """

        self.frame_count += 1

        # Kalman Filter prediction step: calculate priors
        predictions = self._get_bbox_priors()
        matches, unmatched_detections = self.associate_detected_and_predicted_boxes(detections, predictions)

        # Kalman Filter update step: calculate posteriors
        for detection_idx, predicted_idx in matches:
            self.trackers[predicted_idx].update(detections[detection_idx])

        # Create new trackers        
        new_trackers = self.create_trackers(detections, unmatched_detections)
        self.trackers.extend(new_trackers)

        # Remove trackers that have not been updated in a while
        self.trackers = self.filter_dead_trackers()
        
        results = []
        for tracker in self.trackers:
            if self.should_output_tracker(tracker):
                results.append({"predicted_bbox": tracker.posterior_bbox, "id": tracker.id})

        return results

    @staticmethod
    def _calculate_cost_matrix(detections, predictions, dtype=np.float64):
        """
        """
        # TODO: This double loop looks costly. The original implementation vectorizes it, but is hard to read.
        iou_matrix = []
        for det in detections:
            for pred in predictions:
                iou_matrix.append(calculate_iou(det, pred))
        iou_matrix = np.array(iou_matrix, dtype=dtype)
        if len(iou_matrix) > 0:
            return iou_matrix.reshape(-1, len(predictions))
        return iou_matrix


def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    From:https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of 
    # prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
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


def draw_tracking_box(image, observation):
    """
    Modified from: https://github.com/Jeremy26/tracking_course
    """

    left, top, right, bottom = map(int, observation["predicted_bbox"])
    observation_id = observation["id"] 
    
    image = cv2.rectangle(
        img=image, 
        pt1=(left, top),
        pt2=(right, bottom),
        color=id_to_color(observation_id*10),
        thickness=3
    )
    
    image = cv2.putText(
        img=image,
        text=str(observation_id),
        org=(left - 10, top - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=id_to_color(observation_id*10),
        thickness=3
    )
    return image


if __name__ == "__main__":
    
    COMPARE_WITH_YOLO_TRACK = True
    SELECTED_CLASSES = ["car"]
    video_filename = "MOT16-13-raw.mp4"  # https://motchallenge.net/data/MOT16/
    sort_tracker = SortTracker(max_cycles_without_update=3)
    model = YOLO("yolo11l.pt")
    
    class_names = list(model.names.values())
    selected_indices = [class_names.index(option) for option in SELECTED_CLASSES]

    # YOLO tracking is done only to compare its results with my results
    results = model.track(
        video_filename,
        show=COMPARE_WITH_YOLO_TRACK,
        classes=selected_indices,
        stream=True,
        verbose=False
    )
    
    for r in results:
        boxes = [box.xyxy[0].cpu().numpy() for box in r.boxes]
        results = sort_tracker.update(boxes)
        final_image = r.orig_img.copy()
        for observation in results:
            draw_tracking_box(final_image, observation)
        cv2.imshow("", final_image)
        cv2.waitKey(1)
    print("---")


    
