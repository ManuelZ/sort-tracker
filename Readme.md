# SORT Tracker with Kalman Filter

A straightforward implementation of the SORT tracker algorithm. It uses a custom Kalman Filter to help predict where objects will be and matches them with new detections. It uses YOLO for object detection.

- The multi-object tracking system monitors multiple objects (e.g., vehicles) within video sequences.
- A Kalman Filter is employed to estimate an object's position, even when the object is not detected in a given frame.
- The Hungarian Algorithm constructs a cost matrix (often based on IoU) to optimally match new detections with predictions. If no satisfactory match is found, a new tracker is initiated.
- Bounding boxes are annotated with identification numbers for verification of the tracking process.

The Kalman Filter code can be found on: https://github.com/ManuelZ/Kalman-Filter

Some rules to show the tracking boxes differ from the original paper code.


https://github.com/user-attachments/assets/e3cb5306-74a7-4d91-9893-121f2385e023



## Requirements
```
pip install -r requirements.txt
```

## How to run
```
python main.py
```

## References

This implementation reflects my own learning journey, drawing on insights from the structure, code, and techniques found in:
  - https://github.com/abewley/sort/blob/master/sort.py
  - https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition/blob/master/chapter10/sort.py
  - https://github.com/Jeremy26/tracking_course/tree/master