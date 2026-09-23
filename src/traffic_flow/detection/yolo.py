"""Vehicle detection with a YOLO model trained on COCO.

COCO already contains car, truck, bus and motorcycle, which is exactly the
breakdown this project reports, and there are no boxes in this dataset to train
anything of our own on. So the pretrained model is used as-is and the work goes
into running it well on very small objects.

The one setting that matters most is ``imgsz``. The source frames are 320x240,
where a car near the horizon is about ten pixels across - smaller than the
network's finest output cell, so it cannot be detected at native size at all.
Running at a larger ``imgsz`` enlarges the frame first, which is what makes those
vehicles visible to the model.
"""

from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO

from traffic_flow.config import DetectorConfig
from traffic_flow.detection.base import VEHICLE_CLASS_KEY, empty_detections


class YoloDetector:
    """Runs a YOLO model and keeps only the classes we count as vehicles."""

    name = "yolo"

    def __init__(self, config: DetectorConfig):
        self._config = config
        self._model = YOLO(config.model)

    @property
    def config(self) -> DetectorConfig:
        return self._config

    def detect(self, image: np.ndarray) -> sv.Detections:
        config = self._config
        result = self._model.predict(
            image,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            max_det=config.max_detections,
            augment=config.augment,
            classes=sorted(config.class_map),
            verbose=False,
        )[0]

        detections = sv.Detections.from_ultralytics(result)
        if len(detections) == 0:
            return empty_detections()

        # ``classes=`` above already asks the model for these ids only, but the
        # filter is repeated here so a model with a different class order cannot
        # silently let something else through.
        keep = np.array([cid in config.class_map for cid in detections.class_id], dtype=bool)
        detections = detections[keep]
        if len(detections) == 0:
            return empty_detections()

        detections.data[VEHICLE_CLASS_KEY] = np.array(
            [str(config.class_map[cid]) for cid in detections.class_id], dtype=object
        )
        return detections
