"""What every vehicle detector in this project must look like.

The rest of the pipeline only ever talks to this interface, so the YOLO detector
and the plain background-subtraction baseline can be swapped by changing one line
of config. Nothing downstream knows or cares which one is running.

Detections are passed around as ``supervision.Detections``, the shared structure
that the tracker and the drawing code already speak. One project rule rides on
top of it: the vehicle word for each box lives in ``detections.data`` under
:data:`VEHICLE_CLASS_KEY`. Detector class ids stay inside the detector.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import supervision as sv

#: Key in ``Detections.data`` holding the :class:`~traffic_flow.labels.VehicleClass`
#: of each box, as a string. Every detector must fill it.
VEHICLE_CLASS_KEY = "vehicle_class"


@runtime_checkable
class Detector(Protocol):
    """Finds the vehicles in a single frame."""

    name: str

    def detect(self, image: np.ndarray) -> sv.Detections:
        """Return the vehicles visible in one frame.

        The boxes are in the coordinates of the image that was passed in. Any
        resizing a detector does internally is its own business and must be
        undone before returning.
        """
        ...


def empty_detections() -> sv.Detections:
    """An empty result with the vehicle-class field present.

    Returned when a frame has no vehicles. It carries the same fields as a real
    result so callers never have to special-case the empty case.
    """
    detections = sv.Detections.empty()
    detections.data[VEHICLE_CLASS_KEY] = np.array([], dtype=object)
    return detections
