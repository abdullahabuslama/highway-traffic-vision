"""Answers "is this vehicle on the road, and in which lane?".

Everything outside the carriageway is noise for this project: parked cars on the
verge, traffic on the far side, anything the detector invents in the trees. One
polygon drawn over the road decides what counts, and it is the only filter of its
kind in the pipeline.

A vehicle is placed by the middle of the bottom edge of its box, the same point
the homography uses, so "on the road" and "this far down the road" always agree
about where a vehicle is.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from traffic_flow.config import CameraConfig

#: Where on a box we consider the vehicle to be standing.
GROUND_ANCHOR = sv.Position.BOTTOM_CENTER


def ground_points(detections: sv.Detections) -> np.ndarray:
    """The road-contact point of every box, shape (n, 2)."""
    if len(detections) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return detections.get_anchors_coordinates(GROUND_ANCHOR).astype(np.float32)


class RoadZone:
    """The drivable area, plus the lanes inside it."""

    def __init__(self, camera: CameraConfig):
        self._road = camera.road_polygon_array
        self._lane_names = tuple(lane.name for lane in camera.lanes)
        self._lane_polygons = tuple(lane.polygon_array for lane in camera.lanes)
        self.area_px = float(cv2.contourArea(self._road.astype(np.float32)))

    @property
    def polygon(self) -> np.ndarray:
        return self._road

    @property
    def lane_names(self) -> tuple[str, ...]:
        return self._lane_names

    def contains(self, points: np.ndarray) -> np.ndarray:
        """True for each point that sits on the carriageway."""
        return _inside(self._road, points)

    def filter(self, detections: sv.Detections) -> sv.Detections:
        """Drop every detection standing outside the carriageway."""
        if len(detections) == 0:
            return detections
        return detections[self.contains(ground_points(detections))]

    def lane_of(self, points: np.ndarray) -> np.ndarray:
        """Which lane each point falls in, as a lane name, or ``""`` for none.

        Lanes are tested in the order they are declared, so overlapping polygons
        resolve to the first match rather than to an arbitrary one.
        """
        result = np.full(len(points), "", dtype=object)
        if len(points) == 0 or not self._lane_polygons:
            return result
        for name, polygon in zip(self._lane_names, self._lane_polygons, strict=True):
            unassigned = result == ""
            if not unassigned.any():
                break
            hits = _inside(polygon, points)
            result[unassigned & hits] = name
        return result

    def coverage(self, detections: sv.Detections) -> float:
        """Fraction of the road area covered by vehicle boxes.

        This is lane occupancy in the traffic-engineering sense, approximated with
        boxes. It is a useful backup for density because it still means something
        when the tracker struggles: it needs boxes, not identities.

        Overlapping boxes are counted once, by painting them into a mask rather
        than adding areas up.
        """
        if self.area_px <= 0:
            return 0.0
        if len(detections) == 0:
            return 0.0

        x_max = int(np.ceil(self._road[:, 0].max())) + 1
        y_max = int(np.ceil(self._road[:, 1].max())) + 1
        canvas = np.zeros((y_max, x_max), dtype=np.uint8)
        for x1, y1, x2, y2 in detections.xyxy.astype(int):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color=1, thickness=-1)

        road_mask = np.zeros_like(canvas)
        cv2.fillPoly(road_mask, [self._road.astype(np.int32)], color=1)
        return float(np.count_nonzero(canvas & road_mask) / self.area_px)


def _inside(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    contour = polygon.astype(np.float32)
    return np.array(
        [cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in points],
        dtype=bool,
    )
