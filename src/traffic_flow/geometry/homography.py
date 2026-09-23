"""Turns image pixels into metres on the road surface.

A camera looking down a road flattens it: two cars the same distance apart look
far apart near the camera and almost touching near the horizon. Measuring speed
in pixels per frame would therefore report the same vehicle as fast when it is
close and slow when it is far away.

The fix is a homography - a mapping between the image and a flat map of the road
seen from above. It is valid because a road surface really is flat, so once four
points on it are matched to their real positions, every other point on that
surface follows.

A vehicle is placed on that surface by the middle of the bottom edge of its box,
which is roughly where its tyres touch the road. Using the box centre instead
would float the vehicle above the ground and stretch every distance.
"""

from __future__ import annotations

import cv2
import numpy as np

from traffic_flow.config import CameraConfig

#: Metres per second to kilometres per hour.
MPS_TO_KPH = 3.6


class RoadPlane:
    """The mapping between image pixels and metres on the road.

    Built from matching pairs of points: where something is in the picture, and
    where it is on the ground. Four pairs are the minimum; more are averaged.
    """

    def __init__(self, image_points: np.ndarray, world_points: np.ndarray):
        image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        world_points = np.asarray(world_points, dtype=np.float32).reshape(-1, 2)
        if len(image_points) != len(world_points):
            raise ValueError("need the same number of image points and world points")
        if len(image_points) < 4:
            raise ValueError("a homography needs at least 4 point pairs")

        matrix, _ = cv2.findHomography(image_points, world_points, method=0)
        if matrix is None:
            raise ValueError("could not fit a homography to these points")

        self.image_to_world_matrix = matrix
        self.world_to_image_matrix = np.linalg.inv(matrix)

    @classmethod
    def from_config(cls, camera: CameraConfig) -> RoadPlane:
        return cls(camera.image_points_array, camera.world_points_array)

    def to_world(self, image_points: np.ndarray) -> np.ndarray:
        """Where these pixels are on the road, in metres. Shape (n, 2) in, (n, 2) out."""
        return _apply(self.image_to_world_matrix, image_points)

    def to_image(self, world_points: np.ndarray) -> np.ndarray:
        """The reverse: metres on the road back to pixels in the picture."""
        return _apply(self.world_to_image_matrix, world_points)

    def world_distances(self, image_points_a: np.ndarray, image_points_b: np.ndarray) -> np.ndarray:
        """Real distance in metres between matching pairs of image points."""
        world_a = self.to_world(image_points_a)
        world_b = self.to_world(image_points_b)
        return np.linalg.norm(world_a - world_b, axis=1)

    def polygon_length_m(self, image_polygon: np.ndarray) -> float:
        """How long a road region is, along the direction traffic travels.

        Used to turn a vehicle count into a density per kilometre. The polygon is
        mapped to the road plane and measured along its longer world axis, which
        for a stretch of carriageway is the direction of travel.
        """
        world = self.to_world(image_polygon)
        spans = world.max(axis=0) - world.min(axis=0)
        return float(spans.max())

    def polygon_width_m(self, image_polygon: np.ndarray) -> float:
        """How wide a road region is, across the direction of travel."""
        world = self.to_world(image_polygon)
        spans = world.max(axis=0) - world.min(axis=0)
        return float(spans.min())


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Push points through a 3x3 projective matrix."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
