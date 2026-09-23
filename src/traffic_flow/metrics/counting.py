"""How many vehicles there were, by kind, and how many passed the camera.

Two counts answer two different questions and both are reported, because mixing
them up is the usual way a traffic count goes wrong:

- **vehicles seen** - how many distinct vehicles appeared anywhere in the clip.
- **crossings** - how many drove past one fixed line on the road.

Only the second one turns into flow, the vehicles-per-hour figure traffic
engineers use, because flow is defined at a point on the road. In stopped
traffic a clip can show twenty vehicles and have none cross the line, and that
is the correct answer, not a bug.

Crossings are counted from the finished tracks rather than frame by frame. That
way each vehicle is counted once, under the class it was called most often
across its whole path, instead of under whatever the detector guessed in the one
frame it happened to touch the line.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from traffic_flow.labels import VEHICLE_ORDER, VehicleClass
from traffic_flow.tracking.tracks import Track

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class CountMetrics:
    """The counts for one clip."""

    vehicles_seen: int
    crossings: int
    #: Distinct vehicles per class, over the whole clip.
    seen_by_class: dict[VehicleClass, int]
    #: Line crossings per class.
    crossings_by_class: dict[VehicleClass, int]
    #: Crossings scaled to an hourly rate.
    flow_veh_per_hour: float
    #: How long the clip was, which the hourly rate is extrapolated from. About
    #: five seconds here, so the rate is indicative rather than precise.
    duration_s: float

    @property
    def heavy_vehicle_share(self) -> float:
        """Fraction of vehicles that were trucks or buses.

        Worth reporting on its own: heavy vehicles take more road space and
        accelerate more slowly, so a high share changes what the same count means
        for congestion.
        """
        if self.vehicles_seen == 0:
            return 0.0
        heavy = self.seen_by_class.get(VehicleClass.TRUCK, 0) + self.seen_by_class.get(
            VehicleClass.BUS, 0
        )
        return heavy / self.vehicles_seen


Line = tuple[tuple[float, float], tuple[float, float]]


def side_of_line(point: np.ndarray, line: Line) -> float:
    """Which side of the line a point is on: positive one side, negative the other.

    Zero means the point is exactly on the line. The sign is all that is used -
    to tell whether a vehicle has passed the counting line yet.
    """
    (x1, y1), (x2, y2) = line
    along_x, along_y = x2 - x1, y2 - y1
    offset_x, offset_y = float(point[0]) - x1, float(point[1]) - y1
    return along_x * offset_y - along_y * offset_x


def _segments_cross(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    """True when segment a1-a2 and segment b1-b2 touch.

    Works by asking which side of each line the other segment's endpoints fall
    on. If the two ends of each segment sit on opposite sides of the other, the
    segments must cross.

    A point landing exactly on the line counts as a crossing. That case is real:
    a vehicle is only sampled ten times a second, and one of those samples can
    land on the counting line to the pixel. Demanding a strict change of side
    would miss that vehicle entirely.
    """
    segment_a = ((float(a1[0]), float(a1[1])), (float(a2[0]), float(a2[1])))
    segment_b = ((float(b1[0]), float(b1[1])), (float(b2[0]), float(b2[1])))

    d1, d2 = side_of_line(a1, segment_b), side_of_line(a2, segment_b)
    d3, d4 = side_of_line(b1, segment_a), side_of_line(b2, segment_a)
    return (d1 * d2 <= 0) and (d3 * d4 <= 0) and not (d1 == d2 == 0)


def crosses_line(track: Track, line: Line) -> bool:
    """Did this vehicle's path pass through the counting line?"""
    path = track.image_path
    if len(path) < 2:
        return False
    start = np.asarray(line[0], dtype=np.float64)
    end = np.asarray(line[1], dtype=np.float64)
    return any(
        _segments_cross(path[i], path[i + 1], start, end) for i in range(len(path) - 1)
    )


def _count_by_class(tracks: list[Track]) -> dict[VehicleClass, int]:
    """Count vehicles per class, always listing every class, zeros included."""
    votes = Counter(track.vehicle_class for track in tracks)
    return {vehicle: votes.get(vehicle, 0) for vehicle in VEHICLE_ORDER}


def count_vehicles(
    tracks: list[Track],
    counting_line: Line,
    duration_s: float,
) -> CountMetrics:
    """Count the vehicles in a clip and work out the flow past the line."""
    crossing_tracks = [track for track in tracks if crosses_line(track, counting_line)]
    flow = (len(crossing_tracks) / duration_s * SECONDS_PER_HOUR) if duration_s > 0 else 0.0

    return CountMetrics(
        vehicles_seen=len(tracks),
        crossings=len(crossing_tracks),
        seen_by_class=_count_by_class(tracks),
        crossings_by_class=_count_by_class(crossing_tracks),
        flow_veh_per_hour=flow,
        duration_s=duration_s,
    )
