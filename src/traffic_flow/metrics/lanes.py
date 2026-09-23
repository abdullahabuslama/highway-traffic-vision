"""Which lane traffic used, and who changed lanes.

Lane use tells you something a single road-wide number hides: an inside lane
crawling while the outside lane runs free is a different problem from all three
lanes slowing together, and it needs a different fix.

Lane changes are counted from each vehicle's sequence of lanes. The catch is
that a box sitting on a lane line flickers between the two lanes from frame to
frame, and counting every flicker would report far more lane changes than
actually happened. So a vehicle has to hold its new lane for a few frames in a
row before the change is believed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from traffic_flow.tracking.tracks import NO_LANE, FrameRecord, Track

#: How many frames in a row a vehicle must be seen in its new lane before the
#: change counts. Two is enough to reject single-frame flicker at 10 fps.
LANE_HOLD_FRAMES = 2


@dataclass(frozen=True)
class LaneMetrics:
    """How the traffic was spread across the lanes."""

    #: Average vehicles in each lane per frame.
    mean_vehicles_per_lane: dict[str, float]
    #: The lane carrying the most traffic, or ``""`` when no lanes are declared.
    busiest_lane: str
    #: How unevenly the lanes were loaded, from 0 (even) to 1 (all in one lane).
    lane_imbalance: float
    #: Confirmed lane changes in the clip.
    lane_changes: int
    #: Lane changes per vehicle. Rises when drivers are hunting for a faster
    #: lane, which is a useful early sign of congestion building.
    lane_changes_per_vehicle: float


def _debounced_lane_sequence(lanes: list[str], hold_frames: int) -> list[str]:
    """Strip out flicker, leaving the lanes the vehicle really settled in.

    Runs of the same lane shorter than ``hold_frames`` are dropped as noise, and
    frames where the vehicle was in no declared lane are ignored rather than
    treated as a lane of their own.
    """
    settled: list[str] = []
    run_lane: str | None = None
    run_length = 0

    for lane in lanes:
        if lane == NO_LANE:
            continue
        if lane == run_lane:
            run_length += 1
        else:
            run_lane, run_length = lane, 1
        if run_length == hold_frames and (not settled or settled[-1] != run_lane):
            settled.append(run_lane)

    return settled


def count_lane_changes(track: Track, hold_frames: int = LANE_HOLD_FRAMES) -> int:
    """How many times this vehicle moved to a different lane and stayed there."""
    return max(len(_debounced_lane_sequence(track.lanes, hold_frames)) - 1, 0)


def measure_lanes(
    tracks: list[Track],
    frames: list[FrameRecord],
    lane_names: tuple[str, ...],
) -> LaneMetrics:
    """Summarise lane use and lane changing for one clip."""
    changes = sum(count_lane_changes(track) for track in tracks)
    per_vehicle = changes / len(tracks) if tracks else 0.0

    if not lane_names or not frames:
        return LaneMetrics({}, "", 0.0, changes, per_vehicle)

    means = {
        name: float(np.mean([frame.lane_counts.get(name, 0) for frame in frames]))
        for name in lane_names
    }
    busiest = max(means, key=lambda name: means[name])

    return LaneMetrics(
        mean_vehicles_per_lane=means,
        busiest_lane=busiest,
        lane_imbalance=_imbalance(list(means.values())),
        lane_changes=changes,
        lane_changes_per_vehicle=per_vehicle,
    )


def _imbalance(loads: list[float]) -> float:
    """How far lane loads are from being equal, on a 0-to-1 scale.

    Zero means every lane carries the same amount. One means a single lane
    carries everything. It is the Gini coefficient of the lane loads, used
    because it does not care how many lanes there are, so a two-lane and a
    four-lane camera give comparable numbers.
    """
    values = np.sort(np.asarray(loads, dtype=np.float64))
    total = values.sum()
    if total <= 0 or len(values) < 2:
        return 0.0
    n = len(values)
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * values).sum()) / (n * total) - (n + 1) / n)


def lane_share(frames: list[FrameRecord], lane_names: tuple[str, ...]) -> dict[str, float]:
    """Share of all observed vehicles that were in each lane, summing to 1."""
    totals = Counter()
    for frame in frames:
        for name in lane_names:
            totals[name] += frame.lane_counts.get(name, 0)
    grand_total = sum(totals.values())
    if grand_total == 0:
        return {name: 0.0 for name in lane_names}
    return {name: totals[name] / grand_total for name in lane_names}
