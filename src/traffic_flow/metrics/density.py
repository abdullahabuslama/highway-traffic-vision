"""How tightly packed the traffic is.

Density is the measure that says whether a road is busy, and it is what a count
alone cannot tell you: ten vehicles spread over a kilometre is an empty road,
ten vehicles in fifty metres is a queue. It is counted as vehicles per kilometre
per lane, which is the standard unit and the one the level-of-service bands in
the Highway Capacity Manual are written in.

The count comes from the frames, not from the tracks. Every frame is a snapshot
of how many vehicles were on the measured stretch at that instant, and the
average of those snapshots divided by the stretch's length gives the density.

Occupancy is reported next to it as a second opinion. It is the share of the road
area covered by vehicles, and unlike density it needs only boxes, not identities
- so when tracking struggles, occupancy still carries a usable signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_flow.tracking.tracks import FrameRecord

METRES_PER_KM = 1000.0


@dataclass(frozen=True)
class DensityMetrics:
    """How crowded the measured stretch of road was."""

    #: The headline figure: vehicles per kilometre per lane.
    veh_per_km_per_lane: float
    #: Average number of vehicles visible on the stretch at any instant.
    mean_vehicles_in_view: float
    #: The busiest single frame, which shows how bunched the traffic was.
    peak_vehicles_in_view: int
    #: Share of the road area covered by vehicles, averaged over the frames.
    mean_occupancy: float
    #: Length of the measured stretch in metres, from the road polygon.
    road_length_m: float
    lane_count: int


def measure_density(
    frames: list[FrameRecord],
    road_length_m: float,
    lane_count: int,
) -> DensityMetrics:
    """Turn the per-frame snapshots into a density for the whole clip."""
    if not frames or road_length_m <= 0 or lane_count <= 0:
        return DensityMetrics(0.0, 0.0, 0, 0.0, road_length_m, lane_count)

    counts = np.array([frame.vehicle_count for frame in frames], dtype=np.float64)
    occupancies = np.array([frame.occupancy for frame in frames], dtype=np.float64)

    mean_in_view = float(counts.mean())
    road_length_km = road_length_m / METRES_PER_KM

    return DensityMetrics(
        veh_per_km_per_lane=mean_in_view / road_length_km / lane_count,
        mean_vehicles_in_view=mean_in_view,
        peak_vehicles_in_view=int(counts.max()),
        mean_occupancy=float(occupancies.mean()),
        road_length_m=road_length_m,
        lane_count=lane_count,
    )


def flow_from_fundamental_relation(density_veh_per_km: float, speed_kph: float) -> float:
    """Flow implied by the density and speed, in vehicles per hour per lane.

    Traffic obeys a simple identity: flow = density x speed. It is used here as a
    cross-check on the line count. The clips are only about five seconds long, so
    counting crossings gives a noisy hourly rate, while density and speed are
    averaged over every frame and are steadier. When the two disagree badly, the
    tracking is at fault.
    """
    return density_veh_per_km * speed_kph
