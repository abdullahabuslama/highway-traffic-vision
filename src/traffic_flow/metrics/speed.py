"""How fast each vehicle was going, in kilometres per hour.

The obvious method - measure how far a vehicle moved between two frames, divide
by the time - is the wrong one here. The boxes wobble by a pixel or two even on a
parked car, and far down the road one pixel is several metres. Because distance
is always positive, that wobble does not cancel out: it adds to every step and
reports stationary traffic as moving.

So the speed is fitted to the whole path instead. A vehicle crossing this camera
travels almost straight at almost constant speed for a second or two, which makes
its position a straight line against time, and the slope of that line is its
velocity. Fitting uses the Theil-Sen estimator - the median of the slopes of
every pair of points - which both averages the wobble away and ignores the odd
wild point left by the tracker swapping two vehicles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import theilslopes

from traffic_flow.config import MetricsConfig
from traffic_flow.geometry.homography import MPS_TO_KPH
from traffic_flow.labels import VehicleClass
from traffic_flow.tracking.tracks import Track

#: Fewest positions a straight line can be fitted to with any meaning.
MIN_POINTS_FOR_FIT = 3

#: The speed reported when there was no vehicle to measure it on.
#:
#: This is deliberately *not* zero. Zero km/h means traffic is stopped, which is
#: the worst state a road can be in, and an empty road is the best. Six clips in
#: the dataset have no traffic at all - early-morning rain, nothing on the road -
#: and reporting them as 0 km/h would tell a reader the road was jammed exactly
#: when it was clear. "Not a number" says what is actually true: there was
#: nothing to measure. Anything consuming a speed must check :func:`is_measured`.
NOT_MEASURED = float("nan")


@dataclass(frozen=True)
class TrackSpeed:
    """One vehicle's speed, and how much evidence it rests on."""

    track_id: int
    vehicle_class: VehicleClass
    kph: float
    #: How many frames the vehicle was followed for.
    n_points: int
    #: How far it travelled in total, in metres. A vehicle measured over a longer
    #: stretch gives a more trustworthy speed than one seen for half a second.
    distance_m: float
    #: How long it was watched for, in seconds. Needed to weight it correctly
    #: when the clip's overall speed is worked out.
    observed_s: float

    @property
    def mps(self) -> float:
        return self.kph / MPS_TO_KPH


def track_speed(track: Track, config: MetricsConfig) -> TrackSpeed | None:
    """Fit one vehicle's speed, or return ``None`` if it cannot be trusted.

    Returns nothing when the vehicle was seen too briefly to fit a line, or when
    the fitted speed is impossible for this road - which in practice means the
    tracker moved one identity onto a different vehicle.
    """
    if len(track) < max(MIN_POINTS_FOR_FIT, config.min_track_frames):
        return None

    path = track.world_path
    times = track.times
    if np.ptp(times) <= 0:
        return None

    velocity_x = theilslopes(path[:, 0], times).slope
    velocity_y = theilslopes(path[:, 1], times).slope
    kph = float(np.hypot(velocity_x, velocity_y) * MPS_TO_KPH)

    if not np.isfinite(kph) or kph > config.max_plausible_kph:
        return None

    return TrackSpeed(
        track_id=track.track_id,
        vehicle_class=track.vehicle_class,
        kph=kph,
        n_points=len(track),
        distance_m=float(np.linalg.norm(path[-1] - path[0])),
        observed_s=float(np.ptp(times)),
    )


def clip_speeds(tracks: list[Track], config: MetricsConfig) -> list[TrackSpeed]:
    """Fit a speed for every vehicle in a clip that has enough evidence."""
    fitted = (track_speed(track, config) for track in tracks)
    return [speed for speed in fitted if speed is not None]


def space_mean_speed_kph(speeds: list[TrackSpeed]) -> float:
    """The clip's speed, in the sense traffic engineering uses.

    Averaging the vehicles' speeds directly gives the *time* mean speed, which
    overstates how well a road is flowing: fast vehicles pass the camera more
    often and so get counted more than the slow ones stuck in the queue. What is
    wanted instead is the space mean speed, which weights every vehicle by how
    long it spends on the road, and it is the one that belongs in the
    speed-density relationship the rest of the analysis rests on.

    It is computed here the way it is defined - total distance covered by all
    vehicles, divided by the total time they spent covering it - and *not* as the
    harmonic mean of the individual speeds. The two agree whenever every vehicle
    is watched over the same stretch, but only the first is safe to compute.

    The harmonic mean divides by each speed in turn, so one vehicle reading near
    zero contributes an enormous term and drags the answer down to nothing. That
    is not hypothetical: a light-traffic clip with eighteen vehicles flowing at
    90 km/h reported 1.1 km/h, because a single stationary false detection was
    measured at a twentieth of a km/h. Totalling distance and time cannot do
    that - a stopped vehicle simply adds no distance while adding its time, which
    lowers the average by exactly as much as it should.
    """
    total_time = sum(speed.observed_s for speed in speeds)
    if total_time <= 0:
        return NOT_MEASURED
    total_distance = sum(speed.kph * speed.observed_s for speed in speeds)
    return float(total_distance / total_time)


def median_speed_kph(speeds: list[TrackSpeed]) -> float:
    """The middle vehicle's speed. Reported alongside the space mean as a check."""
    if not speeds:
        return NOT_MEASURED
    return float(np.median([speed.kph for speed in speeds]))


def is_measured(kph: float | None | np.ndarray) -> bool | np.ndarray:
    """True when a speed was actually measured, rather than there being no traffic.

    Takes one speed or a whole column of them, so the single-clip path and the
    table-wide rule model ask the same question the same way. ``None`` counts as
    not measured, because that is how a speed arrives back from JSON.
    """
    return np.isfinite(np.asarray(kph, dtype=float))


#: What a reader sees in place of a speed when there was nothing to measure.
NO_TRAFFIC_TEXT = "no traffic"


def describe_speed(kph: float | None, decimals: int = 0) -> str:
    """A speed as a person should read it, or plain words when there is none.

    The one place speeds are turned into text, so the terminal, the video overlay
    and the demo script all say the same thing about an empty road.
    """
    if not is_measured(kph):
        return NO_TRAFFIC_TEXT
    return f"{kph:.{decimals}f} km/h"
