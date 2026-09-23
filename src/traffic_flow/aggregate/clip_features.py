"""Boils one clip's analysis down to a fixed row of numbers.

The congestion classifier needs every clip described the same way, in the same
order, every time. That is what this file is: one declared list of features, each
with the name it is reported under and how to pull it out of a
:class:`~traffic_flow.pipeline.ClipAnalysis`.

Adding a feature means adding one entry to :data:`FEATURES`. The extractor, the
column names, the training code and the report all read that one list, so they
cannot drift apart.

The features are deliberately the traffic parameters themselves - speed, density,
occupancy, counts - and not some abstract descriptor. That way the classifier's
inputs are quantities a traffic engineer already understands, and when it calls a
clip "heavy" the reason can be read straight off the row.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic_flow.labels import VehicleClass
from traffic_flow.pipeline import ClipAnalysis


@dataclass(frozen=True)
class Feature:
    """One number describing a clip, with a plain sentence saying what it is."""

    name: str
    describe: str
    extract: Callable[[ClipAnalysis], float]


def _speed_spread(analysis: ClipAnalysis) -> float:
    """How much the vehicles' speeds differ from each other.

    Free-flowing traffic moves at similar speeds. Traffic that is breaking down
    has some vehicles still moving and others already stopped, so the spread
    widens before the average speed drops much.
    """
    if len(analysis.speeds) < 2:
        return 0.0
    return float(np.std([speed.kph for speed in analysis.speeds]))


def _mean_track_seconds(analysis: ClipAnalysis) -> float:
    """How long a vehicle stays in view on average.

    A direct consequence of speed that does not depend on the homography being
    right: slow vehicles linger. It gives the classifier a second, independent
    route to the same fact.
    """
    if not analysis.confirmed_tracks:
        return 0.0
    return float(np.mean([track.duration_s for track in analysis.confirmed_tracks]))


def _stopped_share(analysis: ClipAnalysis) -> float:
    """Fraction of vehicles moving slower than a walking pace.

    This is what separates heavy from medium traffic: heavy means stopped or
    crawling, not merely slow.
    """
    if not analysis.speeds:
        return 0.0
    stopped = sum(1 for speed in analysis.speeds if speed.kph < STOPPED_KPH)
    return stopped / len(analysis.speeds)


#: Below this, a vehicle counts as stopped or crawling rather than moving.
STOPPED_KPH = 10.0


#: The full feature set, in the order the columns appear.
FEATURES: tuple[Feature, ...] = (
    Feature(
        "space_mean_speed_kph",
        "clip speed, harmonic mean over vehicles",
        lambda a: a.space_mean_speed_kph,
    ),
    Feature(
        "median_speed_kph",
        "the middle vehicle's speed",
        lambda a: a.median_speed_kph,
    ),
    Feature("speed_spread_kph", "how much vehicle speeds differ", _speed_spread),
    Feature("stopped_share", "fraction of vehicles crawling or stopped", _stopped_share),
    Feature(
        "density_veh_per_km_per_lane",
        "vehicles per kilometre per lane",
        lambda a: a.density.veh_per_km_per_lane,
    ),
    Feature(
        "mean_vehicles_in_view",
        "vehicles on the measured stretch at any instant",
        lambda a: a.density.mean_vehicles_in_view,
    ),
    Feature(
        "peak_vehicles_in_view",
        "busiest single frame",
        lambda a: float(a.density.peak_vehicles_in_view),
    ),
    Feature(
        "mean_occupancy",
        "share of road area covered by vehicles",
        lambda a: a.density.mean_occupancy,
    ),
    Feature("vehicles_seen", "distinct vehicles in the clip", lambda a: float(a.counts.vehicles_seen)),
    Feature("crossings", "vehicles passing the counting line", lambda a: float(a.counts.crossings)),
    Feature(
        "flow_veh_per_hour",
        "line crossings scaled to an hourly rate",
        lambda a: a.counts.flow_veh_per_hour,
    ),
    Feature("mean_track_seconds", "how long a vehicle stays in view", _mean_track_seconds),
    Feature(
        "lane_changes_per_vehicle",
        "lane changes per vehicle",
        lambda a: a.lanes.lane_changes_per_vehicle,
    ),
    Feature("lane_imbalance", "how unevenly the lanes are loaded", lambda a: a.lanes.lane_imbalance),
    Feature(
        "heavy_vehicle_share",
        "fraction of vehicles that are trucks or buses",
        lambda a: a.counts.heavy_vehicle_share,
    ),
)

#: Column names, derived from the declaration so the two cannot disagree.
FEATURE_NAMES: tuple[str, ...] = tuple(feature.name for feature in FEATURES)

#: What each feature means, for report tables and the API's documentation.
FEATURE_DESCRIPTIONS: dict[str, str] = {f.name: f.describe for f in FEATURES}


def clip_features(analysis: ClipAnalysis) -> dict[str, float]:
    """One clip as a row of named numbers."""
    return {feature.name: float(feature.extract(analysis)) for feature in FEATURES}


def clip_row(analysis: ClipAnalysis) -> dict[str, float | str]:
    """The feature row plus the per-class counts and the clip's name.

    The per-class counts are reported but kept out of :data:`FEATURES` on
    purpose: which vehicles are present says little about how congested the road
    is, and with 254 clips every extra column is a chance to overfit.
    """
    row: dict[str, float | str] = {"clip_id": analysis.clip_id}
    row.update(clip_features(analysis))
    for vehicle in VehicleClass:
        row[f"seen_{vehicle}"] = float(analysis.counts.seen_by_class.get(vehicle, 0))
    return row


def features_frame(analyses: list[ClipAnalysis]) -> pd.DataFrame:
    """Turn many clips into one table, one row per clip."""
    return pd.DataFrame([clip_row(analysis) for analysis in analyses])
