"""Congestion level from traffic engineering thresholds, with no training.

The dataset's own README says what its three labels mean: light is free-flowing
traffic, medium is traffic at reduced speed, and heavy is stopped or very slow
traffic. That is a statement about speed, so speed is what the rules read, with
density as the second opinion for the case where the road is packed but the few
measurable vehicles happen to be moving.

The density numbers come from the Highway Capacity Manual's level-of-service
bands for a basic freeway segment, converted from vehicles per mile per lane to
per kilometre. Levels A to C, up to about 16 veh/km/lane, are free flow. D and E
run to about 28. Above that is level F, breakdown.

The bands are declared as data and checked worst-first. Changing where the line
falls, or adding a fourth level, edits the list - never the code below it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from traffic_flow.labels import CongestionLevel
from traffic_flow.metrics.speed import is_measured


@dataclass(frozen=True)
class Band:
    """A clip belongs to this level if it is slow enough *or* packed enough."""

    level: CongestionLevel
    #: At or below this speed, the clip is at least this bad.
    speed_at_or_below_kph: float
    #: At or above this density, the clip is at least this bad.
    density_at_or_above: float


#: Checked in order, worst first, so the first band that matches wins.
BANDS: tuple[Band, ...] = (
    Band(CongestionLevel.HEAVY, speed_at_or_below_kph=35.0, density_at_or_above=28.0),
    Band(CongestionLevel.MEDIUM, speed_at_or_below_kph=75.0, density_at_or_above=16.0),
    Band(CongestionLevel.LIGHT, speed_at_or_below_kph=math.inf, density_at_or_above=0.0),
)

#: Columns the rules read. Kept here so a missing column fails loudly.
REQUIRED_COLUMNS = ("space_mean_speed_kph", "density_veh_per_km_per_lane")


class RuleCongestionModel:
    """Applies the level-of-service bands. Learns nothing."""

    name = "rules"

    def __init__(self, bands: tuple[Band, ...] = BANDS):
        self._bands = bands

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> RuleCongestionModel:
        """Here only to satisfy the interface: thresholds are fixed, not fitted."""
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        missing = [name for name in REQUIRED_COLUMNS if name not in features]
        if missing:
            raise KeyError(f"rule model needs these columns: {missing}")

        speed = features["space_mean_speed_kph"].to_numpy(dtype=float)
        density = features["density_veh_per_km_per_lane"].to_numpy(dtype=float)

        # A clip with no measurable vehicle is empty road, so it is light. Only a
        # missing speed means that. A measured speed of zero is stopped traffic -
        # the opposite case - and must reach the bands like any other speed.
        speed = np.where(is_measured(speed), speed, math.inf)

        result = np.full(len(features), str(CongestionLevel.LIGHT), dtype=object)
        undecided = np.ones(len(features), dtype=bool)
        for band in self._bands:
            hits = undecided & (
                (speed <= band.speed_at_or_below_kph) | (density >= band.density_at_or_above)
            )
            result[hits] = str(band.level)
            undecided &= ~hits
        return result

    def predict_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """How far the clip sits from the nearest band edge, squashed to 0-1.

        A clip at 20 km/h is unambiguously heavy; one at 34 km/h only just is.
        This reports that difference instead of claiming certainty either way.
        """
        speed = features["space_mean_speed_kph"].to_numpy(dtype=float)
        edges = np.array(
            [band.speed_at_or_below_kph for band in self._bands if math.isfinite(
                band.speed_at_or_below_kph
            )]
        )
        if edges.size == 0:
            return np.ones(len(features))
        distance = np.abs(speed[:, None] - edges[None, :]).min(axis=1)
        confidence = np.clip(distance / edges.min(), 0.0, 1.0)
        # An empty road is the one case with no doubt at all: nothing is moving
        # slowly because nothing is there.
        return np.where(is_measured(speed), confidence, 1.0)
