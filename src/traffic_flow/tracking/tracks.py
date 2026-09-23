"""What the pipeline remembers about one clip: the vehicles and the frames.

Two records come out of watching a clip, and every traffic number is computed
from one of them:

- a :class:`Track` per vehicle - where it was, frame by frame. Speed, counts and
  lane changes come from these.
- a :class:`FrameRecord` per frame - how many vehicles were on the road at that
  instant, and how much of it they covered. Density and occupancy come from these.

Both are plain containers. They do no maths beyond summarising themselves, so the
rules for turning them into traffic parameters all live under ``metrics/``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from traffic_flow.labels import VehicleClass

#: Lane name used when a vehicle is on the road but in no declared lane.
NO_LANE = ""


@dataclass(frozen=True)
class Observation:
    """One vehicle, seen in one frame."""

    frame_index: int
    time_s: float
    xyxy: tuple[float, float, float, float]
    #: Where the vehicle meets the road, in image pixels.
    ground_xy: tuple[float, float]
    #: The same point in metres on the road surface.
    world_xy: tuple[float, float]
    vehicle_class: VehicleClass
    confidence: float
    lane: str


@dataclass
class Track:
    """One vehicle, followed across the frames it was visible in."""

    track_id: int
    observations: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def vehicle_class(self) -> VehicleClass:
        """The class this vehicle was called most often.

        A single frame's guess flips between car and truck on small objects, so
        the majority across the whole track is the stable answer.
        """
        votes = Counter(obs.vehicle_class for obs in self.observations)
        return votes.most_common(1)[0][0]

    @property
    def first_seen_s(self) -> float:
        return self.observations[0].time_s

    @property
    def last_seen_s(self) -> float:
        return self.observations[-1].time_s

    @property
    def duration_s(self) -> float:
        return self.last_seen_s - self.first_seen_s

    @property
    def world_path(self) -> np.ndarray:
        """Positions in metres, shape (n, 2), in the order they were seen."""
        return np.array([obs.world_xy for obs in self.observations], dtype=np.float64)

    @property
    def image_path(self) -> np.ndarray:
        """The same path in image pixels, for drawing."""
        return np.array([obs.ground_xy for obs in self.observations], dtype=np.float64)

    @property
    def times(self) -> np.ndarray:
        return np.array([obs.time_s for obs in self.observations], dtype=np.float64)

    @property
    def lanes(self) -> list[str]:
        return [obs.lane for obs in self.observations]


@dataclass(frozen=True)
class FrameRecord:
    """A snapshot of the road at one instant."""

    frame_index: int
    time_s: float
    #: How many vehicles were on the carriageway in this frame.
    vehicle_count: int
    #: Fraction of the road area their boxes covered, 0 to 1.
    occupancy: float
    #: Vehicles per class in this frame.
    class_counts: dict[VehicleClass, int]
    #: Vehicles per lane in this frame.
    lane_counts: dict[str, int]


@dataclass
class ClipObservations:
    """Everything seen in one clip, ready to be turned into traffic numbers."""

    clip_id: str
    fps: float
    duration_s: float
    tracks: dict[int, Track] = field(default_factory=dict)
    frames: list[FrameRecord] = field(default_factory=list)

    def record(self, track_id: int, observation: Observation) -> None:
        self.tracks.setdefault(track_id, Track(track_id=track_id)).add(observation)

    def confirmed_tracks(self, min_frames: int) -> list[Track]:
        """Tracks long enough to be a real vehicle rather than detector noise."""
        return [track for track in self.tracks.values() if len(track) >= min_frames]

    @property
    def n_frames(self) -> int:
        return len(self.frames)
