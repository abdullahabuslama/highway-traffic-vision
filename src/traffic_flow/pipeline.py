"""Watches one clip and produces its traffic numbers.

This is the only file that knows the whole order of events. Each piece it calls
knows just its own job:

    frames -> detect -> keep what is on the road -> track -> record
                                                              |
                                        counts, speed, density, lane use

Adding a new traffic parameter means adding a module under ``metrics/`` and one
line in :func:`summarise`. Nothing else in the project has to change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

from traffic_flow.config import CameraConfig, PipelineConfig
from traffic_flow.detection.base import VEHICLE_CLASS_KEY, Detector
from traffic_flow.geometry.homography import RoadPlane
from traffic_flow.geometry.road_zone import RoadZone, ground_points
from traffic_flow.io.video_source import VideoSource
from traffic_flow.labels import VehicleClass
from traffic_flow.metrics.counting import CountMetrics, count_vehicles
from traffic_flow.metrics.density import (
    DensityMetrics,
    flow_from_fundamental_relation,
    measure_density,
)
from traffic_flow.metrics.lanes import LaneMetrics, measure_lanes
from traffic_flow.metrics.speed import (
    TrackSpeed,
    clip_speeds,
    median_speed_kph,
    space_mean_speed_kph,
)
from traffic_flow.tracking.byte_tracker import VehicleTracker
from traffic_flow.tracking.tracks import (
    ClipObservations,
    FrameRecord,
    Observation,
    Track,
)


@dataclass(frozen=True)
class ClipAnalysis:
    """Every traffic number for one clip, plus what it was derived from."""

    clip_id: str
    duration_s: float
    counts: CountMetrics
    density: DensityMetrics
    lanes: LaneMetrics
    #: One speed per vehicle that could be measured.
    speeds: list[TrackSpeed]
    #: The clip's speed, harmonic-mean weighted. See ``metrics.speed``.
    space_mean_speed_kph: float
    median_speed_kph: float
    #: Flow implied by density x speed, as a cross-check on the line count.
    implied_flow_veh_per_hour_per_lane: float
    #: Kept so the annotator can redraw the clip without analysing it twice.
    observations: ClipObservations
    #: Tracks long enough to trust, the ones every metric above was built from.
    confirmed_tracks: list[Track]


class TrafficPipeline:
    """Runs the full analysis on clips from one camera.

    The camera geometry is fixed for the whole dataset, so a pipeline is built
    once and reused across all 254 clips. Only the tracker is rebuilt per clip,
    because vehicle ids must start again for each one.
    """

    def __init__(self, detector: Detector, camera: CameraConfig, config: PipelineConfig):
        self._detector = detector
        self._camera = camera
        self._config = config
        self._plane = RoadPlane.from_config(camera)
        self._zone = RoadZone(camera)
        self._road_length_m = self._plane.polygon_length_m(camera.road_polygon_array)

    @property
    def road_length_m(self) -> float:
        """Length of the measured stretch of road, in metres."""
        return self._road_length_m

    @property
    def road_zone(self) -> RoadZone:
        return self._zone

    @property
    def road_plane(self) -> RoadPlane:
        return self._plane

    def analyse(self, video_path: str | Path, clip_id: str | None = None) -> ClipAnalysis:
        """Watch one clip and return its traffic numbers."""
        video_path = Path(video_path)
        observations = self._watch(video_path, clip_id or video_path.stem)
        return self.summarise(observations)

    def _watch(self, video_path: Path, clip_id: str) -> ClipObservations:
        """Play the clip through detection and tracking, recording what is seen."""
        tracker = VehicleTracker(self._config.tracker, fps=_fps_of(video_path))

        with VideoSource(video_path) as source:
            observations = ClipObservations(
                clip_id=clip_id,
                fps=source.meta.fps,
                duration_s=source.meta.duration_s,
            )
            for frame in source:
                detections = self._zone.filter(self._detector.detect(frame.image))
                tracked = tracker.update(detections)
                self._record_frame(observations, frame.index, frame.time_s, tracked)

        return observations

    def _record_frame(
        self,
        observations: ClipObservations,
        frame_index: int,
        time_s: float,
        tracked: sv.Detections,
    ) -> None:
        """Store one frame's vehicles, both per vehicle and as a snapshot.

        Detections the tracker could not place get no identity. It marks them
        with ``-1`` rather than leaving the field empty, and they must be dropped:
        if they are kept they all share that one id and become a single fake
        vehicle that teleports around the frame.
        """
        tracked = tracked[_has_identity(tracked)]

        anchors = ground_points(tracked)
        world = self._plane.to_world(anchors) if len(anchors) else np.empty((0, 2))
        lanes = self._zone.lane_of(anchors)
        classes = [VehicleClass(name) for name in tracked.data.get(VEHICLE_CLASS_KEY, [])]

        for i, track_id in enumerate(tracked.tracker_id):
            observations.record(
                int(track_id),
                Observation(
                    frame_index=frame_index,
                    time_s=time_s,
                    xyxy=tuple(float(v) for v in tracked.xyxy[i]),
                    ground_xy=(float(anchors[i][0]), float(anchors[i][1])),
                    world_xy=(float(world[i][0]), float(world[i][1])),
                    vehicle_class=classes[i],
                    confidence=float(tracked.confidence[i]),
                    lane=str(lanes[i]),
                ),
            )

        observations.frames.append(
            FrameRecord(
                frame_index=frame_index,
                time_s=time_s,
                vehicle_count=len(tracked),
                occupancy=self._zone.coverage(tracked),
                class_counts=dict(Counter(classes)),
                lane_counts=dict(Counter(str(lane) for lane in lanes)),
            )
        )

    def summarise(self, observations: ClipObservations) -> ClipAnalysis:
        """Turn what was seen into the traffic parameters."""
        metrics_config = self._config.metrics
        tracks = observations.confirmed_tracks(metrics_config.min_track_frames)

        speeds = clip_speeds(tracks, metrics_config)
        space_mean = space_mean_speed_kph(speeds)
        density = measure_density(
            observations.frames,
            road_length_m=self._road_length_m,
            lane_count=max(len(self._camera.lanes), 1),
        )

        return ClipAnalysis(
            clip_id=observations.clip_id,
            duration_s=observations.duration_s,
            counts=count_vehicles(tracks, self._camera.counting_line, observations.duration_s),
            density=density,
            lanes=measure_lanes(tracks, observations.frames, self._zone.lane_names),
            speeds=speeds,
            space_mean_speed_kph=space_mean,
            median_speed_kph=median_speed_kph(speeds),
            implied_flow_veh_per_hour_per_lane=flow_from_fundamental_relation(
                density.veh_per_km_per_lane, space_mean
            ),
            observations=observations,
            confirmed_tracks=tracks,
        )


def _has_identity(tracked: sv.Detections) -> np.ndarray:
    """True for each detection the tracker actually gave a vehicle identity to."""
    if len(tracked) == 0 or tracked.tracker_id is None:
        return np.zeros(len(tracked), dtype=bool)
    ids = np.asarray(tracked.tracker_id)
    return np.array([i is not None and int(i) >= 0 for i in ids], dtype=bool)


def _fps_of(video_path: Path) -> float:
    """Read the frame rate without consuming the clip."""
    with VideoSource(video_path) as source:
        return source.meta.fps
