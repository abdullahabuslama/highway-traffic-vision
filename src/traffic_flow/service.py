"""Analyse one clip and return plain data. The single entry point.

The command line and the HTTP API both call :func:`analyse_clip` and neither has
any analysis logic of its own. That is deliberate: two front ends that each did
their own assembling would drift, and one of them would quietly start reporting
something different from the other.

Everything returned here is ordinary Python - numbers, strings, lists - so it can
be written straight to JSON without another conversion step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_flow.aggregate.clip_features import (
    FEATURE_DESCRIPTIONS,
    clip_features,
)
from traffic_flow.classify.store import DEFAULT_MODEL_PATH, StoredModel, load_model
from traffic_flow.config import CameraConfig, PipelineConfig
from traffic_flow.detection.yolo import YoloDetector
from traffic_flow.labels import VEHICLE_ORDER, CongestionLevel
from traffic_flow.metrics.speed import is_measured
from traffic_flow.pipeline import ClipAnalysis, TrafficPipeline
from traffic_flow.viz.annotate import ClipAnnotator

#: Result key holding where the annotated video was written on this machine.
#: A local file path: fine to show in a terminal, never to send over a network.
LOCAL_VIDEO_PATH_KEY = "annotated_video"


@dataclass
class TrafficService:
    """Holds the loaded model and pipeline so they are built once, not per clip.

    Loading YOLO takes a few seconds, so a server that rebuilt it for every
    request would spend most of its time doing that.
    """

    pipeline: TrafficPipeline
    camera: CameraConfig
    classifier: StoredModel | None

    @classmethod
    def build(
        cls,
        camera_config: str | Path | None = None,
        pipeline_config: str | Path | None = None,
        model_path: str | Path | None = DEFAULT_MODEL_PATH,
    ) -> TrafficService:
        """Load everything once. A missing model is allowed - the traffic numbers
        are measured, not learned, so only the light/medium/heavy call is lost."""
        config = PipelineConfig.load(pipeline_config)
        camera = CameraConfig.load(camera_config)
        pipeline = TrafficPipeline(YoloDetector(config.detector), camera, config)

        classifier: StoredModel | None = None
        if model_path is not None:
            try:
                classifier = load_model(model_path)
            except FileNotFoundError:
                classifier = None

        return cls(pipeline=pipeline, camera=camera, classifier=classifier)

    def analyse(
        self,
        video_path: str | Path,
        annotate_to: str | Path | None = None,
        clip_id: str | None = None,
    ) -> dict[str, Any]:
        """Measure one clip, and optionally write an annotated copy of it.

        ``clip_id`` names the clip in the result. Leave it out and the file's own
        name is used. The API passes it because it saves every upload under a
        random name, which would otherwise be reported as the clip's name.
        """
        analysis = self.pipeline.analyse(video_path, clip_id=clip_id)
        level, confidence = self._classify(analysis)

        result = describe(analysis, self.pipeline.road_length_m, len(self.camera.lanes))
        result["congestion"] = {
            "level": str(level) if level else None,
            "confidence": confidence,
            "source": "model" if self.classifier else "unavailable: no trained model loaded",
        }

        if annotate_to is not None:
            written = ClipAnnotator(self.camera).render(analysis, video_path, annotate_to, level)
            result[LOCAL_VIDEO_PATH_KEY] = str(written)

        return result

    def _classify(self, analysis: ClipAnalysis) -> tuple[CongestionLevel | None, float | None]:
        if self.classifier is None:
            return None, None
        row = pd.DataFrame([clip_features(analysis)])
        level = CongestionLevel(str(self.classifier.model.predict(row)[0]))
        confidence = float(self.classifier.model.predict_confidence(row)[0])
        return level, confidence


def describe(analysis: ClipAnalysis, road_length_m: float, lane_count: int) -> dict[str, Any]:
    """Turn one clip's analysis into plain, JSON-ready data.

    Grouped the way a reader would ask about it - what was there, how fast it was
    going, how tightly packed - rather than the order it was computed in.
    """
    counts = analysis.counts
    density = analysis.density
    lanes = analysis.lanes

    return {
        "clip_id": analysis.clip_id,
        "duration_s": round(analysis.duration_s, 2),
        "measured_over": {
            "road_length_m": round(road_length_m, 1),
            "lanes": lane_count,
        },
        "counts": {
            "vehicles_seen": counts.vehicles_seen,
            "crossed_the_line": counts.crossings,
            "by_class": {str(v): counts.seen_by_class.get(v, 0) for v in VEHICLE_ORDER},
            "heavy_vehicle_share": round(counts.heavy_vehicle_share, 3),
            "flow_veh_per_hour": round(counts.flow_veh_per_hour, 0),
        },
        "speed": {
            "space_mean_kph": _measured_or_none(analysis.space_mean_speed_kph),
            "median_kph": _measured_or_none(analysis.median_speed_kph),
            "vehicles_measured": len(analysis.speeds),
        },
        "density": {
            "veh_per_km_per_lane": round(density.veh_per_km_per_lane, 1),
            "mean_vehicles_in_view": round(density.mean_vehicles_in_view, 1),
            "peak_vehicles_in_view": density.peak_vehicles_in_view,
            "occupancy": round(density.mean_occupancy, 3),
        },
        "lanes": {
            "busiest": lanes.busiest_lane,
            "mean_vehicles_per_lane": {k: round(v, 2) for k, v in lanes.mean_vehicles_per_lane.items()},
            "imbalance": round(lanes.lane_imbalance, 3),
            "lane_changes": lanes.lane_changes,
            "lane_changes_per_vehicle": round(lanes.lane_changes_per_vehicle, 3),
        },
        "features": {
            name: _measured_or_none(value, decimals=4)
            for name, value in clip_features(analysis).items()
        },
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }


def _measured_or_none(value: float, decimals: int = 1) -> float | None:
    """A number ready for JSON, or ``None`` when it could not be measured.

    JSON has no way to write "not a number", and Python's writer would emit a bare
    ``NaN`` that most JSON readers reject outright. ``None`` becomes ``null``,
    which every reader understands as "no value" - and that is the truth for a
    speed on an empty road.
    """
    return round(value, decimals) if is_measured(value) else None
