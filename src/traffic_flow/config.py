"""The shape of the YAML settings, and how to load them.

Everything that varies between runs is declared in ``config/*.yaml`` and read
here into plain dataclasses. Nothing in the pipeline branches on a hard-coded
value: to change a threshold, a model, or which detector classes count as a
vehicle, you edit the YAML, not the code.

Two files, because they change for different reasons:

- ``pipeline.yaml`` is about the algorithm - model, thresholds, smoothing.
- ``camera_*.yaml`` is about one physical camera - where the road is in the
  image and how image pixels map to metres on the ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from traffic_flow.labels import VehicleClass

#: Where the shipped config files live, so scripts can find them without being
#: told every time.
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

Point = tuple[float, float]


def _as_points(raw: list[list[float]]) -> tuple[Point, ...]:
    return tuple((float(x), float(y)) for x, y in raw)


def _as_array(points: tuple[Point, ...]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32)


#: How many decimal places coordinates keep when written back to YAML. Two is
#: a hundredth of a pixel and a centimetre on the road: far finer than anything
#: measured here, and it keeps the file readable.
COORDINATE_DECIMALS = 2


def _plain_points(points: tuple[Point, ...]) -> list[list[float]]:
    """Points as ordinary rounded Python floats, ready for the YAML writer."""
    return [[round(float(x), COORDINATE_DECIMALS), round(float(y), COORDINATE_DECIMALS)]
            for x, y in points]


#: Written in the YAML when the machine should decide for itself.
AUTO_DEVICE = "auto"


def resolve_device(requested: str) -> str:
    """Turn ``auto`` into the best device this machine actually has.

    Any other value is passed through untouched, so a config can still pin the
    run to ``cpu`` or to a particular GPU.
    """
    if requested != AUTO_DEVICE:
        return requested
    import torch  # imported here so reading a config does not pay for loading torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class DetectorConfig:
    """How the vehicle detector is run.

    ``imgsz`` matters more than anything else here. The source video is only
    320x240, so a distant car is about ten pixels wide. Running the detector at
    its native size finds almost nothing; enlarging the frame first is what makes
    those vehicles detectable at all.
    """

    model: str
    imgsz: int
    conf: float
    iou: float
    device: str
    max_detections: int
    #: Test-time augmentation. Runs the model at several scales and merges the
    #: results, which finds more small vehicles but takes roughly twice as long.
    augment: bool
    #: Detector class id -> the vehicle word we report. Any id not listed here is
    #: ignored, which is how people, road signs and birds get dropped.
    class_map: dict[int, VehicleClass]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DetectorConfig:
        return cls(
            model=raw["model"],
            imgsz=int(raw["imgsz"]),
            conf=float(raw["conf"]),
            iou=float(raw["iou"]),
            device=resolve_device(raw["device"]),
            max_detections=int(raw.get("max_detections", 300)),
            augment=bool(raw.get("augment", False)),
            class_map={int(k): VehicleClass(v) for k, v in raw["class_map"].items()},
        )


@dataclass(frozen=True)
class TrackerConfig:
    """How detections are linked into tracks across frames.

    The defaults are loosened for this dataset because it runs at 10 fps. A car
    at motorway speed moves nearly three metres between two frames, so a tracker
    tuned for 25 or 30 fps breaks the identity on every fast vehicle.
    """

    #: A detection this confident can start a new vehicle track.
    track_activation_threshold: float
    #: How many frames a vehicle may be missed for before its identity is dropped.
    lost_track_buffer: int
    #: How much two boxes must overlap to be called the same vehicle. Low here,
    #: because at 10 fps a fast vehicle's box barely overlaps its own box in the
    #: next frame.
    minimum_iou_threshold: float
    #: How many frames a new vehicle must be seen in before it is reported.
    minimum_consecutive_frames: int
    #: The split between ByteTrack's two passes. Detections above this are matched
    #: first; weaker ones are then used only to keep existing vehicles alive.
    high_confidence_threshold: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrackerConfig:
        return cls(
            track_activation_threshold=float(raw["track_activation_threshold"]),
            lost_track_buffer=int(raw["lost_track_buffer"]),
            minimum_iou_threshold=float(raw["minimum_iou_threshold"]),
            minimum_consecutive_frames=int(raw["minimum_consecutive_frames"]),
            high_confidence_threshold=float(raw["high_confidence_threshold"]),
        )


@dataclass(frozen=True)
class MetricsConfig:
    """Thresholds used when turning tracks into traffic numbers."""

    #: Tracks shorter than this are detector noise, not vehicles.
    min_track_frames: int
    #: Speeds above this are impossible on this road and are thrown away.
    max_plausible_kph: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MetricsConfig:
        return cls(
            min_track_frames=int(raw["min_track_frames"]),
            max_plausible_kph=float(raw["max_plausible_kph"]),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Everything in ``pipeline.yaml``."""

    detector: DetectorConfig
    tracker: TrackerConfig
    metrics: MetricsConfig

    @classmethod
    def load(cls, path: str | Path | None = None) -> PipelineConfig:
        path = Path(path) if path else CONFIG_DIR / "pipeline.yaml"
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            detector=DetectorConfig.from_dict(raw["detector"]),
            tracker=TrackerConfig.from_dict(raw["tracker"]),
            metrics=MetricsConfig.from_dict(raw["metrics"]),
        )


@dataclass(frozen=True)
class LaneConfig:
    """One lane, drawn as a polygon in image pixels."""

    name: str
    polygon: tuple[Point, ...]

    @property
    def polygon_array(self) -> np.ndarray:
        return _as_array(self.polygon)


@dataclass(frozen=True)
class CameraConfig:
    """Where the road is in the image, and how pixels become metres.

    All 254 clips come from the same fixed camera, so this file is measured once
    by hand and then reused for the whole dataset.

    ``image_points`` and ``world_points`` are four matching corners on the road
    surface: the first in pixels, the second in metres. Together they define the
    flat-ground mapping that turns a box on screen into a position on the road.
    """

    name: str
    image_size: tuple[int, int]
    image_points: tuple[Point, ...]
    world_points: tuple[Point, ...]
    road_polygon: tuple[Point, ...]
    counting_line: tuple[Point, Point]
    lanes: tuple[LaneConfig, ...] = field(default_factory=tuple)
    #: What the world coordinates mean, written down so a reader knows the
    #: numbers were assumed from road standards rather than surveyed.
    scale_notes: str = ""

    @property
    def image_points_array(self) -> np.ndarray:
        return _as_array(self.image_points)

    @property
    def world_points_array(self) -> np.ndarray:
        return _as_array(self.world_points)

    @property
    def road_polygon_array(self) -> np.ndarray:
        return _as_array(self.road_polygon)

    @classmethod
    def load(cls, path: str | Path | None = None) -> CameraConfig:
        path = Path(path) if path else CONFIG_DIR / "camera_i5.yaml"
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CameraConfig:
        line = _as_points(raw["counting_line"])
        if len(line) != 2:
            raise ValueError("counting_line needs exactly two points")
        return cls(
            name=raw["name"],
            image_size=(int(raw["image_size"][0]), int(raw["image_size"][1])),
            image_points=_as_points(raw["homography"]["image_points"]),
            world_points=_as_points(raw["homography"]["world_points"]),
            road_polygon=_as_points(raw["road_polygon"]),
            counting_line=(line[0], line[1]),
            lanes=tuple(
                LaneConfig(name=lane["name"], polygon=_as_points(lane["polygon"]))
                for lane in raw.get("lanes", [])
            ),
            scale_notes=raw.get("scale_notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Turn the config back into the YAML shape, for the calibration script.

        Coordinates are rounded to plain Python floats. Rounding keeps the file
        readable, and the conversion matters because the calibration works in
        numpy, whose number types the YAML writer refuses to write.
        """
        return {
            "name": self.name,
            "image_size": [int(v) for v in self.image_size],
            "scale_notes": self.scale_notes,
            "homography": {
                "image_points": _plain_points(self.image_points),
                "world_points": _plain_points(self.world_points),
            },
            "road_polygon": _plain_points(self.road_polygon),
            "counting_line": _plain_points(self.counting_line),
            "lanes": [
                {"name": lane.name, "polygon": _plain_points(lane.polygon)}
                for lane in self.lanes
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
