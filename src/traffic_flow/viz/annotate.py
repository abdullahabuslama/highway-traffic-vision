"""Draws the analysis back onto the video.

The source frames are 320x240, which is too small to put a readable label on, so
every frame is enlarged before anything is drawn. All the coordinates coming out
of the pipeline are in original-frame pixels and get scaled on the way in, which
keeps the scaling in this one file and out of the measurement code.

What ends up on screen, and why each piece is there:

- the road outline and lane lines, so a viewer can see what the numbers measured;
- a box per vehicle, coloured by kind, labelled with its speed;
- the counting line, and a running total of what has crossed it;
- a panel with the clip's speed, density and congestion level.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from traffic_flow.config import CameraConfig
from traffic_flow.io.video_source import VideoSource
from traffic_flow.labels import CongestionLevel, VehicleClass
from traffic_flow.metrics.counting import Line, crosses_line, side_of_line
from traffic_flow.metrics.speed import describe_speed
from traffic_flow.pipeline import ClipAnalysis
from traffic_flow.tracking.tracks import Observation

#: How much to enlarge each frame before drawing. 3x turns 320x240 into 960x720.
DRAW_SCALE = 3

#: Colours in BGR, the order OpenCV uses.
CLASS_COLOURS: dict[VehicleClass, tuple[int, int, int]] = {
    VehicleClass.CAR: (120, 220, 90),
    VehicleClass.TRUCK: (60, 160, 255),
    VehicleClass.BUS: (255, 180, 60),
    VehicleClass.MOTORCYCLE: (230, 120, 255),
}

#: Green for free flow, amber for slowing, red for stopped.
LEVEL_COLOURS: dict[CongestionLevel, tuple[int, int, int]] = {
    CongestionLevel.LIGHT: (110, 210, 110),
    CongestionLevel.MEDIUM: (60, 200, 255),
    CongestionLevel.HEAVY: (70, 70, 240),
}

ROAD_COLOUR = (200, 200, 200)
LANE_COLOUR = (150, 150, 150)
LINE_COLOUR = (255, 255, 255)
PANEL_BACKGROUND = (28, 28, 28)
PANEL_TEXT = (240, 240, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class AnnotationStyle:
    """Sizes that depend on the output resolution, kept in one place."""

    scale: int = DRAW_SCALE
    box_thickness: int = 2
    font_scale: float = 0.45
    panel_font_scale: float = 0.5
    panel_height: int = 112
    #: Boxes narrower than this get no speed label. Far down the road a vehicle
    #: is a few pixels wide and its label would cover its neighbours.
    min_label_width_px: int = 26


class ClipAnnotator:
    """Redraws one clip with its analysis on top."""

    def __init__(self, camera: CameraConfig, style: AnnotationStyle | None = None):
        self._camera = camera
        self._style = style or AnnotationStyle()

    def render(
        self,
        analysis: ClipAnalysis,
        video_path: str | Path,
        out_path: str | Path,
        level: CongestionLevel | None = None,
    ) -> Path:
        """Write an annotated copy of the clip and return where it was written.

        ``level`` is the congestion label to show in the panel. It comes from the
        classifier, which sits outside the pipeline, so it is passed in rather
        than looked up here.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        by_frame = _observations_by_frame(analysis)
        speed_of = {speed.track_id: speed.kph for speed in analysis.speeds}
        crossing_ids = _crossing_ids(analysis, self._camera.counting_line)
        scale = self._style.scale

        with VideoSource(video_path) as source:
            size = (source.meta.width * scale, source.meta.height * scale + self._style.panel_height)
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), source.meta.fps, size
            )
            if not writer.isOpened():
                raise OSError(f"cannot write video to {out_path}")

            try:
                crossed_so_far = 0
                seen_crossing: set[int] = set()
                for frame in source:
                    canvas = cv2.resize(
                        frame.image,
                        (source.meta.width * scale, source.meta.height * scale),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    self._draw_road(canvas)

                    observations = by_frame.get(frame.index, [])
                    for track_id, observation in observations:
                        self._draw_vehicle(canvas, track_id, observation, speed_of.get(track_id))
                        if (
                            track_id in crossing_ids
                            and track_id not in seen_crossing
                            and _has_passed_line(observation, self._camera.counting_line)
                        ):
                            seen_crossing.add(track_id)
                            crossed_so_far += 1

                    writer.write(
                        self._with_panel(canvas, analysis, level, crossed_so_far, len(observations))
                    )
            finally:
                writer.release()

        return out_path

    def _draw_road(self, canvas: np.ndarray) -> None:
        scale = self._style.scale
        road = (self._camera.road_polygon_array * scale).astype(np.int32)
        cv2.polylines(canvas, [road], isClosed=True, color=ROAD_COLOUR, thickness=1)

        for lane in self._camera.lanes:
            polygon = (lane.polygon_array * scale).astype(np.int32)
            cv2.polylines(canvas, [polygon], isClosed=True, color=LANE_COLOUR, thickness=1)

        (x1, y1), (x2, y2) = self._camera.counting_line
        cv2.line(
            canvas,
            (int(x1 * scale), int(y1 * scale)),
            (int(x2 * scale), int(y2 * scale)),
            LINE_COLOUR,
            2,
        )

    def _draw_vehicle(
        self,
        canvas: np.ndarray,
        track_id: int,
        observation: Observation,
        kph: float | None,
    ) -> None:
        scale = self._style.scale
        x1, y1, x2, y2 = (int(value * scale) for value in observation.xyxy)
        colour = CLASS_COLOURS.get(observation.vehicle_class, (200, 200, 200))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, self._style.box_thickness)

        # Speed only. In queued traffic there can be thirty vehicles on screen at
        # once, and a label carrying the id and the vehicle kind as well is wider
        # than the vehicle it belongs to - the labels then cover each other and
        # the traffic underneath. The kind is already in the box colour, which
        # the panel spells out.
        if kph is None or (x2 - x1) < self._style.min_label_width_px:
            return
        _draw_label(canvas, f"{kph:.0f}", (x1, y1), colour, self._style.font_scale)

    def _with_panel(
        self,
        canvas: np.ndarray,
        analysis: ClipAnalysis,
        level: CongestionLevel | None,
        crossed: int,
        in_view: int,
    ) -> np.ndarray:
        """Add the summary strip under the picture."""
        height = self._style.panel_height
        panel = np.full((height, canvas.shape[1], 3), PANEL_BACKGROUND, dtype=np.uint8)
        font_scale = self._style.panel_font_scale

        if level is not None:
            colour = LEVEL_COLOURS[level]
            cv2.rectangle(panel, (0, 0), (10, height), colour, -1)
            cv2.putText(
                panel, f"{str(level).upper()} TRAFFIC", (22, 26), FONT, 0.7, colour, 2
            )

        lines = [
            f"speed {describe_speed(analysis.space_mean_speed_kph, decimals=1)}",
            f"density {analysis.density.veh_per_km_per_lane:5.1f} veh/km/lane",
            f"occupancy {analysis.density.mean_occupancy * 100:4.1f}%",
        ]
        for i, text in enumerate(lines):
            cv2.putText(panel, text, (22 + i * 230, 52), FONT, font_scale, PANEL_TEXT, 1)

        counters = (
            f"in view {in_view:2d}   crossed {crossed:2d}   "
            f"seen {analysis.counts.vehicles_seen:2d}   "
            f"flow {analysis.counts.flow_veh_per_hour:5.0f} veh/h"
        )
        cv2.putText(panel, counters, (22, 78), FONT, font_scale, PANEL_TEXT, 1)

        # The box colours say which kind of vehicle each one is, so the panel has
        # to say what the colours mean. Numbers on the boxes are speeds in km/h.
        cv2.putText(panel, "boxes:", (22, 102), FONT, 0.42, PANEL_TEXT, 1)
        offset = 78
        for vehicle, colour in CLASS_COLOURS.items():
            cv2.rectangle(panel, (offset, 92), (offset + 14, 104), colour, -1)
            cv2.putText(panel, str(vehicle), (offset + 20, 102), FONT, 0.42, PANEL_TEXT, 1)
            offset += 26 + len(str(vehicle)) * 9
        cv2.putText(panel, "number on a box = km/h", (offset + 16, 102), FONT, 0.42, PANEL_TEXT, 1)

        return np.vstack([canvas, panel])


def _draw_label(
    canvas: np.ndarray,
    text: str,
    top_left: tuple[int, int],
    colour: tuple[int, int, int],
    font_scale: float,
) -> None:
    """Write text on a filled strip so it stays readable over any background."""
    x, y = top_left
    (width, height), baseline = cv2.getTextSize(text, FONT, font_scale, 1)
    top = max(y - height - baseline - 2, 0)
    cv2.rectangle(canvas, (x, top), (x + width + 6, top + height + baseline + 4), colour, -1)
    cv2.putText(canvas, text, (x + 3, top + height + 2), FONT, font_scale, (20, 20, 20), 1)


def _observations_by_frame(analysis: ClipAnalysis) -> dict[int, list[tuple[int, Observation]]]:
    """Group the stored observations by the frame they came from."""
    grouped: dict[int, list[tuple[int, Observation]]] = defaultdict(list)
    for track in analysis.confirmed_tracks:
        for observation in track.observations:
            grouped[observation.frame_index].append((track.track_id, observation))
    return grouped


def _crossing_ids(analysis: ClipAnalysis, line: Line) -> set[int]:
    """Which vehicles cross the counting line at some point in the clip."""
    return {
        track.track_id for track in analysis.confirmed_tracks if crosses_line(track, line)
    }


def _has_passed_line(observation: Observation, line: Line) -> bool:
    """Is this vehicle already on the far side of the counting line?

    Used only to decide when the on-screen counter ticks up, so the number a
    viewer sees matches the moment the vehicle visibly crosses. The side test is
    the counting module's, so the video and the totals cannot disagree.
    """
    return side_of_line(np.asarray(observation.ground_xy), line) > 0
