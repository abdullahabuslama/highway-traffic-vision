"""Measures this camera once and writes its ``config/camera_*.yaml``.

Every clip in the dataset comes from the same fixed camera, so this runs once and
the result is reused for all 254. It needs no clicking and no hand-drawn
polygons: the road is found from where the traffic actually goes.

What it does, in order:

1. Adds up frame-to-frame change over a sample of clips. Lanes show up as bright
   bands, because that is where vehicles pass. Reading the bands off row by row
   gives the lane centres, and from those, the road outline and the lane
   polygons.
2. Tracks corner features through the light-traffic clips and fits the horizon
   row to how their apparent speed grows down the frame.
3. Uses the lane spacing, now that the horizon is known, to get the camera height.
4. Writes the config with a provisional zoom setting, runs the finished pipeline
   over some light-traffic clips, and rescales the zoom until those clips report
   free-flow speed. That last number is an assumption rather than a measurement,
   and :mod:`traffic_flow.geometry.pinhole_road` explains why it has to be.

Run::

    python scripts/calibrate_camera.py --dataset dataset --out config/camera_i5.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from traffic_flow.config import CameraConfig, LaneConfig, PipelineConfig
from traffic_flow.data.catalog import ClipCatalog, load_catalog
from traffic_flow.detection.yolo import YoloDetector
from traffic_flow.geometry.pinhole_road import (
    FREE_FLOW_MPS,
    US_LANE_WIDTH_M,
    GroundGeometry,
    RoadCameraModel,
    fit_ground_geometry,
    fit_horizon_row,
    focal_for_target_speed,
)
from traffic_flow.geometry.road_survey import (
    LaneBands,
    empty_road_image,
    find_lane_bands,
    lane_polygons,
    motion_energy,
    sample_optical_flow,
)
from traffic_flow.labels import CongestionLevel
from traffic_flow.metrics.speed import median_speed_kph
from traffic_flow.pipeline import TrafficPipeline

#: Rows probed for lane bands. They stop short of the very bottom, where vehicles
#: are large enough to merge into each other, and short of the very top, where
#: the lanes are only a couple of pixels apart.
PROBE_ROWS = tuple(range(80, 225, 10))

#: Columns holding the southbound carriageway. Everything left of this is the
#: opposite carriageway, which travels the other way and must not be measured.
CARRIAGEWAY_COLUMNS = (130, 320)

#: How many lanes this carriageway has. Checked against the bands that are found.
LANE_COUNT = 5

#: Rows the analysis works over. The road outline and lanes are drawn across it.
WORKING_ROWS = (95, 232)

#: Where to count vehicles crossing. Low in the frame, where they are biggest.
COUNTING_ROW = 195

#: Focal length used for the first pass, before it is corrected. Any value works,
#: because reported speed is exactly proportional to it.
PROVISIONAL_FOCAL_PX = 1000.0


def _light_clips(catalog: ClipCatalog, count: int) -> list[str]:
    light = catalog.clips[catalog.clips["label"] == CongestionLevel.LIGHT]
    return list(light.head(count).video_path)


def _counting_line(road: np.ndarray, row: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """The segment across the carriageway at one image row."""
    on_row = road[np.abs(road[:, 1] - row) < 3.0]
    if len(on_row) < 2:
        raise ValueError(f"the road outline does not reach row {row}")
    return (
        (float(on_row[:, 0].min()), float(row)),
        (float(on_row[:, 0].max()), float(row)),
    )


def build_camera(
    model: RoadCameraModel,
    bands: LaneBands,
    image_size: tuple[int, int],
    notes: str,
) -> CameraConfig:
    """Assemble the config: where the road is, and how its pixels become metres."""
    polygons, road = lane_polygons(bands.fit_lane_curves(), row_range=WORKING_ROWS)
    image_points, world_points = model.homography_pairs(
        rows=(float(WORKING_ROWS[0]), float(WORKING_ROWS[1])),
        columns=(float(CARRIAGEWAY_COLUMNS[0]), float(CARRIAGEWAY_COLUMNS[1] - 1)),
    )
    return CameraConfig(
        name="I-5 south at S 188th St, Seattle (UCSD traffic database)",
        image_size=image_size,
        image_points=tuple(map(tuple, image_points)),
        world_points=tuple(map(tuple, world_points)),
        road_polygon=tuple(map(tuple, road)),
        counting_line=_counting_line(road, COUNTING_ROW),
        lanes=tuple(
            LaneConfig(name=f"lane_{i + 1}", polygon=tuple(map(tuple, polygon)))
            for i, polygon in enumerate(polygons)
        ),
        scale_notes=notes,
    )


def measure_light_clip_speed(camera: CameraConfig, clips: list[str]) -> float:
    """Run the finished pipeline over some light clips and report their speed.

    The median over every vehicle measured in every clip. The median is used
    because a light clip still contains the odd slow lorry, and one of those
    should not move the calibration.
    """
    pipeline_config = PipelineConfig.load()
    pipeline = TrafficPipeline(YoloDetector(pipeline_config.detector), camera, pipeline_config)
    speeds = [speed for path in clips for speed in pipeline.analyse(path).speeds]
    return median_speed_kph(speeds)


def _scale_notes(geometry: GroundGeometry, focal: float, target_kph: float) -> str:
    return (
        f"Measured by scripts/calibrate_camera.py. Horizon row {geometry.horizon_row:.1f} is "
        f"fitted to how apparent motion grows down the frame (fit quality "
        f"{geometry.motion_fit_quality:.3f}). Camera height {geometry.camera_height_m:.1f} m "
        f"follows from lane spacing, assuming {US_LANE_WIDTH_M:.3f} m US interstate lanes "
        f"(spread between rows {geometry.height_spread_m:.2f} m). Focal length {focal:.0f} px is "
        f"NOT measured: it is set so light-traffic clips report {target_kph:.0f} km/h, because "
        f"the lane dash markings are not resolvable at 320x240. Absolute speeds scale with that "
        f"choice and densities scale inversely; relative results do not depend on it. World units "
        f"are metres: x is sideways from the camera centre line, y is distance from the camera."
    )


def _draw_overlay(
    background: np.ndarray,
    camera: CameraConfig,
    model: RoadCameraModel,
    out_path: Path,
    scale: int = 3,
) -> None:
    """Save a picture of what was calibrated, to be checked by eye."""
    canvas = cv2.resize(
        background,
        (background.shape[1] * scale, background.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    for metres in range(40, 220, 20):
        row = model.horizon_row + model.depth_constant / metres
        if not 0 < row < background.shape[0]:
            continue
        y = int(row * scale)
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (90, 90, 90), 1)
        cv2.putText(canvas, f"{metres} m", (6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)

    cv2.polylines(
        canvas, [(camera.road_polygon_array * scale).astype(np.int32)], True, (0, 255, 255), 2
    )
    for lane in camera.lanes:
        cv2.polylines(
            canvas, [(lane.polygon_array * scale).astype(np.int32)], True, (0, 180, 255), 1
        )
    (x1, y1), (x2, y2) = camera.counting_line
    cv2.line(canvas, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)),
             (255, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset", help="dataset directory")
    parser.add_argument("--out", default="config/camera_i5.yaml", help="where to write the config")
    parser.add_argument("--overlay", default="outputs/calibration_overlay.png")
    parser.add_argument("--motion-clips", type=int, default=60)
    parser.add_argument("--flow-clips", type=int, default=40)
    parser.add_argument("--speed-clips", type=int, default=25)
    parser.add_argument("--free-flow-kph", type=float, default=FREE_FLOW_MPS * 3.6)
    args = parser.parse_args()

    catalog = load_catalog(args.dataset)
    print(f"catalogue: {len(catalog)} clips")

    motion_clips = list(
        catalog.clips.sample(min(args.motion_clips, len(catalog)), random_state=0).video_path
    )
    print(f"1. finding the lanes from where traffic goes, over {len(motion_clips)} clips ...")
    motion = motion_energy(motion_clips)
    bands = find_lane_bands(
        motion, rows=PROBE_ROWS, lane_count=LANE_COUNT, column_range=CARRIAGEWAY_COLUMNS
    )
    lane_rows, lane_gaps = bands.mean_gaps()
    print(
        f"   {LANE_COUNT} lanes in {len(lane_rows)} rows, "
        f"spacing {lane_gaps.min():.1f}-{lane_gaps.max():.1f} px"
    )

    flow_clips = _light_clips(catalog, args.flow_clips)
    print(f"2. fitting the horizon from apparent motion, over {len(flow_clips)} light clips ...")
    points, moves = sample_optical_flow(
        flow_clips, row_range=WORKING_ROWS, column_range=CARRIAGEWAY_COLUMNS
    )
    horizon, quality = fit_horizon_row(points[:, 1], np.linalg.norm(moves, axis=1))
    print(f"   {len(points)} samples -> horizon row {horizon:.1f}, fit quality {quality:.3f}")

    print("3. camera height from lane spacing ...")
    geometry = fit_ground_geometry(horizon, quality, lane_rows, lane_gaps)
    print(
        f"   {geometry.camera_height_m:.1f} m ({geometry.camera_height_m * 3.281:.0f} ft), "
        f"spread between rows {geometry.height_spread_m:.2f} m"
    )

    image_size = (motion.shape[1], motion.shape[0])
    principal_column = image_size[0] / 2.0
    provisional = RoadCameraModel(geometry, PROVISIONAL_FOCAL_PX, principal_column)
    camera = build_camera(provisional, bands, image_size, notes="provisional, being calibrated")

    speed_clips = _light_clips(catalog, args.speed_clips)
    print(f"4. calibrating the zoom against {len(speed_clips)} light clips ...")
    observed = measure_light_clip_speed(camera, speed_clips)
    focal = focal_for_target_speed(PROVISIONAL_FOCAL_PX, observed, args.free_flow_kph)
    print(f"   at {PROVISIONAL_FOCAL_PX:.0f} px they read {observed:.1f} km/h "
          f"-> focal length {focal:.0f} px")

    model = RoadCameraModel(geometry, focal, principal_column)
    camera = build_camera(
        model, bands, image_size, _scale_notes(geometry, focal, args.free_flow_kph)
    )

    field_of_view = 2 * np.degrees(np.arctan(principal_column / focal))
    print(
        f"   horizontal field of view {field_of_view:.0f} deg, road visible from "
        f"{model.depth_m(WORKING_ROWS[1]):.0f} m to {model.depth_m(WORKING_ROWS[0]):.0f} m"
    )

    print("5. checking the corrected calibration ...")
    achieved = measure_light_clip_speed(camera, speed_clips)
    print(f"   light clips now read {achieved:.1f} km/h (target {args.free_flow_kph:.0f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    camera.save(out)
    print(f"wrote {out}")

    _draw_overlay(empty_road_image(_light_clips(catalog, 8)), camera, model, Path(args.overlay))
    print(f"wrote {args.overlay}")


if __name__ == "__main__":
    main()
