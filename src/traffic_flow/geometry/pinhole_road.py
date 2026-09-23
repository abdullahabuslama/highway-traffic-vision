"""The physical camera model used to calibrate this road, once, by measurement.

Only the calibration step uses this file. At run time the pipeline maps pixels to
metres with the plain homography in :mod:`traffic_flow.geometry.homography`; what
this model does is *produce* that homography's point pairs from things that can
actually be measured in the video.

The model is the standard one for a camera on a pole looking at flat ground. Put
the camera at height ``h`` and let ``v_h`` be the image row of the horizon. Then a
point on the ground at image row ``v`` lies at depth

    Y = f * h / (v - v_h)

and its sideways position is ``X = (u - u0) * Y / f``. Three numbers describe the
whole thing: the horizon row, the camera height, and the focal length in pixels.

Two of the three can be measured from this footage, and one cannot:

- **the horizon row** comes from how the traffic moves. A vehicle at steady speed
  covers the same number of metres every frame, but not the same number of
  pixels: it crawls across the far end of the picture and races across the near
  end. The model says that pixel step grows with the square of the distance below
  the horizon, so plotting the square root of the step against image row gives a
  straight line that crosses zero exactly at the horizon. Tens of thousands of
  measurements go into that line, which matters, because the far end of the road
  is very sensitive to this number.
- **camera height** comes from the lane markings. Lanes are a fixed 3.66 m apart,
  so the number of pixels between two lane centres says how far away that part of
  the road is. With the horizon already known, that fixes the height.
- **focal length** cannot be measured. It is the zoom, and nothing in a picture of
  a road reveals it unless something of known length lies *along* the road. The
  lane dash markings would do it - they repeat every 12.2 m on a US interstate -
  but at 320x240 with worn paint they are not recoverable, which was checked
  rather than assumed.

So the focal length is fixed by the one thing the dataset does tell us: clips
labelled ``light`` are, by the dataset's own definition, free-flowing traffic.
Free flow on this road, signed at 60 mph, is about 100 km/h. The focal length is
set so that the speeds the finished pipeline reports for the light clips come out
at that figure. Calibrating against the pipeline's own output, rather than
against some separate measurement, is what keeps the assumption honest: there is
exactly one number being assumed, and it is the one being reported.

What that means for the results, stated plainly: **the absolute speed scale rests
on that assumption, and the density scale moves with it in the opposite
direction.** Everything else - which clips are faster than which, the shape of the
speed-density curve, and every classification result - is untouched by it,
because it is a single multiplier applied to every clip alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

#: Lane width on a US interstate, 12 feet.
US_LANE_WIDTH_M = 3.6576

#: Free-flow speed assumed for the ``light`` clips, in metres per second.
#: I-5 is signed at 60 mph; free-flowing traffic runs a little above the limit.
FREE_FLOW_MPS = 27.8


@dataclass(frozen=True)
class GroundGeometry:
    """The part of the model that the lane markings determine.

    ``horizon_row`` is where the road would vanish into the distance, in image
    rows. ``camera_height_m`` is how far the camera is above the road surface.
    """

    horizon_row: float
    camera_height_m: float
    #: How well the motion measurements fitted the straight line the model
    #: predicts, from 0 to 1. Near 1 means the flat road model really does
    #: describe this camera.
    motion_fit_quality: float
    #: Spread of the camera height estimates from the individual lane-gap rows,
    #: in metres. Small means every part of the road agrees about the height.
    height_spread_m: float

    def lane_gap_px(self, row: float) -> float:
        """How many pixels apart two lane centres appear at this image row."""
        return US_LANE_WIDTH_M * (row - self.horizon_row) / self.camera_height_m


@dataclass(frozen=True)
class RoadCameraModel:
    """The complete camera model: ground geometry plus the zoom."""

    geometry: GroundGeometry
    focal_px: float
    principal_column: float

    @property
    def horizon_row(self) -> float:
        return self.geometry.horizon_row

    @property
    def camera_height_m(self) -> float:
        return self.geometry.camera_height_m

    @property
    def depth_constant(self) -> float:
        """``f * h``, the number that converts image row into depth."""
        return self.focal_px * self.camera_height_m

    def depth_m(self, rows: np.ndarray | float) -> np.ndarray:
        """How far ahead the ground is at these image rows, in metres."""
        rows = np.asarray(rows, dtype=np.float64)
        below_horizon = np.maximum(rows - self.horizon_row, 1e-6)
        return self.depth_constant / below_horizon

    def to_world(self, image_points: np.ndarray) -> np.ndarray:
        """Turn image points on the road into metres, shape (n, 2).

        The result is ``(sideways, depth)``: sideways is positive to the right of
        the camera's centre line, depth is distance away from the camera.
        """
        points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        depth = self.depth_m(points[:, 1])
        sideways = (points[:, 0] - self.principal_column) * depth / self.focal_px
        return np.column_stack([sideways, depth])

    def to_image(self, world_points: np.ndarray) -> np.ndarray:
        """The reverse: metres on the road back to image pixels."""
        world = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
        depth = np.maximum(world[:, 1], 1e-6)
        rows = self.horizon_row + self.depth_constant / depth
        columns = self.principal_column + world[:, 0] * self.focal_px / depth
        return np.column_stack([columns, rows])

    def homography_pairs(
        self, rows: tuple[float, float], columns: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Four matching image and world points, for the run-time homography.

        The four are the corners of the box formed by two image rows and two
        image columns. A homography fitted through them reproduces this model
        exactly, because the model is itself a projective map of a plane.
        """
        image_points = np.array(
            [
                [columns[0], rows[0]],
                [columns[1], rows[0]],
                [columns[1], rows[1]],
                [columns[0], rows[1]],
            ],
            dtype=np.float64,
        )
        return image_points, self.to_world(image_points)


#: Rows are grouped into bands this many pixels tall before the horizon is
#: fitted, so that a crowded part of the frame does not outvote a sparse one.
MOTION_BAND_PX = 10

#: Motion is summarised per band by this percentile. The median is used rather
#: than the mean because a band always holds some near-stationary readings from
#: background texture, and they would drag a mean down.
MOTION_PERCENTILE = 50.0


def fit_horizon_row(
    rows: np.ndarray,
    displacements_px: np.ndarray,
    band_px: int = MOTION_BAND_PX,
    percentile: float = MOTION_PERCENTILE,
) -> tuple[float, float]:
    """Find the horizon row from how fast things appear to move down the frame.

    At a steady speed the pixel step grows with the square of the distance below
    the horizon, so the square root of the step is a straight line in image row
    that hits zero at the horizon. Fitting the square root rather than the step
    itself is what turns this into an ordinary straight-line fit.

    Returns the horizon row and how well the line fitted, from 0 to 1.
    """
    rows = np.asarray(rows, dtype=np.float64)
    steps = np.asarray(displacements_px, dtype=np.float64)
    if len(rows) < 100:
        raise ValueError("need a few hundred motion measurements to fit the horizon")

    band_centres, band_steps = _summarise_bands(rows, steps, band_px, percentile)
    if len(band_centres) < 4:
        raise ValueError("motion measurements do not span enough of the frame")

    root_steps = np.sqrt(band_steps)
    slope, intercept = np.polyfit(band_centres, root_steps, 1)
    if slope <= 0:
        raise ValueError("apparent motion must grow towards the bottom of the frame")

    predicted = slope * band_centres + intercept
    quality = 1.0 - float(
        np.sum((root_steps - predicted) ** 2) / np.sum((root_steps - root_steps.mean()) ** 2)
    )
    return float(-intercept / slope), quality


def _summarise_bands(
    rows: np.ndarray, values: np.ndarray, band_px: int, percentile: float
) -> tuple[np.ndarray, np.ndarray]:
    """One representative value per horizontal band of the image."""
    edges = np.arange(rows.min(), rows.max() + band_px, band_px)
    centres: list[float] = []
    summaries: list[float] = []
    for low, high in pairwise(edges):
        inside = (rows >= low) & (rows < high)
        if inside.sum() < 30:
            continue
        centres.append(float((low + high) / 2))
        summaries.append(float(np.percentile(values[inside], percentile)))
    return np.array(centres), np.array(summaries)


def fit_ground_geometry(
    horizon_row: float,
    horizon_quality: float,
    lane_rows: np.ndarray,
    lane_gaps_px: np.ndarray,
    lane_width_m: float = US_LANE_WIDTH_M,
) -> GroundGeometry:
    """Add the camera height to an already-known horizon row.

    With the horizon fixed, every row of lane-gap measurements gives its own
    estimate of the height, through ``height = lane_width * (row - horizon) /
    gap``. They are averaged, and how much they disagree is reported: a wide
    spread would mean the road is not as flat as the model assumes.
    """
    rows = np.asarray(lane_rows, dtype=np.float64)
    gaps = np.asarray(lane_gaps_px, dtype=np.float64)
    if len(rows) < 3:
        raise ValueError("need at least three rows of lane-gap measurements")

    heights = lane_width_m * (rows - horizon_row) / gaps
    return GroundGeometry(
        horizon_row=float(horizon_row),
        camera_height_m=float(np.median(heights)),
        motion_fit_quality=float(horizon_quality),
        height_spread_m=float(heights.std()),
    )


def focal_for_target_speed(
    provisional_focal_px: float,
    observed_kph: float,
    target_kph: float = FREE_FLOW_MPS * 3.6,
) -> float:
    """Rescale the focal length so the reported speed hits its target.

    Reported speed is exactly proportional to the focal length - every distance
    on the road plane scales with it - so this is one multiplication, not a
    search. Run the pipeline once at any focal length, see what speed the light
    clips give, and correct it.
    """
    # "Not a number" fails every comparison, so ``observed_kph <= 0`` alone would
    # wave an empty sample through and hand back a focal length of NaN.
    if not np.isfinite(observed_kph) or observed_kph <= 0:
        raise ValueError("no usable speed was measured, cannot set the focal length")
    return provisional_focal_px * target_kph / observed_kph
