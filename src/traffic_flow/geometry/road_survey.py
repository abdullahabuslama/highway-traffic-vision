"""Measures where the road is, and how fast things move on it, from sample clips.

Calibration needs four things out of the raw video, and this module produces all
four. Nothing here decides anything - it measures, and
:mod:`traffic_flow.geometry.pinhole_road` turns the measurements into a camera.

**The empty road.** Take the middle value of every pixel across hundreds of
frames. Anything that moved is gone and the bare road is left, which is what you
want to look at when you are trying to see the markings under the traffic.

**Where the traffic goes.** Add up how much each pixel changes from one frame to
the next, over many clips. Lanes light up as bright bands, because that is where
vehicles pass. This finds the lanes far more reliably than the paint does: the
markings on this road are worn and the video is 320x240, but the traffic itself
is unmistakable.

**The lane centres.** The bright bands, row by row. Their spacing is what the
camera height is derived from.

**How far things move per frame.** Sparse optical flow on the light-traffic clips.
This is what the focal length is derived from.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from traffic_flow.io.video_source import VideoSource

#: Optical-flow settings. The window is large relative to a 320x240 frame because
#: the vehicles are small and low contrast.
_FLOW_PARAMS = {
    "winSize": (15, 15),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
}

#: Movements smaller than this are camera noise rather than traffic.
MIN_FLOW_PX = 0.8


def empty_road_image(video_paths: list[str | Path]) -> np.ndarray:
    """The road with the traffic removed, as the median of every frame.

    Pass several clips. More frames means more chance that every patch of road is
    empty in at least half of them, which is what the median needs in order to
    erase a vehicle that sat still through one whole clip.
    """
    frames: list[np.ndarray] = []
    for path in video_paths:
        with VideoSource(path) as source:
            frames.extend(frame.image for frame in source)
    if not frames:
        raise ValueError("no frames to build a background from")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def motion_energy(video_paths: list[str | Path]) -> np.ndarray:
    """Average frame-to-frame change per pixel, as a float image.

    Bright means "things move here", which on a fixed camera means the lanes.
    """
    total: np.ndarray | None = None
    pairs = 0
    for path in video_paths:
        previous: np.ndarray | None = None
        with VideoSource(path) as source:
            for frame in source:
                grey = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if previous is not None:
                    difference = np.abs(grey - previous)
                    total = difference if total is None else total + difference
                    pairs += 1
                previous = grey
    if total is None or pairs == 0:
        raise ValueError("need at least two frames to measure motion")
    return total / pairs


@dataclass(frozen=True)
class LaneBands:
    """Where the lanes sit, measured row by row.

    ``centres[row]`` holds the image column of each lane centre visible in that
    row, left to right. A row only appears if every lane was found in it, so the
    gaps between them can be compared fairly.
    """

    rows: tuple[int, ...]
    centres: dict[int, tuple[float, ...]]

    @property
    def lane_count(self) -> int:
        return len(next(iter(self.centres.values())))

    def mean_gaps(self) -> tuple[np.ndarray, np.ndarray]:
        """Rows, and the average spacing between neighbouring lanes in each."""
        rows = np.array(self.rows, dtype=np.float64)
        gaps = np.array(
            [float(np.mean(np.diff(self.centres[row]))) for row in self.rows], dtype=np.float64
        )
        return rows, gaps

    def fit_lane_curves(self, degree: int = 2) -> list[np.ndarray]:
        """A smooth curve per lane centre, as polynomial coefficients in row.

        Second degree, because this stretch of road bends. A straight line would
        drift off the lane by a dozen pixels at the ends, which is most of a lane
        width in the distance.
        """
        curves = []
        for lane in range(self.lane_count):
            rows = [row for row in self.rows if lane < len(self.centres[row])]
            columns = [self.centres[row][lane] for row in rows]
            curves.append(np.polyfit(rows, columns, degree))
        return curves


def find_lane_bands(
    motion: np.ndarray,
    rows: tuple[int, ...],
    lane_count: int,
    column_range: tuple[int, int],
    smoothing: float = 1.0,
) -> LaneBands:
    """Find the bright bands traffic leaves in the motion image.

    Only rows where exactly ``lane_count`` bands are visible are kept. A row that
    shows four bands instead of five usually means one lane has run off the side
    of the frame, and its gaps would be measured between the wrong pair.
    """
    smoothed = cv2.GaussianBlur(motion.astype(np.float32), (0, 0), smoothing)
    low, high = column_range
    centres: dict[int, tuple[float, ...]] = {}

    for row in rows:
        strip = smoothed[row, low:high]
        peaks, _ = find_peaks(
            strip,
            height=strip.max() * 0.35,
            distance=3,
            prominence=strip.max() * 0.08,
        )
        if len(peaks) == lane_count:
            centres[row] = tuple(float(low + peak) for peak in peaks)

    if len(centres) < 3:
        raise ValueError(
            f"found {lane_count} lanes in only {len(centres)} rows; "
            "check the row list and column range"
        )
    return LaneBands(rows=tuple(sorted(centres)), centres=centres)


def sample_optical_flow(
    video_paths: list[str | Path],
    row_range: tuple[int, int],
    column_range: tuple[int, int],
    max_corners: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Track corner features one frame to the next and report how far they moved.

    Returns where each feature was, shape (n, 2), and how far it moved in one
    frame, also shape (n, 2). Only movements inside the given window are kept, so
    the opposite carriageway and the roadside do not contribute.
    """
    starts: list[np.ndarray] = []
    moves: list[np.ndarray] = []
    row_low, row_high = row_range
    column_low, column_high = column_range

    for path in video_paths:
        with VideoSource(path) as source:
            frames = [cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY) for frame in source]

        for first, second in pairwise(frames):
            corners = cv2.goodFeaturesToTrack(
                first, maxCorners=max_corners, qualityLevel=0.01, minDistance=4, blockSize=5
            )
            if corners is None:
                continue
            moved, status, _ = cv2.calcOpticalFlowPyrLK(first, second, corners, None, **_FLOW_PARAMS)
            if moved is None:
                continue

            found = status.ravel() == 1
            before = corners[found].reshape(-1, 2)
            after = moved[found].reshape(-1, 2)
            delta = after - before

            keep = (
                (np.linalg.norm(delta, axis=1) > MIN_FLOW_PX)
                & (before[:, 1] >= row_low)
                & (before[:, 1] <= row_high)
                & (before[:, 0] >= column_low)
                & (before[:, 0] <= column_high)
            )
            starts.append(before[keep])
            moves.append(delta[keep])

    if not starts:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.vstack(starts), np.vstack(moves)


def lane_polygons(
    curves: list[np.ndarray],
    row_range: tuple[int, int],
    step: int = 5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Turn the lane centre curves into one polygon per lane, plus the whole road.

    A lane's edges are drawn halfway to its neighbours. The two outermost edges
    are half a lane beyond the outer centres, since there is no neighbour there to
    split the difference with.
    """
    rows = np.arange(row_range[0], row_range[1] + 1, step, dtype=np.float64)
    centres = np.column_stack([np.polyval(curve, rows) for curve in curves])

    half_gaps = np.diff(centres, axis=1) / 2.0
    edges = np.empty((len(rows), centres.shape[1] + 1))
    edges[:, 1:-1] = centres[:, :-1] + half_gaps
    edges[:, 0] = centres[:, 0] - half_gaps[:, 0]
    edges[:, -1] = centres[:, -1] + half_gaps[:, -1]

    polygons = [
        _strip_polygon(rows, edges[:, lane], edges[:, lane + 1])
        for lane in range(centres.shape[1])
    ]
    road = _strip_polygon(rows, edges[:, 0], edges[:, -1])
    return polygons, road


def _strip_polygon(rows: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Close a strip: down one edge, back up the other."""
    down = np.column_stack([left, rows])
    up = np.column_stack([right, rows])[::-1]
    return np.vstack([down, up])
