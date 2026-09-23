"""Writes a reference picture of the road, with a pixel grid drawn over it.

Calibration needs pixel coordinates read off the road, and reading them off a
320x240 frame by eye is guesswork. This enlarges the picture, draws a labelled
grid on it, and saves it. The grid numbers are coordinates in the *original*
frame, so what you read here is what goes into ``config/camera_*.yaml``.

By default the picture is the median of every frame in the clip rather than a
single frame. Taking the middle value at each pixel across the whole clip erases
anything that moved and keeps what stayed still, so the vehicles disappear and
the empty road is left behind with its lane markings unobstructed. That is much
easier to calibrate against than a frame with cars parked on the very lines you
are trying to measure.

Run it once per camera::

    python scripts/export_reference_frame.py --video <clip.avi> --out outputs/reference.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from traffic_flow.geometry.road_survey import empty_road_image
from traffic_flow.io.video_source import read_reference_frame

#: How much to enlarge the frame. The grid labels need room to be readable.
SCALE = 4
#: Grid spacing in original-frame pixels.
GRID_STEP = 20

GRID_COLOUR = (0, 255, 255)
MAJOR_COLOUR = (0, 128, 255)
LABEL_COLOUR = (255, 255, 255)


def draw_grid(image: np.ndarray, scale: int = SCALE, step: int = GRID_STEP) -> np.ndarray:
    """Enlarge a frame and draw a coordinate grid on the enlarged copy."""
    height, width = image.shape[:2]
    canvas = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)

    for x in range(0, width + 1, step):
        major = x % (step * 5) == 0
        cv2.line(
            canvas,
            (x * scale, 0),
            (x * scale, height * scale),
            MAJOR_COLOUR if major else GRID_COLOUR,
            2 if major else 1,
        )
        if major:
            cv2.putText(
                canvas, str(x), (x * scale + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOUR, 1
            )

    for y in range(0, height + 1, step):
        major = y % (step * 5) == 0
        cv2.line(
            canvas,
            (0, y * scale),
            (width * scale, y * scale),
            MAJOR_COLOUR if major else GRID_COLOUR,
            2 if major else 1,
        )
        if major:
            cv2.putText(
                canvas, str(y), (3, y * scale - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOUR, 1
            )

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, nargs="+", help="one or more clips")
    parser.add_argument("--out", required=True, help="where to write the PNG")
    parser.add_argument(
        "--single-frame",
        type=int,
        default=None,
        metavar="N",
        help="use frame N of the first clip instead of the median of all frames",
    )
    parser.add_argument("--plain", action="store_true", help="save without the grid as well")
    parser.add_argument("--scale", type=int, default=SCALE, help="how much to enlarge")
    parser.add_argument("--step", type=int, default=GRID_STEP, help="grid spacing in source pixels")
    args = parser.parse_args()

    if args.single_frame is None:
        frame = empty_road_image(args.video)
    else:
        frame = read_reference_frame(args.video[0], index=args.single_frame)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out), draw_grid(frame, scale=args.scale, step=args.step))
    print(
        f"wrote {out}  (grid labels are coordinates in the original "
        f"{frame.shape[1]}x{frame.shape[0]} frame)"
    )

    if args.plain:
        plain = out.with_name(f"{out.stem}_plain{out.suffix}")
        cv2.imwrite(str(plain), frame)
        print(f"wrote {plain}")


if __name__ == "__main__":
    main()
