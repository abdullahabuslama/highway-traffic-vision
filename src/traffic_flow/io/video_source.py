"""Reads frames out of one traffic clip.

This is the only place that touches OpenCV's video reader, and the only place
that knows about the dataset's one awkward quirk: the first frame of every clip
is corrupted with a second video signal, so reading starts at the frame after it.
The dataset README says so, and ``info.txt`` repeats it in its "start frame"
column, which is 2 for every clip.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np

#: How many frames to throw away at the start of a clip. One, because frame 0 is
#: the corrupted one. ``info.txt`` counts frames from 1, so its "start frame = 2"
#: is this same frame in a different counting scheme.
CORRUPTED_LEAD_FRAMES = 1


@dataclass(frozen=True)
class VideoMeta:
    """The fixed facts about one clip, read from its file header."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def seconds_per_frame(self) -> float:
        """How much time passes between two frames.

        At this dataset's 10 fps that is 0.1 s, which every speed calculation
        divides by.
        """
        return 1.0 / self.fps

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.seconds_per_frame


@dataclass(frozen=True)
class Frame:
    """One image out of a clip, with where and when it sits in that clip.

    ``index`` counts from zero at the first *usable* frame, not at the corrupted
    one, so ``time_s`` and ``index`` always agree with each other.
    """

    index: int
    time_s: float
    image: np.ndarray


class VideoSource:
    """Iterate the usable frames of one clip.

    Used as a context manager so the file handle always closes::

        with VideoSource(path) as source:
            for frame in source:
                ...
    """

    def __init__(self, path: str | Path, skip_lead_frames: int = CORRUPTED_LEAD_FRAMES):
        self.path = Path(path)
        self._skip_lead_frames = skip_lead_frames
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise OSError(f"cannot open video: {self.path}")
        self.meta = self._read_meta()

    def _read_meta(self) -> VideoMeta:
        capture = self._capture
        raw_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return VideoMeta(
            path=self.path,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            n_frames=max(raw_count - self._skip_lead_frames, 0),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def __iter__(self) -> Iterator[Frame]:
        """Yield every usable frame in order, dropping the corrupted lead frames.

        Seeking is avoided on purpose: these clips use an old MPEG-4 variant
        where seeking lands on the wrong frame with some OpenCV builds, and the
        clips are only about fifty frames long, so reading and discarding is both
        safer and fast.
        """
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        seconds_per_frame = self.meta.seconds_per_frame

        for _ in range(self._skip_lead_frames):
            if not self._capture.read()[0]:
                return

        index = 0
        while True:
            ok, image = self._capture.read()
            if not ok:
                return
            yield Frame(index=index, time_s=index * seconds_per_frame, image=image)
            index += 1

    def read_all(self) -> list[Frame]:
        """Read the whole clip into memory.

        Safe here because a clip is about 50 frames of 320x240, roughly 11 MB.
        Handy for anything that needs a second pass, such as background models.
        """
        return list(self)


def read_reference_frame(path: str | Path, index: int = 0) -> np.ndarray:
    """Grab a single usable frame, for calibration screens and thumbnails."""
    with VideoSource(path) as source:
        for frame in source:
            if frame.index == index:
                return frame.image
    raise IndexError(f"{path} has no usable frame at index {index}")
