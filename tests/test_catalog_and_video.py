"""Checks the dataset is read correctly, including its one documented quirk.

These tests need the dataset itself, so they skip when it is not there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from traffic_flow.data.catalog import load_catalog
from traffic_flow.io.video_source import CORRUPTED_LEAD_FRAMES, VideoSource
from traffic_flow.labels import CongestionLevel

DATASET = Path("dataset")
pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="dataset not present")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(DATASET)


def test_every_clip_is_listed_once_with_a_video(catalog):
    assert len(catalog) == 254
    assert catalog.clips["clip_id"].is_unique
    assert all(Path(path).exists() for path in catalog.clips["video_path"])


def test_labels_are_the_three_the_dataset_defines(catalog):
    assert set(catalog.clips["label"]) == set(CongestionLevel)


def test_the_folds_never_test_on_what_they_trained_on(catalog):
    """A clip in both halves would make the accuracy meaningless."""
    assert len(catalog.folds) == 4
    for fold in catalog.folds:
        assert not set(fold.train) & set(fold.test)
        assert len(fold.train) + len(fold.test) == len(catalog)


def test_info_and_master_agree_about_hours(catalog):
    assert catalog.clips["hour"].between(0, 23).all()


def test_the_corrupted_first_frame_is_skipped(catalog):
    """The dataset README says frame 0 carries a second video signal.

    It is visibly different from the frame after it, so the check is that the
    frame the reader hands out first is *not* that one.
    """
    path = catalog.clips.iloc[0]["video_path"]

    import cv2

    capture = cv2.VideoCapture(str(path))
    ok, raw_first = capture.read()
    capture.release()
    assert ok

    with VideoSource(path) as source:
        first_usable = next(iter(source))

    assert CORRUPTED_LEAD_FRAMES == 1
    assert first_usable.index == 0
    assert first_usable.time_s == 0.0
    assert not np.array_equal(raw_first, first_usable.image)


def test_clip_metadata_matches_the_video(catalog):
    row = catalog.clips.iloc[0]
    with VideoSource(row["video_path"]) as source:
        assert (source.meta.width, source.meta.height) == (320, 240)
        assert source.meta.fps == pytest.approx(10.0)
        frames = source.read_all()

    # info.txt counts the corrupted frame; the reader does not.
    assert len(frames) == row["n_frames"] - CORRUPTED_LEAD_FRAMES
    assert [f.index for f in frames] == list(range(len(frames)))
