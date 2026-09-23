"""Links detections in one frame to the same vehicle in the next.

ByteTrack is used because of how it handles weak detections. Most trackers throw
away low-confidence boxes; ByteTrack keeps them for a second pass and tries to
match them to vehicles it is already following. That is exactly the failure mode
here - a car far down the road scores badly for a few frames, and a stricter
tracker would cut its identity in half and report two short vehicles instead of
one long one.

The settings are loosened for 10 fps. The gap between frames is 100 ms, in which
a vehicle at motorway speed moves nearly three metres, so its box in one frame
barely overlaps its box in the next. A tracker tuned for 25 or 30 fps breaks the
identity of every fast vehicle here.

This wrapper exists so those settings come from the config file, and so the
pipeline sees one small interface instead of the library's.
"""

from __future__ import annotations

import supervision as sv
from trackers import ByteTrackTracker

from traffic_flow.config import TrackerConfig


class VehicleTracker:
    """ByteTrack, set up for this dataset's frame rate."""

    def __init__(self, config: TrackerConfig, fps: float):
        self._tracker = ByteTrackTracker(
            lost_track_buffer=config.lost_track_buffer,
            frame_rate=float(fps),
            track_activation_threshold=config.track_activation_threshold,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
            minimum_iou_threshold=config.minimum_iou_threshold,
            high_conf_det_threshold=config.high_confidence_threshold,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Attach a ``tracker_id`` to each detection it can follow.

        Detections the tracker is not yet confident about come back without an
        id; the pipeline drops those rather than inventing an identity for them.
        """
        return self._tracker.update(detections)

    def reset(self) -> None:
        """Forget every vehicle. Called between clips so ids start again at 1."""
        self._tracker.reset()
