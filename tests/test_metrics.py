"""Checks the rules that turn tracks into traffic numbers."""

from __future__ import annotations

import numpy as np
import pytest

from traffic_flow.config import MetricsConfig
from traffic_flow.labels import VehicleClass
from traffic_flow.metrics.counting import count_vehicles, crosses_line, side_of_line
from traffic_flow.metrics.density import flow_from_fundamental_relation, measure_density
from traffic_flow.metrics.lanes import count_lane_changes, measure_lanes
from traffic_flow.metrics.speed import (
    NO_TRAFFIC_TEXT,
    clip_speeds,
    describe_speed,
    is_measured,
    median_speed_kph,
    space_mean_speed_kph,
    track_speed,
)
from traffic_flow.tracking.tracks import NO_LANE, FrameRecord, Observation, Track

METRICS = MetricsConfig(min_track_frames=3, max_plausible_kph=160.0)


def _track(
    track_id: int,
    metres_per_frame: float,
    n: int = 10,
    vehicle: VehicleClass = VehicleClass.CAR,
    lanes: list[str] | None = None,
    start_x: float = 0.0,
) -> Track:
    """A vehicle moving straight at a steady speed, for the maths to chew on."""
    track = Track(track_id=track_id)
    for i in range(n):
        lane = lanes[i] if lanes else "lane_1"
        track.add(
            Observation(
                frame_index=i,
                time_s=i * 0.1,
                xyxy=(0.0, 0.0, 10.0, 10.0),
                ground_xy=(start_x + i * 2.0, 200.0 - i * 3.0),
                world_xy=(0.0, i * metres_per_frame),
                vehicle_class=vehicle,
                confidence=0.5,
                lane=lane,
            )
        )
    return track


def test_speed_is_read_off_a_steady_track():
    """2.78 m per frame at 10 fps is 27.8 m/s, which is 100 km/h."""
    speed = track_speed(_track(1, metres_per_frame=2.78), METRICS)
    assert speed is not None
    assert speed.kph == pytest.approx(100.0, rel=0.01)


def test_a_wobbling_box_does_not_invent_speed():
    """A parked vehicle whose box jitters must still read as stopped.

    This is why the speed is fitted to the whole path instead of being measured
    step by step: step distances are always positive, so noise alone would add up
    to a speed.
    """
    rng = np.random.default_rng(0)
    track = Track(track_id=1)
    for i in range(12):
        track.add(
            Observation(
                frame_index=i,
                time_s=i * 0.1,
                xyxy=(0.0, 0.0, 10.0, 10.0),
                ground_xy=(100.0, 200.0),
                world_xy=(rng.normal(0, 0.4), 60.0 + rng.normal(0, 0.4)),
                vehicle_class=VehicleClass.CAR,
                confidence=0.5,
                lane="lane_1",
            )
        )
    speed = track_speed(track, METRICS)
    assert speed is not None
    assert speed.kph < 12.0


def test_short_and_impossible_tracks_are_thrown_away():
    assert track_speed(_track(1, 2.0, n=2), METRICS) is None
    assert track_speed(_track(2, metres_per_frame=10.0), METRICS) is None  # 360 km/h


def test_space_mean_speed_is_not_the_plain_average():
    """It sits below the plain average whenever speeds differ.

    That is the whole point of it. A plain average counts each vehicle once and
    so flatters a road that is part-blocked, because the fast vehicles stream
    past while the slow ones sit there. Weighting by time on the road instead
    gives the queue the weight it deserves.

    The two vehicles here cover roughly the same 25 m of road, but the slow one
    takes three times as long over it - which is exactly what happens in the
    video, and is where the extra weight comes from.
    """
    fast = _track(1, 2.78, n=10)  # 25 m in 0.9 s
    slow = _track(2, 0.93, n=30)  # 27 m in 2.9 s
    speeds = clip_speeds([fast, slow], METRICS)

    assert len(speeds) == 2
    arithmetic = np.mean([s.kph for s in speeds])
    assert space_mean_speed_kph(speeds) < arithmetic


def test_one_stationary_false_detection_cannot_flatten_the_clip_speed():
    """The reason this is a distance-over-time sum and not a harmonic mean.

    A harmonic mean divides by every speed, so a single vehicle measured at a
    twentieth of a km/h contributes a term twenty times larger than everything
    else put together, and a freely flowing road reports as stopped. This is a
    real case: a light clip of eighteen vehicles once reported 1.1 km/h.
    """
    flowing = [_track(i, 2.5, n=10) for i in range(17)]
    stuck = _track(99, 0.0014, n=10)  # about 0.05 km/h - a static false positive
    speeds = clip_speeds([*flowing, stuck], METRICS)

    assert len(speeds) == 18
    assert space_mean_speed_kph(speeds) > 80.0


def test_an_empty_road_has_no_speed_rather_than_zero():
    """Zero km/h means a jam. An empty road is the opposite, and must not say zero.

    Six clips in the dataset have no traffic at all. Reporting them as 0 km/h
    would tell a reader the road was stopped exactly when it was clear.
    """
    assert not is_measured(space_mean_speed_kph([]))
    assert not is_measured(median_speed_kph([]))
    assert describe_speed(space_mean_speed_kph([])) == NO_TRAFFIC_TEXT


def test_a_measured_speed_reads_as_a_speed():
    speeds = clip_speeds([_track(1, 2.78)], METRICS)
    assert is_measured(space_mean_speed_kph(speeds))
    assert describe_speed(space_mean_speed_kph(speeds)) == "100 km/h"


def test_is_measured_answers_for_a_whole_column_and_for_json_nulls():
    column = np.array([90.0, float("nan"), 0.0])
    assert list(is_measured(column)) == [True, False, True]
    assert not is_measured(None)


def test_genuinely_stopped_traffic_still_reports_as_stopped():
    """The guard above must not paper over a real jam."""
    speeds = clip_speeds([_track(i, 0.02, n=10) for i in range(10)], METRICS)
    assert space_mean_speed_kph(speeds) < 10.0


def test_a_vehicle_that_drives_through_the_line_is_counted():
    line = ((0.0, 185.0), (320.0, 185.0))
    assert crosses_line(_track(1, 2.78), line) is True


def test_a_vehicle_that_stops_short_of_the_line_is_not():
    """Its path ends before the line, so nothing has passed the counter yet."""
    line = ((0.0, 100.0), (320.0, 100.0))
    assert crosses_line(_track(1, 2.78, n=5), line) is False


def test_side_of_line_separates_the_two_sides():
    line = ((0.0, 100.0), (320.0, 100.0))
    assert side_of_line(np.array([160.0, 150.0]), line) > 0
    assert side_of_line(np.array([160.0, 50.0]), line) < 0


def test_counts_are_split_by_kind_and_scaled_to_an_hour():
    tracks = [
        _track(1, 2.78, vehicle=VehicleClass.CAR),
        _track(2, 2.78, vehicle=VehicleClass.TRUCK),
    ]
    counts = count_vehicles(tracks, ((0.0, 185.0), (320.0, 185.0)), duration_s=5.0)
    assert counts.vehicles_seen == 2
    assert counts.crossings == 2
    assert counts.seen_by_class[VehicleClass.TRUCK] == 1
    assert counts.seen_by_class[VehicleClass.BUS] == 0
    assert counts.flow_veh_per_hour == pytest.approx(2 / 5.0 * 3600)
    assert counts.heavy_vehicle_share == pytest.approx(0.5)


def _frames(counts: list[int]) -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_index=i,
            time_s=i * 0.1,
            vehicle_count=n,
            occupancy=n / 40.0,
            class_counts={VehicleClass.CAR: n},
            lane_counts={"lane_1": n},
        )
        for i, n in enumerate(counts)
    ]


def test_density_is_per_kilometre_and_per_lane():
    """10 vehicles over 100 m of 5-lane road is 20 per km per lane."""
    density = measure_density(_frames([10] * 6), road_length_m=100.0, lane_count=5)
    assert density.veh_per_km_per_lane == pytest.approx(20.0)
    assert density.mean_vehicles_in_view == pytest.approx(10.0)
    assert density.peak_vehicles_in_view == 10


def test_an_empty_clip_has_no_density():
    density = measure_density([], road_length_m=100.0, lane_count=5)
    assert density.veh_per_km_per_lane == 0.0


def test_flow_is_density_times_speed():
    assert flow_from_fundamental_relation(20.0, 100.0) == pytest.approx(2000.0)


def test_one_frame_of_lane_flicker_is_not_a_lane_change():
    lanes = ["lane_1"] * 5 + ["lane_2"] + ["lane_1"] * 5
    assert count_lane_changes(_track(1, 2.78, n=11, lanes=lanes)) == 0


def test_a_vehicle_that_settles_in_a_new_lane_has_changed_lane():
    lanes = ["lane_1"] * 5 + ["lane_2"] * 5
    assert count_lane_changes(_track(1, 2.78, n=10, lanes=lanes)) == 1


def test_frames_outside_any_lane_are_ignored():
    lanes = ["lane_1"] * 4 + [NO_LANE] * 2 + ["lane_1"] * 4
    assert count_lane_changes(_track(1, 2.78, n=10, lanes=lanes)) == 0


def test_lane_imbalance_runs_from_even_to_one_sided():
    even = measure_lanes([], _frames_per_lane({"a": 5, "b": 5}), ("a", "b"))
    lopsided = measure_lanes([], _frames_per_lane({"a": 10, "b": 0}), ("a", "b"))
    assert even.lane_imbalance == pytest.approx(0.0, abs=1e-9)
    assert lopsided.lane_imbalance > even.lane_imbalance
    assert lopsided.busiest_lane == "a"


def _frames_per_lane(counts: dict[str, int]) -> list[FrameRecord]:
    return [
        FrameRecord(
            frame_index=0,
            time_s=0.0,
            vehicle_count=sum(counts.values()),
            occupancy=0.1,
            class_counts={},
            lane_counts=dict(counts),
        )
    ]
