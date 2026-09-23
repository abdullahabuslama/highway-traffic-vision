"""Checks the congestion classifiers and how they are scored."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from traffic_flow.aggregate.clip_features import FEATURE_NAMES
from traffic_flow.classify.baseline import MajorityCongestionModel
from traffic_flow.classify.evaluation import evaluate
from traffic_flow.classify.learned import LearnedCongestionModel
from traffic_flow.classify.rules import RuleCongestionModel
from traffic_flow.classify.store import load_model, save_model
from traffic_flow.data.catalog import ClipCatalog, Fold
from traffic_flow.labels import CongestionLevel


def _features(speeds: list[float], densities: list[float]) -> pd.DataFrame:
    """A features table with the two columns the rules read, and zeros elsewhere."""
    table = pd.DataFrame({name: 0.0 for name in FEATURE_NAMES}, index=range(len(speeds)))
    table["space_mean_speed_kph"] = speeds
    table["density_veh_per_km_per_lane"] = densities
    table["clip_id"] = [f"clip_{i}" for i in range(len(speeds))]
    return table


def test_rules_read_free_flow_as_light_and_a_crawl_as_heavy():
    model = RuleCongestionModel()
    predicted = model.predict(_features([100.0, 55.0, 12.0], [8.0, 20.0, 40.0]))
    assert list(predicted) == [
        str(CongestionLevel.LIGHT),
        str(CongestionLevel.MEDIUM),
        str(CongestionLevel.HEAVY),
    ]


def test_a_packed_road_is_heavy_even_when_the_few_measured_vehicles_move():
    """Density is the second opinion for when speed alone would be misleading."""
    model = RuleCongestionModel()
    assert model.predict(_features([90.0], [45.0]))[0] == str(CongestionLevel.HEAVY)


def test_an_empty_road_with_no_measurable_vehicle_is_light():
    """No vehicles means nothing to measure, not a stationary jam."""
    model = RuleCongestionModel()
    empty = _features([float("nan")], [0.0])
    assert model.predict(empty)[0] == str(CongestionLevel.LIGHT)
    assert model.predict_confidence(empty)[0] == pytest.approx(1.0)


def test_a_measured_standstill_is_heavy_not_empty():
    """Zero km/h that was actually measured is a jam, not an empty road.

    Before speeds had an explicit "not measured" value, zero did double duty and
    a genuine standstill would have been waved through as free flow.
    """
    model = RuleCongestionModel()
    assert model.predict(_features([0.0], [0.0]))[0] == str(CongestionLevel.HEAVY)


def test_rules_complain_when_a_column_they_need_is_missing():
    model = RuleCongestionModel()
    with pytest.raises(KeyError):
        model.predict(pd.DataFrame({"something_else": [1.0]}))


def test_the_baseline_always_answers_the_commonest_level():
    labels = pd.Series(["light"] * 8 + ["heavy"] * 2)
    model = MajorityCongestionModel().fit(_features([0.0] * 10, [0.0] * 10), labels)
    predicted = model.predict(_features([0.0] * 3, [0.0] * 3))
    assert set(predicted) == {"light"}
    assert model.predict_confidence(_features([0.0], [0.0]))[0] == pytest.approx(0.8)


def test_the_learned_model_separates_what_is_separable():
    rng = np.random.default_rng(0)
    fast = _features(list(rng.normal(100, 5, 40)), list(rng.normal(8, 2, 40)))
    slow = _features(list(rng.normal(10, 5, 40)), list(rng.normal(50, 5, 40)))
    table = pd.concat([fast, slow], ignore_index=True)
    labels = pd.Series(["light"] * 40 + ["heavy"] * 40)

    model = LearnedCongestionModel("logistic").fit(table, labels)
    assert (model.predict(table) == labels.to_numpy()).mean() > 0.95
    assert model.predict_confidence(table).max() <= 1.0


def test_the_learned_model_refuses_a_table_missing_a_feature():
    model = LearnedCongestionModel("logistic")
    with pytest.raises(KeyError):
        model.fit(pd.DataFrame({"clip_id": ["a"]}), pd.Series(["light"]))


def test_an_unknown_estimator_fails_at_once():
    with pytest.raises(KeyError):
        LearnedCongestionModel("magic")


def _tiny_catalog(n: int = 12) -> tuple[ClipCatalog, pd.DataFrame]:
    features = _features(
        speeds=[100.0, 12.0] * (n // 2),
        densities=[8.0, 45.0] * (n // 2),
    )
    clips = pd.DataFrame(
        {
            "index": range(n),
            "clip_id": features["clip_id"],
            "label": ["light", "heavy"] * (n // 2),
            "weather": ["clear", "rain"] * (n // 2),
            "has_lens_drops": [False] * n,
        }
    )
    folds = (
        Fold(number=0, train=tuple(range(n - 4)), test=tuple(range(n - 4, n))),
        Fold(number=1, train=tuple(range(4, n)), test=tuple(range(4))),
    )
    return ClipCatalog(clips=clips, folds=folds), features


def test_evaluation_scores_every_fold_and_pools_the_predictions():
    catalog, features = _tiny_catalog()
    report = evaluate(RuleCongestionModel, features, catalog)

    assert len(report.folds) == 2
    assert len(report.predictions) == sum(fold.n_test for fold in report.folds)
    assert report.mean_accuracy == pytest.approx(1.0)
    assert set(report.confusion().index) == {"light", "medium", "heavy"}

    by_weather = report.accuracy_by("weather")
    assert set(by_weather.index) == {"clear", "rain"}
    assert by_weather["n_clips"].sum() == len(report.predictions)


def test_evaluation_refuses_features_the_catalogue_does_not_know():
    catalog, features = _tiny_catalog()
    stranger = features.copy()
    stranger.loc[0, "clip_id"] = "not_in_the_catalogue"
    with pytest.raises(ValueError):
        evaluate(RuleCongestionModel, stranger, catalog)


def test_a_saved_model_comes_back_predicting_the_same_thing(tmp_path):
    table = _features([100.0] * 20 + [10.0] * 20, [8.0] * 20 + [50.0] * 20)
    labels = pd.Series(["light"] * 20 + ["heavy"] * 20)
    model = LearnedCongestionModel("logistic").fit(table, labels)

    path = save_model(model, tmp_path / "model.pkl", provenance="a test")
    reloaded = load_model(path)

    assert reloaded.provenance == "a test"
    assert list(reloaded.model.predict(table)) == list(model.predict(table))


def test_loading_a_model_trained_on_other_features_is_refused(tmp_path, monkeypatch):
    """A model fed a different feature set would not crash - it would be wrong."""
    from traffic_flow.classify import store

    path = save_model(object(), tmp_path / "model.pkl")
    monkeypatch.setattr(store, "FEATURE_NAMES", ("something", "different"))
    with pytest.raises(ValueError, match="different features"):
        store.load_model(path)


def test_a_missing_model_file_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_model"):
        load_model(tmp_path / "nope.pkl")


def test_an_unmeasured_speed_becomes_json_null():
    """JSON cannot write NaN, and Python's writer emits a bare NaN that readers reject."""
    import json

    from traffic_flow.service import _measured_or_none

    assert _measured_or_none(float("nan")) is None
    assert _measured_or_none(91.44) == 91.4
    json.loads(json.dumps({"speed": _measured_or_none(float("nan"))}))
