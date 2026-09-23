"""Draws the report figures and prints the headline findings.

Reads the table that ``run_dataset.py`` produced, scores the best classifier on
the dataset's own folds, and writes the charts.

Run::

    python scripts/build_report.py --features outputs/clip_features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traffic_flow.classify.baseline import MajorityCongestionModel
from traffic_flow.classify.evaluation import evaluate
from traffic_flow.classify.learned import DEFAULT_ESTIMATOR, LearnedCongestionModel
from traffic_flow.data.catalog import load_catalog
from traffic_flow.labels import CONGESTION_ORDER
from traffic_flow.viz import report as figures


def _headline(table: pd.DataFrame) -> None:
    """Print the numbers a reader of the charts will want beside them."""
    print("\nmeasured traffic parameters, by the dataset's own label:")
    summary = (
        table.groupby("label", observed=True)
        .agg(
            clips=("clip_id", "size"),
            speed_kph=("space_mean_speed_kph", "median"),
            density=("density_veh_per_km_per_lane", "median"),
            occupancy=("mean_occupancy", "median"),
            vehicles_seen=("vehicles_seen", "median"),
            stopped_share=("stopped_share", "median"),
        )
        .reindex([str(level) for level in CONGESTION_ORDER])
        .round(2)
    )
    print(summary.to_string())

    busiest = table.groupby("hour", observed=True)["space_mean_speed_kph"].median().idxmin()
    slowest = table.groupby("hour", observed=True)["space_mean_speed_kph"].median().min()
    print(f"\nslowest hour of day: {busiest}:00, median speed {slowest:.0f} km/h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--features", default="outputs/clip_features.parquet")
    parser.add_argument("--out", default="outputs/figures")
    parser.add_argument("--model", default=DEFAULT_ESTIMATOR)
    args = parser.parse_args()

    catalog = load_catalog(args.dataset)
    features = pd.read_parquet(args.features)
    table = catalog.clips.merge(features, on="clip_id", validate="one_to_one")
    table["label"] = table["label"].astype(str)

    report = evaluate(lambda: LearnedCongestionModel(args.model), features, catalog)
    baseline = evaluate(MajorityCongestionModel, features, catalog)
    print(
        f"{report.model_name}: {report.mean_accuracy:.3f} accuracy over {len(report.folds)} folds"
        f"  (answering 'light' every time scores {baseline.mean_accuracy:.3f})"
    )

    _headline(table)

    written = figures.build_all(table, report, baseline, Path(args.out))
    print("\nfigures:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
