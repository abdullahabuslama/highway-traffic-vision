"""Scores every congestion classifier on the dataset's own four folds.

Reads the table that ``run_dataset.py`` produced, fits each model on each fold's
training half, tests it on the other half, and prints the results side by side
with the do-nothing baseline.

The models are declared in one list. Adding another is one line there.

Run::

    python scripts/evaluate.py --features outputs/clip_features.parquet
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from traffic_flow.classify.base import CongestionModel
from traffic_flow.classify.baseline import MajorityCongestionModel
from traffic_flow.classify.evaluation import EvaluationReport, evaluate
from traffic_flow.classify.learned import LearnedCongestionModel
from traffic_flow.classify.rules import RuleCongestionModel
from traffic_flow.data.catalog import load_catalog

#: Every model that gets scored, in the order they are reported. The baseline
#: comes first on purpose: it is the number the others have to beat.
MODELS: tuple[Callable[[], CongestionModel], ...] = (
    MajorityCongestionModel,
    RuleCongestionModel,
    lambda: LearnedCongestionModel("logistic"),
    lambda: LearnedCongestionModel("gradient_boosting"),
)


def _print_report(report: EvaluationReport) -> None:
    per_fold = "  ".join(f"f{fold.fold}={fold.accuracy:.3f}" for fold in report.folds)
    print(f"\n=== {report.model_name} ===")
    print(
        f"accuracy {report.mean_accuracy:.3f} +/- {report.accuracy_spread:.3f}   "
        f"macro-F1 {report.mean_macro_f1:.3f}   ({per_fold})"
    )
    print("\nconfusion (rows = truth, columns = predicted):")
    print(report.confusion().to_string())
    print("\nby weather:")
    print(report.accuracy_by("weather").to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--features", default="outputs/clip_features.parquet")
    parser.add_argument("--out", default="outputs/evaluation.csv")
    args = parser.parse_args()

    catalog = load_catalog(args.dataset)
    features = pd.read_parquet(args.features)
    print(f"{len(features)} clips scored on {len(catalog.folds)} folds")

    summary = []
    predictions = []
    for build in MODELS:
        report = evaluate(build, features, catalog)
        _print_report(report)
        summary.append(
            {
                "model": report.model_name,
                "accuracy": report.mean_accuracy,
                "accuracy_spread": report.accuracy_spread,
                "macro_f1": report.mean_macro_f1,
            }
        )
        predictions.append(report.predictions.assign(model=report.model_name))

    table = pd.DataFrame(summary).sort_values("accuracy", ascending=False)
    print("\n=== summary ===")
    print(table.to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out.with_name(f"{out.stem}_predictions.csv"), index=False
    )
    print(f"\nwrote {out} and {out.with_name(f'{out.stem}_predictions.csv')}")


if __name__ == "__main__":
    main()
