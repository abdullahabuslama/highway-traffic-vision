"""Fits the congestion classifier on every clip and saves it.

The fold-by-fold scores in ``evaluate.py`` are the honest measure of how well
this model works, because there each model is tested on clips it never saw. This
script does something different: it fits one final model on *all* 254 clips, for
use on new footage. More training data makes a better model; it just cannot be
scored on the data it learned from.

The saved file records which score it earned in the proper evaluation, so the
number travels with the model instead of being remembered separately.

Run::

    python scripts/train_model.py --features outputs/clip_features.parquet
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from traffic_flow.classify.evaluation import evaluate
from traffic_flow.classify.learned import DEFAULT_ESTIMATOR, LearnedCongestionModel
from traffic_flow.classify.store import DEFAULT_MODEL_PATH, save_model
from traffic_flow.data.catalog import load_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--features", default="outputs/clip_features.parquet")
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--estimator", default=DEFAULT_ESTIMATOR)
    args = parser.parse_args()

    catalog = load_catalog(args.dataset)
    features = pd.read_parquet(args.features)
    table = catalog.clips.merge(features, on="clip_id", validate="one_to_one")

    report = evaluate(lambda: LearnedCongestionModel(args.estimator), features, catalog)
    print(
        f"{args.estimator}: {report.mean_accuracy:.3f} +/- {report.accuracy_spread:.3f} "
        f"accuracy over {len(report.folds)} folds"
    )

    model = LearnedCongestionModel(args.estimator).fit(table, table["label"])
    provenance = (
        f"{args.estimator} fitted on all {len(table)} clips of {Path(args.dataset).name} "
        f"on {datetime.now(UTC).date().isoformat()}. Scored {report.mean_accuracy:.3f} accuracy "
        f"(macro-F1 {report.mean_macro_f1:.3f}) on the dataset's four evaluation folds, "
        f"where each model was tested only on clips it had not seen."
    )
    path = save_model(model, args.out, provenance)
    print(f"wrote {path}")
    print(provenance)

    weights = model.explain()
    if len(weights):
        print("\nwhat the model leans on most:")
        print(weights.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
