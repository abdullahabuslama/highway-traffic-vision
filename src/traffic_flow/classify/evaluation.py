"""Scores a congestion classifier on the splits the dataset ships with.

The dataset comes with four train/test splits in ``EvalSet_train`` and
``EvalSet_test``. Using them, rather than inventing our own split, is what makes
the accuracy here comparable with the published results on this database.

A fresh model is fitted for every fold. Reusing one fitted model would let it
carry knowledge of a fold's test clips into the next fold, and the resulting
number would be flattering and wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from traffic_flow.classify.base import CongestionModel
from traffic_flow.data.catalog import ClipCatalog
from traffic_flow.labels import CONGESTION_ORDER

#: Column holding the true label in the joined table.
LABEL_COLUMN = "label"
#: Column holding a clip's ``ImageMaster`` index, which the folds refer to.
INDEX_COLUMN = "index"


@dataclass(frozen=True)
class FoldResult:
    """What happened on one of the four folds."""

    fold: int
    n_train: int
    n_test: int
    accuracy: float
    macro_f1: float
    #: One row per test clip: its name, true label, predicted label, confidence.
    predictions: pd.DataFrame


@dataclass(frozen=True)
class EvaluationReport:
    """A classifier's score across all four folds."""

    model_name: str
    folds: tuple[FoldResult, ...]
    #: Every fold's test predictions stacked together.
    predictions: pd.DataFrame

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([fold.accuracy for fold in self.folds]))

    @property
    def accuracy_spread(self) -> float:
        """How much the folds disagree. A wide spread means a fragile result."""
        return float(np.std([fold.accuracy for fold in self.folds]))

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean([fold.macro_f1 for fold in self.folds]))

    def confusion(self) -> pd.DataFrame:
        """Where the mistakes go, pooled over the folds.

        Confusing light with medium is a small error; confusing light with heavy
        means the measurements are wrong, not the threshold. The table separates
        those two cases, which a single accuracy number cannot.
        """
        order = [str(level) for level in CONGESTION_ORDER]
        matrix = confusion_matrix(
            self.predictions["true"], self.predictions["predicted"], labels=order
        )
        return pd.DataFrame(matrix, index=order, columns=order)

    def accuracy_by(self, column: str) -> pd.DataFrame:
        """Accuracy broken down by a clip property, such as weather.

        The rain and lens-drop clips are the hard ones. Reporting them inside one
        overall average hides exactly the failure a deployment would hit.
        """
        table = self.predictions.copy()
        table["correct"] = table["true"] == table["predicted"]
        grouped = table.groupby(column)["correct"]
        return pd.DataFrame({"n_clips": grouped.size(), "accuracy": grouped.mean()})


def evaluate(
    build_model: Callable[[], CongestionModel],
    features: pd.DataFrame,
    catalog: ClipCatalog,
    carry_columns: tuple[str, ...] = ("weather", "has_lens_drops"),
) -> EvaluationReport:
    """Fit and score one classifier on every fold.

    ``features`` needs a ``clip_id`` column; the clip's label, fold membership and
    conditions are taken from the catalog, so a features table can never disagree
    with the dataset about what a clip is.
    """
    table = catalog.clips.merge(features, on="clip_id", how="inner", validate="one_to_one")
    if len(table) != len(features):
        raise ValueError("some clips in the features table are not in the catalog")

    model_name = build_model().name
    fold_results: list[FoldResult] = []
    all_predictions: list[pd.DataFrame] = []

    for fold in catalog.folds:
        train = table[table[INDEX_COLUMN].isin(fold.train)]
        test = table[table[INDEX_COLUMN].isin(fold.test)]

        model = build_model().fit(train, train[LABEL_COLUMN])
        predicted = model.predict(test)
        confidence = model.predict_confidence(test)
        truth = test[LABEL_COLUMN].astype(str).to_numpy()

        predictions = pd.DataFrame(
            {
                "fold": fold.number,
                "clip_id": test["clip_id"].to_numpy(),
                "true": truth,
                "predicted": predicted.astype(str),
                "confidence": confidence,
            }
        )
        for column in carry_columns:
            predictions[column] = test[column].to_numpy()

        fold_results.append(
            FoldResult(
                fold=fold.number,
                n_train=len(train),
                n_test=len(test),
                accuracy=float((predictions["true"] == predictions["predicted"]).mean()),
                macro_f1=float(f1_score(truth, predicted.astype(str), average="macro")),
                predictions=predictions,
            )
        )
        all_predictions.append(predictions)

    return EvaluationReport(
        model_name=model_name,
        folds=tuple(fold_results),
        predictions=pd.concat(all_predictions, ignore_index=True),
    )
