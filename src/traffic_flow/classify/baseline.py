"""The do-nothing classifier, there to give every other number something to beat.

This dataset is lopsided: 165 of its 254 clips are light traffic, 45 medium and
44 heavy. So a model that ignores the video entirely and answers "light" every
time is right about 65% of the time.

That number has to be on the page. Without it, an accuracy of 70% sounds like a
working system when it is barely better than a constant, and an accuracy of 90%
cannot be judged at all. It is reported alongside the real models, scored on the
same folds, through the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MajorityCongestionModel:
    """Always answers with whichever level was most common in training."""

    name = "majority-baseline"

    def __init__(self) -> None:
        self._answer: str | None = None
        self._share = 0.0

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> MajorityCongestionModel:
        counts = labels.astype(str).value_counts()
        self._answer = str(counts.index[0])
        self._share = float(counts.iloc[0] / counts.sum())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self._answer is None:
            raise RuntimeError("the baseline must be fitted before it can predict")
        return np.full(len(features), self._answer, dtype=object)

    def predict_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """How common that answer was in training. The same for every clip."""
        return np.full(len(features), self._share, dtype=float)
