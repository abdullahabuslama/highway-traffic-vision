"""Congestion level learned from the dataset's own labels.

The model is deliberately small. There are 254 clips, about 190 of them in each
fold's training half, described by fifteen numbers. A neural network on that much
data would memorise it and explain nothing.

Two estimators are offered and both are reported. On overall accuracy they tie,
so the choice between them is made on the clips that matter, and that is where
logistic regression wins - see :data:`DEFAULT_ESTIMATOR`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from traffic_flow.aggregate.clip_features import FEATURE_NAMES

#: The estimators on offer. Adding one is a line here; nothing else changes.
ESTIMATORS: dict[str, Callable[[], object]] = {
    "logistic": lambda: LogisticRegression(
        max_iter=2000,
        # The dataset has far more light clips than heavy ones. Balancing the
        # weights stops the model from scoring well by simply calling
        # everything light.
        class_weight="balanced",
    ),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        learning_rate=0.1,
        # Shallow and few, because 190 training rows overfit a deep forest fast.
        random_state=0,
    ),
}

#: Logistic regression is the default: 95.3% against 93.7% for gradient boosting
#: on the dataset's folds, with the same spread between folds (1.5 points).
#:
#: Overall accuracy is the weaker reason, though. 165 of the 254 clips are light
#: traffic, and both models get 164 of those right, so accuracy mostly measures
#: how well a model recognises an empty road. What a traffic department needs is
#: for congestion to be caught, and that is where the two differ: the linear
#: model catches 39 of the 44 heavy clips against 36. Its macro-F1, which weights
#: the three levels equally, is 0.918 against 0.887. And its coefficients can be
#: read, so every call it makes can be explained by the measurements behind it.
#:
#: This was gradient boosting until two measurement bugs were fixed - a single
#: stationary false detection dragging a clip's speed to near zero, and empty
#: roads reported as 0 km/h rather than "not measured". On the broken features
#: gradient boosting led, 94.1% against 92.5%. Why it led there is not
#: established; one possibility is that its trees had fitted the artefacts.
DEFAULT_ESTIMATOR = "logistic"


class LearnedCongestionModel:
    """Fits one of the estimators to the clip features."""

    def __init__(self, estimator: str = DEFAULT_ESTIMATOR, features: tuple[str, ...] = FEATURE_NAMES):
        if estimator not in ESTIMATORS:
            raise KeyError(f"unknown estimator {estimator!r}, pick one of {sorted(ESTIMATORS)}")
        self.name = estimator
        self._features = features
        self._pipeline = Pipeline(
            [
                # A clip with no measurable vehicle leaves gaps; fill them with
                # the training median rather than dropping the clip.
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("estimator", ESTIMATORS[estimator]()),
            ]
        )

    def _matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self._features if name not in features]
        if missing:
            raise KeyError(f"features table is missing columns: {missing}")
        return features[list(self._features)]

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> LearnedCongestionModel:
        self._pipeline.fit(self._matrix(features), labels.astype(str))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict(self._matrix(features))

    def predict_confidence(self, features: pd.DataFrame) -> np.ndarray:
        return self._pipeline.predict_proba(self._matrix(features)).max(axis=1)

    @property
    def classes(self) -> np.ndarray:
        return self._pipeline.named_steps["estimator"].classes_

    def explain(self) -> pd.DataFrame:
        """Which features the fitted model leans on, strongest first.

        Only meaningful for the linear model, where a coefficient is the pull one
        feature has on one class. Returns an empty table otherwise, because a
        boosted forest's importances mean something different and mixing the two
        in the same report would mislead.
        """
        estimator = self._pipeline.named_steps["estimator"]
        if not hasattr(estimator, "coef_"):
            return pd.DataFrame(columns=["feature", "class", "weight"])

        rows = [
            {"feature": name, "class": str(label), "weight": float(weight)}
            for label, weights in zip(estimator.classes_, estimator.coef_, strict=True)
            for name, weight in zip(self._features, weights, strict=True)
        ]
        table = pd.DataFrame(rows)
        return table.reindex(table["weight"].abs().sort_values(ascending=False).index)
