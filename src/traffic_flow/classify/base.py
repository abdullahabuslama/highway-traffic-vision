"""What a congestion classifier must look like.

Two of them exist and both are reported, because they answer slightly different
questions:

- the **learned** one is fitted to the dataset's own labels and gives the best
  accuracy number;
- the **rule** one reads thresholds straight from traffic engineering and needs
  no training, so it explains its answer and works on a road nobody has labelled.

They share this interface so the evaluation code runs them both the same way and
neither gets a quietly easier test than the other.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class CongestionModel(Protocol):
    """Decides whether a clip is light, medium or heavy traffic."""

    name: str

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> CongestionModel:
        """Learn from labelled clips. A rule-based model just returns itself."""
        ...

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """One :class:`~traffic_flow.labels.CongestionLevel` per row."""
        ...

    def predict_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """How sure the model is about each row, from 0 to 1."""
        ...
