"""Saves a fitted congestion model to disk and loads it back.

The pipeline measures traffic without any training at all - speed, density and
counts come out of geometry. Only the final light/medium/heavy call is learned,
so this is the one piece that has to be carried from the training machine to
wherever the analysis runs.

What is saved is the fitted model together with the feature names it was fitted
on. Loading checks those names, because a model silently fed a differently
ordered feature vector would not fail - it would just be wrong.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traffic_flow.aggregate.clip_features import FEATURE_NAMES

#: Where the trained model lands unless told otherwise.
DEFAULT_MODEL_PATH = Path("models/congestion_model.pkl")

#: Bumped when the saved shape changes, so an old file fails loudly.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class StoredModel:
    """A fitted model plus what it needs to be used correctly."""

    model: Any
    feature_names: tuple[str, ...]
    #: Free text: what it was trained on and how it scored, for the record.
    provenance: str


def save_model(model: Any, path: str | Path = DEFAULT_MODEL_PATH, provenance: str = "") -> Path:
    """Write a fitted model, with the feature names it expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "model": model,
        "feature_names": tuple(FEATURE_NAMES),
        "provenance": provenance,
    }
    path.write_bytes(pickle.dumps(payload))
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> StoredModel:
    """Read a fitted model back, refusing one that no longer matches the features."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path}. Run scripts/train_model.py first."
        )

    payload = pickle.loads(path.read_bytes())
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"{path} was written by format version {payload.get('version')}, "
            f"this build expects {FORMAT_VERSION}. Retrain it."
        )

    stored = tuple(payload["feature_names"])
    if stored != tuple(FEATURE_NAMES):
        missing = set(FEATURE_NAMES) - set(stored)
        extra = set(stored) - set(FEATURE_NAMES)
        raise ValueError(
            f"{path} was trained on different features. "
            f"missing now: {sorted(missing)}; no longer produced: {sorted(extra)}. Retrain it."
        )

    return StoredModel(
        model=payload["model"],
        feature_names=stored,
        provenance=payload.get("provenance", ""),
    )
