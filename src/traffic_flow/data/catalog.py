"""Reads the dataset's own index files into one table.

The UCSD traffic database describes itself with three plain-text files that each
hold a different piece:

- ``ImageMaster`` gives every clip an index number and its traffic label.
- ``info.txt``    gives the same clips their date, time, weather and frame count.
- ``EvalSet_train`` / ``EvalSet_test`` list which index numbers belong to the
  training and test half of each of the four evaluation folds.

Every later step reads the single table this module builds, so the parsing rules
live in one place and the rest of the code never opens those files again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_flow.labels import CongestionLevel, Weather

#: Column names of ``info.txt``, taken from its own header comment line.
_INFO_COLUMNS = [
    "clip_id",
    "date",
    "timestamp",
    "direction",
    "day_night",
    "weather",
    "start_frame",
    "n_frames",
    "label",
    "notes",
]

#: Column names of ``ImageMaster``, also from its header comment line.
_MASTER_COLUMNS = ["index", "clip_id", "label"]

#: Clips whose ``notes`` field contains this word have water on the lens. They
#: stay in the dataset, but results are reported separately for them.
LENS_DROPS_NOTE = "drops on lens"


@dataclass(frozen=True)
class Fold:
    """One train/test split that the dataset ships with.

    ``train`` and ``test`` hold ``ImageMaster`` index numbers, not row positions
    in the clip table, which is why the catalog keeps that index as a column.
    """

    number: int
    train: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class ClipCatalog:
    """Everything the dataset says about itself.

    ``clips`` is one row per video with its label, conditions and file path.
    ``folds`` is the four evaluation splits, in the order the files list them.
    """

    clips: pd.DataFrame
    folds: tuple[Fold, ...]

    def __len__(self) -> int:
        return len(self.clips)

    def fold(self, number: int) -> Fold:
        """Return one fold by its number, counting from zero."""
        return self.folds[number]

    def split(self, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the clip table into the train half and test half of one fold."""
        is_train = self.clips["index"].isin(fold.train)
        is_test = self.clips["index"].isin(fold.test)
        return self.clips[is_train].copy(), self.clips[is_test].copy()


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read one of the dataset's tab-separated index files.

    The files start with a ``#`` header comment and the last column may be empty,
    which is why the column names are passed in rather than inferred.
    """
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        names=columns,
        header=None,
        dtype=str,
        keep_default_na=False,
    )


def _read_fold_file(path: Path) -> list[tuple[int, ...]]:
    """Read ``EvalSet_train`` or ``EvalSet_test``: one comma-separated row per fold."""
    rows: list[tuple[int, ...]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(tuple(int(value) for value in line.split(",")))
    return rows


def _load_folds(dataset_dir: Path) -> tuple[Fold, ...]:
    """Pair the rows of the train file with the rows of the test file."""
    train_rows = _read_fold_file(dataset_dir / "EvalSet_train")
    test_rows = _read_fold_file(dataset_dir / "EvalSet_test")
    if len(train_rows) != len(test_rows):
        raise ValueError(
            f"EvalSet_train has {len(train_rows)} folds but "
            f"EvalSet_test has {len(test_rows)}"
        )
    return tuple(
        Fold(number=i, train=train, test=test)
        for i, (train, test) in enumerate(zip(train_rows, test_rows, strict=True))
    )


def load_catalog(dataset_dir: str | Path) -> ClipCatalog:
    """Build the clip table and the fold list from a dataset directory.

    The two index files are joined on the clip name. They are checked against
    each other, because a disagreement about a clip's label would quietly poison
    every accuracy number computed later.
    """
    dataset_dir = Path(dataset_dir)

    master = _read_table(dataset_dir / "ImageMaster", _MASTER_COLUMNS)
    info = _read_table(dataset_dir / "info.txt", _INFO_COLUMNS)

    clips = master.merge(info, on="clip_id", how="left", suffixes=("", "_info"))

    missing = clips["label_info"].isna()
    if missing.any():
        names = ", ".join(clips.loc[missing, "clip_id"].head())
        raise ValueError(f"clips listed in ImageMaster but missing from info.txt: {names}")

    disagreed = clips["label"] != clips["label_info"]
    if disagreed.any():
        names = ", ".join(clips.loc[disagreed, "clip_id"].head())
        raise ValueError(f"ImageMaster and info.txt disagree on the label of: {names}")
    clips = clips.drop(columns=["label_info"])

    clips["index"] = clips["index"].astype(int)
    clips["start_frame"] = clips["start_frame"].astype(int)
    clips["n_frames"] = clips["n_frames"].astype(int)
    clips["label"] = clips["label"].map(CongestionLevel)
    clips["weather"] = clips["weather"].map(Weather)
    clips["notes"] = clips["notes"].str.strip()
    clips["has_lens_drops"] = clips["notes"].str.contains(LENS_DROPS_NOTE, case=False)
    # "16.01638" means the clip was recorded in the 16:00 hour. The part after the
    # dot is a sequence number, not minutes, so only the hour is usable.
    clips["hour"] = clips["timestamp"].str.split(".").str[0].astype(int)
    clips["video_path"] = clips["clip_id"].map(
        lambda name: str(dataset_dir / "video" / f"{name}.avi")
    )

    absent = [path for path in clips["video_path"] if not Path(path).exists()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} videos are missing, first is {absent[0]}")

    clips = clips.sort_values("index").reset_index(drop=True)
    return ClipCatalog(clips=clips, folds=_load_folds(dataset_dir))
