"""Renders a few annotated clips to show what the analysis sees.

Picks one clip per traffic level plus one in the rain, so the demo covers both
the easy case and the awkward one, and writes an annotated copy of each.

Run::

    python scripts/render_demos.py --out outputs/demos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traffic_flow.classify.store import DEFAULT_MODEL_PATH
from traffic_flow.data.catalog import load_catalog
from traffic_flow.labels import CONGESTION_ORDER, Weather
from traffic_flow.metrics.speed import describe_speed
from traffic_flow.service import TrafficService

#: Which clips to show, and what each one is there to demonstrate.
DEMOS: tuple[tuple[str, dict[str, object]], ...] = (
    ("light", {"label": "light", "weather": Weather.CLEAR}),
    ("medium", {"label": "medium"}),
    ("heavy", {"label": "heavy"}),
    ("rain", {"weather": Weather.RAIN, "has_lens_drops": True}),
)


def _pick(clips: pd.DataFrame, wanted: dict[str, object]) -> pd.Series | None:
    """The first clip matching every condition, or nothing if there is none."""
    matching = clips
    for column, value in wanted.items():
        matching = matching[matching[column] == value]
    return matching.iloc[0] if len(matching) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="outputs/demos")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    catalog = load_catalog(args.dataset)
    service = TrafficService.build(model_path=args.model)
    if service.classifier is None:
        print("note: no trained model found, so the videos carry no congestion banner")

    out_dir = Path(args.out)
    for name, wanted in DEMOS:
        clip = _pick(catalog.clips, wanted)
        if clip is None:
            print(f"{name}: no clip matches {wanted}, skipping")
            continue

        target = out_dir / f"{name}_{clip['clip_id']}.mp4"
        result = service.analyse(clip["video_path"], annotate_to=target)
        predicted = result["congestion"]["level"] or "-"
        print(
            f"{name:7s} {clip['clip_id']}  truth={clip['label']:6s} predicted={predicted:6s}  "
            f"{result['counts']['vehicles_seen']:2d} vehicles  "
            f"{describe_speed(result['speed']['space_mean_kph'], decimals=1):>11s}  "
            f"{result['density']['veh_per_km_per_lane']:5.1f} veh/km/lane"
        )
        print(f"        -> {target}")

    print(f"\nlevels covered: {', '.join(str(level) for level in CONGESTION_ORDER)}")


if __name__ == "__main__":
    main()
