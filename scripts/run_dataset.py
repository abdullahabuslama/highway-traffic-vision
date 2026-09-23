"""Analyses every clip in the dataset and writes one table of traffic numbers.

This is the slow step - it runs the detector over roughly thirteen thousand
frames - so it is done once and everything downstream reads its output. The
table has one row per clip: its speed, density, occupancy, counts and lane use,
next to the label the dataset gives it.

Run::

    python scripts/run_dataset.py --dataset dataset --out outputs/clip_features.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from traffic_flow.aggregate.clip_features import clip_row
from traffic_flow.config import CameraConfig, PipelineConfig
from traffic_flow.data.catalog import load_catalog
from traffic_flow.detection.yolo import YoloDetector
from traffic_flow.pipeline import TrafficPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--camera", default=None, help="camera config, defaults to camera_i5.yaml")
    parser.add_argument("--pipeline", default=None, help="pipeline config, defaults to pipeline.yaml")
    parser.add_argument("--out", default="outputs/clip_features.parquet")
    parser.add_argument("--limit", type=int, default=None, help="only the first N clips, for a smoke test")
    args = parser.parse_args()

    catalog = load_catalog(args.dataset)
    pipeline_config = PipelineConfig.load(args.pipeline)
    camera = CameraConfig.load(args.camera)
    pipeline = TrafficPipeline(YoloDetector(pipeline_config.detector), camera, pipeline_config)

    clips = catalog.clips if args.limit is None else catalog.clips.head(args.limit)
    print(
        f"{len(clips)} clips, {len(camera.lanes)} lanes, "
        f"{pipeline.road_length_m:.0f} m of road, on {pipeline_config.detector.device}"
    )

    started = time.time()
    rows = []
    for record in tqdm(list(clips.itertuples()), unit="clip"):
        analysis = pipeline.analyse(record.video_path, record.clip_id)
        rows.append(clip_row(analysis))

    table = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    elapsed = time.time() - started
    print(f"wrote {out}: {len(table)} rows x {len(table.columns)} columns")
    print(f"took {elapsed / 60:.1f} min, {elapsed / max(len(table), 1):.1f} s per clip")


if __name__ == "__main__":
    main()
