"""The command line. A thin wrapper over :mod:`traffic_flow.service`.

No analysis happens here - this file only reads arguments, calls the service, and
prints what comes back. The HTTP API in :mod:`traffic_flow.api` does the same
thing over the network, so the two always report the same numbers.

    traffic-flow analyse dataset/video/cctv052x2004080613x00015.avi
    traffic-flow analyse <clip.avi> --annotate outputs/demo.mp4
    traffic-flow serve
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from traffic_flow.classify.store import DEFAULT_MODEL_PATH
from traffic_flow.metrics.speed import describe_speed
from traffic_flow.service import TrafficService

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def analyse(
    video: Annotated[Path, typer.Argument(help="the clip to analyse")],
    annotate: Annotated[Path | None, typer.Option(help="also write an annotated copy here")] = None,
    out: Annotated[Path | None, typer.Option(help="write the JSON result here too")] = None,
    camera: Annotated[Path | None, typer.Option(help="camera config")] = None,
    pipeline: Annotated[Path | None, typer.Option(help="pipeline config")] = None,
    model: Annotated[Path, typer.Option(help="trained congestion model")] = DEFAULT_MODEL_PATH,
    quiet: Annotated[bool, typer.Option(help="print only the summary line")] = False,
) -> None:
    """Measure the traffic in one clip."""
    service = TrafficService.build(camera, pipeline, model)
    result = service.analyse(video, annotate_to=annotate)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    if quiet:
        typer.echo(_one_line(result))
    else:
        typer.echo(json.dumps(result, indent=2))
        typer.echo("\n" + _one_line(result))


def _one_line(result: dict) -> str:
    """The whole clip in a sentence, for a terminal that is being watched."""
    level = result["congestion"]["level"] or "unclassified"
    return (
        f"{result['clip_id']}: {level} traffic - "
        f"{result['counts']['vehicles_seen']} vehicles, "
        f"{describe_speed(result['speed']['space_mean_kph'])}, "
        f"{result['density']['veh_per_km_per_lane']:.0f} veh/km/lane, "
        f"{result['density']['occupancy'] * 100:.0f}% occupancy"
    )


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="address to listen on")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="port to listen on")] = 8000,
) -> None:
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run("traffic_flow.api:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
