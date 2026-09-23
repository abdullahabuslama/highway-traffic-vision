"""The HTTP API. A thin wrapper over :mod:`traffic_flow.service`.

Upload a clip, get its traffic parameters back as JSON, and optionally fetch an
annotated copy of the video. No analysis happens in this file; it only moves
files and calls the service, so the API and the command line can never disagree
about a number.

Start it with ``traffic-flow serve``, then open http://127.0.0.1:8000/docs for
the interactive documentation FastAPI generates from this file.

Endpoints:

- ``GET  /health``         - is the service up, and is a trained model loaded?
- ``POST /analyse``        - upload a clip, get its traffic parameters
- ``GET  /analyse/{id}/video`` - the annotated clip from that request
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from traffic_flow.aggregate.clip_features import FEATURE_DESCRIPTIONS
from traffic_flow.service import LOCAL_VIDEO_PATH_KEY, TrafficService

#: Video containers the pipeline can read. The dataset itself is .avi.
ALLOWED_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}

#: Uploads and rendered videos live here until the process exits.
_WORK_DIR = Path(tempfile.gettempdir()) / "traffic-flow-api"

#: Built once at start-up: loading the detector takes a few seconds.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the detector and the trained model once, when the server starts."""
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    _state["service"] = TrafficService.build()
    yield
    _state.clear()


app = FastAPI(
    title="Traffic Flow Analysis",
    version="0.1.0",
    summary="Vehicle counts, speed, density and congestion level from fixed-camera video.",
    lifespan=lifespan,
)


def _service() -> TrafficService:
    service = _state.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail="the service is still starting up")
    return service


@app.get("/health")
def health() -> dict[str, Any]:
    """Whether the service is ready, and what it can currently answer."""
    service = _state.get("service")
    return {
        "ready": service is not None,
        "congestion_model_loaded": bool(service and service.classifier),
        "lanes": len(service.camera.lanes) if service else None,
        "road_length_m": round(service.pipeline.road_length_m, 1) if service else None,
        "measures": list(FEATURE_DESCRIPTIONS),
    }


@app.post("/analyse")
def analyse(
    clip: UploadFile = File(..., description="a video clip from the calibrated camera"),
    annotate: bool = Query(False, description="also render an annotated copy of the clip"),
) -> dict[str, Any]:
    """Measure the traffic in an uploaded clip.

    The camera calibration is fixed, so a clip from a different camera will be
    measured against the wrong road and give meaningless distances. Recalibrate
    with ``scripts/calibrate_camera.py`` before pointing this at new footage.
    """
    name = _uploaded_name(clip.filename)
    suffix = name.suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {suffix!r}; expected one of {sorted(ALLOWED_SUFFIXES)}",
        )

    request_id = uuid.uuid4().hex
    upload_path = _WORK_DIR / f"{request_id}{suffix}"
    try:
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(clip.file, handle)
    finally:
        clip.file.close()

    annotated_path = _WORK_DIR / f"{request_id}.mp4" if annotate else None
    try:
        # Never empty: a name that passed the extension check has a stem.
        result = _service().analyse(upload_path, annotate_to=annotated_path, clip_id=name.stem)
    except OSError as error:
        raise HTTPException(status_code=422, detail=f"could not read the clip: {error}") from error
    finally:
        upload_path.unlink(missing_ok=True)

    # The service reports where it wrote the video on disk. That is useful from
    # the command line, but over the network it would hand every caller this
    # server's folder layout and user name. Callers get the download URL instead.
    result.pop(LOCAL_VIDEO_PATH_KEY, None)

    result["request_id"] = request_id
    if annotated_path is not None:
        result["annotated_video_url"] = f"/analyse/{request_id}/video"
    return result


def _uploaded_name(filename: str | None) -> Path:
    """The uploaded file's name, without any folders the client sent with it.

    The one place an upload's name is read, so the extension check and the
    clip's label can never disagree about what the file is called. Some clients
    send a full Windows path, and on a Linux server Python does not treat a
    backslash as a folder separator, so backslashes are handled here explicitly.

    The name is only ever a label and an extension. The upload itself is saved
    under a random id, so a hostile name such as ``../../x.avi`` can mislabel its
    own result but cannot choose where anything is written.
    """
    return Path((filename or "").replace("\\", "/").rsplit("/", 1)[-1])


@app.get("/analyse/{request_id}/video")
def annotated_video(request_id: str) -> FileResponse:
    """Fetch the annotated clip produced by an earlier request."""
    if not request_id.isalnum():
        raise HTTPException(status_code=400, detail="bad request id")

    path = _WORK_DIR / f"{request_id}.mp4"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="no annotated video for that request; call /analyse with annotate=true",
        )
    return FileResponse(path, media_type="video/mp4", filename=f"{request_id}.mp4")
