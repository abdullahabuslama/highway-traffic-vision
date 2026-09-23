"""Checks the HTTP API's handling of uploads and what its responses reveal.

A stand-in replaces the real service, so these run in milliseconds and load no
detector. What they check is the API's own job: naming the clip, hiding the
server's files, and never letting a caller choose where an upload is written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from traffic_flow import api
from traffic_flow.service import LOCAL_VIDEO_PATH_KEY

CLIP_NAME = "cctv052x2004080516x01646"


class FakeService:
    """Records what the API asked for, and answers like the real service."""

    def __init__(self) -> None:
        self.video_paths: list[Path] = []
        self.clip_ids: list[str | None] = []

    def analyse(self, video_path, annotate_to=None, clip_id=None):
        self.video_paths.append(Path(video_path))
        self.clip_ids.append(clip_id)
        result = {"clip_id": clip_id}
        if annotate_to is not None:
            result[LOCAL_VIDEO_PATH_KEY] = str(annotate_to)
        return result


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An API client wired to the stand-in service and a throwaway folder.

    The client is not used as a context manager, so the start-up hook that loads
    the real detector never runs.
    """
    fake = FakeService()
    monkeypatch.setattr(api, "_WORK_DIR", tmp_path)
    monkeypatch.setitem(api._state, "service", fake)
    test_client = TestClient(api.app)
    test_client.fake = fake
    test_client.work_dir = tmp_path
    return test_client


def _upload(client: TestClient, filename: str, annotate: bool = False):
    return client.post(
        "/analyse",
        params={"annotate": annotate},
        files={"clip": (filename, b"not really a video", "video/x-msvideo")},
    )


def test_the_clip_is_named_after_the_uploaded_file(client):
    """The bug this file was written for: the result was named after a random id."""
    response = _upload(client, f"{CLIP_NAME}.avi")

    assert response.status_code == 200
    body = response.json()
    assert body["clip_id"] == CLIP_NAME
    assert body["clip_id"] != body["request_id"]


def test_the_servers_own_file_path_is_never_sent_back(client):
    """A local path would give every caller this machine's folders and user name."""
    response = _upload(client, f"{CLIP_NAME}.avi", annotate=True)
    body = response.json()

    assert LOCAL_VIDEO_PATH_KEY not in body
    assert str(client.work_dir) not in response.text
    assert body["annotated_video_url"] == f"/analyse/{body['request_id']}/video"


def test_a_hostile_filename_cannot_choose_where_the_upload_is_written(client):
    """The upload is saved under a random id inside the work folder, whatever it is called."""
    _upload(client, "../../evil.avi")

    saved = client.fake.video_paths[0]
    assert saved.parent == client.work_dir
    assert "evil" not in saved.name


@pytest.mark.parametrize(
    ("filename", "status"),
    [
        ("notes.txt", 415),
        # ``.avi`` on its own is a hidden file with no extension, not a video.
        (".avi", 415),
        ("clip", 415),
        # A part with no filename is not a file upload at all, so FastAPI's own
        # request checking refuses it before the endpoint ever runs.
        ("", 422),
    ],
)
def test_anything_that_is_not_a_named_video_is_refused(client, filename, status):
    response = _upload(client, filename)
    assert response.status_code == status
    assert client.fake.clip_ids == []


@pytest.mark.parametrize(
    ("sent", "stem"),
    [
        (f"{CLIP_NAME}.avi", CLIP_NAME),
        (f"C:\\Users\\someone\\clips\\{CLIP_NAME}.avi", CLIP_NAME),
        (f"/home/someone/{CLIP_NAME}.avi", CLIP_NAME),
        ("../../evil.avi", "evil"),
    ],
)
def test_the_name_is_read_the_same_way_on_every_platform(sent, stem):
    """Name and extension both come from this one parse, folders stripped."""
    name = api._uploaded_name(sent)
    assert name.stem == stem
    assert name.suffix == ".avi"
