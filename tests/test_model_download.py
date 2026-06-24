from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mimo_transcriber.asr.model_download import ModelDownloadError, ensure_model


class _FakeHub:
    """A tiny HTTP server mimicking the HF file/resolve routes with Range support."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        # filename -> list of HTTP statuses to return once each, then 200/206.
        self.glitches: dict[str, list[int]] = {}
        self.range_requests: list[str] = []
        self.get_count = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.fake = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _send_bytes(self, status: int, body: bytes, ctype: str, headers=None):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or []):
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_HEAD(self):  # noqa: N802
                path = self.path
                if "/resolve/main/" in path:
                    name = path.rsplit("/resolve/main/", 1)[1]
                    data = fake.files.get(name)
                    if data is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    self._send_bytes(200, b"", "application/octet-stream",
                                     [("Content-Length", str(len(data)))])
                    return
                self.send_response(404)
                self.end_headers()

            def do_GET(self):  # noqa: N802
                path = self.path
                if path.startswith("/api/models/"):
                    body = json.dumps(
                        {"siblings": [{"rfilename": f} for f in fake.files]}
                    ).encode()
                    self._send_bytes(200, body, "application/json")
                    return
                if "/resolve/main/" in path:
                    name = path.rsplit("/resolve/main/", 1)[1]
                    data = fake.files.get(name)
                    if data is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    # inject a one-off glitch if scheduled
                    queue = fake.glitches.get(name)
                    if queue:
                        status = queue.pop(0)
                        self.send_response(status)
                        self.end_headers()
                        return
                    fake.get_count += 1
                    range_header = self.headers.get("Range")
                    if range_header:
                        fake.range_requests.append(range_header)
                        start = int(range_header.split("=")[1].split("-")[0])
                        chunk = data[start:]
                        self.send_response(206)
                        self.send_header("Content-Length", str(len(chunk)))
                        self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
                        self.end_headers()
                        self.wfile.write(chunk)
                    else:
                        self._send_bytes(200, data, "application/octet-stream")
                    return
                self.send_response(404)
                self.end_headers()

        return Handler

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> "_FakeHub":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


REPO = "org/model"


def _seed_small_files(hub: _FakeHub) -> None:
    hub.files["config.json"] = b'{"a": 1}'
    hub.files["weights.safetensors"] = bytes(range(64))


def test_ensure_model_downloads_all_files(tmp_path: Path) -> None:
    with _FakeHub() as hub:
        _seed_small_files(hub)
        target = ensure_model(REPO, cache_dir=tmp_path, base_url=hub.base_url, max_retries=1)
        assert (target / "config.json").read_bytes() == b'{"a": 1}'
        assert (target / "weights.safetensors").read_bytes() == bytes(range(64))
        assert (target / ".complete").exists()


def test_ensure_model_is_idempotent_when_complete(tmp_path: Path) -> None:
    with _FakeHub() as hub:
        _seed_small_files(hub)
        ensure_model(REPO, cache_dir=tmp_path, base_url=hub.base_url, max_retries=1)
        before = hub.get_count
        target = ensure_model(REPO, cache_dir=tmp_path, base_url=hub.base_url, max_retries=1)
        # marker short-circuits: no further file fetches
        assert hub.get_count == before
        assert (target / ".complete").exists()


def test_ensure_model_resumes_partial_part_file(tmp_path: Path) -> None:
    with _FakeHub() as hub:
        _seed_small_files(hub)
        sanitized = REPO.replace("/", "__")
        target = tmp_path / sanitized
        target.mkdir(parents=True)
        # simulate an interrupted download: 10 bytes already present
        partial = bytes(range(10))
        (target / "weights.safetensors.part").write_bytes(partial)

        ensure_model(REPO, cache_dir=tmp_path, base_url=hub.base_url, max_retries=1)

        assert "bytes=10-" in hub.range_requests
        assert (target / "weights.safetensors").read_bytes() == bytes(range(64))


def test_ensure_model_retries_after_transient_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mimo_transcriber.asr.model_download._retry_backoff", lambda _: 0.0)
    with _FakeHub() as hub:
        _seed_small_files(hub)
        # weights fail once with 500, then succeed
        hub.glitches["weights.safetensors"] = [500]
        target = ensure_model(REPO, cache_dir=tmp_path, base_url=hub.base_url, max_retries=3)
        assert (target / "weights.safetensors").read_bytes() == bytes(range(64))


def test_ensure_model_raises_after_exhausting_retries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mimo_transcriber.asr.model_download._retry_backoff", lambda _: 0.0)
    with _FakeHub() as hub:
        hub.files["config.json"] = b'{"a": 1}'
        # every attempt fails
        hub.glitches["config.json"] = [500, 500, 500]
        with pytest.raises(ModelDownloadError, match="config.json"):
            ensure_model(REPO, cache_dir=tmp_path, base_url=hub.base_url, max_retries=3)
