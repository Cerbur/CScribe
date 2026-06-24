from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mimo_transcriber.asr.mimo import MimoAsrEngine, extract_content, openai_request
from mimo_transcriber.models import SegmentStatus, SpeakerSegment
from mimo_transcriber.progress import NullProgressReporter


class RecordingReporter(NullProgressReporter):
    def __init__(self) -> None:
        self.retries: list[tuple[str, int, int]] = []
        self.completions: list[tuple[bool, str]] = []

    def segment_retrying(self, segment_id: str, retry_number: int, max_retries: int) -> None:
        self.retries.append((segment_id, retry_number, max_retries))

    def segment_completed(self, success: bool) -> None:
        self.completions.append((success, ""))


def test_extract_content_handles_string_and_content_objects() -> None:
    assert extract_content("  你好   world  ") == "你好 world"
    assert extract_content([SimpleNamespace(text=" hello "), {"text": "世界"}]) == "hello 世界"


@pytest.mark.asyncio
async def test_retries_then_returns_results_in_index_order(tmp_path: Path) -> None:
    calls = 0

    async def request(data_url: str, language: str) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("slow")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f" text {calls} "))]
        )

    async def no_sleep(seconds: float) -> None:
        return None

    paths = []
    for index in range(2):
        path = tmp_path / f"segment_{index:04d}.mp3"
        path.write_bytes(b"audio")
        paths.append(path)
    segments = [
        SpeakerSegment(1, 1, 2, "B"),
        SpeakerSegment(0, 0, 1, "A"),
    ]
    transcriber = MimoAsrEngine(
        request=request,
        model="mimo-v2.5-asr",
        language="auto",
        concurrency=1,
        requests_per_minute=1000,
        max_retries=1,
        sleep=no_sleep,
    )
    result = await transcriber.transcribe_all(list(zip(segments, paths)), False)
    assert [item.index for item in result] == [0, 1]
    assert all(item.status is SegmentStatus.SUCCESS for item in result)
    assert calls == 3


@pytest.mark.asyncio
async def test_retry_events_report_one_through_three(tmp_path: Path) -> None:
    reporter = RecordingReporter()

    class HttpError(RuntimeError):
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def completion(text: str) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    async def no_sleep(seconds: float) -> None:
        return None

    request = AsyncMock(side_effect=[
        TimeoutError("1"),
        ConnectionError("2"),
        HttpError(500),
        completion("ok"),
    ])
    audio_path = tmp_path / "s0000.mp3"
    audio_path.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    transcriber = MimoAsrEngine(
        request=request,
        model="mimo-v2.5-asr",
        language="auto",
        concurrency=1,
        requests_per_minute=1000,
        max_retries=3,
        reporter=reporter,
        sleep=no_sleep,
    )
    result = await transcriber.transcribe_one(segment, audio_path)
    assert reporter.retries == [
        ("s0000", 1, 3),
        ("s0000", 2, 3),
        ("s0000", 3, 3),
    ]
    assert result.status is SegmentStatus.SUCCESS


@pytest.mark.asyncio
async def test_empty_response_is_not_retried(tmp_path: Path) -> None:
    async def request(data_url: str, language: str) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" "))]
        )

    async def no_sleep(seconds: float) -> None:
        return None

    audio_path = tmp_path / "s0000.mp3"
    audio_path.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    request_mock = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=" "))]
    ))
    transcriber = MimoAsrEngine(
        request=request_mock,
        model="mimo-v2.5-asr",
        language="auto",
        concurrency=1,
        requests_per_minute=1000,
        max_retries=3,
        reporter=RecordingReporter(),
        sleep=no_sleep,
    )
    result = await transcriber.transcribe_one(segment, audio_path)
    assert request_mock.await_count == 1
    assert result.status is SegmentStatus.FAILED


@pytest.mark.asyncio
async def test_openai_request_uses_configured_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: Client())

    request = openai_request("key", model="custom-model")
    await request("data:audio/mp3;base64,abc", "en")

    assert captured["model"] == "custom-model"


@pytest.mark.asyncio
async def test_openai_request_sends_prompt(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr("openai.AsyncOpenAI", FakeClient)

    request = openai_request("key", prompt="Facebook prompt")
    await request("data:audio/mp3;base64,abc", "zh")

    assert captured["extra_body"]["asr_options"]["prompt"] == "Facebook prompt"


@pytest.mark.asyncio
async def test_mimo_engine_applies_explicit_term_corrections(tmp_path: Path) -> None:
    audio = tmp_path / "s.mp3"
    audio.write_bytes(b"audio")

    class Message:
        content = "飞书 和 格拉布"

    class Choice:
        message = Message()

    class Completion:
        choices = [Choice()]

    async def request(data_url, language):
        return Completion()

    engine = MimoAsrEngine(
        request=request,
        model="mimo-v2.5-asr",
        language="zh",
        concurrency=1,
        requests_per_minute=60,
        max_retries=0,
        term_replacements={"飞书": "Facebook", "格拉布": "Grab"},
    )

    result = await engine.transcribe_one(
        SpeakerSegment(0, 0, 1, "A", segment_id="s0000"), audio
    )

    assert result.text == "Facebook 和 Grab"
