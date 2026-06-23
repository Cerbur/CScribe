from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_transcriber.mimo_asr import MiMoTranscriber, extract_content
from mimo_transcriber.models import SegmentStatus, SpeakerSegment


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
    transcriber = MiMoTranscriber(
        request=request,
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
