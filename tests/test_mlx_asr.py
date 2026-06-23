from pathlib import Path

import pytest

from mimo_transcriber.asr.base import AsrConfig
from mimo_transcriber.asr.mlx import MlxAsrEngine
from mimo_transcriber.models import SegmentStatus, SpeakerSegment


@pytest.mark.asyncio
async def test_mlx_engine_transcribes_text(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, **kwargs})
        return {"text": " hello  world "}

    audio = tmp_path / "s0000.mp3"
    audio.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    engine = MlxAsrEngine(AsrConfig(provider="mlx", language="zh"), fake_transcribe)

    result = await engine.transcribe_one(segment, audio)

    assert result.text == "hello world"
    assert result.status is SegmentStatus.SUCCESS
    assert calls == [{
        "path": str(audio),
        "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
        "language": "zh",
    }]


@pytest.mark.asyncio
async def test_mlx_engine_omits_language_for_auto(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"text": "ok"}

    audio = tmp_path / "s0000.mp3"
    audio.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    engine = MlxAsrEngine(AsrConfig(provider="mlx"), fake_transcribe)

    await engine.transcribe_one(segment, audio)

    assert "language" not in calls[0]


@pytest.mark.asyncio
async def test_mlx_engine_marks_empty_text_failed(tmp_path: Path) -> None:
    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        return {"text": " "}

    audio = tmp_path / "s0000.mp3"
    audio.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    engine = MlxAsrEngine(AsrConfig(provider="mlx"), fake_transcribe)

    result = await engine.transcribe_one(segment, audio)

    assert result.status is SegmentStatus.FAILED
    assert result.text == "[该片段识别失败]"


@pytest.mark.asyncio
async def test_mlx_engine_returns_results_in_sort_order(tmp_path: Path) -> None:
    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        return {"text": Path(path).stem}

    paths = []
    for name in ("s0001.mp3", "s0000.mp3"):
        path = tmp_path / name
        path.write_bytes(b"audio")
        paths.append(path)
    segments = [
        SpeakerSegment(1, 1, 2, "B", segment_id="s0001"),
        SpeakerSegment(0, 0, 1, "A", segment_id="s0000"),
    ]
    engine = MlxAsrEngine(AsrConfig(provider="mlx"), fake_transcribe)

    result = await engine.transcribe_all(list(zip(segments, paths)), False)

    assert [item.segment_id for item in result] == ["s0000", "s0001"]
