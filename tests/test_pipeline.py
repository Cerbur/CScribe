from datetime import datetime
from pathlib import Path

import pytest

from mimo_transcriber.config import AppConfig
from mimo_transcriber.devices import DeviceDecision
from mimo_transcriber.diarization import DiarizationError, DiarizationResult
from mimo_transcriber.models import AudioMetadata, SegmentStatus, SpeakerSegment
from mimo_transcriber.pipeline import (
    PipelineDependencies,
    prepare_audio_segments,
    run_pipeline,
)


def diarization_result(segments: list[SpeakerSegment]) -> DiarizationResult:
    return DiarizationResult(
        segments=segments,
        decision=DeviceDecision(
            requested_device="cpu",
            selected_device="cpu",
        ),
    )


@pytest.mark.asyncio
async def test_partial_failure_writes_output_and_returns_two(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 2, "aac", 48000, 2, datetime(2026, 1, 1, 9))

    async def transcribe(items, fail_fast):
        successful = items[0][0]
        successful.text = "你好"
        successful.status = SegmentStatus.SUCCESS
        failed = items[1][0]
        failed.text = "[该片段识别失败]"
        failed.status = SegmentStatus.FAILED
        failed.error = "timeout"
        return [successful, failed]

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
            SpeakerSegment(-1, 1, 2, "B"),
        ]),
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        transcribe=transcribe,
    )
    result = await run_pipeline(
        AppConfig(input_path=source, output_path=output, num_speakers=2),
        "mimo",
        "hf",
        dependencies,
    )
    assert result.exit_code == 2
    assert output.exists()
    assert "[该片段识别失败]" in output.read_text()


def test_oversize_segment_is_split_until_payload_is_accepted(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.wav"
    normalized.write_bytes(b"wav")
    calls: list[tuple[float, float]] = []

    def slice_audio(source: Path, segment: SpeakerSegment, target: Path) -> None:
        calls.append((segment.start, segment.end))
        target.write_bytes(b"mp3")

    def payload_fits(path: Path, segment: SpeakerSegment) -> bool:
        return segment.duration <= 2.5

    items = prepare_audio_segments(
        normalized,
        [SpeakerSegment(0, 0, 10, "A", "说话人 1")],
        tmp_path,
        slice_audio,
        payload_fits,
    )
    assert [(item.start, item.end) for item, _ in items] == [
        (0, 2.5),
        (2.5, 5),
        (5, 7.5),
        (7.5, 10),
    ]
    assert [item.index for item, _ in items] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_fail_fast_does_not_write_formal_output(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48000, 2, None)

    async def fail(items, fail_fast):
        raise RuntimeError("first failed segment")

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
        ]),
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, segment: True,
        transcribe=fail,
    )
    with pytest.raises(RuntimeError, match="first failed segment"):
        await run_pipeline(
            AppConfig(
                input_path=source,
                output_path=output,
                num_speakers=1,
                fail_fast=True,
            ),
            "mimo",
            "hf",
            dependencies,
        )
    assert output.exists() is False


@pytest.mark.asyncio
async def test_pipeline_normalizes_and_creates_preflight_only_once(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48_000, 2, None)
    calls: list[str] = []

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "完成"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    def normalize(source_path: Path, target: Path) -> None:
        calls.append("normalize")
        target.write_bytes(b"wav")

    def preflight(source_path: Path, target: Path) -> None:
        calls.append("preflight")
        target.write_bytes(b"sample")

    def diarize(*args, **kwargs):
        calls.append("diarize")
        return diarization_result([SpeakerSegment(-1, 0, 1, "A")])

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=normalize,
        create_preflight=preflight,
        diarize=diarize,
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, segment: True,
        transcribe=transcribe,
    )

    await run_pipeline(
        AppConfig(
            input_path=source,
            output_path=output,
            num_speakers=1,
            device="mps",
        ),
        "mimo",
        "hf",
        dependencies,
    )

    assert calls == ["normalize", "preflight", "diarize"]


@pytest.mark.asyncio
async def test_fatal_cpu_fallback_failure_stops_before_slicing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    metadata = AudioMetadata(source, 1, "aac", 48_000, 2, None)
    sliced: list[bool] = []

    def fail_diarization(*args, **kwargs):
        raise DiarizationError("MPS 完整运行失败，CPU 回退也失败")

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=fail_diarization,
        slice_audio=lambda source, segment, target: sliced.append(True),
    )

    with pytest.raises(DiarizationError, match="CPU 回退也失败"):
        await run_pipeline(
            AppConfig(input_path=source, device="mps"),
            "mimo",
            "hf",
            dependencies,
        )

    assert sliced == []
