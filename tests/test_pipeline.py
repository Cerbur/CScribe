import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

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
        segment = items[0][0]
        if segment.segment_id == "s0000":
            segment.text = "你好"
            segment.status = SegmentStatus.SUCCESS
        else:
            segment.text = "[该片段识别失败]"
            segment.status = SegmentStatus.FAILED
            segment.error = "timeout"
        return [segment]

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


@pytest.mark.asyncio
async def test_recovery_skips_ready_normalization_and_diarization(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 2, "aac", 48000, 2, datetime(2026, 1, 1, 9))

    norm_calls = 0
    diar_calls = 0

    def _normalize(src: Path, target: Path) -> None:
        nonlocal norm_calls
        norm_calls += 1
        target.write_bytes(b"wav")

    def _diarize(*args, **kwargs):
        nonlocal diar_calls
        diar_calls += 1
        return diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
            SpeakerSegment(-1, 1, 2, "B"),
        ])

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "ok"
        segment.status = SegmentStatus.SUCCESS
        raise asyncio.CancelledError()

    deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=_normalize,
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=_diarize,
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        transcribe=transcribe,
    )

    with pytest.raises(asyncio.CancelledError):
        await run_pipeline(
            AppConfig(input_path=source, output_path=output, num_speakers=2),
            "mimo",
            "hf",
            deps,
            cache_root=tmp_path,
        )

    assert norm_calls == 1
    assert diar_calls == 1

    async def _recover_transcribe(items, fail_fast):
        for seg, _path in items:
            seg.text = "完成"
            seg.status = SegmentStatus.SUCCESS
        return [item[0] for item in items]

    recover_deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=_normalize,
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=_diarize,
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        transcribe=_recover_transcribe,
    )

    await run_pipeline(
        AppConfig(input_path=source, output_path=output, num_speakers=2),
        "mimo",
        "hf",
        recover_deps,
        cache_root=tmp_path,
    )

    assert norm_calls == 1
    assert diar_calls == 1
    assert output.exists()


@pytest.mark.asyncio
async def test_successful_cached_transcript_is_not_requested_again(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 3, "aac", 48000, 2, None)

    request_counts: dict[str, int] = {}

    async def _transcribe_counting(items, fail_fast):
        for seg, _path in items:
            request_counts[seg.segment_id] = request_counts.get(seg.segment_id, 0) + 1
            seg.text = "ok"
            seg.status = SegmentStatus.SUCCESS
        raise asyncio.CancelledError()

    deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda src, target: target.write_bytes(b"wav"),
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
            SpeakerSegment(-1, 1.5, 2.5, "B"),
        ]),
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, seg: True,
        transcribe=_transcribe_counting,
    )

    config = AppConfig(input_path=source, output_path=output, num_speakers=2)
    with pytest.raises(asyncio.CancelledError):
        await run_pipeline(config, "mimo", "hf", deps, cache_root=tmp_path)

    assert request_counts.get("s0000", 0) == 1
    assert request_counts.get("s0001", 0) == 1

    async def _recover_transcribe(items, fail_fast):
        for seg, _path in items:
            request_counts[seg.segment_id] = request_counts.get(seg.segment_id, 0) + 1
            seg.text = "完成"
            seg.status = SegmentStatus.SUCCESS
        return [item[0] for item in items]

    recover_deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda src, target: target.write_bytes(b"wav"),
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
            SpeakerSegment(-1, 1.5, 2.5, "B"),
        ]),
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, seg: True,
        transcribe=_recover_transcribe,
    )

    await run_pipeline(config, "mimo", "hf", recover_deps, cache_root=tmp_path)

    assert request_counts["s0000"] == 1
    assert request_counts["s0001"] == 1


def test_oversize_segment_retains_stable_base_id_and_derives_children(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "normalized.wav"
    normalized.write_bytes(b"wav")

    def slice_audio(source: Path, segment: SpeakerSegment, target: Path) -> None:
        target.write_bytes(b"mp3")

    def fits(path: Path, segment: SpeakerSegment) -> bool:
        return segment.duration <= 7.0

    items = prepare_audio_segments(
        normalized,
        [SpeakerSegment(0, 0, 10, "A", segment_id="s0000")],
        tmp_path,
        slice_audio,
        fits,
    )
    assert len(items) == 2
    child_ids = {item.segment_id for item, _ in items}
    assert "s0000.0" in child_ids
    assert "s0000.1" in child_ids


@pytest.mark.asyncio
async def test_success_removes_task_cache_but_keeps_debug_json_when_requested(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48000, 2, None)

    async def _transcribe_ok(items, fail_fast):
        seg = items[0][0]
        seg.text = "你好"
        seg.status = SegmentStatus.SUCCESS
        return [seg]

    deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda src, target: target.write_bytes(b"wav"),
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
        ]),
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        transcribe=_transcribe_ok,
    )

    from mimo_transcriber.cache import TaskPaths as TP, fingerprint_input as fp
    tp = TP.for_run(
        AppConfig(input_path=source, output_path=output, debug_json=True),
        fp(source),
        tmp_path,
    )

    result = await run_pipeline(
        AppConfig(input_path=source, output_path=output, debug_json=True, num_speakers=1),
        "mimo",
        "hf",
        deps,
        cache_root=tmp_path,
    )
    assert result.exit_code == 0
    assert output.exists()
    assert output.with_suffix(".segments.json").exists()
    assert not tp.work_dir.exists()


@pytest.mark.asyncio
async def test_partial_failure_keeps_cache_and_returns_two(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 2, "aac", 48000, 2, datetime(2026, 1, 1, 9))

    from mimo_transcriber.cache import TaskPaths as TP, fingerprint_input as fp

    async def _transcribe_partial(items, fail_fast):
        segment = items[0][0]
        if segment.segment_id == "s0000":
            segment.text = "你好"
            segment.status = SegmentStatus.SUCCESS
        else:
            segment.text = "[该片段识别失败]"
            segment.status = SegmentStatus.FAILED
            segment.error = "timeout"
        return [segment]

    deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda src, target: target.write_bytes(b"wav"),
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
            SpeakerSegment(-1, 1, 2, "B"),
        ]),
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        transcribe=_transcribe_partial,
    )

    tp = TP.for_run(
        AppConfig(input_path=source, output_path=output, num_speakers=2),
        fp(source),
        tmp_path,
    )

    result = await run_pipeline(
        AppConfig(input_path=source, output_path=output, num_speakers=2),
        "mimo",
        "hf",
        deps,
        cache_root=tmp_path,
    )
    assert result.exit_code == 2
    assert tp.work_dir.exists()
    assert "[该片段识别失败]" in output.read_text()
