import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from mimo_transcriber.asr.base import RuntimeConfig
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
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
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
            RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
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
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
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
            RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
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
            RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
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
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
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
        await run_pipeline(config, RuntimeConfig(hf_token="hf", mimo_api_key="mimo"), deps, cache_root=tmp_path)

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

    await run_pipeline(config, RuntimeConfig(hf_token="hf", mimo_api_key="mimo"), recover_deps, cache_root=tmp_path)

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
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
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
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
        deps,
        cache_root=tmp_path,
    )
    assert result.exit_code == 2
    assert tp.work_dir.exists()
    assert "[该片段识别失败]" in output.read_text()


@pytest.mark.asyncio
async def test_end_to_end_resume_from_interruption_and_complete(
    tmp_path: Path,
) -> None:
    """Simulate interruption after one segment succeeds, then recover."""
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 3, "aac", 48000, 2, None)

    from mimo_transcriber.cache import TaskPaths as TP, fingerprint_input as fp

    request_counts: dict[str, int] = {}
    normalize_count = 0
    diarize_count = 0

    def _normalize(src: Path, target: Path) -> None:
        nonlocal normalize_count
        normalize_count += 1
        target.write_bytes(b"wav")

    def _diarize(*args, **kwargs):
        nonlocal diarize_count
        diarize_count += 1
        return diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
            SpeakerSegment(-1, 1.5, 2.5, "B"),
        ])

    async def _interrupting_transcribe(items, fail_fast):
        segment = items[0][0]
        request_counts[segment.segment_id] = request_counts.get(segment.segment_id, 0) + 1
        segment.text = "ok"
        segment.status = SegmentStatus.SUCCESS
        raise asyncio.CancelledError()

    interrupting_deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=_normalize,
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=_diarize,
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, seg: True,
        transcribe=_interrupting_transcribe,
    )

    config = AppConfig(input_path=source, output_path=output, num_speakers=2)

    with pytest.raises(asyncio.CancelledError):
        await run_pipeline(config, RuntimeConfig(hf_token="hf", mimo_api_key="mimo"), interrupting_deps, cache_root=tmp_path)

    # Second run: all segments succeed
    async def _recover_transcribe(items, fail_fast):
        segment = items[0][0]
        request_counts[segment.segment_id] = request_counts.get(segment.segment_id, 0) + 1
        segment.text = "完成"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    recover_deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=_normalize,
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=_diarize,
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        transcribe=_recover_transcribe,
    )

    tp = TP.for_run(config, fp(source), tmp_path)
    second = await run_pipeline(config, RuntimeConfig(hf_token="hf", mimo_api_key="mimo"), recover_deps, cache_root=tmp_path)

    assert normalize_count == 1
    assert diarize_count == 1
    assert request_counts.get("s0000", 0) == 1
    assert request_counts.get("s0001", 0) == 1
    assert second.exit_code == 0
    assert output.exists()
    assert not tp.work_dir.exists()


@pytest.mark.asyncio
async def test_resume_with_many_ready_slices_does_not_deadlock(tmp_path: Path) -> None:
    """Regression: on resume, pre-sliced segments were enqueued into the bounded
    transcription queue *before* any transcription worker existed. With more ready
    slices than the queue capacity (maxsize = max(2, 2*concurrency) = 4), the
    enqueue loop blocked forever on the Nth put. This must complete, not hang."""
    from mimo_transcriber.cache import TaskPaths as TP, fingerprint_input as fp
    from mimo_transcriber.manifest import (
        ManifestStore,
        SegmentRecord,
        TaskIdentity,
        TaskManifest,
    )

    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 6, "aac", 48000, 2, None)
    config = AppConfig(input_path=source, output_path=output, num_speakers=6)

    fingerprint = fp(source)
    paths = TP.for_run(config, fingerprint, tmp_path)

    # Build a recovered manifest: normalize + diarize already done, six segments
    # whose slices already exist on disk and are marked ready (> queue capacity).
    paths.normalized.parent.mkdir(parents=True, exist_ok=True)
    paths.normalized.write_bytes(b"wav")
    paths.audio_dir.mkdir(parents=True, exist_ok=True)
    segments: list[SpeakerSegment] = []
    records: list[SegmentRecord] = []
    for index in range(6):
        seg = SpeakerSegment(index, float(index), float(index + 1), "A", segment_id=f"s{index:04d}")
        slice_path = paths.audio_dir / f"{seg.segment_id}.mp3"
        slice_path.write_bytes(b"mp3-bytes")
        segments.append(seg)
        records.append(
            SegmentRecord(
                segment=seg,
                slice_status="ready",
                slice_bytes=slice_path.stat().st_size,
                transcript_status="pending",
            )
        )
    manifest = TaskManifest(
        identity=TaskIdentity(
            task_hash=paths.task_hash,
            fingerprint_size=fingerprint.size,
            fingerprint_mtime_ns=fingerprint.mtime_ns,
            fingerprint_sha256=fingerprint.content_sha256,
        ),
        metadata_source_path=str(source),
        metadata_duration=metadata.duration_seconds,
        metadata_codec=metadata.codec,
        metadata_sample_rate=metadata.sample_rate,
        metadata_channels=metadata.channels,
        metadata_creation_time=None,
        normalize_status="ready",
        diarization_status="ready",
        segments=records,
    )
    ManifestStore(paths.manifest).save(manifest)

    transcribed: list[str] = []

    async def transcribe(items, fail_fast):
        for seg, _path in items:
            transcribed.append(seg.segment_id)
            seg.text = "完成"
            seg.status = SegmentStatus.SUCCESS
        return [item[0] for item in items]

    deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda src, target: target.write_bytes(b"wav"),
        create_preflight=lambda src, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([]),
        slice_audio=lambda src, seg, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, seg: True,
        transcribe=transcribe,
    )

    result = await asyncio.wait_for(
        run_pipeline(config, RuntimeConfig(hf_token="hf", mimo_api_key="mimo"), deps, cache_root=tmp_path),
        timeout=20.0,
    )

    assert result.exit_code == 0
    assert sorted(transcribed) == [f"s{i:04d}" for i in range(6)]
    assert output.exists()


@pytest.mark.asyncio
async def test_pipeline_records_success_through_manifest_projection(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48_000, 2, None)

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "projected"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
        ]),
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, segment: True,
        transcribe=transcribe,
    )

    result = await run_pipeline(
        AppConfig(input_path=source, output_path=output, num_speakers=1),
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
        deps,
        cache_root=tmp_path,
    )

    assert result.exit_code == 0
    assert result.outcome.segments[0].text == "projected"
    assert output.read_text(encoding="utf-8")


def test_pipeline_includes_terms_in_asr_cache_identity(tmp_path: Path) -> None:
    terms = tmp_path / "terms.txt"
    terms.write_text("Facebook\n", encoding="utf-8")
    config = AppConfig(
        input_path=tmp_path / "in.m4a",
        asr="mimo",
        asr_prompt="技术会议",
        terms_file=terms,
    )

    identity = config.asr_cache_identity()

    assert identity["settings"]["prompt_digest"].startswith("sha256:")
    assert identity["settings"]["term_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_passes_paragraph_config_to_formatter(tmp_path, monkeypatch):
    captured = {}

    def fake_write_outputs(outcome, recording_time, output_path, debug_json, paragraph_config=None):
        captured["paragraph_config"] = paragraph_config
        output_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("mimo_transcriber.pipeline.write_outputs", fake_write_outputs)

    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48000, 2, datetime(2026, 1, 1, 9))

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "完成"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
        ]),
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, segment: True,
        transcribe=transcribe,
    )

    await run_pipeline(
        AppConfig(
            input_path=source,
            output_path=output,
            num_speakers=1,
            paragraph_mode="aggressive",
        ),
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
        dependencies,
        cache_root=tmp_path,
    )

    assert captured["paragraph_config"].enabled is True
    assert captured["paragraph_config"].mode == "aggressive"
