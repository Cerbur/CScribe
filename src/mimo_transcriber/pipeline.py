from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from mimo_transcriber.audio import create_preflight_sample, normalize_audio, probe_audio, slice_mp3, payload_fits
from mimo_transcriber.cache import TaskAlreadyRunningError, TaskLock, TaskPaths, fingerprint_input
from mimo_transcriber.asr.base import AsrConfig, RuntimeConfig
from mimo_transcriber.asr.factory import create_asr_engine
from mimo_transcriber.config import AppConfig
from mimo_transcriber.events import (
    AudioSlice,
    SegmentTotalChanged,
    SliceFailed,
    SliceReady,
    TranscriptFailed,
    TranscriptSucceeded,
)
from mimo_transcriber.state_worker import RunStateProjector, run_state_worker
from mimo_transcriber.diarization import DiarizationResult, run_diarization
from mimo_transcriber.formatter import write_outputs
from mimo_transcriber.keywords import extract_keywords
from mimo_transcriber.manifest import ManifestStore, SegmentRecord, TaskIdentity, TaskManifest
from mimo_transcriber.models import RunSummary, SegmentStatus, SpeakerSegment, TranscriptionOutcome
from mimo_transcriber.paths import task_cache_dir
from mimo_transcriber.progress import NullProgressReporter, ProgressReporter
from mimo_transcriber.segments import process_segments, split_segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineDependencies:
    probe: Callable[[Path], Any] = probe_audio
    normalize: Callable[[Path, Path], None] = normalize_audio
    create_preflight: Callable[[Path, Path], None] = create_preflight_sample
    diarize: Callable[..., DiarizationResult] = run_diarization
    slice_audio: Callable[[Path, SpeakerSegment, Path], None] = slice_mp3
    payload_fits: Callable[[Path, SpeakerSegment], bool] = payload_fits
    transcribe: Callable[
        [list[tuple[SpeakerSegment, Path]], bool],
        Awaitable[list[SpeakerSegment]],
    ] | None = None


@dataclass(frozen=True)
class PipelineResult:
    outcome: TranscriptionOutcome
    exit_code: int


def prepare_audio_segments(
    normalized: Path,
    segments: list[SpeakerSegment],
    temp: Path,
    slice_audio: Callable[[Path, SpeakerSegment, Path], None],
    fits: Callable[[Path, SpeakerSegment], bool],
) -> list[tuple[SpeakerSegment, Path]]:
    pending = list(segments)
    accepted: list[tuple[SpeakerSegment, Path]] = []
    while pending:
        segment = pending.pop(0)
        path = temp / "candidate.mp3"
        slice_audio(normalized, segment, path)
        if fits(path, segment):
            accepted.append((segment, path.with_name(f"segment_{len(accepted):04d}.mp3")))
            path.replace(accepted[-1][1])
        else:
            path.unlink(missing_ok=True)
            pending[0:0] = split_segment(segment)
    for index, (segment, _) in enumerate(accepted):
        segment.index = index
    return accepted


def _identity_from_fingerprint(task_hash: str, fingerprint) -> TaskIdentity:
    return TaskIdentity(
        task_hash=task_hash,
        fingerprint_size=fingerprint.size,
        fingerprint_mtime_ns=fingerprint.mtime_ns,
        fingerprint_sha256=fingerprint.content_sha256,
    )


def _load_or_init_manifest(
    store: ManifestStore,
    config: AppConfig,
    fingerprint,
    paths: TaskPaths,
    probe_fn: Callable[[Path], Any],
) -> tuple[TaskManifest, bool]:
    import shutil

    def _init_new() -> TaskManifest:
        logger.debug("初始化新 manifest: %s", store.path)
        metadata = probe_fn(config.input_path)
        identity = _identity_from_fingerprint(paths.task_hash, fingerprint)
        manifest = TaskManifest.new(identity, metadata)
        store.save(manifest)
        return manifest

    try:
        manifest = store.load()
        if manifest.identity.fingerprint_sha256 != fingerprint.content_sha256:
            logger.info("Manifest 指纹不匹配，清理旧缓存并重新初始化")
            if paths.work_dir.exists():
                shutil.rmtree(paths.work_dir, ignore_errors=True)
            return _init_new(), True
        logger.debug("从缓存加载 manifest: %s", store.path)
        store.reconcile_artifacts(manifest, paths.work_dir)
        store.reset_retryable_work(manifest)
        return manifest, False
    except (FileNotFoundError, Exception):
        return _init_new(), True


def _manage_target_index(paths: TaskPaths) -> None:
    """原子管理 target_index：检测陈旧缓存并写入当前 task_hash。"""
    import json as _json
    import os as _os
    import shutil as _shutil

    paths.target_index.parent.mkdir(parents=True, exist_ok=True)
    if paths.target_index.exists():
        try:
            existing = _json.loads(paths.target_index.read_text(encoding="utf-8"))
            old_hash = existing.get("task_hash")
            if old_hash and old_hash != paths.task_hash:
                old_work_dir = paths.root / old_hash
                if old_work_dir.exists():
                    logger.info("输入指纹变化，清理旧工作目录: %s", old_work_dir)
                    _shutil.rmtree(old_work_dir, ignore_errors=True)
        except Exception:
            logger.debug("读取 target_index 失败，忽略并覆盖", exc_info=True)

    tmp_index = paths.target_index.with_name(
        f".{paths.target_index.name}.tmp"
    )
    payload = _json.dumps({"task_hash": paths.task_hash}, ensure_ascii=False)
    tmp_index.write_text(payload, encoding="utf-8")
    _os.replace(tmp_index, paths.target_index)


async def run_pipeline(
    config: AppConfig,
    runtime: RuntimeConfig,
    dependencies: PipelineDependencies = PipelineDependencies(),
    cache_root: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> PipelineResult:
    if reporter is None:
        reporter = NullProgressReporter()
    if cache_root is None:
        cache_root = task_cache_dir()

    started = time.monotonic()
    fingerprint = fingerprint_input(config.input_path)
    paths = TaskPaths.for_run(config, fingerprint, cache_root)

    try:
        with TaskLock(paths.lock) as _lock:
            # Manage target_index: detect stale cache, then atomically write current task_hash
            _manage_target_index(paths)

            store = ManifestStore(paths.manifest)
            manifest, _ = _load_or_init_manifest(store, config, fingerprint, paths, dependencies.probe)

            # Stage 1: Audio probe (recoverable)
            reporter.start_stage("音频探测")
            if manifest.normalize_status == "ready" and paths.normalized.exists() and paths.normalized.stat().st_size > 0:
                metadata = dependencies.probe(config.input_path)
                logger.debug("音频探测复用缓存")
            else:
                metadata = dependencies.probe(config.input_path)

            # Stage 2: Normalize (recoverable)
            reporter.start_stage("标准化音频")
            normalized = paths.normalized
            if manifest.normalize_status == "ready" and normalized.exists() and normalized.stat().st_size > 0:
                logger.debug("标准化音频复用缓存")
            else:
                normalized.parent.mkdir(parents=True, exist_ok=True)
                dependencies.normalize(config.input_path, normalized)
                manifest.normalize_status = "ready"
                store.save(manifest)

            # Stage 3: Preflight
            reporter.start_stage("MPS 预检")
            preflight = paths.preflight
            if normalized.exists():
                preflight.parent.mkdir(parents=True, exist_ok=True)
                dependencies.create_preflight(normalized, preflight)

            # Stage 4: Diarization (recoverable)
            reporter.start_stage("说话人分离")
            if manifest.diarization_status == "ready" and manifest.segments:
                # Recover segments from manifest
                logger.debug("说话人分离复用缓存，片段数: %d", len(manifest.segments))
                segments = [item.segment for item in manifest.segments]
                reporter.set_segment_total(len(segments))
            else:
                diarization = dependencies.diarize(
                    normalized,
                    preflight,
                    runtime.hf_token,
                    config.device,
                    config.resolved_num_speakers(),
                    config.min_speakers,
                    config.max_speakers,
                    model_id=config.diarization_model,
                )
                raw = diarization.segments
                segments = process_segments(raw, metadata.duration_seconds)
                manifest.diarization_status = "ready"
                manifest.diarization_device = diarization.decision.selected_device
                manifest.segments = [SegmentRecord.from_segment(seg) for seg in segments]
                store.save(manifest)

            reporter.set_segment_total(len(segments))

            # Stage 5: Slice and transcribe with bounded pipeline
            reporter.start_stage("正在处理音频片段")
            completed = await _run_segment_workers(
                config=config,
                normalized=normalized,
                segments=segments,
                manifest=manifest,
                store=store,
                paths=paths,
                dependencies=dependencies,
                runtime=runtime,
                reporter=reporter,
            )

            outcome_has_failure = any(
                item.status is SegmentStatus.FAILED for item in completed
            )
            successful_texts = [
                item.text or "" for item in completed if item.status is SegmentStatus.SUCCESS
            ]
            outcome = TranscriptionOutcome(
                metadata=metadata,
                segments=completed,
                keywords=extract_keywords(successful_texts, config.keyword_count),
                summary=RunSummary(
                    elapsed_seconds=time.monotonic() - started,
                    speakers=len({item.raw_speaker for item in completed}),
                    segments=len(completed),
                    succeeded=sum(item.status is SegmentStatus.SUCCESS for item in completed),
                    failed=sum(item.status is SegmentStatus.FAILED for item in completed),
                    output_path=config.resolved_output_path,
                    temp_path=paths.work_dir if outcome_has_failure else None,
                ),
            )

            recording_time = metadata.creation_time or datetime.fromtimestamp(
                config.input_path.stat().st_mtime
            )
            write_outputs(
                outcome, recording_time, config.resolved_output_path, config.debug_json
            )

            manifest.output_written = True
            store.save(manifest)

            exit_code = 2 if outcome_has_failure else 0
            reporter.finish(
                outcome.summary.succeeded,
                outcome.summary.failed,
                outcome.summary.elapsed_seconds,
            )

            if not outcome_has_failure:
                _cleanup_task(paths)

            return PipelineResult(outcome, exit_code)

    except asyncio.CancelledError:
        reporter.finish(0, 0, time.monotonic() - started)
        reporter.close()
        raise
    except TaskAlreadyRunningError:
        reporter.close()
        raise


async def _run_segment_workers(
    config: AppConfig,
    normalized: Path,
    segments: list[SpeakerSegment],
    manifest: TaskManifest,
    store: ManifestStore,
    paths: TaskPaths,
    dependencies: PipelineDependencies,
    runtime: RuntimeConfig,
    reporter: ProgressReporter,
) -> list[SpeakerSegment]:
    import json as _json
    import os as _os

    # Determine work items from manifest records
    slice_queue: asyncio.Queue[SegmentRecord | None] = asyncio.Queue()
    transcribe_queue: asyncio.Queue[tuple[SegmentRecord, Path] | None] = asyncio.Queue(
        maxsize=max(2, 2 * config.concurrency)
    )

    completed: dict[str, SpeakerSegment] = {}
    cancel_event = asyncio.Event()

    state_queue: asyncio.Queue[object | None] = asyncio.Queue()
    projector = RunStateProjector(manifest, store, reporter)
    state_task = asyncio.create_task(run_state_worker(state_queue, projector))

    # Count recovered (already successful) segments
    segments_by_id: dict[str, SpeakerSegment] = {s.segment_id: s for s in segments}

    # Ensure manifest records match current segments
    existing_ids = {r.segment.segment_id: r for r in manifest.segments}
    work_items: list[SegmentRecord] = []
    for seg in segments:
        record = existing_ids.get(seg.segment_id, SegmentRecord.from_segment(seg))
        record.segment = seg
        work_items.append(record)
        # Count already-successful transcripts
        if record.transcript_status == "success":
            completed[seg.segment_id] = seg
            await state_queue.put(
                TranscriptSucceeded(seg.segment_id, seg.text or "")
            )
    manifest.segments = work_items
    store.save(manifest)

    logger.debug(
        "恢复已完成片段: %d",
        sum(1 for _ in work_items if _.transcript_status == "success"),
    )

    # Pre-create ASR engine for transcription workers
    if dependencies.transcribe is None:
        client = create_asr_engine(
            AsrConfig(
                provider=config.asr,
                stt_model=config.stt_model,
                language=config.language,
            ),
            runtime,
            event_sink=None,
            concurrency=config.concurrency,
            requests_per_minute=config.requests_per_minute,
            max_retries=config.max_retries,
        )
    else:
        client = None

    # Enqueue work: pending/failed segments need slicing; ready slices go direct to transcription.
    # Pre-sliced segments are collected and fed to the transcription queue only AFTER the
    # transcription workers start. The queue is bounded, so enqueueing into it before any
    # consumer exists would block forever once it fills (more ready slices than capacity) and
    # deadlock the pipeline on resume.
    ready_to_transcribe: list[tuple[SegmentRecord, Path]] = []
    pending = 0
    paths.audio_dir.mkdir(parents=True, exist_ok=True)
    for record in work_items:
        if record.segment.segment_id in completed:
            continue
        if record.slice_status == "ready":
            audio_path = paths.audio_dir / f"{record.segment.segment_id}.mp3"
            if audio_path.exists() and audio_path.stat().st_size > 0:
                ready_to_transcribe.append((record, audio_path))
                pending += 1
                continue
        # Need slicing
        record.slice_status = "pending"
        record.slice_bytes = 0
        slice_queue.put_nowait(record)
        pending += 1

    if pending == 0:
        logger.debug("所有片段已完成，无待处理工作")
        await state_queue.put(None)
        await state_task
        return sorted(
            projector.snapshot_completed_segments(),
            key=lambda s: s.sort_key(),
        )

    # Slice worker
    async def _slice_worker() -> None:
        while True:
            record = await slice_queue.get()
            if record is None or cancel_event.is_set():
                slice_queue.task_done()
                break
            try:
                seg = record.segment
                if seg.segment_id in segments_by_id:
                    seg = segments_by_id[seg.segment_id]
                candidate = paths.audio_dir / f"{seg.segment_id}_tmp.mp3"
                target = paths.audio_dir / f"{seg.segment_id}.mp3"

                await asyncio.to_thread(dependencies.slice_audio, normalized, seg, candidate)

                if dependencies.payload_fits(candidate, seg):
                    size = candidate.stat().st_size
                    _os.replace(candidate, target)
                    record.slice_status = "ready"
                    record.slice_bytes = size
                    await state_queue.put(SliceReady(seg.segment_id, target, size))
                    await transcribe_queue.put((record, target))
                else:
                    candidate.unlink(missing_ok=True)
                    children = split_segment(seg)
                    logger.debug(
                        "[%s] 切片过大 (duration=%.1fs)，拆分为 %d 个子片段",
                        seg.segment_id, seg.duration, len(children),
                    )
                    child_records = [SegmentRecord.from_segment(c) for c in children]
                    # Replace parent record in manifest with children atomically
                    idx = next(i for i, r in enumerate(manifest.segments) if r.segment.segment_id == record.segment.segment_id)
                    manifest.segments[idx:idx + 1] = child_records
                    await state_queue.put(SegmentTotalChanged(len(manifest.segments)))
                    store.save(manifest)
                    for cr in child_records:
                        if cr.segment.segment_id not in segments_by_id:
                            segments_by_id[cr.segment.segment_id] = cr.segment
                        slice_queue.put_nowait(cr)
            except Exception as exc:
                logger.error("切片失败 %s: %s", record.segment.segment_id, exc)
                record.slice_status = "failed"
                record.error = str(exc)
                if record.segment.segment_id in segments_by_id:
                    segments_by_id[record.segment.segment_id] = record.segment
                await state_queue.put(
                    SliceFailed(record.segment.segment_id, str(exc))
                )
            finally:
                slice_queue.task_done()

    # Transcription worker
    async def _transcribe_worker() -> None:
        while True:
            entry = await transcribe_queue.get()
            if entry is None or cancel_event.is_set():
                transcribe_queue.task_done()
                break
            record, audio_path = entry
            try:
                try:
                    if client is not None:
                        result = await client.transcribe_one(record.segment, audio_path)
                        record.text = result.text
                        record.transcript_status = str(result.status) if hasattr(result, 'status') else "success"
                        record.error = result.error if hasattr(result, 'error') else None
                    else:
                        batch_result = await dependencies.transcribe(
                            [(record.segment, audio_path)], False
                        )
                        result = batch_result[0] if isinstance(batch_result, list) else batch_result
                        record.text = result.text
                        record.transcript_status = str(result.status) if hasattr(result, 'status') else "success"
                        record.error = result.error if hasattr(result, 'error') else None

                    if record.segment.segment_id in segments_by_id:
                        segments_by_id[record.segment.segment_id].text = record.text
                        segments_by_id[record.segment.segment_id].status = SegmentStatus(
                            record.transcript_status
                        )

                    if (
                        record.transcript_status == "failed"
                        or record.segment.status is SegmentStatus.FAILED
                    ):
                        await state_queue.put(
                            TranscriptFailed(
                                record.segment.segment_id,
                                record.error or "transcription failed",
                            )
                        )
                    else:
                        await state_queue.put(
                            TranscriptSucceeded(
                                record.segment.segment_id,
                                record.text or "",
                            )
                        )
                except asyncio.CancelledError:
                    # Sync record from segment in case it was modified in-place before cancel
                    record.text = record.segment.text
                    record.transcript_status = str(record.segment.status)
                    if record.segment.segment_id in segments_by_id:
                        segments_by_id[record.segment.segment_id].text = record.text
                        segments_by_id[record.segment.segment_id].status = SegmentStatus(
                            record.transcript_status
                        )
                    await state_queue.put(
                        TranscriptSucceeded(
                            record.segment.segment_id,
                            record.segment.text or "",
                        )
                    )
                    raise
                except Exception as exc:
                    if config.fail_fast:
                        cancel_event.set()
                    logger.error(
                        "转写失败 %s (duration=%.1fs, %s): %s",
                        record.segment.segment_id, record.segment.duration,
                        type(exc).__name__, str(exc)[:300],
                    )
                    record.transcript_status = "failed"
                    record.error = f"{type(exc).__name__}: {exc}"
                    record.segment.text = "[该片段识别失败]"
                    record.segment.status = SegmentStatus.FAILED
                    await state_queue.put(
                        TranscriptFailed(
                            record.segment.segment_id,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
            finally:
                transcribe_queue.task_done()

    # Start workers
    slice_workers = [asyncio.create_task(_slice_worker()) for _ in range(2)]
    transcribe_workers = [
        asyncio.create_task(_transcribe_worker())
        for _ in range(config.concurrency)
    ]

    # Feed pre-sliced segments now that transcription workers are draining the queue.
    # The bounded put applies backpressure but can no longer deadlock.
    for record, audio_path in ready_to_transcribe:
        await transcribe_queue.put((record, audio_path))

    try:
        # Wait for all work to be queued
        await slice_queue.join()
        # Signal slice workers to stop
        for _ in slice_workers:
            slice_queue.put_nowait(None)
        await asyncio.gather(*slice_workers)

        # Wait for transcription to complete
        await transcribe_queue.join()
        for _ in transcribe_workers:
            transcribe_queue.put_nowait(None)
        await asyncio.gather(*transcribe_workers)

        # Close state worker
        await state_queue.join()
        await state_queue.put(None)
        await state_task

    except asyncio.CancelledError:
        cancel_event.set()
        # Drain queues and cancel workers
        while not slice_queue.empty():
            slice_queue.get_nowait()
            slice_queue.task_done()
        while not transcribe_queue.empty():
            transcribe_queue.get_nowait()
            transcribe_queue.task_done()
        for w in slice_workers + transcribe_workers:
            w.cancel()
        state_task.cancel()
        await asyncio.gather(
            *slice_workers, *transcribe_workers, state_task,
            return_exceptions=True,
        )
        store.save(manifest)
        raise

    if cancel_event.is_set() and config.fail_fast:
        store.save(manifest)
        # Find the first failed segment error
        for record in manifest.segments:
            if record.transcript_status == "failed":
                raise RuntimeError(record.error or "片段识别失败")

    return sorted(
        projector.snapshot_completed_segments(),
        key=lambda item: item.sort_key(),
    )


def _cleanup_task(paths: TaskPaths) -> None:
    import shutil
    if paths.work_dir.exists():
        shutil.rmtree(paths.work_dir, ignore_errors=True)
    if paths.target_index.exists():
        paths.target_index.unlink(missing_ok=True)
