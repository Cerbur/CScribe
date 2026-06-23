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
from mimo_transcriber.config import AppConfig
from mimo_transcriber.diarization import DiarizationResult, run_diarization
from mimo_transcriber.formatter import write_outputs
from mimo_transcriber.keywords import extract_keywords
from mimo_transcriber.manifest import ManifestStore, SegmentRecord, TaskIdentity, TaskManifest
from mimo_transcriber.mimo_asr import MiMoTranscriber, openai_request
from mimo_transcriber.models import RunSummary, SegmentStatus, SpeakerSegment, TranscriptionOutcome
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
) -> TaskManifest:
    try:
        manifest = store.load()
        logger.debug("从缓存加载 manifest: %s", store.path)
        store.reconcile_artifacts(manifest, paths.work_dir)
        store.reset_retryable_work(manifest)
        return manifest
    except (FileNotFoundError, Exception):
        logger.debug("初始化新 manifest: %s", store.path)
        metadata = probe_fn(config.input_path)
        identity = _identity_from_fingerprint(paths.task_hash, fingerprint)
        manifest = TaskManifest.new(identity, metadata)
        store.save(manifest)
        return manifest


async def run_pipeline(
    config: AppConfig,
    mimo_key: str,
    hf_token: str,
    dependencies: PipelineDependencies = PipelineDependencies(),
    cache_root: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> PipelineResult:
    if reporter is None:
        reporter = NullProgressReporter()
    if cache_root is None:
        cache_root = Path("/tmp/cscribe")

    started = time.monotonic()
    fingerprint = fingerprint_input(config.input_path)
    paths = TaskPaths.for_run(config, fingerprint, cache_root)

    try:
        with TaskLock(paths.lock) as _lock:
            store = ManifestStore(paths.manifest)
            manifest = _load_or_init_manifest(store, config, fingerprint, paths, dependencies.probe)

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
                    hf_token,
                    config.device,
                    config.num_speakers,
                    config.min_speakers,
                    config.max_speakers,
                )
                raw = diarization.segments
                segments = process_segments(raw, metadata.duration_seconds)
                manifest.diarization_status = "ready"
                manifest.diarization_device = diarization.decision.selected_device
                manifest.segments = [SegmentRecord.from_segment(seg) for seg in segments]
                store.save(manifest)

            reporter.set_segment_total(len(segments))

            # Stage 5: Slice and transcribe
            reporter.start_stage("正在处理音频片段")
            items = prepare_audio_segments(
                normalized,
                segments,
                paths.work_dir,
                dependencies.slice_audio,
                dependencies.payload_fits,
            )

            transcribe = dependencies.transcribe
            if transcribe is None:
                client = MiMoTranscriber(
                    request=openai_request(mimo_key),
                    language=config.language,
                    concurrency=config.concurrency,
                    requests_per_minute=config.requests_per_minute,
                    max_retries=config.max_retries,
                    reporter=reporter,
                )
                transcribe = client.transcribe_all

            completed = await transcribe(items, config.fail_fast)

            for seg in completed:
                for record in manifest.segments:
                    if record.segment.segment_id == seg.segment_id:
                        record.text = seg.text
                        record.transcript_status = str(seg.status)
                        record.error = seg.error
                        break

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

    except TaskAlreadyRunningError:
        reporter.close()
        raise


def _cleanup_task(paths: TaskPaths) -> None:
    import shutil
    if paths.work_dir.exists():
        shutil.rmtree(paths.work_dir, ignore_errors=True)
    if paths.target_index.exists():
        paths.target_index.unlink(missing_ok=True)
