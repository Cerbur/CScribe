from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from mimo_transcriber.audio import (
    create_preflight_sample,
    normalize_audio,
    payload_fits,
    probe_audio,
    slice_mp3,
    workspace,
)
from mimo_transcriber.config import AppConfig
from mimo_transcriber.diarization import DiarizationResult, run_diarization
from mimo_transcriber.formatter import write_outputs
from mimo_transcriber.keywords import extract_keywords
from mimo_transcriber.mimo_asr import MiMoTranscriber, openai_request
from mimo_transcriber.models import RunSummary, SegmentStatus, SpeakerSegment, TranscriptionOutcome
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


async def run_pipeline(
    config: AppConfig,
    mimo_key: str,
    hf_token: str,
    dependencies: PipelineDependencies = PipelineDependencies(),
) -> PipelineResult:
    started = time.monotonic()
    metadata = dependencies.probe(config.input_path)
    with workspace(config.keep_temp) as temp:
        normalized = temp / "normalized.wav"
        dependencies.normalize(config.input_path, normalized)
        preflight = temp / "preflight.wav"
        dependencies.create_preflight(normalized, preflight)
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
        items = prepare_audio_segments(
            normalized,
            segments,
            temp,
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
            )
            transcribe = client.transcribe_all
        completed = await transcribe(items, config.fail_fast)
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
                temp_path=temp if config.keep_temp else None,
            ),
        )
        recording_time = metadata.creation_time or datetime.fromtimestamp(
            config.input_path.stat().st_mtime
        )
        write_outputs(
            outcome, recording_time, config.resolved_output_path, config.debug_json
        )
    return PipelineResult(outcome, 2 if outcome.has_failures else 0)
