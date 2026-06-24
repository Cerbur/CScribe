from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mimo_transcriber.models import SpeakerSegment

StabilityMode = Literal["conservative", "balanced", "aggressive"]


@dataclass(frozen=True)
class SpeakerStabilityConfig:
    enabled: bool = True
    mode: StabilityMode = "balanced"


@dataclass(frozen=True)
class SpeakerStabilityDiagnostics:
    enabled: bool
    mode: str
    dropped_overlaps: int = 0
    relabeled_islands: int = 0


@dataclass(frozen=True)
class StabilizedSegments:
    segments: list[SpeakerSegment]
    diagnostics: SpeakerStabilityDiagnostics


def stabilize_speakers(
    segments: list[SpeakerSegment],
    config: SpeakerStabilityConfig,
) -> StabilizedSegments:
    if not config.enabled:
        return StabilizedSegments(
            list(segments),
            SpeakerStabilityDiagnostics(False, config.mode),
        )
    ordered = sorted(segments, key=lambda item: item.sort_key())
    deduped, dropped = _drop_duplicate_overlaps(ordered)
    smoothed, relabeled = _smooth_islands(deduped, config.mode)
    if not smoothed and ordered:
        smoothed = ordered
    _renumber(smoothed)
    return StabilizedSegments(
        smoothed,
        SpeakerStabilityDiagnostics(True, config.mode, dropped, relabeled),
    )


def _drop_duplicate_overlaps(segments: list[SpeakerSegment]) -> tuple[list[SpeakerSegment], int]:
    result: list[SpeakerSegment] = []
    dropped = 0
    for item in segments:
        if result and item.raw_speaker != result[-1].raw_speaker and _overlap_ratio(result[-1], item) >= 0.8:
            keep_existing = _context_score(result, result[-1].raw_speaker) >= _context_score(result, item.raw_speaker)
            if keep_existing or result[-1].duration >= item.duration:
                dropped += 1
                continue
            result[-1] = item
            dropped += 1
            continue
        result.append(item)
    return result, dropped


def _smooth_islands(segments: list[SpeakerSegment], mode: StabilityMode) -> tuple[list[SpeakerSegment], int]:
    max_duration, max_gap = {
        "conservative": (1.2, 0.5),
        "balanced": (2.0, 1.0),
        "aggressive": (3.0, 1.5),
    }[mode]
    result = list(segments)
    relabeled = 0
    for index in range(1, len(result) - 1):
        prev = result[index - 1]
        cur = result[index]
        nxt = result[index + 1]
        if (
            prev.raw_speaker == nxt.raw_speaker
            and cur.raw_speaker != prev.raw_speaker
            and cur.duration <= max_duration
            and cur.start - prev.end <= max_gap
            and nxt.start - cur.end <= max_gap
        ):
            cur.raw_speaker = prev.raw_speaker
            cur.display_speaker = prev.display_speaker
            relabeled += 1
    return result, relabeled


def _overlap_ratio(left: SpeakerSegment, right: SpeakerSegment) -> float:
    overlap = min(left.end, right.end) - max(left.start, right.start)
    if overlap <= 0:
        return 0.0
    return overlap / min(left.duration, right.duration)


def _context_score(result: list[SpeakerSegment], speaker: str) -> int:
    return sum(1 for item in result[-2:] if item.raw_speaker == speaker)


def _renumber(segments: list[SpeakerSegment]) -> None:
    names: dict[str, str] = {}
    for index, item in enumerate(segments):
        item.index = index
        item.segment_id = f"s{index:04d}"
        if item.raw_speaker not in names:
            names[item.raw_speaker] = f"说话人 {len(names) + 1}"
        item.display_speaker = names[item.raw_speaker]
