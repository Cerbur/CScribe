from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class SegmentStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class AudioMetadata:
    source_path: Path
    duration_seconds: float
    codec: str
    sample_rate: int
    channels: int
    creation_time: datetime | None


@dataclass
class SpeakerSegment:
    index: int
    start: float
    end: float
    raw_speaker: str
    display_speaker: str | None = None
    text: str | None = None
    status: SegmentStatus = SegmentStatus.PENDING
    error: str | None = None
    segment_id: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def sort_key(self) -> tuple[float, float, str]:
        return (self.start, self.end, self.segment_id)


@dataclass
class RunSummary:
    elapsed_seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    speakers: int = 0
    segments: int = 0
    succeeded: int = 0
    failed: int = 0
    output_path: Path | None = None
    temp_path: Path | None = None


@dataclass
class TranscriptionOutcome:
    metadata: AudioMetadata
    segments: list[SpeakerSegment]
    keywords: list[str] = field(default_factory=list)
    summary: RunSummary = field(default_factory=RunSummary)

    @property
    def has_failures(self) -> bool:
        return any(segment.status is SegmentStatus.FAILED for segment in self.segments)
