from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from mimo_transcriber.models import SpeakerSegment


@dataclass(frozen=True)
class AudioSlice:
    segment_id: str
    segment: SpeakerSegment
    path: Path


@dataclass(frozen=True)
class StageStarted:
    name: str


@dataclass(frozen=True)
class SegmentTotalChanged:
    total: int


@dataclass(frozen=True)
class SliceReady:
    segment_id: str
    path: Path
    bytes: int


@dataclass(frozen=True)
class SliceFailed:
    segment_id: str
    error: str


@dataclass(frozen=True)
class SegmentsExpanded:
    parent_id: str
    children: list[SpeakerSegment]


@dataclass(frozen=True)
class TranscriptRetrying:
    segment_id: str
    retry_number: int
    max_retries: int


@dataclass(frozen=True)
class TranscriptSucceeded:
    segment_id: str
    text: str


@dataclass(frozen=True)
class TranscriptFailed:
    segment_id: str
    error: str


PipelineEvent: TypeAlias = (
    StageStarted
    | SegmentTotalChanged
    | SliceReady
    | SliceFailed
    | SegmentsExpanded
    | TranscriptRetrying
    | TranscriptSucceeded
    | TranscriptFailed
)
