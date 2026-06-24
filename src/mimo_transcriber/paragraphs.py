from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mimo_transcriber.models import SegmentStatus, SpeakerSegment

ParagraphMode = Literal["conservative", "balanced", "aggressive"]
FAILED_TEXT = "[该片段识别失败]"
SENTENCE_ENDINGS = ("。", "！", "？", ".", "!", "?")
CONTINUATIONS = (
    "然后", "所以", "但是", "而且", "就是", "那", "这个", "因为",
    "OK", "ok", "and", "so", "but", "then",
)


@dataclass
class TranscriptBlock:
    index: int
    start: float
    end: float
    raw_speaker: str
    display_speaker: str | None
    text: str
    source_segment_ids: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ParagraphConfig:
    enabled: bool = True
    mode: ParagraphMode = "balanced"
    gap: float | None = None
    max_duration: float | None = None
    max_chars: int = 900


def build_transcript_blocks(
    segments: list[SpeakerSegment],
    config: ParagraphConfig,
) -> list[TranscriptBlock]:
    ordered = sorted(segments, key=lambda item: item.sort_key())
    blocks: list[TranscriptBlock] = []
    for segment in ordered:
        raw = segment.text or ""
        text = " ".join(raw.split()) if config.enabled else raw
        block = TranscriptBlock(
            index=-1,
            start=segment.start,
            end=segment.end,
            raw_speaker=segment.raw_speaker,
            display_speaker=segment.display_speaker,
            text=text,
            source_segment_ids=[segment.segment_id],
        )
        if config.enabled and blocks and _can_merge(blocks[-1], block, segment, config):
            _merge_into(blocks[-1], block)
        else:
            blocks.append(block)
    for index, block in enumerate(blocks):
        block.index = index
    return blocks


def _can_merge(
    left: TranscriptBlock,
    right: TranscriptBlock,
    right_segment: SpeakerSegment,
    config: ParagraphConfig,
) -> bool:
    if left.raw_speaker != right.raw_speaker:
        return False
    if not left.text or not right.text:
        return False
    if left.text == FAILED_TEXT or right.text == FAILED_TEXT:
        return False
    if right_segment.status is SegmentStatus.FAILED:
        return False
    gap = right.start - left.end
    if gap < 0:
        gap = 0
    if gap > _gap(config):
        return False
    if right.end - left.start > _max_duration(config):
        return False
    if len(left.text) + len(right.text) > config.max_chars:
        return False
    if config.mode == "aggressive":
        return not (left.text.endswith(("?", "？")) and len(right.text) <= 12)
    return _soft_boundary(left.text, right.text) or gap <= (_gap(config) / 2)


def _gap(config: ParagraphConfig) -> float:
    if config.gap is not None:
        return config.gap
    return {"conservative": 1.0, "balanced": 2.0, "aggressive": 3.0}[config.mode]


def _max_duration(config: ParagraphConfig) -> float:
    if config.max_duration is not None:
        return config.max_duration
    return {"conservative": 75.0, "balanced": 120.0, "aggressive": 180.0}[config.mode]


def _soft_boundary(left: str, right: str) -> bool:
    return (not left.endswith(SENTENCE_ENDINGS)) or right.startswith(CONTINUATIONS)


def _merge_into(left: TranscriptBlock, right: TranscriptBlock) -> None:
    left.end = right.end
    left.text = _join_text(left.text, right.text)
    left.source_segment_ids.extend(right.source_segment_ids)


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and right[0].isascii():
        return f"{left} {right}"
    return f"{left}{right}"
