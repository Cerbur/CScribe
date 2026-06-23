from __future__ import annotations

import logging

from mimo_transcriber.models import SpeakerSegment

logger = logging.getLogger(__name__)


def process_segments(
    raw: list[SpeakerSegment],
    duration: float,
    min_duration: float = 0.4,
    merge_gap: float = 0.8,
    max_duration: float = 45.0,
) -> list[SpeakerSegment]:
    clipped = _clip_and_sort(raw, duration)
    names = _speaker_names(clipped)
    for item in clipped:
        item.display_speaker = names[item.raw_speaker]
    without_short = _merge_or_drop_short(clipped, min_duration, merge_gap)
    merged = _merge_adjacent(without_short, merge_gap)
    split = _split_long(merged, max_duration)
    for index, item in enumerate(split):
        item.index = index
    return split


def _clip_and_sort(
    raw: list[SpeakerSegment], duration: float
) -> list[SpeakerSegment]:
    ordered = sorted(enumerate(raw), key=lambda pair: (pair[1].start, pair[0]))
    result: list[SpeakerSegment] = []
    for _, item in ordered:
        start = max(0.0, min(item.start, duration))
        end = max(0.0, min(item.end, duration))
        if end <= start:
            logger.debug("丢弃非法区间 %.3f-%.3f", item.start, item.end)
            continue
        result.append(SpeakerSegment(-1, start, end, item.raw_speaker))
    return result


def _speaker_names(items: list[SpeakerSegment]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if item.raw_speaker not in result:
            result[item.raw_speaker] = f"说话人 {len(result) + 1}"
    return result


def _merge_pair(left: SpeakerSegment, right: SpeakerSegment) -> SpeakerSegment:
    return SpeakerSegment(
        -1,
        min(left.start, right.start),
        max(left.end, right.end),
        left.raw_speaker,
        left.display_speaker,
    )


def _merge_or_drop_short(
    items: list[SpeakerSegment], minimum: float, gap: float
) -> list[SpeakerSegment]:
    result = list(items)
    index = 0
    while index < len(result):
        item = result[index]
        if item.duration >= minimum:
            index += 1
            continue
        previous = result[index - 1] if index > 0 else None
        following = result[index + 1] if index + 1 < len(result) else None
        if (
            previous
            and previous.raw_speaker == item.raw_speaker
            and item.start - previous.end <= gap
        ):
            result[index - 1] = _merge_pair(previous, item)
            result.pop(index)
        elif (
            following
            and following.raw_speaker == item.raw_speaker
            and following.start - item.end <= gap
        ):
            result[index + 1] = _merge_pair(item, following)
            result.pop(index)
        else:
            logger.debug("跳过无法合并的短区间 %.3f-%.3f", item.start, item.end)
            result.pop(index)
    return result


def _merge_adjacent(
    items: list[SpeakerSegment], gap: float
) -> list[SpeakerSegment]:
    result: list[SpeakerSegment] = []
    for item in items:
        if (
            result
            and result[-1].raw_speaker == item.raw_speaker
            and item.start - result[-1].end <= gap
        ):
            result[-1] = _merge_pair(result[-1], item)
        else:
            result.append(item)
    return result


def _split_long(
    items: list[SpeakerSegment], maximum: float
) -> list[SpeakerSegment]:
    result: list[SpeakerSegment] = []
    for item in items:
        start = item.start
        while item.end - start > maximum:
            result.append(
                SpeakerSegment(-1, start, start + maximum, item.raw_speaker, item.display_speaker)
            )
            start += maximum
        result.append(
            SpeakerSegment(-1, start, item.end, item.raw_speaker, item.display_speaker)
        )
    return result
