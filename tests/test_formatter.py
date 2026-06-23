from datetime import datetime
from pathlib import Path

from mimo_transcriber.formatter import (
    format_duration,
    format_timestamp,
    render_transcript,
)
from mimo_transcriber.models import (
    AudioMetadata,
    SegmentStatus,
    SpeakerSegment,
    TranscriptionOutcome,
)


def test_time_formats() -> None:
    assert format_timestamp(62.9) == "01:02"
    assert format_timestamp(3661.2) == "01:01:01"
    assert format_duration(513) == "8分钟 33秒"
    assert format_duration(4113) == "1小时 8分钟 33秒"


def test_renders_exact_transcript_layout() -> None:
    metadata = AudioMetadata(Path("input.m4a"), 63, "aac", 48000, 2, None)
    outcome = TranscriptionOutcome(
        metadata=metadata,
        keywords=["Agent", "RAG"],
        segments=[
            SpeakerSegment(
                0, 2, 4, "A", "说话人 1", "你好。", SegmentStatus.SUCCESS
            )
        ],
    )
    rendered = render_transcript(outcome, datetime(2026, 6, 15, 23, 32))
    assert rendered == (
        "2026年6月15日 下午 11:32|1分钟 3秒\n\n"
        "关键词:\nAgent、RAG\n\n"
        "文字记录:\n说话人 1 00:02\n你好。\n"
    )


def test_render_transcript_uses_time_then_segment_id_order() -> None:
    metadata = AudioMetadata(Path("input.m4a"), 3, "aac", 48_000, 2, None)
    outcome = TranscriptionOutcome(
        metadata=metadata,
        segments=[
            SpeakerSegment(1, 2, 3, "B", "说话人 2", "后", segment_id="s0001"),
            SpeakerSegment(0, 0, 1, "A", "说话人 1", "先", segment_id="s0000"),
        ],
    )
    rendered = render_transcript(outcome, datetime(2026, 6, 15, 10, 0))
    assert rendered.index("先") < rendered.index("后")
