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
