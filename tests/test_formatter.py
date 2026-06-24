from datetime import datetime
from pathlib import Path

from mimo_transcriber.formatter import (
    format_duration,
    format_timestamp,
    render_transcript,
    write_outputs,
)
from mimo_transcriber.models import (
    AudioMetadata,
    SegmentStatus,
    SpeakerSegment,
    TranscriptionOutcome,
)
from mimo_transcriber.paragraphs import ParagraphConfig
from mimo_transcriber.speaker_stability import SpeakerStabilityDiagnostics


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


def test_debug_json_can_include_speaker_stability(tmp_path: Path) -> None:
    metadata = AudioMetadata(tmp_path / "input.m4a", 3, "aac", 48000, 2, None)
    outcome = TranscriptionOutcome(
        metadata=metadata,
        segments=[
            SpeakerSegment(0, 0, 1, "A", "说话人 1", "你好", SegmentStatus.SUCCESS),
        ],
    )
    outcome.speaker_stability = SpeakerStabilityDiagnostics(
        enabled=True,
        mode="balanced",
        dropped_overlaps=1,
        relabeled_islands=2,
    )
    output = tmp_path / "out.txt"

    write_outputs(outcome, datetime(2026, 6, 24, 10, 0), output, debug_json=True)

    debug = output.with_suffix(".segments.json").read_text(encoding="utf-8")
    assert '"speaker_stability"' in debug
    assert '"dropped_overlaps": 1' in debug


def outcome_with_segments(tmp_path: Path) -> TranscriptionOutcome:
    return TranscriptionOutcome(
        metadata=AudioMetadata(
            source_path=tmp_path / "meeting.m4a",
            duration_seconds=10,
            codec="aac",
            sample_rate=44100,
            channels=1,
            creation_time=None,
        ),
        segments=[
            SpeakerSegment(0, 0, 3, "A", "说话人 1", "第一句。", SegmentStatus.SUCCESS, segment_id="s0000"),
            SpeakerSegment(1, 3.5, 5, "A", "说话人 1", "然后第二句。", SegmentStatus.SUCCESS, segment_id="s0001"),
        ],
        keywords=[],
    )


def test_render_transcript_uses_paragraph_blocks(tmp_path: Path) -> None:
    text = render_transcript(
        outcome_with_segments(tmp_path),
        datetime(2026, 6, 24, 10, 0),
        ParagraphConfig(),
    )

    assert text.count("说话人 1 00:00") == 1
    assert "第一句。然后第二句。" in text
    assert "说话人 1 00:03" not in text


def test_write_outputs_debug_json_includes_blocks(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"

    write_outputs(
        outcome_with_segments(tmp_path),
        datetime(2026, 6, 24, 10, 0),
        output,
        debug_json=True,
        paragraph_config=ParagraphConfig(),
    )

    debug = output.with_suffix(".segments.json").read_text(encoding="utf-8")
    assert '"segments"' in debug
    assert '"blocks"' in debug
    assert '"source_segment_ids": [' in debug
