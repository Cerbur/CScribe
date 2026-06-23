from pathlib import Path

from mimo_transcriber.models import (
    AudioMetadata,
    SegmentStatus,
    SpeakerSegment,
    TranscriptionOutcome,
)


def test_speaker_segment_duration_and_default_state() -> None:
    segment = SpeakerSegment(
        index=0, start=1.25, end=3.75, raw_speaker="SPEAKER_00"
    )
    assert segment.duration == 2.5
    assert segment.status is SegmentStatus.PENDING
    assert segment.text is None


def test_transcription_outcome_reports_partial_failure() -> None:
    metadata = AudioMetadata(
        source_path=Path("meeting.m4a"),
        duration_seconds=10.0,
        codec="aac",
        sample_rate=48_000,
        channels=2,
        creation_time=None,
    )
    outcome = TranscriptionOutcome(
        metadata=metadata,
        segments=[
            SpeakerSegment(
                index=0,
                start=0,
                end=1,
                raw_speaker="A",
                status=SegmentStatus.FAILED,
            )
        ],
    )
    assert outcome.has_failures is True
