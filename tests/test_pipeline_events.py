from pathlib import Path

from mimo_transcriber.events import AudioSlice, SliceReady, TranscriptSucceeded
from mimo_transcriber.models import SpeakerSegment


def test_audio_slice_carries_segment_and_path() -> None:
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    audio_slice = AudioSlice("s0000", segment, Path("s0000.mp3"))

    assert audio_slice.segment_id == "s0000"
    assert audio_slice.segment is segment
    assert audio_slice.path == Path("s0000.mp3")


def test_events_are_immutable() -> None:
    event = SliceReady("s0000", Path("s0000.mp3"), 123)

    try:
        event.bytes = 456
    except Exception as exc:
        assert type(exc).__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("event should be frozen")


def test_transcript_event_has_segment_identity() -> None:
    event = TranscriptSucceeded("s0000", "hello")

    assert event.segment_id == "s0000"
    assert event.text == "hello"
