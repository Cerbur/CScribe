from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.segments import process_segments


def seg(start: float, end: float, speaker: str) -> SpeakerSegment:
    return SpeakerSegment(-1, start, end, speaker)


def test_merges_same_speaker_with_small_gap() -> None:
    result = process_segments([seg(0, 1, "A"), seg(1.5, 2, "A")], 10)
    assert [(item.start, item.end) for item in result] == [(0, 2)]


def test_does_not_merge_different_speakers() -> None:
    result = process_segments([seg(0, 1, "A"), seg(1.1, 2, "B")], 10)
    assert len(result) == 2


def test_short_segment_prefers_previous_same_speaker() -> None:
    result = process_segments(
        [seg(0, 1, "A"), seg(1.1, 1.3, "A"), seg(1.4, 2, "A")], 10
    )
    assert [(item.start, item.end) for item in result] == [(0, 2)]


def test_splits_long_segment_contiguously() -> None:
    result = process_segments([seg(0, 100, "A")], 100)
    assert [(item.start, item.end) for item in result] == [
        (0, 45),
        (45, 90),
        (90, 100),
    ]
