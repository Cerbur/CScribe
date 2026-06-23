from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.segments import process_segments, split_segment


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


def test_process_segments_assigns_stable_ids() -> None:
    result = process_segments([seg(0, 1, "A"), seg(2, 3, "B")], 10)
    assert [item.segment_id for item in result] == ["s0000", "s0001"]


def test_split_segment_derives_child_ids() -> None:
    parent = SpeakerSegment(0, 0, 10, "A", segment_id="s0007")
    children = split_segment(parent)
    assert [item.segment_id for item in children] == ["s0007.0", "s0007.1"]
