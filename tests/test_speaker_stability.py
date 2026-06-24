from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.speaker_stability import SpeakerStabilityConfig, stabilize_speakers


def seg(start: float, end: float, speaker: str, segment_id: str = "") -> SpeakerSegment:
    return SpeakerSegment(
        index=0,
        start=start,
        end=end,
        raw_speaker=speaker,
        display_speaker="说话人 1" if speaker == "A" else "说话人 2",
        segment_id=segment_id,
    )


def test_disabled_stabilizer_returns_segments_unchanged() -> None:
    segments = [seg(0, 1, "A", "s0000")]

    result = stabilize_speakers(segments, SpeakerStabilityConfig(enabled=False))

    assert result.segments == segments
    assert result.diagnostics.enabled is False


def test_drops_highly_overlapped_duplicate_by_context() -> None:
    result = stabilize_speakers([
        seg(0, 2, "A", "s0000"),
        seg(2.1, 4.1, "A", "s0001"),
        seg(2.2, 4.0, "B", "s0002"),
        seg(4.2, 6, "A", "s0003"),
    ], SpeakerStabilityConfig())

    assert [item.raw_speaker for item in result.segments] == ["A", "A", "A"]
    assert result.diagnostics.dropped_overlaps == 1


def test_relabels_short_speaker_island_between_same_speaker() -> None:
    result = stabilize_speakers([
        seg(0, 3, "A", "s0000"),
        seg(3.2, 4.0, "B", "s0001"),
        seg(4.2, 7, "A", "s0002"),
    ], SpeakerStabilityConfig(mode="balanced"))

    assert [item.raw_speaker for item in result.segments] == ["A", "A", "A"]
    assert result.diagnostics.relabeled_islands == 1


def test_does_not_relabel_long_turn() -> None:
    result = stabilize_speakers([
        seg(0, 3, "A", "s0000"),
        seg(3.2, 7.0, "B", "s0001"),
        seg(7.2, 9, "A", "s0002"),
    ], SpeakerStabilityConfig(mode="balanced"))

    assert [item.raw_speaker for item in result.segments] == ["A", "B", "A"]


def test_reassigns_indexes_and_segment_ids_after_drop() -> None:
    result = stabilize_speakers([
        seg(0, 2, "A", "old0"),
        seg(0.1, 1.9, "B", "old1"),
    ], SpeakerStabilityConfig())

    assert [(item.index, item.segment_id) for item in result.segments] == [(0, "s0000")]
