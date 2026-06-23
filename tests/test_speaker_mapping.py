from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.segments import process_segments


def test_speaker_numbers_follow_first_appearance() -> None:
    raw = [
        SpeakerSegment(-1, 0, 1, "SPEAKER_09"),
        SpeakerSegment(-1, 2, 3, "SPEAKER_01"),
        SpeakerSegment(-1, 4, 5, "SPEAKER_09"),
    ]
    result = process_segments(raw, 5)
    assert [item.display_speaker for item in result] == [
        "说话人 1",
        "说话人 2",
        "说话人 1",
    ]
    assert [item.index for item in result] == [0, 1, 2]
