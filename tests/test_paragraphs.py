from mimo_transcriber.models import SegmentStatus, SpeakerSegment
from mimo_transcriber.paragraphs import ParagraphConfig, build_transcript_blocks


def seg(
    start: float,
    end: float,
    speaker: str = "SPEAKER_00",
    text: str = "hello",
    segment_id: str = "",
) -> SpeakerSegment:
    return SpeakerSegment(
        index=0,
        start=start,
        end=end,
        raw_speaker=speaker,
        display_speaker="说话人 1" if speaker == "SPEAKER_00" else "说话人 2",
        text=text,
        status=SegmentStatus.SUCCESS,
        segment_id=segment_id,
    )


def test_balanced_merges_same_speaker_short_gap() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, text="我们先聊 Facebook。", segment_id="s0000"),
        seg(4, 7, text="然后看 Grab 的例子。", segment_id="s0001"),
    ], ParagraphConfig())

    assert len(blocks) == 1
    assert blocks[0].start == 0
    assert blocks[0].end == 7
    assert blocks[0].text == "我们先聊 Facebook。然后看 Grab 的例子。"
    assert blocks[0].source_segment_ids == ["s0000", "s0001"]


def test_different_speakers_do_not_merge() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, "SPEAKER_00", "你怎么看？", "s0000"),
        seg(3.2, 5, "SPEAKER_01", "我同意。", "s0001"),
    ], ParagraphConfig())

    assert len(blocks) == 2


def test_long_gap_does_not_merge() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, text="第一段。", segment_id="s0000"),
        seg(6, 8, text="第二段。", segment_id="s0001"),
    ], ParagraphConfig())

    assert len(blocks) == 2


def test_failed_segment_stays_separate() -> None:
    failed = seg(3.5, 5, text="[该片段识别失败]", segment_id="s0001")
    failed.status = SegmentStatus.FAILED

    blocks = build_transcript_blocks([
        seg(0, 3, text="正常文本。", segment_id="s0000"),
        failed,
    ], ParagraphConfig())

    assert len(blocks) == 2


def test_off_mode_keeps_one_block_per_segment() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, text="第一段", segment_id="s0000"),
        seg(3.1, 4, text="第二段", segment_id="s0001"),
    ], ParagraphConfig(enabled=False, mode="balanced"))

    assert [block.source_segment_ids for block in blocks] == [["s0000"], ["s0001"]]


def test_disabled_mode_preserves_raw_segment_text() -> None:
    blocks = build_transcript_blocks(
        [seg(0, 3, text="  hi  there  ", segment_id="s0000")],
        ParagraphConfig(enabled=False),
    )
    assert blocks[0].text == "  hi  there  "
