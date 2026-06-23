import io

import pytest

from mimo_transcriber.progress import TerminalProgressReporter


def test_non_tty_emits_plain_stage_and_retry_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    reporter = TerminalProgressReporter(io.StringIO(), is_tty=False)
    reporter.start_stage("说话人分离")
    reporter.segment_retrying("s0007", 2, 3)
    reporter.close()
    assert "说话人分离" in caplog.text
    assert "s0007 重试 2/3" in caplog.text


def test_tty_renders_slice_and_transcript_counts() -> None:
    stream = io.StringIO()
    reporter = TerminalProgressReporter(stream, is_tty=True)
    reporter.start_stage("正在处理音频片段")
    reporter.set_segment_total(3)
    reporter.segment_sliced()
    reporter.segment_completed(True)
    reporter.close()
    output = stream.getvalue()
    assert "切片 1/3" in output
    assert "转写 1/3" in output
