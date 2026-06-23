import io

import pytest

from mimo_transcriber.progress import TerminalProgressReporter


def test_non_tty_emits_plain_stage_and_retry_to_stream() -> None:
    """Non-TTY 模式下阶段名和重试信息直接写入流，不经过 logging。"""
    stream = io.StringIO()
    reporter = TerminalProgressReporter(stream, is_tty=False)
    reporter.start_stage("说话人分离")
    reporter.segment_retrying("s0007", 2, 3)
    reporter.close()
    output = stream.getvalue()
    assert "说话人分离" in output
    assert "s0007 重试 2/3" in output


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


def test_close_is_idempotent() -> None:
    """Calling close() multiple times should not raise."""
    stream = io.StringIO()
    reporter = TerminalProgressReporter(stream, is_tty=True)
    reporter.start_stage("测试阶段")
    reporter.close()
    reporter.close()  # second close must not raise
    reporter.close()  # third close must not raise


def test_finish_writes_summary_to_stream() -> None:
    """finish() 直接写入流，不经过 logging。"""
    stream = io.StringIO()
    reporter = TerminalProgressReporter(stream, is_tty=False)
    reporter.finish(success_count=5, failure_count=2, elapsed_seconds=3.14)
    output = stream.getvalue()
    assert "5" in output
    assert "2" in output
    assert "3.14" in output


def test_finish_is_idempotent() -> None:
    """Calling finish() multiple times should not raise."""
    stream = io.StringIO()
    reporter = TerminalProgressReporter(stream, is_tty=True)
    reporter.start_stage("测试")
    reporter.finish(success_count=1, failure_count=0, elapsed_seconds=0.0)
    reporter.finish(success_count=1, failure_count=0, elapsed_seconds=0.0)
    reporter.close()
