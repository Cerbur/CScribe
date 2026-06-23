from pathlib import Path

import pytest

from mimo_transcriber.cache import (
    TaskAlreadyRunningError,
    TaskLock,
    TaskPaths,
    fingerprint_input,
)
from mimo_transcriber.config import AppConfig


def test_fingerprint_changes_when_edge_content_changes(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"a" * (2 * 1024 * 1024 + 8))
    before = fingerprint_input(source)
    source.write_bytes(b"b" + source.read_bytes()[1:])
    after = fingerprint_input(source)
    assert before.content_sha256 != after.content_sha256


def test_runtime_tuning_does_not_change_task_hash(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)
    first = AppConfig(input_path=source, concurrency=2, max_retries=1)
    second = AppConfig(input_path=source, concurrency=8, max_retries=3)
    assert TaskPaths.for_run(first, fingerprint, tmp_path).task_hash == (
        TaskPaths.for_run(second, fingerprint, tmp_path).task_hash
    )


def test_second_live_lock_is_rejected(tmp_path: Path) -> None:
    first = TaskLock(tmp_path / "task.lock")
    second = TaskLock(tmp_path / "task.lock")
    first.acquire()
    with pytest.raises(TaskAlreadyRunningError):
        second.acquire()
    first.release()
