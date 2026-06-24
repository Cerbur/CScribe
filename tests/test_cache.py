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


def test_task_hash_changes_when_asr_identity_changes(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    mlx = TaskPaths.for_run(
        AppConfig(input_path=source, asr="mlx"),
        fingerprint,
        tmp_path,
    )
    mimo = TaskPaths.for_run(
        AppConfig(input_path=source, asr="mimo", stt_model="mimo-v2.5-asr"),
        fingerprint,
        tmp_path,
    )

    assert mlx.task_hash != mimo.task_hash


def test_task_hash_ignores_asr_runtime_controls(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    first = TaskPaths.for_run(
        AppConfig(input_path=source, concurrency=1, max_retries=0),
        fingerprint,
        tmp_path,
    )
    second = TaskPaths.for_run(
        AppConfig(input_path=source, concurrency=8, max_retries=5),
        fingerprint,
        tmp_path,
    )

    assert first.task_hash == second.task_hash


def test_task_hash_changes_with_diarization_model(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    first = TaskPaths.for_run(AppConfig(input_path=source), fingerprint, tmp_path)
    second = TaskPaths.for_run(
        AppConfig(input_path=source, diarization_model="local/other-model"),
        fingerprint,
        tmp_path,
    )

    assert first.task_hash != second.task_hash


def test_task_hash_changes_with_diarization_stabilizer(tmp_path: Path) -> None:
    """The stabilizer output is cached in the manifest, so its mode must be part
    of the cache identity — otherwise toggling --diarization-stabilizer has no
    effect on a cached run and the documented comparison workflow breaks."""
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    balanced = TaskPaths.for_run(
        AppConfig(input_path=source, diarization_stabilizer="balanced"),
        fingerprint,
        tmp_path,
    )
    off = TaskPaths.for_run(
        AppConfig(input_path=source, diarization_stabilizer="off"),
        fingerprint,
        tmp_path,
    )

    assert balanced.task_hash != off.task_hash


def test_task_hash_ignores_paragraph_settings(tmp_path: Path) -> None:
    """Paragraph merging is a render-only layer; it must not invalidate the
    ASR/diarization cache, so re-running with different paragraph settings can
    reuse cached transcripts."""
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    default = TaskPaths.for_run(AppConfig(input_path=source), fingerprint, tmp_path)
    tuned = TaskPaths.for_run(
        AppConfig(
            input_path=source,
            paragraph_mode="aggressive",
            paragraph_gap=3.0,
            paragraph_max_duration=180.0,
            paragraph_max_chars=1200,
        ),
        fingerprint,
        tmp_path,
    )

    assert default.task_hash == tuned.task_hash
