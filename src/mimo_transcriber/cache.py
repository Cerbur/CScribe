from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimo_transcriber.config import AppConfig
from mimo_transcriber.paths import task_cache_dir

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
PROCESSING_RULES_VERSION = 1
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_RATE = 16_000
AUDIO_CODEC = "pcm_s16le"
SUBSONIC_AUDIO_CODEC = "aac"


@dataclass(frozen=True)
class InputFingerprint:
    path: str
    size: int
    mtime_ns: int
    content_sha256: str


def fingerprint_input(path: Path) -> InputFingerprint:
    resolved = path.resolve()
    stat = resolved.stat()
    size = stat.st_size

    if size < 2 * 1024 * 1024:
        content_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    else:
        with open(resolved, "rb") as stream:
            head = stream.read(1024 * 1024)
            stream.seek(-1024 * 1024, os.SEEK_END)
            tail = stream.read(1024 * 1024)
        content_sha256 = hashlib.sha256(head + tail).hexdigest()

    return InputFingerprint(
        path=str(resolved),
        size=size,
        mtime_ns=stat.st_mtime_ns,
        content_sha256=content_sha256,
    )


@dataclass(frozen=True)
class TaskPaths:
    root: Path
    task_hash: str
    work_dir: Path
    manifest: Path
    lock: Path
    normalized: Path
    preflight: Path
    audio_dir: Path
    target_index: Path

    @classmethod
    def for_run(
        cls,
        config: AppConfig,
        fingerprint: InputFingerprint,
        root: Path = task_cache_dir(),
    ) -> TaskPaths:
        params = config.cache_parameters()
        identity = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "processing_version": PROCESSING_RULES_VERSION,
            "input_path": fingerprint.path,
            "input_fingerprint": {
                "size": fingerprint.size,
                "mtime_ns": fingerprint.mtime_ns,
                "content_sha256": fingerprint.content_sha256,
            },
            "output_path": str(config.resolved_output_path.resolve()),
            "params": params,
            "audio_constants": {
                "channels": AUDIO_CHANNELS,
                "sample_rate": AUDIO_SAMPLE_RATE,
                "codec": AUDIO_CODEC,
                "subsonic_codec": SUBSONIC_AUDIO_CODEC,
            },
        }
        raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        task_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        work_dir = root / task_hash
        resolved = config.resolved_output_path.resolve()
        target_hash = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        return cls(
            root=root,
            task_hash=task_hash,
            work_dir=work_dir,
            manifest=work_dir / "manifest.json",
            lock=work_dir / "task.lock",
            normalized=work_dir / "normalized.wav",
            preflight=work_dir / "preflight.wav",
            audio_dir=work_dir / "audio",
            target_index=root / "targets" / f"{target_hash}.json",
        )


class TaskAlreadyRunningError(RuntimeError):
    pass


def _process_probe(pid: int) -> float | None:
    """Return the process start time as a Unix timestamp, or None."""
    import subprocess

    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "LC_TIME": "C"},
        )
        line = result.stdout.strip()
        if not line:
            return None
        import time as _time
        from datetime import datetime as _dt

        parsed = _dt.strptime(line, "%a %b %d %H:%M:%S %Y")
        return parsed.timestamp()
    except Exception:
        return None


@dataclass
class TaskLock:
    path: Path
    _fd: int | None = None

    def __enter__(self) -> TaskLock:
        self.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "process_started": _process_probe(os.getpid()),
                "run_id": run_id,
            }
        )
        for _ in range(2):
            try:
                fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC,
                    0o644,
                )
            except FileExistsError:
                stale = self._read_lock()
                if stale is not None and not self._is_alive(stale):
                    logger.debug("检测到陈旧锁，接管任务")
                    self.path.unlink(missing_ok=True)
                    continue
                raise TaskAlreadyRunningError("相同任务正在运行")
            else:
                os.write(fd, payload.encode("utf-8"))
                os.fsync(fd)
                self._fd = fd
                return
        raise TaskAlreadyRunningError("相同任务正在运行（陈旧锁接管失败）")

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self.path.unlink(missing_ok=True)

    def _read_lock(self) -> dict[str, Any] | None:
        try:
            with open(self.path, "r") as stream:
                return json.load(stream)
        except Exception:
            return None

    @staticmethod
    def _is_alive(record: dict[str, Any]) -> bool:
        pid = record.get("pid")
        if not isinstance(pid, int):
            return False
        expected_start = record.get("process_started")
        actual_start = _process_probe(pid)
        if actual_start is None:
            return False
        if expected_start is not None and isinstance(expected_start, (int, float)):
            return abs(expected_start - actual_start) < 5.0
        return True
