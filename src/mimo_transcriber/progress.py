from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Protocol, TextIO, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ProgressReporter(Protocol):
    def start_stage(self, stage: str, detail: str | None = None) -> None: ...
    def set_segment_total(self, total: int) -> None: ...
    def segment_sliced(self) -> None: ...
    def segment_completed(self, success: bool) -> None: ...
    def segment_retrying(self, segment_id: str, retry_number: int, max_retries: int) -> None: ...
    def finish(self, success_count: int, failure_count: int, elapsed_seconds: float) -> None: ...
    def close(self) -> None: ...


class NullProgressReporter:
    def start_stage(self, stage: str, detail: str | None = None) -> None:
        pass

    def set_segment_total(self, total: int) -> None:
        pass

    def segment_sliced(self) -> None:
        pass

    def segment_completed(self, success: bool) -> None:
        pass

    def segment_retrying(self, segment_id: str, retry_number: int, max_retries: int) -> None:
        pass

    def finish(self, success_count: int, failure_count: int, elapsed_seconds: float) -> None:
        pass

    def close(self) -> None:
        pass


class TerminalProgressReporter:
    def __init__(
        self,
        stream: TextIO | None = None,
        is_tty: bool | None = None,
    ) -> None:
        self._stream = stream or sys.stderr
        self._is_tty = is_tty if is_tty is not None else self._stream.isatty()
        self._lock = threading.Lock()
        self._progress = None
        self._task_slice = None
        self._task_transcribe = None
        self._slice_total = 0
        self._slice_done = 0
        self._transcribe_done = 0
        self._transcribe_total = 0
        self._rich_available = False

        if self._is_tty:
            try:
                from rich.console import Console
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                )

                self._console = Console(file=self._stream, force_terminal=True)
                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    console=self._console,
                    refresh_per_second=10,
                )
                self._rich_available = True
            except Exception:
                logger.debug("Rich 初始化失败，回退到普通日志", exc_info=True)
                self._is_tty = False

    def _safe(self, action: Callable[[], None]) -> None:
        try:
            with self._lock:
                action()
        except Exception:
            if self._rich_available:
                logger.debug("进度渲染故障，切换到普通日志", exc_info=True)
                self._rich_available = False
                self._is_tty = False
            else:
                logger.debug("进度回调异常", exc_info=True)

    def start_stage(self, stage: str, detail: str | None = None) -> None:
        label = stage if detail is None else f"{stage}｜{detail}"

        def _start() -> None:
            if self._rich_available and self._progress is not None:
                if not self._progress.task_ids:
                    self._progress.start()
                self._task_slice = self._progress.add_task(
                    label, total=None, visible=True
                )
            else:
                logger.info(label)

        self._safe(_start)

    def set_segment_total(self, total: int) -> None:
        def _set() -> None:
            self._slice_total = total
            self._transcribe_total = total

        self._safe(_set)

    def segment_sliced(self) -> None:
        def _inc() -> None:
            self._slice_done += 1
            self._update_progress()

        self._safe(_inc)

    def segment_completed(self, success: bool) -> None:
        def _inc() -> None:
            self._transcribe_done += 1
            self._update_progress()

        self._safe(_inc)

    def segment_retrying(self, segment_id: str, retry_number: int, max_retries: int) -> None:
        msg = f"{segment_id} 重试 {retry_number}/{max_retries}"

        def _retry() -> None:
            if self._rich_available and self._progress is not None and self._task_slice is not None:
                self._progress.update(
                    self._task_slice,
                    description=f"正在处理音频片段｜{msg}",
                )
            else:
                logger.info(msg)

        self._safe(_retry)

    def finish(self, success_count: int, failure_count: int, elapsed_seconds: float) -> None:
        def _finish() -> None:
            if self._rich_available and self._progress is not None:
                self._progress.stop()
            logger.info(
                "已完成｜%d 成功｜%d 失败｜耗时 %.2f 秒",
                success_count,
                failure_count,
                elapsed_seconds,
            )

        self._safe(_finish)

    def close(self) -> None:
        def _close() -> None:
            if self._rich_available and self._progress is not None:
                try:
                    self._progress.stop()
                except Exception:
                    pass

        try:
            self._safe(_close)
        except Exception:
            pass

    def _update_progress(self) -> None:
        if self._rich_available and self._progress is not None and self._task_slice is not None:
            desc = (
                f"正在处理音频片段｜切片 {self._slice_done}/{self._slice_total}"
                f"｜转写 {self._transcribe_done}/{self._transcribe_total}"
            )
            self._progress.update(self._task_slice, description=desc)
