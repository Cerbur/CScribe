from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mimo_transcriber.asr.base import AsrConfig
from mimo_transcriber.models import SegmentStatus, SpeakerSegment

FAILED_TEXT = "[该片段识别失败]"
MlxTranscribe = Callable[..., dict[str, Any]]


class MlxAsrEngine:
    def __init__(
        self,
        config: AsrConfig,
        transcribe: MlxTranscribe | None = None,
    ) -> None:
        self.config = config
        self._transcribe = transcribe
        self._lock = asyncio.Lock()

    @property
    def cache_identity(self) -> dict[str, object]:
        return self.config.cache_identity()

    async def transcribe_one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        try:
            async with self._lock:
                result = await asyncio.to_thread(self._call_transcribe, path)
            text = " ".join(str(result.get("text", "")).split())
            if not text:
                raise ValueError("MLX Whisper 返回了空文本")
            segment.text = text
            segment.status = SegmentStatus.SUCCESS
            segment.error = None
            return segment
        except Exception as exc:
            segment.text = FAILED_TEXT
            segment.status = SegmentStatus.FAILED
            segment.error = f"{type(exc).__name__}: {exc}"
            return segment

    async def transcribe_all(
        self,
        items: list[tuple[SpeakerSegment, Path]],
        fail_fast: bool,
    ) -> list[SpeakerSegment]:
        results: list[SpeakerSegment] = []
        for segment, path in items:
            result = await self.transcribe_one(segment, path)
            if fail_fast and result.status is SegmentStatus.FAILED:
                raise RuntimeError(result.error or "片段识别失败")
            results.append(result)
        return sorted(results, key=lambda item: item.sort_key())

    def _call_transcribe(self, path: Path) -> dict[str, Any]:
        transcribe = self._transcribe
        if transcribe is None:
            import mlx_whisper

            transcribe = mlx_whisper.transcribe
            self._transcribe = transcribe
        kwargs: dict[str, object] = {
            "path_or_hf_repo": self.config.resolved_model(),
        }
        if self.config.language != "auto":
            kwargs["language"] = self.config.language
        return transcribe(str(path), **kwargs)
