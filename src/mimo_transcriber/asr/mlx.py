from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mimo_transcriber.asr.base import AsrConfig
from mimo_transcriber.models import SegmentStatus, SpeakerSegment

FAILED_TEXT = "[该片段识别失败]"
MlxTranscribe = Callable[..., dict[str, Any]]
ModelResolver = Callable[[str], str | Path]


class MlxAsrEngine:
    def __init__(
        self,
        config: AsrConfig,
        transcribe: MlxTranscribe | None = None,
        model_resolver: ModelResolver | None = None,
    ) -> None:
        self.config = config
        self._transcribe = transcribe
        # When set, resolves the model id to a local path (downloading it once).
        # When None, the raw model id is passed through so mlx_whisper's own
        # (flaky) download is used — this keeps the unit tests dependency-free.
        self._model_resolver = model_resolver
        self._lock = asyncio.Lock()
        self._resolve_lock = threading.Lock()
        self._resolved_model: str | Path | None = None

    @property
    def cache_identity(self) -> dict[str, object]:
        return self.config.cache_identity()

    def _resolve_model_ref(self) -> str | Path:
        if self._model_resolver is None:
            return self.config.resolved_model()
        if self._resolved_model is None:
            # _call_transcribe runs in worker threads (asyncio.to_thread); the
            # lock ensures the (potentially long) model download happens once.
            with self._resolve_lock:
                if self._resolved_model is None:
                    self._resolved_model = self._model_resolver(
                        self.config.resolved_model()
                    )
        return self._resolved_model

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
            "path_or_hf_repo": str(self._resolve_model_ref()),
        }
        if self.config.language != "auto":
            kwargs["language"] = self.config.language
        return transcribe(str(path), **kwargs)
