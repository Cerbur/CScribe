from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mimo_transcriber.asr.base import AsrEventSink
from mimo_transcriber.audio import encoded_audio_data
from mimo_transcriber.models import SegmentStatus, SpeakerSegment

Request = Callable[[str, str], Awaitable[Any]]
Sleep = Callable[[float], Awaitable[None]]


def extract_content(content: Any) -> str:
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            value = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
            if value:
                parts.append(str(value))
        raw = " ".join(parts)
    else:
        raw = str(getattr(content, "text", "") or "")
    return " ".join(raw.split())


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(exc, (TimeoutError, ConnectionError)) or status == 429 or (
        isinstance(status, int) and status >= 500
    )


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute
        self.lock = asyncio.Lock()
        self.next_time = 0.0
        self._consecutive_429s = 0
        self._base_cooldown = 15.0

    async def wait(self, sleep: Sleep) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            if delay:
                await sleep(delay)
            self.next_time = max(now, self.next_time) + self.interval

    async def report_429(self, sleep: Sleep) -> None:
        """Global cooldown when any worker receives a 429.

        Each consecutive 429 increases the cooldown exponentially so that
        all workers back off together — this breaks the retry-storm cycle
        where every worker retries simultaneously and re-triggers the limit.
        """
        async with self.lock:
            self._consecutive_429s += 1
            cooldown = min(self._base_cooldown * (2 ** (self._consecutive_429s - 1)), 120.0)
            self.next_time = max(self.next_time, time.monotonic() + cooldown)

    def reset_429_counter(self) -> None:
        self._consecutive_429s = 0


class MimoAsrEngine:
    def __init__(
        self,
        request: Request,
        model: str,
        language: str,
        concurrency: int,
        requests_per_minute: int,
        max_retries: int,
        sleep: Sleep = asyncio.sleep,
        reporter: object = None,
        event_sink: AsrEventSink | None = None,
    ) -> None:
        from mimo_transcriber.progress import NullProgressReporter

        self.request = request
        self.model = model
        self.language = language
        self.semaphore = asyncio.Semaphore(concurrency)
        self.limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.sleep = sleep
        self.reporter = reporter if reporter is not None else NullProgressReporter()
        self.event_sink = event_sink

    @property
    def cache_identity(self) -> dict[str, object]:
        return {
            "kind": "asr-engine",
            "engine": "mimo",
            "identity_version": 1,
            "settings": {
                "model": self.model,
                "language": self.language,
            },
        }

    async def transcribe_one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        # 文件内容只读取一次，避免每次重试重复编码
        data_url = encoded_audio_data(path)
        _logger.debug(
            "[%s] 转写开始 (duration=%.1fs, size=%d bytes, max_retries=%d)",
            segment.segment_id, segment.duration, path.stat().st_size,
            self.max_retries,
        )
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    # Reset 429 counter on first attempt to avoid stale cooldown
                    if attempt == 0:
                        self.limiter.reset_429_counter()
                    await self.limiter.wait(self.sleep)
                    completion = await self.request(data_url, self.language)
                text = extract_content(completion.choices[0].message.content)
                if not text:
                    raise ValueError("MiMo 返回了空文本")
                segment.text = text
                segment.status = SegmentStatus.SUCCESS
                self.reporter.segment_completed(True)
                _logger.debug("[%s] 转写成功", segment.segment_id)
                return segment
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                is_429 = status == 429
                retryable = is_retryable(exc)
                exc_type = type(exc).__name__
                exc_msg = str(exc)[:200]
                _logger.warning(
                    "[%s] 转写失败 (attempt %d/%d, retryable=%s, 429=%s): %s: %s",
                    segment.segment_id, attempt + 1, self.max_retries + 1,
                    retryable, is_429, exc_type, exc_msg,
                )
                if attempt < self.max_retries and retryable:
                    retry_number = attempt + 1
                    if is_429:
                        # Global cooldown — all workers back off together
                        await self.limiter.report_429(self.sleep)
                        # Longer per-segment backoff for rate limits (10-20s)
                        backoff = 10.0 + random.uniform(0, 10.0)
                    else:
                        backoff = (2**attempt) + random.uniform(0, 0.5)
                    _logger.debug(
                        "[%s] 重试 %d/%d，等待 %.1fs",
                        segment.segment_id, retry_number, self.max_retries, backoff,
                    )
                    if self.event_sink is not None:
                        from mimo_transcriber.events import TranscriptRetrying

                        await self.event_sink(
                            TranscriptRetrying(
                                segment.segment_id,
                                retry_number,
                                self.max_retries,
                            )
                        )
                    else:
                        self.reporter.segment_retrying(
                            segment.segment_id,
                            retry_number,
                            self.max_retries,
                        )
                    await self.sleep(backoff)
                    continue
                segment.text = "[该片段识别失败]"
                segment.status = SegmentStatus.FAILED
                segment.error = f"{exc_type}: {exc}"
                self.reporter.segment_completed(False)
                return segment
        self.reporter.segment_completed(False)
        return segment

    async def transcribe_all(
        self, items: list[tuple[SpeakerSegment, Path]], fail_fast: bool
    ) -> list[SpeakerSegment]:
        if fail_fast:
            results: list[SpeakerSegment] = []
            for segment, path in items:
                result = await self.transcribe_one(segment, path)
                if result.status is SegmentStatus.FAILED:
                    raise RuntimeError(result.error or "片段识别失败")
                results.append(result)
        else:
            results = list(
                await asyncio.gather(
                    *(self.transcribe_one(segment, path) for segment, path in items)
                )
            )
        return sorted(results, key=lambda item: item.sort_key())


def openai_request(api_key: str, timeout: float = 120.0) -> Request:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.xiaomimimo.com/v1",
        timeout=timeout,
        max_retries=0,  # SDK 内部重试已关闭，由 MimoAsrEngine 统一管理
    )

    async def request(data_url: str, language: str) -> Any:
        return await client.chat.completions.create(
            model="mimo-v2.5-asr",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": data_url},
                }],
            }],
            extra_body={"asr_options": {"language": language}},
        )

    return request
