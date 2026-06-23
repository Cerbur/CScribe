from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

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

    async def wait(self, sleep: Sleep) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            if delay:
                await sleep(delay)
            self.next_time = max(now, self.next_time) + self.interval


class MiMoTranscriber:
    def __init__(
        self,
        request: Request,
        language: str,
        concurrency: int,
        requests_per_minute: int,
        max_retries: int,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.request = request
        self.language = language
        self.semaphore = asyncio.Semaphore(concurrency)
        self.limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.sleep = sleep

    async def _one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        data_url = encoded_audio_data(path)
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    await self.limiter.wait(self.sleep)
                    completion = await self.request(data_url, self.language)
                text = extract_content(completion.choices[0].message.content)
                if not text:
                    raise ValueError("MiMo 返回了空文本")
                segment.text = text
                segment.status = SegmentStatus.SUCCESS
                return segment
            except Exception as exc:
                if attempt < self.max_retries and is_retryable(exc):
                    await self.sleep((2**attempt) + random.uniform(0, 0.25))
                    continue
                segment.text = "[该片段识别失败]"
                segment.status = SegmentStatus.FAILED
                segment.error = str(exc)
                return segment
        return segment

    async def transcribe_all(
        self, items: list[tuple[SpeakerSegment, Path]], fail_fast: bool
    ) -> list[SpeakerSegment]:
        if fail_fast:
            results: list[SpeakerSegment] = []
            for segment, path in items:
                result = await self._one(segment, path)
                if result.status is SegmentStatus.FAILED:
                    raise RuntimeError(result.error or "片段识别失败")
                results.append(result)
        else:
            results = list(
                await asyncio.gather(*(self._one(segment, path) for segment, path in items))
            )
        return sorted(results, key=lambda item: item.index)


def openai_request(api_key: str, timeout: float = 120.0) -> Request:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key, base_url="https://api.xiaomimimo.com/v1", timeout=timeout
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
