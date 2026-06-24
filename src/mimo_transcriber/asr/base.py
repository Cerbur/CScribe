from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from mimo_transcriber.config import Language
from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.terms import prompt_digest

AsrProvider = Literal["mlx", "mimo"]
AsrEventSink = Callable[[object], Awaitable[None]]

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MIMO_MODEL = "mimo-v2.5-asr"


@dataclass(frozen=True)
class AsrConfig:
    provider: AsrProvider = "mlx"
    stt_model: str | None = None
    language: Language = "auto"
    prompt: str | None = None
    term_count: int = 0
    term_correction: bool = True
    terms_file: Path | None = None

    def resolved_model(self) -> str:
        if self.stt_model:
            return self.stt_model
        if self.provider == "mimo":
            return DEFAULT_MIMO_MODEL
        return DEFAULT_MLX_MODEL

    def cache_identity(self) -> dict[str, object]:
        engine = "mimo" if self.provider == "mimo" else "mlx-whisper"
        settings: dict[str, object] = {
            "model": self.resolved_model(),
            "language": self.language,
        }
        digest = prompt_digest(self.prompt)
        if digest is not None:
            settings["prompt_digest"] = digest
            settings["term_count"] = self.term_count
            settings["term_correction"] = self.term_correction
        return {
            "kind": "asr-engine",
            "engine": engine,
            "identity_version": 1,
            "settings": settings,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    hf_token: str
    mimo_api_key: str | None = None


class AsrEngine(Protocol):
    @property
    def cache_identity(self) -> Mapping[str, object]:
        ...

    async def transcribe_one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        ...

    async def transcribe_all(
        self,
        items: list[tuple[SpeakerSegment, Path]],
        fail_fast: bool,
    ) -> list[SpeakerSegment]:
        ...
