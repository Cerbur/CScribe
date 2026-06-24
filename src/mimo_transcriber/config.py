from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from mimo_transcriber.asr.base import AsrProvider, RuntimeConfig
else:
    AsrProvider = Literal["mlx", "mimo"]

from dotenv import load_dotenv

from mimo_transcriber.speaker_stability import SpeakerStabilityConfig

Device = Literal["auto", "cpu", "cuda", "mps"]
Language = Literal["auto", "zh", "en"]
ConversationMode = Literal["auto", "two-person", "multi"]
DiarizationStabilizer = Literal["off", "conservative", "balanced", "aggressive"]


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    output_path: Path | None = None
    num_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 6
    language: Language = "auto"
    device: Device = "auto"
    conversation_mode: ConversationMode = "auto"
    diarization_stabilizer: DiarizationStabilizer = "balanced"
    concurrency: int = 2
    requests_per_minute: int = 20
    max_retries: int = 3
    keyword_count: int = 20
    asr: AsrProvider = "mlx"
    stt_model: str | None = None
    debug_json: bool = False
    fail_fast: bool = False
    debug: bool = False
    verbose: bool = False

    @property
    def resolved_output_path(self) -> Path:
        return self.output_path or self.input_path.with_suffix(".txt")

    def resolved_num_speakers(self) -> int | None:
        if self.num_speakers is not None:
            return self.num_speakers
        if self.conversation_mode == "two-person":
            return 2
        return None

    def speaker_stability_config(self) -> SpeakerStabilityConfig:
        mode = "balanced" if self.diarization_stabilizer == "off" else self.diarization_stabilizer
        return SpeakerStabilityConfig(
            enabled=self.diarization_stabilizer != "off",
            mode=mode,
        )

    def asr_cache_identity(self) -> dict[str, object]:
        from mimo_transcriber.asr.base import AsrConfig

        return AsrConfig(
            provider=self.asr,
            stt_model=self.stt_model,
            language=self.language,
        ).cache_identity()

    def cache_parameters(self) -> dict[str, object]:
        return {
            "num_speakers": self.resolved_num_speakers(),
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "language": self.language,
            "device": self.device,
            "keyword_count": self.keyword_count,
            "asr": self.asr_cache_identity(),
        }

    def validate_arguments(self) -> None:
        if self.num_speakers is not None and self.num_speakers <= 0:
            raise ConfigError("--num-speakers 必须大于 0")
        if self.min_speakers <= 0 or self.min_speakers > self.max_speakers:
            raise ConfigError("--min-speakers 必须大于 0 且不能超过 --max-speakers")
        if self.concurrency <= 0:
            raise ConfigError("--concurrency 必须大于 0")
        if self.requests_per_minute <= 0:
            raise ConfigError("--requests-per-minute 必须大于 0")
        if self.max_retries < 0 or self.keyword_count < 0:
            raise ConfigError("--max-retries 和 --keyword-count 不能为负数")
        if self.asr not in ("mlx", "mimo"):
            raise ConfigError("--asr 必须是 mlx 或 mimo")


def resolve_device(
    requested: Device, cuda_available: Callable[[], bool] | None = None
) -> Literal["cpu", "cuda"]:
    if requested == "mps":
        raise ConfigError("MPS 必须通过实验性 diarization 预检选择")
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available
    available = cuda_available()
    if requested == "cuda" and not available:
        raise ConfigError("请求了 CUDA，但当前环境不可用")
    if requested == "cuda":
        return "cuda"
    if requested == "auto" and platform.system() == "Linux" and available:
        return "cuda"
    return "cpu"


def validate_runtime(config: AppConfig) -> RuntimeConfig:
    from mimo_transcriber.asr.base import RuntimeConfig

    load_dotenv(override=False)
    config.validate_arguments()
    if not config.input_path.is_file() or not os.access(config.input_path, os.R_OK):
        raise ConfigError(f"输入文件不存在或不可读: {config.input_path}")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise ConfigError(f"未找到 {command}；macOS 可运行: brew install ffmpeg")
    mimo_key = os.getenv("MIMO_API_KEY", "")
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise ConfigError("缺少 HF_TOKEN，请写入环境变量或 .env")
    if config.asr == "mimo" and not mimo_key:
        raise ConfigError("缺少 MIMO_API_KEY，请写入环境变量或 .env")
    config.resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    return RuntimeConfig(
        hf_token=hf_token,
        mimo_api_key=mimo_key or None,
    )
