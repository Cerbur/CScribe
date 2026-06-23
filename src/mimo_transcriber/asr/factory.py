from __future__ import annotations

from mimo_transcriber.asr.base import AsrConfig, AsrEngine, AsrEventSink, RuntimeConfig
from mimo_transcriber.asr.mimo import MimoAsrEngine, openai_request
from mimo_transcriber.asr.mlx import MlxAsrEngine
from mimo_transcriber.config import ConfigError


def create_asr_engine(
    config: AsrConfig,
    runtime: RuntimeConfig,
    event_sink: AsrEventSink | None,
    *,
    concurrency: int = 2,
    requests_per_minute: int = 20,
    max_retries: int = 3,
) -> AsrEngine:
    if config.provider == "mlx":
        return MlxAsrEngine(config)
    if config.provider == "mimo":
        if not runtime.mimo_api_key:
            raise ConfigError("缺少 MIMO_API_KEY，请写入环境变量或 .env")
        return MimoAsrEngine(
            request=openai_request(runtime.mimo_api_key, model=config.resolved_model()),
            model=config.resolved_model(),
            language=config.language,
            concurrency=concurrency,
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
            event_sink=event_sink,
        )
    raise ConfigError(f"未知 ASR: {config.provider}")
