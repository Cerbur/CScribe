from __future__ import annotations

from mimo_transcriber.asr.base import AsrConfig, AsrEngine, AsrEventSink, RuntimeConfig
from mimo_transcriber.asr.mimo import MimoAsrEngine, openai_request
from mimo_transcriber.asr.mlx import MlxAsrEngine
from mimo_transcriber.asr.model_download import ensure_model, make_stderr_progress
from mimo_transcriber.config import ConfigError
from mimo_transcriber.terms import TermConfig, build_terms_prompt, parse_terms_file


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
        progress = make_stderr_progress()

        def _resolve(repo_id: str) -> str:
            return str(ensure_model(repo_id, on_progress=progress))

        return MlxAsrEngine(config, model_resolver=_resolve)
    if config.provider == "mimo":
        if not runtime.mimo_api_key:
            raise ConfigError("缺少 MIMO_API_KEY，请写入环境变量或 .env")
        term_config = parse_terms_file(config.terms_file) if config.terms_file else TermConfig()
        prompt = build_terms_prompt(config.prompt, term_config.terms)
        return MimoAsrEngine(
            request=openai_request(runtime.mimo_api_key, model=config.resolved_model()),
            model=config.resolved_model(),
            language=config.language,
            concurrency=concurrency,
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
            event_sink=event_sink,
            prompt=prompt,
            term_replacements=term_config.replacements if config.term_correction else {},
        )
    raise ConfigError(f"未知 ASR: {config.provider}")
