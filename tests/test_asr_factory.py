import pytest

from mimo_transcriber.asr.base import AsrConfig, RuntimeConfig
from mimo_transcriber.asr.factory import create_asr_engine
from mimo_transcriber.asr.mimo import MimoAsrEngine
from mimo_transcriber.config import ConfigError


def test_factory_creates_mimo_engine() -> None:
    engine = create_asr_engine(
        AsrConfig(provider="mimo", language="en"),
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo-key"),
        event_sink=None,
    )

    assert isinstance(engine, MimoAsrEngine)
    assert engine.cache_identity["engine"] == "mimo"


def test_factory_requires_mimo_key_for_mimo_engine() -> None:
    with pytest.raises(ConfigError, match="缺少 MIMO_API_KEY"):
        create_asr_engine(
            AsrConfig(provider="mimo"),
            RuntimeConfig(hf_token="hf", mimo_api_key=None),
            event_sink=None,
        )


def test_factory_creates_mlx_engine() -> None:
    from mimo_transcriber.asr.mlx import MlxAsrEngine

    engine = create_asr_engine(
        AsrConfig(provider="mlx"),
        RuntimeConfig(hf_token="hf"),
        event_sink=None,
    )

    assert isinstance(engine, MlxAsrEngine)
    assert engine.cache_identity["engine"] == "mlx-whisper"
