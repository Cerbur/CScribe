from mimo_transcriber.asr.base import AsrConfig


def test_default_asr_config_is_mlx_with_default_model() -> None:
    config = AsrConfig()

    assert config.provider == "mlx"
    assert config.resolved_model() == "mlx-community/whisper-large-v3-turbo"
    assert config.cache_identity() == {
        "kind": "asr-engine",
        "engine": "mlx-whisper",
        "identity_version": 1,
        "settings": {
            "model": "mlx-community/whisper-large-v3-turbo",
            "language": "auto",
        },
    }


def test_mimo_asr_config_uses_mimo_identity() -> None:
    config = AsrConfig(provider="mimo", stt_model="mimo-v2.5-asr", language="zh")

    assert config.resolved_model() == "mimo-v2.5-asr"
    assert config.cache_identity() == {
        "kind": "asr-engine",
        "engine": "mimo",
        "identity_version": 1,
        "settings": {
            "model": "mimo-v2.5-asr",
            "language": "zh",
        },
    }


def test_custom_model_changes_identity() -> None:
    default = AsrConfig()
    custom = AsrConfig(stt_model="mlx-community/whisper-small")

    assert default.cache_identity() != custom.cache_identity()
    assert custom.cache_identity()["settings"] == {
        "model": "mlx-community/whisper-small",
        "language": "auto",
    }
