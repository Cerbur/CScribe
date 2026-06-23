from pathlib import Path

import pytest

from mimo_transcriber.config import AppConfig, ConfigError, resolve_device


def test_num_speakers_must_be_positive(tmp_path: Path) -> None:
    source = tmp_path / "recording.m4a"
    source.write_bytes(b"audio")
    with pytest.raises(ConfigError, match="num-speakers"):
        AppConfig(input_path=source, num_speakers=0).validate_arguments()


def test_minimum_cannot_exceed_maximum(tmp_path: Path) -> None:
    source = tmp_path / "recording.m4a"
    source.write_bytes(b"audio")
    with pytest.raises(ConfigError, match="min-speakers"):
        AppConfig(input_path=source, min_speakers=4, max_speakers=2).validate_arguments()


def test_auto_device_on_macos_stays_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert resolve_device("auto", cuda_available=lambda: True) == "cpu"


def test_auto_device_on_linux_uses_available_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert resolve_device("auto", cuda_available=lambda: True) == "cuda"


def test_explicit_mps_is_reserved_for_diarization_selector() -> None:
    with pytest.raises(ConfigError, match="MPS"):
        resolve_device("mps", cuda_available=lambda: False)
