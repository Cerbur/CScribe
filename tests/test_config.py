from pathlib import Path

import pytest

from mimo_transcriber.config import AppConfig, ConfigError, resolve_device, validate_runtime


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


def test_default_mlx_runtime_does_not_require_mimo_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")

    runtime = validate_runtime(AppConfig(input_path=source))

    assert runtime.hf_token == "hf-token"
    assert runtime.mimo_api_key is None


def test_mimo_runtime_requires_mimo_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.setattr("shutil.which", lambda command: f"/usr/bin/{command}")

    with pytest.raises(ConfigError, match="缺少 MIMO_API_KEY"):
        validate_runtime(AppConfig(input_path=source, asr="mimo"))


def test_terms_file_must_exist(tmp_path: Path) -> None:
    config = AppConfig(input_path=tmp_path / "in.m4a", terms_file=tmp_path / "missing.txt")

    with pytest.raises(ConfigError, match="--terms-file"):
        config.validate_arguments()


def test_asr_prompt_blank_is_allowed(tmp_path: Path) -> None:
    config = AppConfig(input_path=tmp_path / "in.m4a", asr_prompt="   ")

    config.validate_arguments()
