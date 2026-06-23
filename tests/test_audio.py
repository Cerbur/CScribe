import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from mimo_transcriber.audio import (
    create_preflight_sample,
    encoded_audio_data,
    normalize_audio,
    probe_audio,
)


def test_encoded_audio_data_uses_mpeg_data_url(tmp_path: Path) -> None:
    source = tmp_path / "part.mp3"
    source.write_bytes(b"abc")
    assert encoded_audio_data(source) == "data:audio/mpeg;base64,YWJj"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_probe_and_normalize_generated_audio(tmp_path: Path) -> None:
    wav = tmp_path / "source.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\0\0\0\0" * 48_000)
    m4a = tmp_path / "source.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-c:a", "aac", str(m4a)],
        check=True,
        capture_output=True,
    )
    metadata = probe_audio(m4a)
    assert metadata.channels == 2
    normalized = tmp_path / "normalized.wav"
    normalize_audio(m4a, normalized)
    normalized_metadata = probe_audio(normalized)
    assert normalized_metadata.channels == 1
    assert normalized_metadata.sample_rate == 16_000


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_create_preflight_sample_caps_normalized_audio_at_ten_seconds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "normalized.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * 16_000 * 12)

    target = tmp_path / "preflight.wav"
    create_preflight_sample(source, target)

    metadata = probe_audio(target)
    assert metadata.channels == 1
    assert metadata.sample_rate == 16_000
    assert metadata.duration_seconds == pytest.approx(10.0, abs=0.05)
