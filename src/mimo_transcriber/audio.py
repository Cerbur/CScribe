from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from mimo_transcriber.models import AudioMetadata, SpeakerSegment

MAX_BASE64_BYTES = 10 * 1024 * 1024


class AudioError(RuntimeError):
    pass


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments, check=True, capture_output=True, text=True, shell=False
        )
    except subprocess.CalledProcessError as exc:
        summary = (exc.stderr or str(exc)).strip().splitlines()[-1]
        raise AudioError(f"{arguments[0]} 执行失败: {summary}") from exc


def probe_audio(path: Path) -> AudioMetadata:
    result = _run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = next(
        (item for item in payload["streams"] if item.get("codec_type") == "audio"),
        None,
    )
    if stream is None:
        raise AudioError("输入文件没有音频流")
    tags = payload.get("format", {}).get("tags", {})
    creation = tags.get("creation_time") or stream.get("tags", {}).get("creation_time")
    creation_time = (
        datetime.fromisoformat(creation.replace("Z", "+00:00")) if creation else None
    )
    return AudioMetadata(
        source_path=path,
        duration_seconds=float(payload["format"]["duration"]),
        codec=str(stream.get("codec_name", "unknown")),
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        creation_time=creation_time,
    )


def normalize_audio(source: Path, target: Path) -> None:
    _run([
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(target),
    ])


def create_preflight_sample(
    source: Path,
    target: Path,
    duration_seconds: float = 10.0,
) -> None:
    _run([
        "ffmpeg",
        "-y",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ])


def slice_mp3(source: Path, segment: SpeakerSegment, target: Path) -> None:
    _run([
        "ffmpeg", "-y", "-ss", f"{segment.start:.3f}", "-to", f"{segment.end:.3f}",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
        str(target),
    ])


def encoded_audio_data(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes())
    if len(encoded) > MAX_BASE64_BYTES:
        raise AudioError("音频片段 Base64 超过 10 MB，需要继续拆分")
    return f"data:audio/mpeg;base64,{encoded.decode('ascii')}"


@contextmanager
def workspace(keep: bool) -> Iterator[Path]:
    if keep:
        path = Path(tempfile.mkdtemp(prefix="mimo-transcriber-"))
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="mimo-transcriber-") as value:
        yield Path(value)


def payload_fits(path: Path, segment: SpeakerSegment) -> bool:
    """仅用文件尺寸判断 Base64 编码后是否超限，避免实际编码的开销。

    base64.b64encode 输出长度公式：
        base64_len = 4 * ceil(file_size / 3)
    加上 "data:audio/mpeg;base64," 前缀 24 字节。
    """
    prefix_len = 24  # "data:audio/mpeg;base64,"
    base64_len = 4 * ((path.stat().st_size + 2) // 3)
    return (base64_len + prefix_len) <= MAX_BASE64_BYTES
