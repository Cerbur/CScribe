# M4A Multi-Speaker Transcriber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python CLI that diarizes local M4A recordings, transcribes speaker segments through MiMo-V2.5-ASR, extracts keywords, and atomically writes a timestamped UTF-8 transcript.

**Architecture:** A thin CLI validates configuration and invokes a single pipeline. Focused modules own media commands, pyannote adaptation, segment transformations, MiMo concurrency, keyword extraction, and output formatting; typed dataclasses are the contracts between them. Heavy model imports and real network calls remain behind adapters so unit tests run without credentials or downloads.

**Tech Stack:** Python 3.11, uv, argparse, asyncio, OpenAI Python SDK, pyannote.audio, PyTorch, jieba, python-dotenv, FFmpeg/ffprobe, pytest, pytest-asyncio.

## Global Constraints

- Support macOS Apple Silicon with CPU and Linux x86_64 with optional NVIDIA CUDA.
- `--device auto` selects CUDA when available on Linux and CPU otherwise; do not select MPS.
- Use `pyannote/speaker-diarization-community-1`.
- Use MiMo base URL `https://api.xiaomimimo.com/v1` and model `mimo-v2.5-asr`.
- Normalize audio to mono, 16 kHz, PCM S16LE WAV; encode segments as mono, 16 kHz, 48 kbps MP3.
- Never log API keys, Hugging Face tokens, or complete Base64 payloads.
- Only segmented MP3 audio is sent to MiMo; the complete original recording stays local.
- Default request timeout is 120 seconds; defaults are concurrency 4, 80 requests/minute, 3 retries, and 20 keywords.
- Segment rules are 0.4-second minimum, 0.8-second same-speaker merge gap, and 45-second maximum.
- Partial ASR failures write `[该片段识别失败]`, produce output, and exit 2; `--fail-fast` produces no formal output.
- Unit tests must not require real credentials, model downloads, or network access.

---

## File Map

- `pyproject.toml`: Python version, runtime dependencies, development dependencies, pytest configuration.
- `.gitignore`: virtual environment, credentials, caches, model artifacts, generated media and transcripts.
- `.env.example`: names of required secrets with empty values.
- `src/mimo_transcriber/models.py`: shared dataclasses and status enum.
- `src/mimo_transcriber/config.py`: CLI-independent configuration and startup validation.
- `src/mimo_transcriber/audio.py`: ffprobe, FFmpeg normalization, slicing, Base64 size checks, temporary workspace.
- `src/mimo_transcriber/segments.py`: diarization interval normalization, speaker naming, merging and splitting.
- `src/mimo_transcriber/diarization.py`: device selection and pyannote return-shape adapter.
- `src/mimo_transcriber/mimo_asr.py`: request construction, content parsing, rate limiting, retry and concurrent transcription.
- `src/mimo_transcriber/keywords.py`: local keyword extraction and filtering.
- `src/mimo_transcriber/formatter.py`: Chinese time formatting, TXT/JSON rendering and atomic writes.
- `src/mimo_transcriber/pipeline.py`: end-to-end orchestration, timing, partial failures and exit codes.
- `src/mimo_transcriber/cli.py`: argparse surface, logging and user-facing errors.
- `src/mimo_transcriber/__main__.py`: module entry point.
- `README.md`: project overview, Quick Start, setup, usage, privacy and limitations.
- `tests/`: isolated unit tests plus local FFmpeg integration tests.

---

### Task 1: Project Foundation and Shared Models

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/mimo_transcriber/__init__.py`
- Create: `src/mimo_transcriber/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `SegmentStatus`, `AudioMetadata`, `SpeakerSegment`, `RunSummary`, and `TranscriptionOutcome`.

- [ ] **Step 1: Write the failing model tests**

```python
# tests/test_models.py
from pathlib import Path

from mimo_transcriber.models import (
    AudioMetadata,
    SegmentStatus,
    SpeakerSegment,
    TranscriptionOutcome,
)


def test_speaker_segment_duration_and_default_state() -> None:
    segment = SpeakerSegment(
        index=0, start=1.25, end=3.75, raw_speaker="SPEAKER_00"
    )
    assert segment.duration == 2.5
    assert segment.status is SegmentStatus.PENDING
    assert segment.text is None


def test_transcription_outcome_reports_partial_failure() -> None:
    metadata = AudioMetadata(
        source_path=Path("meeting.m4a"),
        duration_seconds=10.0,
        codec="aac",
        sample_rate=48_000,
        channels=2,
        creation_time=None,
    )
    outcome = TranscriptionOutcome(
        metadata=metadata,
        segments=[
            SpeakerSegment(
                index=0,
                start=0,
                end=1,
                raw_speaker="A",
                status=SegmentStatus.FAILED,
            )
        ],
    )
    assert outcome.has_failures is True
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `uv run pytest tests/test_models.py -v`

Expected: FAIL because the package and model types do not exist.

- [ ] **Step 3: Add project metadata and dependencies**

```toml
# pyproject.toml
[project]
name = "mimo-transcriber"
version = "0.1.0"
description = "Local multi-speaker M4A diarization and MiMo transcription CLI"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
  "jieba>=0.42.1",
  "openai>=1.68.0",
  "pyannote.audio>=3.3.2",
  "python-dotenv>=1.0.1",
  "torch>=2.2",
]

[project.scripts]
mimo-transcriber = "mimo_transcriber.cli:main"

[dependency-groups]
dev = [
  "pytest>=8.3",
  "pytest-asyncio>=0.25",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mimo_transcriber"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
asyncio_mode = "auto"
```

```text
# .gitignore
.env
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
*.wav
*.mp3
*.segments.json
```

```text
# .env.example
MIMO_API_KEY=
HF_TOKEN=
```

- [ ] **Step 4: Implement the shared models**

```python
# src/mimo_transcriber/__init__.py
"""M4A multi-speaker transcription."""

__version__ = "0.1.0"
```

```python
# src/mimo_transcriber/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class SegmentStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class AudioMetadata:
    source_path: Path
    duration_seconds: float
    codec: str
    sample_rate: int
    channels: int
    creation_time: datetime | None


@dataclass
class SpeakerSegment:
    index: int
    start: float
    end: float
    raw_speaker: str
    display_speaker: str | None = None
    text: str | None = None
    status: SegmentStatus = SegmentStatus.PENDING
    error: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RunSummary:
    elapsed_seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    speakers: int = 0
    segments: int = 0
    succeeded: int = 0
    failed: int = 0
    output_path: Path | None = None
    temp_path: Path | None = None


@dataclass
class TranscriptionOutcome:
    metadata: AudioMetadata
    segments: list[SpeakerSegment]
    keywords: list[str] = field(default_factory=list)
    summary: RunSummary = field(default_factory=RunSummary)

    @property
    def has_failures(self) -> bool:
        return any(segment.status is SegmentStatus.FAILED for segment in self.segments)
```

- [ ] **Step 5: Install and run the model tests**

Run: `uv sync && uv run pytest tests/test_models.py -v`

Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/mimo_transcriber tests/test_models.py uv.lock
git commit -m "build: initialize transcriber project"
```

---

### Task 2: Configuration Validation and CLI Arguments

**Files:**
- Create: `src/mimo_transcriber/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `pathlib.Path`.
- Produces: `AppConfig`, `ConfigError`, `validate_runtime(config)`, `resolve_device(requested)`.

- [ ] **Step 1: Write failing configuration tests**

```python
# tests/test_config.py
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


def test_auto_device_does_not_select_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert resolve_device("auto", cuda_available=lambda: True) == "cpu"
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_config.py -v`

Expected: FAIL because `mimo_transcriber.config` does not exist.

- [ ] **Step 3: Implement typed configuration and validation**

```python
# src/mimo_transcriber/config.py
from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from dotenv import load_dotenv

Device = Literal["auto", "cpu", "cuda"]
Language = Literal["auto", "zh", "en"]


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
    concurrency: int = 4
    requests_per_minute: int = 80
    max_retries: int = 3
    keyword_count: int = 20
    keep_temp: bool = False
    debug_json: bool = False
    fail_fast: bool = False
    verbose: bool = False

    @property
    def resolved_output_path(self) -> Path:
        return self.output_path or self.input_path.with_suffix(".txt")

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


def resolve_device(
    requested: Device, cuda_available: Callable[[], bool] | None = None
) -> Literal["cpu", "cuda"]:
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


def validate_runtime(config: AppConfig) -> tuple[str, str]:
    load_dotenv(override=False)
    config.validate_arguments()
    if not config.input_path.is_file() or not os.access(config.input_path, os.R_OK):
        raise ConfigError(f"输入文件不存在或不可读: {config.input_path}")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise ConfigError(f"未找到 {command}；macOS 可运行: brew install ffmpeg")
    mimo_key = os.getenv("MIMO_API_KEY", "")
    hf_token = os.getenv("HF_TOKEN", "")
    if not mimo_key:
        raise ConfigError("缺少 MIMO_API_KEY，请写入环境变量或 .env")
    if not hf_token:
        raise ConfigError("缺少 HF_TOKEN，请写入环境变量或 .env")
    config.resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    return mimo_key, hf_token
```

- [ ] **Step 4: Run the configuration tests**

Run: `uv run pytest tests/test_config.py -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/config.py tests/test_config.py
git commit -m "feat: validate transcriber configuration"
```

---

### Task 3: Segment Normalization, Speaker Mapping, Merging and Splitting

**Files:**
- Create: `src/mimo_transcriber/segments.py`
- Create: `tests/test_segments.py`
- Create: `tests/test_speaker_mapping.py`

**Interfaces:**
- Consumes: raw `SpeakerSegment` values and audio duration.
- Produces: `process_segments(raw, duration, min_duration=0.4, merge_gap=0.8, max_duration=45.0) -> list[SpeakerSegment]`.

- [ ] **Step 1: Write failing transformation tests**

```python
# tests/test_segments.py
from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.segments import process_segments


def seg(start: float, end: float, speaker: str) -> SpeakerSegment:
    return SpeakerSegment(-1, start, end, speaker)


def test_merges_same_speaker_with_small_gap() -> None:
    result = process_segments([seg(0, 1, "A"), seg(1.5, 2, "A")], 10)
    assert [(item.start, item.end) for item in result] == [(0, 2)]


def test_does_not_merge_different_speakers() -> None:
    result = process_segments([seg(0, 1, "A"), seg(1.1, 2, "B")], 10)
    assert len(result) == 2


def test_short_segment_prefers_previous_same_speaker() -> None:
    result = process_segments(
        [seg(0, 1, "A"), seg(1.1, 1.3, "A"), seg(1.4, 2, "A")], 10
    )
    assert [(item.start, item.end) for item in result] == [(0, 2)]


def test_splits_long_segment_contiguously() -> None:
    result = process_segments([seg(0, 100, "A")], 100)
    assert [(item.start, item.end) for item in result] == [
        (0, 45),
        (45, 90),
        (90, 100),
    ]
```

```python
# tests/test_speaker_mapping.py
from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.segments import process_segments


def test_speaker_numbers_follow_first_appearance() -> None:
    raw = [
        SpeakerSegment(-1, 0, 1, "SPEAKER_09"),
        SpeakerSegment(-1, 2, 3, "SPEAKER_01"),
        SpeakerSegment(-1, 4, 5, "SPEAKER_09"),
    ]
    result = process_segments(raw, 5)
    assert [item.display_speaker for item in result] == [
        "说话人 1",
        "说话人 2",
        "说话人 1",
    ]
    assert [item.index for item in result] == [0, 1, 2]
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_segments.py tests/test_speaker_mapping.py -v`

Expected: FAIL because `process_segments` does not exist.

- [ ] **Step 3: Implement the deterministic transformation pipeline**

```python
# src/mimo_transcriber/segments.py
from __future__ import annotations

import logging

from mimo_transcriber.models import SpeakerSegment

logger = logging.getLogger(__name__)


def process_segments(
    raw: list[SpeakerSegment],
    duration: float,
    min_duration: float = 0.4,
    merge_gap: float = 0.8,
    max_duration: float = 45.0,
) -> list[SpeakerSegment]:
    clipped = _clip_and_sort(raw, duration)
    names = _speaker_names(clipped)
    for item in clipped:
        item.display_speaker = names[item.raw_speaker]
    without_short = _merge_or_drop_short(clipped, min_duration, merge_gap)
    merged = _merge_adjacent(without_short, merge_gap)
    split = _split_long(merged, max_duration)
    for index, item in enumerate(split):
        item.index = index
    return split


def _clip_and_sort(
    raw: list[SpeakerSegment], duration: float
) -> list[SpeakerSegment]:
    ordered = sorted(enumerate(raw), key=lambda pair: (pair[1].start, pair[0]))
    result: list[SpeakerSegment] = []
    for _, item in ordered:
        start = max(0.0, min(item.start, duration))
        end = max(0.0, min(item.end, duration))
        if end <= start:
            logger.debug("丢弃非法区间 %.3f-%.3f", item.start, item.end)
            continue
        result.append(SpeakerSegment(-1, start, end, item.raw_speaker))
    return result


def _speaker_names(items: list[SpeakerSegment]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if item.raw_speaker not in result:
            result[item.raw_speaker] = f"说话人 {len(result) + 1}"
    return result


def _merge_pair(left: SpeakerSegment, right: SpeakerSegment) -> SpeakerSegment:
    return SpeakerSegment(
        -1,
        min(left.start, right.start),
        max(left.end, right.end),
        left.raw_speaker,
        left.display_speaker,
    )


def _merge_or_drop_short(
    items: list[SpeakerSegment], minimum: float, gap: float
) -> list[SpeakerSegment]:
    result = list(items)
    index = 0
    while index < len(result):
        item = result[index]
        if item.duration >= minimum:
            index += 1
            continue
        previous = result[index - 1] if index > 0 else None
        following = result[index + 1] if index + 1 < len(result) else None
        if (
            previous
            and previous.raw_speaker == item.raw_speaker
            and item.start - previous.end <= gap
        ):
            result[index - 1] = _merge_pair(previous, item)
            result.pop(index)
        elif (
            following
            and following.raw_speaker == item.raw_speaker
            and following.start - item.end <= gap
        ):
            result[index + 1] = _merge_pair(item, following)
            result.pop(index)
        else:
            logger.debug("跳过无法合并的短区间 %.3f-%.3f", item.start, item.end)
            result.pop(index)
    return result


def _merge_adjacent(
    items: list[SpeakerSegment], gap: float
) -> list[SpeakerSegment]:
    result: list[SpeakerSegment] = []
    for item in items:
        if (
            result
            and result[-1].raw_speaker == item.raw_speaker
            and item.start - result[-1].end <= gap
        ):
            result[-1] = _merge_pair(result[-1], item)
        else:
            result.append(item)
    return result


def _split_long(
    items: list[SpeakerSegment], maximum: float
) -> list[SpeakerSegment]:
    result: list[SpeakerSegment] = []
    for item in items:
        start = item.start
        while item.end - start > maximum:
            result.append(
                SpeakerSegment(-1, start, start + maximum, item.raw_speaker, item.display_speaker)
            )
            start += maximum
        result.append(
            SpeakerSegment(-1, start, item.end, item.raw_speaker, item.display_speaker)
        )
    return result
```

- [ ] **Step 4: Run transformation tests**

Run: `uv run pytest tests/test_segments.py tests/test_speaker_mapping.py -v`

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/segments.py tests/test_segments.py tests/test_speaker_mapping.py
git commit -m "feat: normalize diarization segments"
```

---

### Task 4: Audio Metadata, FFmpeg Conversion and Temporary Workspace

**Files:**
- Create: `src/mimo_transcriber/audio.py`
- Create: `tests/test_audio.py`

**Interfaces:**
- Produces: `probe_audio(path) -> AudioMetadata`, `normalize_audio(source, target)`, `slice_mp3(wav, segment, target)`, `encoded_audio_data(path) -> str`, `workspace(keep)`.

- [ ] **Step 1: Write failing pure and FFmpeg integration tests**

```python
# tests/test_audio.py
import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from mimo_transcriber.audio import encoded_audio_data, normalize_audio, probe_audio


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
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_audio.py -v`

Expected: FAIL because audio functions do not exist.

- [ ] **Step 3: Implement external command and media helpers**

```python
# src/mimo_transcriber/audio.py
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
```

- [ ] **Step 4: Run audio tests**

Run: `uv run pytest tests/test_audio.py -v`

Expected: pure test PASS; FFmpeg test PASS when installed or SKIP with the declared reason.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/audio.py tests/test_audio.py
git commit -m "feat: add FFmpeg audio processing"
```

---

### Task 5: Pyannote Diarization Adapter

**Files:**
- Create: `src/mimo_transcriber/diarization.py`
- Create: `tests/test_diarization.py`

**Interfaces:**
- Consumes: normalized WAV path, HF token, resolved device and speaker constraints.
- Produces: `diarize_audio(...) -> list[SpeakerSegment]`; supports `output.speaker_diarization` and direct annotation outputs.

- [ ] **Step 1: Write failing adapter tests with fake pipeline output**

```python
# tests/test_diarization.py
from pathlib import Path
from types import SimpleNamespace

from mimo_transcriber.diarization import diarize_audio


class Annotation:
    def itertracks(self, yield_label: bool):
        assert yield_label is True
        yield SimpleNamespace(start=0.2, end=1.8), None, "SPEAKER_07"


class Pipeline:
    def __init__(self) -> None:
        self.kwargs = {}

    def __call__(self, path: str, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(speaker_diarization=Annotation())


def test_adapter_extracts_current_community_output() -> None:
    pipeline = Pipeline()
    result = diarize_audio(
        Path("normalized.wav"),
        token="secret",
        device="cpu",
        num_speakers=2,
        min_speakers=1,
        max_speakers=6,
        pipeline_factory=lambda token, device: pipeline,
    )
    assert [(item.start, item.end, item.raw_speaker) for item in result] == [
        (0.2, 1.8, "SPEAKER_07")
    ]
    assert pipeline.kwargs == {"num_speakers": 2}
```

- [ ] **Step 2: Verify the adapter test fails**

Run: `uv run pytest tests/test_diarization.py -v`

Expected: FAIL because the diarization adapter does not exist.

- [ ] **Step 3: Implement lazy model loading and output adaptation**

```python
# src/mimo_transcriber/diarization.py
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mimo_transcriber.models import SpeakerSegment

MODEL_ID = "pyannote/speaker-diarization-community-1"


class DiarizationError(RuntimeError):
    pass


def _default_pipeline(token: str, device: str) -> Any:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline.to(torch.device(device))
    return pipeline


def diarize_audio(
    path: Path,
    token: str,
    device: str,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    pipeline_factory: Callable[[str, str], Any] = _default_pipeline,
) -> list[SpeakerSegment]:
    pipeline = pipeline_factory(token, device)
    kwargs = (
        {"num_speakers": num_speakers}
        if num_speakers is not None
        else {"min_speakers": min_speakers, "max_speakers": max_speakers}
    )
    try:
        output = pipeline(str(path), **kwargs)
        annotation = getattr(output, "speaker_diarization", output)
        return [
            SpeakerSegment(-1, float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    except Exception as exc:
        raise DiarizationError(f"说话人分离失败: {exc}") from exc
```

- [ ] **Step 4: Run adapter tests**

Run: `uv run pytest tests/test_diarization.py -v`

Expected: 1 test PASS without downloading pyannote.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/diarization.py tests/test_diarization.py
git commit -m "feat: adapt pyannote diarization output"
```

---

### Task 6: MiMo Request Parsing, Retry, Rate Limit and Ordered Concurrency

**Files:**
- Create: `src/mimo_transcriber/mimo_asr.py`
- Create: `tests/test_mimo_asr.py`

**Interfaces:**
- Consumes: `SpeakerSegment` plus corresponding MP3 path.
- Produces: `MiMoTranscriber.transcribe_all(items, fail_fast) -> list[SpeakerSegment]`.

- [ ] **Step 1: Write failing request, response and retry tests**

```python
# tests/test_mimo_asr.py
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_transcriber.mimo_asr import MiMoTranscriber, extract_content
from mimo_transcriber.models import SegmentStatus, SpeakerSegment


def test_extract_content_handles_string_and_content_objects() -> None:
    assert extract_content("  你好   world  ") == "你好 world"
    assert extract_content([SimpleNamespace(text=" hello "), {"text": "世界"}]) == "hello 世界"


@pytest.mark.asyncio
async def test_retries_then_returns_results_in_index_order(tmp_path: Path) -> None:
    calls = 0

    async def request(data_url: str, language: str) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("slow")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f" text {calls} "))]
        )

    async def no_sleep(seconds: float) -> None:
        return None

    paths = []
    for index in range(2):
        path = tmp_path / f"segment_{index:04d}.mp3"
        path.write_bytes(b"audio")
        paths.append(path)
    segments = [
        SpeakerSegment(1, 1, 2, "B"),
        SpeakerSegment(0, 0, 1, "A"),
    ]
    transcriber = MiMoTranscriber(
        request=request,
        language="auto",
        concurrency=1,
        requests_per_minute=1000,
        max_retries=1,
        sleep=no_sleep,
    )
    result = await transcriber.transcribe_all(list(zip(segments, paths)), False)
    assert [item.index for item in result] == [0, 1]
    assert all(item.status is SegmentStatus.SUCCESS for item in result)
    assert calls == 3
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_mimo_asr.py -v`

Expected: FAIL because `MiMoTranscriber` does not exist.

- [ ] **Step 3: Implement the injectable asynchronous transcriber**

```python
# src/mimo_transcriber/mimo_asr.py
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mimo_transcriber.audio import encoded_audio_data
from mimo_transcriber.models import SegmentStatus, SpeakerSegment

Request = Callable[[str, str], Awaitable[Any]]
Sleep = Callable[[float], Awaitable[None]]


def extract_content(content: Any) -> str:
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            value = item.get("text") if isinstance(item, dict) else getattr(item, "text", "")
            if value:
                parts.append(str(value))
        raw = " ".join(parts)
    else:
        raw = str(getattr(content, "text", "") or "")
    return " ".join(raw.split())


def is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(exc, (TimeoutError, ConnectionError)) or status == 429 or (
        isinstance(status, int) and status >= 500
    )


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute
        self.lock = asyncio.Lock()
        self.next_time = 0.0

    async def wait(self, sleep: Sleep) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            if delay:
                await sleep(delay)
            self.next_time = max(now, self.next_time) + self.interval


class MiMoTranscriber:
    def __init__(
        self,
        request: Request,
        language: str,
        concurrency: int,
        requests_per_minute: int,
        max_retries: int,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.request = request
        self.language = language
        self.semaphore = asyncio.Semaphore(concurrency)
        self.limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.sleep = sleep

    async def _one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        data_url = encoded_audio_data(path)
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    await self.limiter.wait(self.sleep)
                    completion = await self.request(data_url, self.language)
                text = extract_content(completion.choices[0].message.content)
                if not text:
                    raise ValueError("MiMo 返回了空文本")
                segment.text = text
                segment.status = SegmentStatus.SUCCESS
                return segment
            except Exception as exc:
                if attempt < self.max_retries and is_retryable(exc):
                    await self.sleep((2**attempt) + random.uniform(0, 0.25))
                    continue
                segment.text = "[该片段识别失败]"
                segment.status = SegmentStatus.FAILED
                segment.error = str(exc)
                return segment
        return segment

    async def transcribe_all(
        self, items: list[tuple[SpeakerSegment, Path]], fail_fast: bool
    ) -> list[SpeakerSegment]:
        if fail_fast:
            results: list[SpeakerSegment] = []
            for segment, path in items:
                result = await self._one(segment, path)
                if result.status is SegmentStatus.FAILED:
                    raise RuntimeError(result.error or "片段识别失败")
                results.append(result)
        else:
            results = list(
                await asyncio.gather(*(self._one(segment, path) for segment, path in items))
            )
        return sorted(results, key=lambda item: item.index)


def openai_request(api_key: str, timeout: float = 120.0) -> Request:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key, base_url="https://api.xiaomimimo.com/v1", timeout=timeout
    )

    async def request(data_url: str, language: str) -> Any:
        return await client.chat.completions.create(
            model="mimo-v2.5-asr",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": data_url},
                }],
            }],
            extra_body={"asr_options": {"language": language}},
        )

    return request
```

- [ ] **Step 4: Run MiMo tests**

Run: `uv run pytest tests/test_mimo_asr.py -v`

Expected: 2 tests PASS without network access.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/mimo_asr.py tests/test_mimo_asr.py
git commit -m "feat: transcribe segments with retry and limits"
```

---

### Task 7: Keyword Extraction and Atomic Output Formatting

**Files:**
- Create: `src/mimo_transcriber/keywords.py`
- Create: `src/mimo_transcriber/formatter.py`
- Create: `tests/test_keywords.py`
- Create: `tests/test_formatter.py`

**Interfaces:**
- Produces: `extract_keywords(texts, count)`, `render_transcript(outcome, recording_time)`, `write_outputs(...)`.

- [ ] **Step 1: Write failing keyword and formatter tests**

```python
# tests/test_keywords.py
from mimo_transcriber.keywords import extract_keywords


def test_keywords_keep_meaningful_english_and_drop_noise() -> None:
    result = extract_keywords(
        ["我们使用 Java Spring Redis Agent RAG 构建系统。", "123 了 的 和"],
        10,
    )
    assert "Java" in result
    assert "Agent" in result
    assert "123" not in result
```

```python
# tests/test_formatter.py
from datetime import datetime
from pathlib import Path

from mimo_transcriber.formatter import (
    format_duration,
    format_timestamp,
    render_transcript,
)
from mimo_transcriber.models import (
    AudioMetadata,
    SegmentStatus,
    SpeakerSegment,
    TranscriptionOutcome,
)


def test_time_formats() -> None:
    assert format_timestamp(62.9) == "01:02"
    assert format_timestamp(3661.2) == "01:01:01"
    assert format_duration(513) == "8分钟 33秒"
    assert format_duration(4113) == "1小时 8分钟 33秒"


def test_renders_exact_transcript_layout() -> None:
    metadata = AudioMetadata(Path("input.m4a"), 63, "aac", 48000, 2, None)
    outcome = TranscriptionOutcome(
        metadata=metadata,
        keywords=["Agent", "RAG"],
        segments=[
            SpeakerSegment(
                0, 2, 4, "A", "说话人 1", "你好。", SegmentStatus.SUCCESS
            )
        ],
    )
    rendered = render_transcript(outcome, datetime(2026, 6, 15, 23, 32))
    assert rendered == (
        "2026年6月15日 下午 11:32|1分钟 3秒\n\n"
        "关键词:\nAgent、RAG\n\n"
        "文字记录:\n说话人 1 00:02\n你好。\n"
    )
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/test_keywords.py tests/test_formatter.py -v`

Expected: FAIL because keyword and formatter modules do not exist.

- [ ] **Step 3: Implement keyword filtering**

```python
# src/mimo_transcriber/keywords.py
from __future__ import annotations

import re

import jieba.analyse

STOPWORDS = {"的", "了", "和", "是", "在", "我", "你", "他", "她", "它"}


def _valid(value: str) -> bool:
    token = value.strip()
    if not token or token in STOPWORDS or token.isdigit():
        return False
    if len(token) == 1 and (not token.isascii() or re.fullmatch(r"\W", token)):
        return False
    return True


def extract_keywords(texts: list[str], count: int) -> list[str]:
    if count == 0:
        return []
    candidates = jieba.analyse.extract_tags(
        "\n".join(texts), topK=max(count * 3, count)
    )
    result: list[str] = []
    for candidate in candidates:
        if _valid(candidate) and candidate not in result:
            result.append(candidate)
        if len(result) == count:
            break
    return result
```

- [ ] **Step 4: Implement formatting and atomic writes**

```python
# src/mimo_transcriber/formatter.py
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mimo_transcriber.models import TranscriptionOutcome


def format_timestamp(seconds: float) -> str:
    total = math.floor(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{secs:02d}"
        if hours
        else f"{minutes:02d}:{secs:02d}"
    )


def format_duration(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    prefix = f"{hours}小时 " if hours else ""
    return f"{prefix}{minutes}分钟 {secs}秒"


def format_recording_time(value: datetime) -> str:
    local = value.astimezone() if value.tzinfo else value
    period = "上午" if local.hour < 12 else "下午"
    hour = local.hour % 12 or 12
    return f"{local.year}年{local.month}月{local.day}日 {period} {hour}:{local.minute:02d}"


def render_transcript(outcome: TranscriptionOutcome, recording_time: datetime) -> str:
    first = (
        f"{format_recording_time(recording_time)}|"
        f"{format_duration(outcome.metadata.duration_seconds)}"
    )
    blocks = [
        f"{segment.display_speaker} {format_timestamp(segment.start)}\n{segment.text or ''}"
        for segment in outcome.segments
    ]
    transcript = "\n\n".join(blocks)
    return (
        f"{first}\n\n关键词:\n{'、'.join(outcome.keywords)}\n\n"
        f"文字记录:\n{transcript}\n"
    )


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_outputs(
    outcome: TranscriptionOutcome,
    recording_time: datetime,
    output_path: Path,
    debug_json: bool,
) -> None:
    _atomic_write(output_path, render_transcript(outcome, recording_time))
    if debug_json:
        payload = {
            "source": outcome.metadata.source_path.name,
            "duration_seconds": outcome.metadata.duration_seconds,
            "speakers": len({item.raw_speaker for item in outcome.segments}),
            "segments": [asdict(item) for item in outcome.segments],
        }
        json_path = output_path.with_suffix(".segments.json")
        _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
```

- [ ] **Step 5: Run formatting tests**

Run: `uv run pytest tests/test_keywords.py tests/test_formatter.py -v`

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/keywords.py src/mimo_transcriber/formatter.py tests/test_keywords.py tests/test_formatter.py
git commit -m "feat: format transcripts and extract keywords"
```

---

### Task 8: End-to-End Pipeline and Failure Semantics

**Files:**
- Create: `src/mimo_transcriber/pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `src/mimo_transcriber/audio.py`
- Modify: `src/mimo_transcriber/segments.py`

**Interfaces:**
- Consumes: `AppConfig`, credentials and the adapters from Tasks 3–7.
- Produces: `prepare_audio_segments(...)`, `run_pipeline(config, mimo_key, hf_token) -> PipelineResult`; exit code is 0 or 2.

- [ ] **Step 1: Write failing pipeline tests with injected adapters**

```python
# tests/test_pipeline.py
from datetime import datetime
from pathlib import Path

import pytest

from mimo_transcriber.config import AppConfig
from mimo_transcriber.models import AudioMetadata, SegmentStatus, SpeakerSegment
from mimo_transcriber.pipeline import (
    PipelineDependencies,
    prepare_audio_segments,
    run_pipeline,
)


@pytest.mark.asyncio
async def test_partial_failure_writes_output_and_returns_two(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 2, "aac", 48000, 2, datetime(2026, 1, 1, 9))

    async def transcribe(items, fail_fast):
        successful = items[0][0]
        successful.text = "你好"
        successful.status = SegmentStatus.SUCCESS
        failed = items[1][0]
        failed.text = "[该片段识别失败]"
        failed.status = SegmentStatus.FAILED
        failed.error = "timeout"
        return [successful, failed]

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        diarize=lambda *args, **kwargs: [
            SpeakerSegment(-1, 0, 1, "A"),
            SpeakerSegment(-1, 1, 2, "B"),
        ],
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        transcribe=transcribe,
    )
    result = await run_pipeline(
        AppConfig(input_path=source, output_path=output, num_speakers=2),
        "mimo",
        "hf",
        dependencies,
    )
    assert result.exit_code == 2
    assert output.exists()
    assert "[该片段识别失败]" in output.read_text()


def test_oversize_segment_is_split_until_payload_is_accepted(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.wav"
    normalized.write_bytes(b"wav")
    calls: list[tuple[float, float]] = []

    def slice_audio(source: Path, segment: SpeakerSegment, target: Path) -> None:
        calls.append((segment.start, segment.end))
        target.write_bytes(b"mp3")

    def payload_fits(path: Path, segment: SpeakerSegment) -> bool:
        return segment.duration <= 2.5

    items = prepare_audio_segments(
        normalized,
        [SpeakerSegment(0, 0, 10, "A", "说话人 1")],
        tmp_path,
        slice_audio,
        payload_fits,
    )
    assert [(item.start, item.end) for item, _ in items] == [
        (0, 2.5),
        (2.5, 5),
        (5, 7.5),
        (7.5, 10),
    ]
    assert [item.index for item, _ in items] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_fail_fast_does_not_write_formal_output(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48000, 2, None)

    async def fail(items, fail_fast):
        raise RuntimeError("first failed segment")

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        diarize=lambda *args, **kwargs: [SpeakerSegment(-1, 0, 1, "A")],
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, segment: True,
        transcribe=fail,
    )
    with pytest.raises(RuntimeError, match="first failed segment"):
        await run_pipeline(
            AppConfig(
                input_path=source,
                output_path=output,
                num_speakers=1,
                fail_fast=True,
            ),
            "mimo",
            "hf",
            dependencies,
        )
    assert output.exists() is False
```

- [ ] **Step 2: Verify the pipeline test fails**

Run: `uv run pytest tests/test_pipeline.py -v`

Expected: FAIL because pipeline types do not exist.

- [ ] **Step 3: Add Base64 oversize subdivision before slicing**

```python
# append to src/mimo_transcriber/segments.py
def split_segment(segment: SpeakerSegment) -> list[SpeakerSegment]:
    midpoint = segment.start + segment.duration / 2
    return [
        SpeakerSegment(-1, segment.start, midpoint, segment.raw_speaker, segment.display_speaker),
        SpeakerSegment(-1, midpoint, segment.end, segment.raw_speaker, segment.display_speaker),
    ]
```

```python
# append to src/mimo_transcriber/audio.py
def payload_fits(path: Path, segment: SpeakerSegment) -> bool:
    encoded = base64.b64encode(path.read_bytes())
    return len(encoded) <= MAX_BASE64_BYTES
```

- [ ] **Step 4: Implement the orchestrator and dependency bundle**

```python
# src/mimo_transcriber/pipeline.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from mimo_transcriber.audio import (
    normalize_audio,
    payload_fits,
    probe_audio,
    slice_mp3,
    workspace,
)
from mimo_transcriber.config import AppConfig, resolve_device
from mimo_transcriber.diarization import diarize_audio
from mimo_transcriber.formatter import write_outputs
from mimo_transcriber.keywords import extract_keywords
from mimo_transcriber.mimo_asr import MiMoTranscriber, openai_request
from mimo_transcriber.models import RunSummary, SegmentStatus, SpeakerSegment, TranscriptionOutcome
from mimo_transcriber.segments import process_segments, split_segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineDependencies:
    probe: Callable[[Path], Any] = probe_audio
    normalize: Callable[[Path, Path], None] = normalize_audio
    diarize: Callable[..., list[SpeakerSegment]] = diarize_audio
    slice_audio: Callable[[Path, SpeakerSegment, Path], None] = slice_mp3
    payload_fits: Callable[[Path, SpeakerSegment], bool] = payload_fits
    transcribe: Callable[
        [list[tuple[SpeakerSegment, Path]], bool],
        Awaitable[list[SpeakerSegment]],
    ] | None = None


@dataclass(frozen=True)
class PipelineResult:
    outcome: TranscriptionOutcome
    exit_code: int


def prepare_audio_segments(
    normalized: Path,
    segments: list[SpeakerSegment],
    temp: Path,
    slice_audio: Callable[[Path, SpeakerSegment, Path], None],
    fits: Callable[[Path, SpeakerSegment], bool],
) -> list[tuple[SpeakerSegment, Path]]:
    pending = list(segments)
    accepted: list[tuple[SpeakerSegment, Path]] = []
    while pending:
        segment = pending.pop(0)
        path = temp / "candidate.mp3"
        slice_audio(normalized, segment, path)
        if fits(path, segment):
            accepted.append((segment, path.with_name(f"segment_{len(accepted):04d}.mp3")))
            path.replace(accepted[-1][1])
        else:
            path.unlink(missing_ok=True)
            pending[0:0] = split_segment(segment)
    for index, (segment, _) in enumerate(accepted):
        segment.index = index
    return accepted


async def run_pipeline(
    config: AppConfig,
    mimo_key: str,
    hf_token: str,
    dependencies: PipelineDependencies = PipelineDependencies(),
) -> PipelineResult:
    started = time.monotonic()
    metadata = dependencies.probe(config.input_path)
    with workspace(config.keep_temp) as temp:
        normalized = temp / "normalized.wav"
        dependencies.normalize(config.input_path, normalized)
        raw = dependencies.diarize(
            normalized,
            hf_token,
            resolve_device(config.device),
            config.num_speakers,
            config.min_speakers,
            config.max_speakers,
        )
        segments = process_segments(raw, metadata.duration_seconds)
        items = prepare_audio_segments(
            normalized,
            segments,
            temp,
            dependencies.slice_audio,
            dependencies.payload_fits,
        )
        transcribe = dependencies.transcribe
        if transcribe is None:
            client = MiMoTranscriber(
                request=openai_request(mimo_key),
                language=config.language,
                concurrency=config.concurrency,
                requests_per_minute=config.requests_per_minute,
                max_retries=config.max_retries,
            )
            transcribe = client.transcribe_all
        completed = await transcribe(items, config.fail_fast)
        successful_texts = [
            item.text or "" for item in completed if item.status is SegmentStatus.SUCCESS
        ]
        outcome = TranscriptionOutcome(
            metadata=metadata,
            segments=completed,
            keywords=extract_keywords(successful_texts, config.keyword_count),
            summary=RunSummary(
                elapsed_seconds=time.monotonic() - started,
                speakers=len({item.raw_speaker for item in completed}),
                segments=len(completed),
                succeeded=sum(item.status is SegmentStatus.SUCCESS for item in completed),
                failed=sum(item.status is SegmentStatus.FAILED for item in completed),
                output_path=config.resolved_output_path,
                temp_path=temp if config.keep_temp else None,
            ),
        )
        recording_time = metadata.creation_time or datetime.fromtimestamp(
            config.input_path.stat().st_mtime
        )
        write_outputs(
            outcome, recording_time, config.resolved_output_path, config.debug_json
        )
    return PipelineResult(outcome, 2 if outcome.has_failures else 0)
```

- [ ] **Step 5: Run oversize and fail-fast pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`

Expected: partial-failure, recursive subdivision and fail-fast tests PASS.

- [ ] **Step 6: Run the complete suite**

Run: `uv run pytest -v`

Expected: all tests PASS; only FFmpeg integration tests may be skipped when FFmpeg is absent.

- [ ] **Step 7: Commit**

```bash
git add src/mimo_transcriber/pipeline.py src/mimo_transcriber/audio.py src/mimo_transcriber/segments.py tests/test_pipeline.py
git commit -m "feat: orchestrate transcription pipeline"
```

---

### Task 9: User-Facing CLI, Logging and Exit Codes

**Files:**
- Create: `src/mimo_transcriber/cli.py`
- Create: `src/mimo_transcriber/__main__.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces: `build_parser()`, `async_main(argv) -> int`, `main() -> Never`.

- [ ] **Step 1: Write failing CLI tests**

```python
# tests/test_cli.py
from mimo_transcriber.cli import build_parser


def test_parser_exposes_required_defaults() -> None:
    args = build_parser().parse_args(["meeting.m4a"])
    assert args.language == "auto"
    assert args.device == "auto"
    assert args.concurrency == 4
    assert args.requests_per_minute == 80
    assert args.max_retries == 3
    assert args.keyword_count == 20
```

- [ ] **Step 2: Verify CLI test fails**

Run: `uv run pytest tests/test_cli.py -v`

Expected: FAIL because `cli.py` does not exist.

- [ ] **Step 3: Implement parser, logging and error boundary**

```python
# src/mimo_transcriber/cli.py
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Sequence

from mimo_transcriber.config import AppConfig, ConfigError, validate_runtime
from mimo_transcriber.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimo-transcriber",
        description="将多人 M4A 录音按说话人转写为带时间戳的 TXT。",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument("--language", choices=("auto", "zh", "en"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=int, default=80)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--keyword-count", type=int, default=20)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--debug-json", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = AppConfig(
        input_path=args.input,
        output_path=args.output,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=args.language,
        device=args.device,
        concurrency=args.concurrency,
        requests_per_minute=args.requests_per_minute,
        max_retries=args.max_retries,
        keyword_count=args.keyword_count,
        keep_temp=args.keep_temp,
        debug_json=args.debug_json,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
    )
    try:
        mimo_key, hf_token = validate_runtime(config)
        result = await run_pipeline(config, mimo_key, hf_token)
        logging.info("输出文件: %s", result.outcome.summary.output_path)
        logging.info(
            "片段: %d 成功 / %d 失败；耗时 %.2f 秒",
            result.outcome.summary.succeeded,
            result.outcome.summary.failed,
            result.outcome.summary.elapsed_seconds,
        )
        return result.exit_code
    except (ConfigError, RuntimeError) as exc:
        logging.error("%s", exc)
        if args.verbose:
            logging.exception("详细错误")
        return 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))
```

```python
# src/mimo_transcriber/__main__.py
from mimo_transcriber.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests and help command**

Run: `uv run pytest tests/test_cli.py -v && uv run python -m mimo_transcriber --help`

Expected: test PASS and help lists every documented option without loading pyannote.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/cli.py src/mimo_transcriber/__main__.py tests/test_cli.py
git commit -m "feat: expose transcription CLI"
```

---

### Task 10: README, Full Verification and Real M1 Pro Smoke Test

**Files:**
- Create: `README.md`
- Modify: tests or implementation only for failures found by verification.

**Interfaces:**
- Documents the final CLI and does not introduce new runtime behavior.

- [ ] **Step 1: Write the README with project introduction and Quick Start**

````markdown
# MiMo 多说话人录音转写 CLI

一个本地运行的 Python CLI：使用 FFmpeg 标准化 M4A，使用 pyannote 按音色区分说话人，将切割后的短 MP3 片段发送给 MiMo-V2.5-ASR，最后生成带说话人、时间戳和关键词的 UTF-8 TXT。

完整原始录音不会直接发送给 MiMo；只有 diarization 后的音频片段会上传。

## Quick Start

```bash
brew install ffmpeg
uv sync
export MIMO_API_KEY="..."
export HF_TOKEN="..."
uv run python -m mimo_transcriber meeting.m4a --num-speakers 2
```

## 环境要求

- Python 3.11
- uv
- FFmpeg 与 ffprobe
- macOS Apple Silicon 使用 CPU
- Linux 可选 NVIDIA CUDA

## Hugging Face 授权

登录 Hugging Face，接受 `pyannote/speaker-diarization-community-1` 的模型条款，然后创建 Read 权限 Token 并保存为 `HF_TOKEN`。首次运行会下载模型。

## 使用示例

```bash
uv run python -m mimo_transcriber meeting.m4a --num-speakers 3 --language zh
uv run python -m mimo_transcriber meeting.m4a --language auto
uv run python -m mimo_transcriber meeting.m4a --debug-json --keep-temp --verbose
```

## 准确率与隐私边界

- 说话人编号只在当前录音内有效，不能跨录音识别真实人物。
- 重叠讲话、极短插话、背景噪声和低质量音频会降低准确率。
- CPU 可以运行，长音频的 diarization 可能较慢。
- 音频片段会发送给 MiMo API；完整原始文件不会直接上传。
````

Continue the README with this concrete content:

````markdown
## Linux 安装

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

安装 uv 后运行 `uv sync`。带 NVIDIA GPU 的 Linux 可以使用 `--device cuda`；否则使用默认的 `--device auto`。

## 环境变量

复制 `.env.example` 为 `.env`，或在 shell 中设置：

```bash
export MIMO_API_KEY="..."
export HF_TOKEN="..."
```

已有环境变量优先于 `.env`。不要把 `.env` 或真实 Token 提交到 Git。

## CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 本地 M4A 输入文件 |
| `-o, --output` | 输入同名 TXT | 输出路径 |
| `--num-speakers` | 自动估计 | 已知准确说话人数 |
| `--min-speakers` | `1` | 自动估计下限 |
| `--max-speakers` | `6` | 自动估计上限 |
| `--language` | `auto` | `auto`、`zh` 或 `en` |
| `--device` | `auto` | `auto`、`cpu` 或 `cuda` |
| `--concurrency` | `4` | MiMo 最大并发 |
| `--requests-per-minute` | `80` | 全局每分钟请求上限 |
| `--max-retries` | `3` | 首次失败后的最大重试次数 |
| `--keyword-count` | `20` | 关键词数量 |
| `--keep-temp` | 关闭 | 保留 WAV/MP3 临时文件 |
| `--debug-json` | 关闭 | 额外生成 `.segments.json` |
| `--fail-fast` | 关闭 | 首段最终失败即停止且不写正式 TXT |
| `-v, --verbose` | 关闭 | 输出调试日志和阶段耗时 |

## 输出与退出码

TXT 第一行是录音时间和时长，随后是关键词与按时间排序的说话人片段。退出码 `0` 表示全部成功，`1` 表示启动或关键阶段失败，`2` 表示 TXT 已生成但存在失败片段。

## 常见问题

- `ffmpeg/ffprobe not found`：macOS 运行 `brew install ffmpeg`；Ubuntu/Debian 安装 `ffmpeg` 包。
- Hugging Face 401/403：确认已接受 Community-1 模型条款，并使用 Read 权限 `HF_TOKEN`。
- CUDA 不可用：改用 `--device cpu`，或检查 NVIDIA 驱动和 PyTorch CUDA 环境。
- MiMo 429/5xx/超时：程序会按指数退避重试；可降低 `--concurrency` 或 `--requests-per-minute`。
- 某些片段显示 `[该片段识别失败]`：查看 verbose 日志；默认仍会生成结果并返回退出码 2。
````

- [ ] **Step 2: Run static project verification**

Run:

```bash
uv sync
uv run pytest
uv run python -m mimo_transcriber --help
```

Expected: dependency sync succeeds, all tests pass, and help prints without loading/downloading pyannote.

- [ ] **Step 3: Confirm credentials are inherited without printing them**

Run:

```bash
zsh -lc 'test -n "$MIMO_API_KEY" && test -n "$HF_TOKEN"'
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Run the real short-recording smoke test**

Run:

```bash
zsh -lc 'cd /Users/yuancheng/Documents/Code/CScribe && uv run python -m mimo_transcriber "/Users/yuancheng/Downloads/新录音 15.m4a" --num-speakers 2 --language auto --debug-json --verbose'
```

Expected:

- pyannote downloads/loads `speaker-diarization-community-1`.
- macOS selects CPU.
- The 37.7-second ALAC input is normalized and split.
- All successful segments contain non-empty MiMo text.
- `新录音 15.txt` and `新录音 15.segments.json` are created next to the source.
- Exit code is 0 when every segment succeeds, or 2 with explicit failed placeholders.

- [ ] **Step 5: Inspect generated output without exposing credentials**

Run:

```bash
sed -n '1,120p' "/Users/yuancheng/Downloads/新录音 15.txt"
```

Expected: Chinese recording time and duration, keyword section, two stable speaker labels, ordered timestamps, transcript blocks, and a final newline. No `SPEAKER_` labels or secrets appear.

- [ ] **Step 6: Report measured behavior and accuracy limits**

Record in the handoff:

- Total and per-stage elapsed time.
- Number of speakers and final segments.
- Successful and failed segment counts.
- Output paths.
- Manual notes on speaker attribution and transcription accuracy.
- The remaining limitations around overlap, short utterances and CPU speed.

- [ ] **Step 7: Commit documentation and final verification fixes**

```bash
git add README.md src tests pyproject.toml uv.lock
git commit -m "docs: add quick start and usage guide"
```
