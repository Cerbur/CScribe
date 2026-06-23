# Local ASR Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CScribe default to local MLX Whisper ASR while keeping MiMo as an optional engine behind a clean ASR, worker, and state-projection boundary.

**Architecture:** Add an `asr/` package with provider-neutral config, cache identity, factory, MLX engine, and MiMo engine. Refactor pipeline workers so slice/ASR workers emit events while a single `StateWorker` projects those events into manifest state and terminal progress. Keep the existing resumable pipeline behavior, but remove direct `store.save()` and `reporter.*()` calls from business workers.

**Tech Stack:** Python 3.11, asyncio queues/workers, dataclasses, pytest, pytest-asyncio, pyannote.audio, ffmpeg/ffprobe, openai SDK, mlx-whisper.

## Global Constraints

- Default ASR engine is `mlx`.
- MiMo is selected explicitly with `--asr mimo`.
- `MIMO_API_KEY` is required only when `--asr mimo`.
- `HF_TOKEN` is still required for pyannote diarization.
- `ffmpeg` and `ffprobe` remain required for all modes.
- ASR model and provider details stay inside the ASR layer; pipeline treats cache identity as opaque JSON.
- Slice workers and ASR workers must not call `ManifestStore.save()` or `ProgressReporter` directly.
- `StateWorker` is the only layer that projects slice/transcript events into manifest and progress.
- Unit tests must not download real MLX models or call real MiMo APIs.
- Real MLX model execution is verified only by manual smoke command after unit tests pass.

---

## File Structure

- Create `src/mimo_transcriber/asr/__init__.py`: export public ASR config, runtime config, factory, and engine protocol.
- Create `src/mimo_transcriber/asr/base.py`: define `AsrProvider`, `AsrConfig`, `RuntimeConfig`, `AsrEngine`, `AsrEventSink`, and cache identity helpers.
- Create `src/mimo_transcriber/asr/factory.py`: create concrete ASR engines from config/runtime/event sink.
- Create `src/mimo_transcriber/asr/mimo.py`: move existing MiMo transcription implementation behind `MimoAsrEngine`.
- Create `src/mimo_transcriber/asr/mlx.py`: implement local MLX Whisper engine.
- Create `src/mimo_transcriber/events.py`: define state/progress events and `AudioSlice`.
- Create `src/mimo_transcriber/state_worker.py`: implement `RunStateProjector`.
- Modify `src/mimo_transcriber/config.py`: add ASR config fields, runtime validation, and cache identity.
- Modify `src/mimo_transcriber/cache.py`: replace hard-coded `MIMO_MODEL_ID` with opaque ASR identity.
- Modify `src/mimo_transcriber/pipeline.py`: use ASR factory/engine, state queue, and `RunStateProjector`.
- Modify `src/mimo_transcriber/cli.py`: parse `--asr` and `--stt-model`, pass `RuntimeConfig`.
- Modify `pyproject.toml`: add `mlx-whisper` dependency.
- Modify `README.md`: document local default and optional MiMo.
- Create or modify tests listed in each task.

---

### Task 1: ASR Config, Runtime Config, and Cache Identity

**Files:**
- Create: `src/mimo_transcriber/asr/__init__.py`
- Create: `src/mimo_transcriber/asr/base.py`
- Modify: `src/mimo_transcriber/config.py`
- Modify: `src/mimo_transcriber/cache.py`
- Test: `tests/test_asr_config.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `AsrConfig(provider: AsrProvider = "mlx", stt_model: str | None = None, language: Language = "auto")`
- Produces: `RuntimeConfig(hf_token: str, mimo_api_key: str | None = None)`
- Produces: `AsrConfig.resolved_model() -> str`
- Produces: `AsrConfig.cache_identity() -> dict[str, object]`
- Modifies: `AppConfig.asr: AsrProvider`, `AppConfig.stt_model: str | None`
- Modifies: `validate_runtime(config: AppConfig) -> RuntimeConfig`

- [ ] **Step 1: Write failing ASR config tests**

Add `tests/test_asr_config.py`:

```python
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
```

- [ ] **Step 2: Write failing runtime validation tests**

Append to `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from mimo_transcriber.config import AppConfig, ConfigError, validate_runtime


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
```

- [ ] **Step 3: Update cache identity tests**

Append to `tests/test_cache.py`:

```python
from pathlib import Path

from mimo_transcriber.cache import TaskPaths, fingerprint_input
from mimo_transcriber.config import AppConfig


def test_task_hash_changes_when_asr_identity_changes(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    mlx = TaskPaths.for_run(
        AppConfig(input_path=source, asr="mlx"),
        fingerprint,
        tmp_path,
    )
    mimo = TaskPaths.for_run(
        AppConfig(input_path=source, asr="mimo", stt_model="mimo-v2.5-asr"),
        fingerprint,
        tmp_path,
    )

    assert mlx.task_hash != mimo.task_hash


def test_task_hash_ignores_asr_runtime_controls(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    first = TaskPaths.for_run(
        AppConfig(input_path=source, concurrency=1, max_retries=0),
        fingerprint,
        tmp_path,
    )
    second = TaskPaths.for_run(
        AppConfig(input_path=source, concurrency=8, max_retries=5),
        fingerprint,
        tmp_path,
    )

    assert first.task_hash == second.task_hash
```

- [ ] **Step 4: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_asr_config.py tests/test_config.py tests/test_cache.py -q
```

Expected: FAIL with import errors for `mimo_transcriber.asr` or missing `AppConfig.asr`.

- [ ] **Step 5: Implement ASR base types**

Create `src/mimo_transcriber/asr/base.py`:

```python
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from mimo_transcriber.config import Language
from mimo_transcriber.models import SpeakerSegment

AsrProvider = Literal["mlx", "mimo"]
AsrEventSink = Callable[[object], Awaitable[None]]

DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MIMO_MODEL = "mimo-v2.5-asr"


@dataclass(frozen=True)
class AsrConfig:
    provider: AsrProvider = "mlx"
    stt_model: str | None = None
    language: Language = "auto"

    def resolved_model(self) -> str:
        if self.stt_model:
            return self.stt_model
        if self.provider == "mimo":
            return DEFAULT_MIMO_MODEL
        return DEFAULT_MLX_MODEL

    def cache_identity(self) -> dict[str, object]:
        engine = "mimo" if self.provider == "mimo" else "mlx-whisper"
        return {
            "kind": "asr-engine",
            "engine": engine,
            "identity_version": 1,
            "settings": {
                "model": self.resolved_model(),
                "language": self.language,
            },
        }


@dataclass(frozen=True)
class RuntimeConfig:
    hf_token: str
    mimo_api_key: str | None = None


class AsrEngine(Protocol):
    @property
    def cache_identity(self) -> Mapping[str, object]:
        ...

    async def transcribe_one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        ...

    async def transcribe_all(
        self,
        items: list[tuple[SpeakerSegment, Path]],
        fail_fast: bool,
    ) -> list[SpeakerSegment]:
        ...
```

Create `src/mimo_transcriber/asr/__init__.py`:

```python
from mimo_transcriber.asr.base import (
    AsrConfig,
    AsrEngine,
    AsrEventSink,
    AsrProvider,
    RuntimeConfig,
)

__all__ = [
    "AsrConfig",
    "AsrEngine",
    "AsrEventSink",
    "AsrProvider",
    "RuntimeConfig",
]
```

- [ ] **Step 6: Wire config and cache identity**

Modify `src/mimo_transcriber/config.py`:

```python
from typing import Callable, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from mimo_transcriber.asr.base import AsrProvider, RuntimeConfig
else:
    AsrProvider = Literal["mlx", "mimo"]
```

Add fields to `AppConfig`:

```python
    asr: AsrProvider = "mlx"
    stt_model: str | None = None
```

Add method to `AppConfig`:

```python
    def asr_cache_identity(self) -> dict[str, object]:
        from mimo_transcriber.asr.base import AsrConfig

        return AsrConfig(
            provider=self.asr,
            stt_model=self.stt_model,
            language=self.language,
        ).cache_identity()
```

Add to `cache_parameters()`:

```python
            "asr": self.asr_cache_identity(),
```

Modify `validate_arguments()`:

```python
        if self.asr not in ("mlx", "mimo"):
            raise ConfigError("--asr 必须是 mlx 或 mimo")
```

Modify `validate_runtime()` return:

```python
def validate_runtime(config: AppConfig) -> RuntimeConfig:
    from mimo_transcriber.asr.base import RuntimeConfig
```

Replace MiMo key handling:

```python
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
```

Modify `src/mimo_transcriber/cache.py`:

```python
# Remove MIMO_MODEL_ID = "mimo-v2.5-asr"
```

Remove `"mimo_model": MIMO_MODEL_ID,` from the identity dict. The ASR identity is already included through `config.cache_parameters()`.

- [ ] **Step 7: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_asr_config.py tests/test_config.py tests/test_cache.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mimo_transcriber/asr/__init__.py src/mimo_transcriber/asr/base.py src/mimo_transcriber/config.py src/mimo_transcriber/cache.py tests/test_asr_config.py tests/test_config.py tests/test_cache.py
git commit -m "feat: add asr config identity"
```

---

### Task 2: CLI ASR Flags

**Files:**
- Modify: `src/mimo_transcriber/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `AppConfig(asr=..., stt_model=...)`
- Produces: CLI flags `--asr {mlx,mimo}` and `--stt-model MODEL`

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_cli.py`:

```python
from mimo_transcriber.cli import build_parser


def test_parser_defaults_to_mlx_asr() -> None:
    args = build_parser().parse_args(["meeting.m4a"])

    assert args.asr == "mlx"
    assert args.stt_model is None


def test_parser_accepts_mimo_asr_and_model() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--asr",
        "mimo",
        "--stt-model",
        "mimo-v2.5-asr",
    ])

    assert args.asr == "mimo"
    assert args.stt_model == "mimo-v2.5-asr"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_cli.py -q
```

Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'asr'`.

- [ ] **Step 3: Add CLI arguments and config wiring**

Modify `build_parser()` in `src/mimo_transcriber/cli.py`:

```python
    parser.add_argument("--asr", choices=("mlx", "mimo"), default="mlx")
    parser.add_argument("--stt-model")
```

Modify `AppConfig(...)` construction:

```python
        asr=args.asr,
        stt_model=args.stt_model,
```

Modify runtime variable names:

```python
        runtime = validate_runtime(config)
        result = await run_pipeline(config, runtime, reporter=reporter)
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
uv run pytest tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/cli.py tests/test_cli.py
git commit -m "feat: add asr cli flags"
```

---

### Task 3: Pipeline Events and RunStateProjector

**Files:**
- Create: `src/mimo_transcriber/events.py`
- Create: `src/mimo_transcriber/state_worker.py`
- Test: `tests/test_pipeline_events.py`
- Test: `tests/test_state_worker.py`

**Interfaces:**
- Produces: `AudioSlice(segment_id: str, segment: SpeakerSegment, path: Path)`
- Produces: event dataclasses `StageStarted`, `SegmentTotalChanged`, `SliceReady`, `SliceFailed`, `SegmentsExpanded`, `TranscriptRetrying`, `TranscriptSucceeded`, `TranscriptFailed`
- Produces: `RunStateProjector.handle(event: PipelineEvent) -> None`
- Produces: `RunStateProjector.snapshot_completed_segments() -> list[SpeakerSegment]`

- [ ] **Step 1: Write failing event tests**

Create `tests/test_pipeline_events.py`:

```python
from pathlib import Path

from mimo_transcriber.events import AudioSlice, SliceReady, TranscriptSucceeded
from mimo_transcriber.models import SpeakerSegment


def test_audio_slice_carries_segment_and_path() -> None:
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    audio_slice = AudioSlice("s0000", segment, Path("s0000.mp3"))

    assert audio_slice.segment_id == "s0000"
    assert audio_slice.segment is segment
    assert audio_slice.path == Path("s0000.mp3")


def test_events_are_immutable() -> None:
    event = SliceReady("s0000", Path("s0000.mp3"), 123)

    try:
        event.bytes = 456
    except Exception as exc:
        assert type(exc).__name__ == "FrozenInstanceError"
    else:
        raise AssertionError("event should be frozen")


def test_transcript_event_has_segment_identity() -> None:
    event = TranscriptSucceeded("s0000", "hello")

    assert event.segment_id == "s0000"
    assert event.text == "hello"
```

- [ ] **Step 2: Write failing StateWorker tests**

Create `tests/test_state_worker.py`:

```python
from pathlib import Path

from mimo_transcriber.events import (
    SegmentTotalChanged,
    SliceFailed,
    SliceReady,
    TranscriptFailed,
    TranscriptRetrying,
    TranscriptSucceeded,
)
from mimo_transcriber.manifest import SegmentRecord, TaskIdentity, TaskManifest
from mimo_transcriber.models import AudioMetadata, SegmentStatus, SpeakerSegment
from mimo_transcriber.progress import NullProgressReporter
from mimo_transcriber.state_worker import RunStateProjector


class RecordingStore:
    def __init__(self) -> None:
        self.saved = 0

    def save(self, manifest: TaskManifest) -> None:
        self.saved += 1


class RecordingReporter(NullProgressReporter):
    def __init__(self) -> None:
        self.total: int | None = None
        self.sliced = 0
        self.completed: list[bool] = []
        self.retries: list[tuple[str, int, int]] = []

    def set_segment_total(self, total: int) -> None:
        self.total = total

    def segment_sliced(self) -> None:
        self.sliced += 1

    def segment_completed(self, success: bool) -> None:
        self.completed.append(success)

    def segment_retrying(self, segment_id: str, retry_number: int, max_retries: int) -> None:
        self.retries.append((segment_id, retry_number, max_retries))


def manifest_with_segment(segment: SpeakerSegment) -> TaskManifest:
    metadata = AudioMetadata(Path("input.m4a"), 1, "aac", 48000, 2, None)
    manifest = TaskManifest.new(
        identity=TaskIdentity(
            task_hash="abc",
            fingerprint_size=1,
            fingerprint_mtime_ns=0,
            fingerprint_sha256="ff",
        ),
        metadata=metadata,
    )
    manifest.segments = [SegmentRecord.from_segment(segment)]
    return manifest


def test_projector_updates_slice_and_transcript_state() -> None:
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    manifest = manifest_with_segment(segment)
    store = RecordingStore()
    reporter = RecordingReporter()
    projector = RunStateProjector(manifest, store, reporter)

    projector.handle(SegmentTotalChanged(1))
    projector.handle(SliceReady("s0000", Path("s0000.mp3"), 123))
    projector.handle(TranscriptSucceeded("s0000", "hello"))

    record = manifest.segments[0]
    assert record.slice_status == "ready"
    assert record.slice_bytes == 123
    assert record.text == "hello"
    assert record.transcript_status == "success"
    assert reporter.total == 1
    assert reporter.sliced == 1
    assert reporter.completed == [True]
    assert store.saved >= 2
    assert projector.snapshot_completed_segments()[0].text == "hello"


def test_projector_handles_retry_without_manifest_save() -> None:
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    manifest = manifest_with_segment(segment)
    store = RecordingStore()
    reporter = RecordingReporter()
    projector = RunStateProjector(manifest, store, reporter)

    projector.handle(TranscriptRetrying("s0000", 1, 3))

    assert reporter.retries == [("s0000", 1, 3)]
    assert store.saved == 0


def test_projector_marks_failures() -> None:
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    manifest = manifest_with_segment(segment)
    store = RecordingStore()
    reporter = RecordingReporter()
    projector = RunStateProjector(manifest, store, reporter)

    projector.handle(SliceFailed("s0000", "ffmpeg failed"))
    projector.handle(TranscriptFailed("s0000", "empty text"))

    completed = projector.snapshot_completed_segments()[0]
    assert completed.status is SegmentStatus.FAILED
    assert completed.text == "[该片段识别失败]"
    assert reporter.completed == [False, False]
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_pipeline_events.py tests/test_state_worker.py -q
```

Expected: FAIL with missing `mimo_transcriber.events`.

- [ ] **Step 4: Implement events**

Create `src/mimo_transcriber/events.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from mimo_transcriber.models import SpeakerSegment


@dataclass(frozen=True)
class AudioSlice:
    segment_id: str
    segment: SpeakerSegment
    path: Path


@dataclass(frozen=True)
class StageStarted:
    name: str


@dataclass(frozen=True)
class SegmentTotalChanged:
    total: int


@dataclass(frozen=True)
class SliceReady:
    segment_id: str
    path: Path
    bytes: int


@dataclass(frozen=True)
class SliceFailed:
    segment_id: str
    error: str


@dataclass(frozen=True)
class SegmentsExpanded:
    parent_id: str
    children: list[SpeakerSegment]


@dataclass(frozen=True)
class TranscriptRetrying:
    segment_id: str
    retry_number: int
    max_retries: int


@dataclass(frozen=True)
class TranscriptSucceeded:
    segment_id: str
    text: str


@dataclass(frozen=True)
class TranscriptFailed:
    segment_id: str
    error: str


PipelineEvent: TypeAlias = (
    StageStarted
    | SegmentTotalChanged
    | SliceReady
    | SliceFailed
    | SegmentsExpanded
    | TranscriptRetrying
    | TranscriptSucceeded
    | TranscriptFailed
)
```

- [ ] **Step 5: Implement RunStateProjector**

Create `src/mimo_transcriber/state_worker.py`:

```python
from __future__ import annotations

from typing import Protocol

from mimo_transcriber.events import (
    PipelineEvent,
    SegmentTotalChanged,
    SliceFailed,
    SliceReady,
    TranscriptFailed,
    TranscriptRetrying,
    TranscriptSucceeded,
)
from mimo_transcriber.manifest import SegmentRecord, TaskManifest
from mimo_transcriber.models import SegmentStatus, SpeakerSegment
from mimo_transcriber.progress import ProgressReporter

FAILED_TEXT = "[该片段识别失败]"


class ManifestStoreLike(Protocol):
    def save(self, manifest: TaskManifest) -> None:
        ...


class RunStateProjector:
    def __init__(
        self,
        manifest: TaskManifest,
        store: ManifestStoreLike,
        reporter: ProgressReporter,
    ) -> None:
        self.manifest = manifest
        self.store = store
        self.reporter = reporter
        self.segments_by_id: dict[str, SpeakerSegment] = {
            record.segment.segment_id: record.segment for record in manifest.segments
        }
        self.completed: dict[str, SpeakerSegment] = {}

    def handle(self, event: PipelineEvent) -> None:
        if isinstance(event, SegmentTotalChanged):
            self.reporter.set_segment_total(event.total)
            return
        if isinstance(event, SliceReady):
            record = self._record(event.segment_id)
            record.slice_status = "ready"
            record.slice_bytes = event.bytes
            self.reporter.segment_sliced()
            self.store.save(self.manifest)
            return
        if isinstance(event, SliceFailed):
            record = self._record(event.segment_id)
            record.slice_status = "failed"
            record.error = event.error
            self._mark_failed(record.segment, event.error)
            self.reporter.segment_sliced()
            self.reporter.segment_completed(False)
            self.store.save(self.manifest)
            return
        if isinstance(event, TranscriptRetrying):
            self.reporter.segment_retrying(
                event.segment_id,
                event.retry_number,
                event.max_retries,
            )
            return
        if isinstance(event, TranscriptSucceeded):
            record = self._record(event.segment_id)
            record.text = event.text
            record.transcript_status = "success"
            record.error = None
            segment = record.segment
            segment.text = event.text
            segment.status = SegmentStatus.SUCCESS
            segment.error = None
            self.completed[event.segment_id] = segment
            self.reporter.segment_completed(True)
            self.store.save(self.manifest)
            return
        if isinstance(event, TranscriptFailed):
            record = self._record(event.segment_id)
            record.transcript_status = "failed"
            record.error = event.error
            self._mark_failed(record.segment, event.error)
            self.reporter.segment_completed(False)
            self.store.save(self.manifest)
            return
        raise TypeError(f"Unsupported pipeline event: {type(event).__name__}")

    def snapshot_completed_segments(self) -> list[SpeakerSegment]:
        return sorted(self.completed.values(), key=lambda segment: segment.sort_key())

    def _record(self, segment_id: str) -> SegmentRecord:
        for record in self.manifest.segments:
            if record.segment.segment_id == segment_id:
                return record
        segment = self.segments_by_id.get(segment_id)
        if segment is None:
            segment = SpeakerSegment(-1, 0, 0, segment_id, segment_id=segment_id)
            self.segments_by_id[segment_id] = segment
        record = SegmentRecord.from_segment(segment)
        self.manifest.segments.append(record)
        return record

    def _mark_failed(self, segment: SpeakerSegment, error: str) -> None:
        segment.text = FAILED_TEXT
        segment.status = SegmentStatus.FAILED
        segment.error = error
        self.completed[segment.segment_id] = segment
```

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_pipeline_events.py tests/test_state_worker.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mimo_transcriber/events.py src/mimo_transcriber/state_worker.py tests/test_pipeline_events.py tests/test_state_worker.py
git commit -m "feat: add pipeline state events"
```

---

### Task 4: Move MiMo Behind AsrEngine

**Files:**
- Create: `src/mimo_transcriber/asr/mimo.py`
- Create: `src/mimo_transcriber/asr/factory.py`
- Modify: `src/mimo_transcriber/mimo_asr.py`
- Modify: `tests/test_mimo_asr.py`
- Test: `tests/test_asr_factory.py`

**Interfaces:**
- Produces: `MimoAsrEngine(config: AsrConfig, api_key: str, event_sink: AsrEventSink | None, ...)`
- Produces: `create_asr_engine(config: AsrConfig, runtime: RuntimeConfig, event_sink: AsrEventSink | None, reporter: object | None = None) -> AsrEngine`
- Keeps: `mimo_transcriber.mimo_asr.MiMoTranscriber` import as compatibility shim for existing tests until pipeline migration is done.

- [ ] **Step 1: Write failing factory tests**

Create `tests/test_asr_factory.py`:

```python
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
```

- [ ] **Step 2: Update MiMo tests to import new class**

In `tests/test_mimo_asr.py`, change:

```python
from mimo_transcriber.mimo_asr import MiMoTranscriber, extract_content
```

to:

```python
from mimo_transcriber.asr.mimo import MimoAsrEngine, extract_content, openai_request
```

Replace each `MiMoTranscriber(` with `MimoAsrEngine(`. Add `model="mimo-v2.5-asr"` when constructing the engine:

```python
    transcriber = MimoAsrEngine(
        request=request,
        model="mimo-v2.5-asr",
        language="auto",
        concurrency=1,
        requests_per_minute=1000,
        max_retries=1,
        sleep=no_sleep,
    )
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_asr_factory.py tests/test_mimo_asr.py -q
```

Expected: FAIL with missing `mimo_transcriber.asr.mimo`.

- [ ] **Step 4: Create MiMo engine module**

Copy the current contents of `src/mimo_transcriber/mimo_asr.py` into `src/mimo_transcriber/asr/mimo.py`, then rename the class and add model/cache support:

```python
class MimoAsrEngine:
    def __init__(
        self,
        request: Request,
        model: str,
        language: str,
        concurrency: int,
        requests_per_minute: int,
        max_retries: int,
        sleep: Sleep = asyncio.sleep,
        reporter: object = None,
        event_sink: AsrEventSink | None = None,
    ) -> None:
        from mimo_transcriber.progress import NullProgressReporter

        self.request = request
        self.model = model
        self.language = language
        self.semaphore = asyncio.Semaphore(concurrency)
        self.limiter = RateLimiter(requests_per_minute)
        self.max_retries = max_retries
        self.sleep = sleep
        self.reporter = reporter if reporter is not None else NullProgressReporter()
        self.event_sink = event_sink

    @property
    def cache_identity(self) -> dict[str, object]:
        return {
            "kind": "asr-engine",
            "engine": "mimo",
            "identity_version": 1,
            "settings": {
                "model": self.model,
                "language": self.language,
            },
        }
```

Inside retry handling, keep existing reporter call and also send an event when provided:

```python
                    if self.event_sink is not None:
                        from mimo_transcriber.events import TranscriptRetrying

                        await self.event_sink(
                            TranscriptRetrying(
                                segment.segment_id,
                                retry_number,
                                self.max_retries,
                            )
                        )
                    else:
                        self.reporter.segment_retrying(
                            segment.segment_id,
                            retry_number,
                            self.max_retries,
                        )
```

Do not change the request-level model selection in this task. Keep:

```python
def openai_request(api_key: str, timeout: float = 120.0) -> Request:
```

and keep `model="mimo-v2.5-asr"` in the API call. The engine `model` becomes cache identity first; Task 8 wires custom `stt_model` into request creation.

- [ ] **Step 5: Add factory**

Create `src/mimo_transcriber/asr/factory.py`:

```python
from __future__ import annotations

from mimo_transcriber.asr.base import AsrConfig, AsrEngine, AsrEventSink, RuntimeConfig
from mimo_transcriber.asr.mimo import MimoAsrEngine, openai_request
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
    if config.provider == "mimo":
        if not runtime.mimo_api_key:
            raise ConfigError("缺少 MIMO_API_KEY，请写入环境变量或 .env")
        return MimoAsrEngine(
            request=openai_request(runtime.mimo_api_key),
            model=config.resolved_model(),
            language=config.language,
            concurrency=concurrency,
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
            event_sink=event_sink,
        )
    raise ConfigError(f"未知 ASR: {config.provider}")
```

- [ ] **Step 6: Keep compatibility shim**

Replace `src/mimo_transcriber/mimo_asr.py` with:

```python
from mimo_transcriber.asr.mimo import (
    MimoAsrEngine,
    RateLimiter,
    extract_content,
    is_retryable,
    openai_request,
)

MiMoTranscriber = MimoAsrEngine

__all__ = [
    "MimoAsrEngine",
    "MiMoTranscriber",
    "RateLimiter",
    "extract_content",
    "is_retryable",
    "openai_request",
]
```

- [ ] **Step 7: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_asr_factory.py tests/test_mimo_asr.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/mimo_transcriber/asr/mimo.py src/mimo_transcriber/asr/factory.py src/mimo_transcriber/mimo_asr.py tests/test_mimo_asr.py tests/test_asr_factory.py
git commit -m "refactor: move mimo behind asr engine"
```

---

### Task 5: Pipeline Uses RuntimeConfig and AsrEngine

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `src/mimo_transcriber/cli.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `RuntimeConfig`
- Consumes: `create_asr_engine(...)`
- Keeps: `PipelineDependencies.transcribe` for tests during transition
- Produces: `run_pipeline(config: AppConfig, runtime: RuntimeConfig, ...) -> PipelineResult`

- [ ] **Step 1: Update pipeline tests to pass RuntimeConfig**

In `tests/test_pipeline.py`, add:

```python
from mimo_transcriber.asr.base import RuntimeConfig
```

Replace calls like:

```python
await run_pipeline(config, "mimo", "hf", deps, cache_root=tmp_path)
```

with:

```python
await run_pipeline(
    config,
    RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
    deps,
    cache_root=tmp_path,
)
```

For tests using positional `reporter`, use named arguments:

```python
result = await run_pipeline(
    AppConfig(input_path=source, output_path=output, num_speakers=2),
    RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
    dependencies,
)
```

- [ ] **Step 2: Run pipeline tests to verify failure**

Run:

```bash
uv run pytest tests/test_pipeline.py tests/test_cli.py -q
```

Expected: FAIL because `run_pipeline` still expects `mimo_key, hf_token`.

- [ ] **Step 3: Change run_pipeline signature and diarization token usage**

Modify `src/mimo_transcriber/pipeline.py` imports:

```python
from mimo_transcriber.asr.base import AsrConfig, RuntimeConfig
from mimo_transcriber.asr.factory import create_asr_engine
```

Change signature:

```python
async def run_pipeline(
    config: AppConfig,
    runtime: RuntimeConfig,
    dependencies: PipelineDependencies = PipelineDependencies(),
    cache_root: Path | None = None,
    reporter: ProgressReporter | None = None,
) -> PipelineResult:
```

Change diarization call:

```python
                    runtime.hf_token,
```

Change `_run_segment_workers(...)` call:

```python
                runtime=runtime,
```

Change `_run_segment_workers` signature:

```python
    runtime: RuntimeConfig,
```

- [ ] **Step 4: Create engine inside worker transition path**

In `_run_segment_workers`, replace MiMo client construction with:

```python
    if dependencies.transcribe is None:
        client = create_asr_engine(
            AsrConfig(
                provider=config.asr,
                stt_model=config.stt_model,
                language=config.language,
            ),
            runtime,
            event_sink=None,
            concurrency=config.concurrency,
            requests_per_minute=config.requests_per_minute,
            max_retries=config.max_retries,
        )
    else:
        client = None
```

Keep the existing `dependencies.transcribe` path unchanged.

- [ ] **Step 5: Update CLI runtime usage**

In `src/mimo_transcriber/cli.py`, ensure the runtime code is:

```python
        runtime = validate_runtime(config)
        result = await run_pipeline(config, runtime, reporter=reporter)
```

- [ ] **Step 6: Run tests to verify pass**

Run:

```bash
uv run pytest tests/test_pipeline.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mimo_transcriber/pipeline.py src/mimo_transcriber/cli.py tests/test_pipeline.py tests/test_cli.py
git commit -m "refactor: pass asr runtime to pipeline"
```

---

### Task 6: StateWorker Projection in Segment Workers

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `src/mimo_transcriber/state_worker.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_state_worker.py`

**Interfaces:**
- Consumes: `RunStateProjector.handle(event)`
- Produces: segment workers that send `SliceReady`, `SliceFailed`, `TranscriptSucceeded`, `TranscriptFailed`, and `TranscriptRetrying` events
- Keeps: final `PipelineResult` behavior and resumable manifest semantics

- [ ] **Step 1: Add a pipeline regression test for worker isolation**

Append to `tests/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_pipeline_records_success_through_manifest_projection(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48_000, 2, None)

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "projected"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    deps = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 1, "A"),
        ]),
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, segment: True,
        transcribe=transcribe,
    )

    result = await run_pipeline(
        AppConfig(input_path=source, output_path=output, num_speakers=1),
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
        deps,
        cache_root=tmp_path,
    )

    assert result.exit_code == 0
    assert result.outcome.segments[0].text == "projected"
    assert output.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify current behavior still passes**

Run:

```bash
uv run pytest tests/test_pipeline.py::test_pipeline_records_success_through_manifest_projection -q
```

Expected: PASS before refactor. This protects behavior during the internal rewrite.

- [ ] **Step 3: Add async state worker helper**

Extend `src/mimo_transcriber/state_worker.py`:

```python
import asyncio


async def run_state_worker(
    queue: asyncio.Queue[PipelineEvent | None],
    projector: RunStateProjector,
) -> None:
    while True:
        event = await queue.get()
        try:
            if event is None:
                return
            projector.handle(event)
        finally:
            queue.task_done()
```

- [ ] **Step 4: Refactor `_run_segment_workers` to create state queue/projector**

In `src/mimo_transcriber/pipeline.py`, import:

```python
from mimo_transcriber.events import (
    AudioSlice,
    SegmentTotalChanged,
    SliceFailed,
    SliceReady,
    TranscriptFailed,
    TranscriptSucceeded,
)
from mimo_transcriber.state_worker import RunStateProjector, run_state_worker
```

Inside `_run_segment_workers`, after queues are created:

```python
    state_queue: asyncio.Queue[object | None] = asyncio.Queue()
    projector = RunStateProjector(manifest, store, reporter)
    state_task = asyncio.create_task(run_state_worker(state_queue, projector))
```

When setting total:

```python
    await state_queue.put(SegmentTotalChanged(len(segments)))
```

- [ ] **Step 5: Make slice worker emit events**

Replace direct manifest/reporter updates in slice success with:

```python
                    await state_queue.put(SliceReady(seg.segment_id, target, size))
                    await transcribe_queue.put((record, target))
```

Replace slice failure block with:

```python
                await state_queue.put(
                    SliceFailed(record.segment.segment_id, str(exc))
                )
                completed[record.segment.segment_id] = record.segment
```

Keep existing `completed[...]` assignments only until final snapshot is wired. Remove direct `store.save(manifest)`, `reporter.segment_sliced()`, and `reporter.segment_completed(False)` from slice worker.

- [ ] **Step 6: Make ASR worker emit transcript events**

After successful result:

```python
                        await state_queue.put(
                            TranscriptSucceeded(record.segment.segment_id, record.text or "")
                        )
```

After final exception:

```python
                    await state_queue.put(
                        TranscriptFailed(
                            record.segment.segment_id,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
```

Remove direct `store.save(manifest)` and `reporter.segment_completed(...)` calls in the worker path. Keep segment mutation from fake dependency path if needed, but make event projection authoritative for final output.

- [ ] **Step 7: Use projector snapshot at the end**

Before returning from `_run_segment_workers`, close state queue:

```python
        await state_queue.join()
        await state_queue.put(None)
        await state_task
```

Return:

```python
    return sorted(projector.snapshot_completed_segments(), key=lambda item: item.sort_key())
```

In cancellation handling, cancel `state_task`:

```python
        state_task.cancel()
        await asyncio.gather(state_task, return_exceptions=True)
```

- [ ] **Step 8: Run pipeline tests**

Run:

```bash
uv run pytest tests/test_pipeline.py tests/test_state_worker.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/mimo_transcriber/pipeline.py src/mimo_transcriber/state_worker.py tests/test_pipeline.py tests/test_state_worker.py
git commit -m "refactor: project worker state through events"
```

---

### Task 7: MLX ASR Engine

**Files:**
- Create: `src/mimo_transcriber/asr/mlx.py`
- Modify: `src/mimo_transcriber/asr/factory.py`
- Modify: `pyproject.toml`
- Test: `tests/test_mlx_asr.py`
- Test: `tests/test_asr_factory.py`

**Interfaces:**
- Produces: `MlxAsrEngine(config: AsrConfig, transcribe: Callable[..., dict[str, object]] | None = None)`
- Produces: MLX factory branch for `AsrConfig(provider="mlx")`

- [ ] **Step 1: Write failing MLX engine tests**

Create `tests/test_mlx_asr.py`:

```python
from pathlib import Path

import pytest

from mimo_transcriber.asr.base import AsrConfig
from mimo_transcriber.asr.mlx import MlxAsrEngine
from mimo_transcriber.models import SegmentStatus, SpeakerSegment


@pytest.mark.asyncio
async def test_mlx_engine_transcribes_text(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        calls.append({"path": path, **kwargs})
        return {"text": " hello  world "}

    audio = tmp_path / "s0000.mp3"
    audio.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    engine = MlxAsrEngine(AsrConfig(provider="mlx", language="zh"), fake_transcribe)

    result = await engine.transcribe_one(segment, audio)

    assert result.text == "hello world"
    assert result.status is SegmentStatus.SUCCESS
    assert calls == [{
        "path": str(audio),
        "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
        "language": "zh",
    }]


@pytest.mark.asyncio
async def test_mlx_engine_omits_language_for_auto(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"text": "ok"}

    audio = tmp_path / "s0000.mp3"
    audio.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    engine = MlxAsrEngine(AsrConfig(provider="mlx"), fake_transcribe)

    await engine.transcribe_one(segment, audio)

    assert "language" not in calls[0]


@pytest.mark.asyncio
async def test_mlx_engine_marks_empty_text_failed(tmp_path: Path) -> None:
    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        return {"text": " "}

    audio = tmp_path / "s0000.mp3"
    audio.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    engine = MlxAsrEngine(AsrConfig(provider="mlx"), fake_transcribe)

    result = await engine.transcribe_one(segment, audio)

    assert result.status is SegmentStatus.FAILED
    assert result.text == "[该片段识别失败]"


@pytest.mark.asyncio
async def test_mlx_engine_returns_results_in_sort_order(tmp_path: Path) -> None:
    def fake_transcribe(path: str, **kwargs: object) -> dict[str, object]:
        return {"text": Path(path).stem}

    paths = []
    for name in ("s0001.mp3", "s0000.mp3"):
        path = tmp_path / name
        path.write_bytes(b"audio")
        paths.append(path)
    segments = [
        SpeakerSegment(1, 1, 2, "B", segment_id="s0001"),
        SpeakerSegment(0, 0, 1, "A", segment_id="s0000"),
    ]
    engine = MlxAsrEngine(AsrConfig(provider="mlx"), fake_transcribe)

    result = await engine.transcribe_all(list(zip(segments, paths)), False)

    assert [item.segment_id for item in result] == ["s0000", "s0001"]
```

- [ ] **Step 2: Extend factory tests**

Append to `tests/test_asr_factory.py`:

```python
from mimo_transcriber.asr.mlx import MlxAsrEngine


def test_factory_creates_mlx_engine() -> None:
    engine = create_asr_engine(
        AsrConfig(provider="mlx"),
        RuntimeConfig(hf_token="hf"),
        event_sink=None,
    )

    assert isinstance(engine, MlxAsrEngine)
    assert engine.cache_identity["engine"] == "mlx-whisper"
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_mlx_asr.py tests/test_asr_factory.py -q
```

Expected: FAIL with missing `mimo_transcriber.asr.mlx`.

- [ ] **Step 4: Implement MLX engine**

Create `src/mimo_transcriber/asr/mlx.py`:

```python
from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mimo_transcriber.asr.base import AsrConfig
from mimo_transcriber.models import SegmentStatus, SpeakerSegment

FAILED_TEXT = "[该片段识别失败]"
MlxTranscribe = Callable[..., dict[str, Any]]


class MlxAsrEngine:
    def __init__(
        self,
        config: AsrConfig,
        transcribe: MlxTranscribe | None = None,
    ) -> None:
        self.config = config
        self._transcribe = transcribe
        self._lock = asyncio.Lock()

    @property
    def cache_identity(self) -> dict[str, object]:
        return self.config.cache_identity()

    async def transcribe_one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        try:
            async with self._lock:
                result = await asyncio.to_thread(self._call_transcribe, path)
            text = " ".join(str(result.get("text", "")).split())
            if not text:
                raise ValueError("MLX Whisper 返回了空文本")
            segment.text = text
            segment.status = SegmentStatus.SUCCESS
            segment.error = None
            return segment
        except Exception as exc:
            segment.text = FAILED_TEXT
            segment.status = SegmentStatus.FAILED
            segment.error = f"{type(exc).__name__}: {exc}"
            return segment

    async def transcribe_all(
        self,
        items: list[tuple[SpeakerSegment, Path]],
        fail_fast: bool,
    ) -> list[SpeakerSegment]:
        results: list[SpeakerSegment] = []
        for segment, path in items:
            result = await self.transcribe_one(segment, path)
            if fail_fast and result.status is SegmentStatus.FAILED:
                raise RuntimeError(result.error or "片段识别失败")
            results.append(result)
        return sorted(results, key=lambda item: item.sort_key())

    def _call_transcribe(self, path: Path) -> dict[str, Any]:
        transcribe = self._transcribe
        if transcribe is None:
            import mlx_whisper

            transcribe = mlx_whisper.transcribe
            self._transcribe = transcribe
        kwargs: dict[str, object] = {
            "path_or_hf_repo": self.config.resolved_model(),
        }
        if self.config.language != "auto":
            kwargs["language"] = self.config.language
        return transcribe(str(path), **kwargs)
```

- [ ] **Step 5: Add factory branch**

Modify `src/mimo_transcriber/asr/factory.py`:

```python
from mimo_transcriber.asr.mlx import MlxAsrEngine
```

Add branch before final error:

```python
    if config.provider == "mlx":
        return MlxAsrEngine(config)
```

- [ ] **Step 6: Add dependency**

Modify `pyproject.toml` dependencies:

```toml
  "mlx-whisper>=0.4.2",
```

If `uv sync` later selects a newer compatible version, keep the lockfile generated by the install step.

- [ ] **Step 7: Run tests**

Run:

```bash
uv run pytest tests/test_mlx_asr.py tests/test_asr_factory.py -q
```

Expected: PASS.

- [ ] **Step 8: Sync lockfile if dependency was added**

Run:

```bash
uv lock
```

Expected: `uv.lock` updates successfully. If network is unavailable, record that lockfile update must be completed with network access before merge.

- [ ] **Step 9: Commit**

```bash
git add src/mimo_transcriber/asr/mlx.py src/mimo_transcriber/asr/factory.py pyproject.toml uv.lock tests/test_mlx_asr.py tests/test_asr_factory.py
git commit -m "feat: add mlx asr engine"
```

---

### Task 8: Full CLI Runtime, Docs, and Verification

**Files:**
- Modify: `README.md`
- Modify: `src/mimo_transcriber/asr/mimo.py`
- Modify: `src/mimo_transcriber/asr/factory.py`
- Test: `tests/test_mimo_asr.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: documented default local mode
- Produces: documented optional MiMo mode
- Produces: `--stt-model` honored for MiMo API request model

- [ ] **Step 1: Add MiMo custom model request test**

Append to `tests/test_mimo_asr.py`:

```python
@pytest.mark.asyncio
async def test_openai_request_uses_configured_model(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: Client())

    request = openai_request("key", model="custom-model")
    await request("data:audio/mp3;base64,abc", "en")

    assert captured["model"] == "custom-model"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_mimo_asr.py::test_openai_request_uses_configured_model -q
```

Expected: FAIL because `openai_request` does not accept `model`.

- [ ] **Step 3: Wire custom MiMo model**

Modify `src/mimo_transcriber/asr/mimo.py`:

```python
def openai_request(api_key: str, model: str = "mimo-v2.5-asr", timeout: float = 120.0) -> Request:
```

Inside request:

```python
            model=model,
```

Modify `src/mimo_transcriber/asr/factory.py`:

```python
            request=openai_request(runtime.mimo_api_key, model=config.resolved_model()),
```

- [ ] **Step 4: Update README introduction and quick start**

Modify README top paragraphs to:

```markdown
# CScribe

本地运行的多人录音转写 CLI。CScribe 使用 pyannote 区分说话人，默认通过本地 MLX Whisper 转写切分后的音频片段，并生成带时间戳、说话人和关键词的 TXT。

默认模式不会上传音频片段。需要使用 MiMo ASR 时，可以通过 `--asr mimo` 显式选择远端后端。
```

Modify `.env` example section:

```markdown
默认本地转写只需要：

```dotenv
HF_TOKEN=你的_Hugging_Face_Token
```

使用 MiMo 时额外填写：

```dotenv
MIMO_API_KEY=你的_MiMo_API_Key
```
```

- [ ] **Step 5: Update README CLI table**

Add rows:

```markdown
| `--asr {mlx,mimo}` | `mlx` | ASR 引擎；默认本地 MLX Whisper，`mimo` 为远端 MiMo |
| `--stt-model MODEL` | 引擎默认值 | STT 模型；由所选 ASR 引擎解释 |
```

Change existing MiMo rows:

```markdown
| `--concurrency N` | `2` | ASR worker 数量；MiMo 可并发请求，MLX 首版内部串行推理 |
| `--requests-per-minute N` | `20` | MiMo 每分钟请求上限；本地 MLX 忽略 |
```

- [ ] **Step 6: Replace README ASR model section**

Replace “怎么换 ASR 模型” section with:

```markdown
## 怎么选择 ASR 引擎和 STT 模型

默认使用本地 MLX Whisper：

```bash
uv run mimo-transcriber meeting.m4a
```

指定本地模型：

```bash
uv run mimo-transcriber meeting.m4a --asr mlx --stt-model mlx-community/whisper-small
```

使用 MiMo：

```bash
uv run mimo-transcriber meeting.m4a --asr mimo --stt-model mimo-v2.5-asr
```

`--stt-model` 对上层流水线透明，由所选 ASR 引擎解释。切换 ASR 引擎或模型会改变缓存身份，避免复用旧模型的转写结果。
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
uv run pytest tests/test_mimo_asr.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 8: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 9: Manual smoke checks**

Run help:

```bash
uv run mimo-transcriber --help
```

Expected: help includes `--asr {mlx,mimo}` and `--stt-model MODEL`.

Run local mode only if a small local audio sample is available:

```bash
uv run mimo-transcriber /Users/yuancheng/Downloads/新录音\ 15.m4a --num-speakers 2 --asr mlx --debug
```

Expected: command starts without requiring `MIMO_API_KEY`; first MLX run may download the selected model; output TXT is produced or clear MLX dependency/model error is shown.

- [ ] **Step 10: Commit**

```bash
git add README.md src/mimo_transcriber/asr/mimo.py src/mimo_transcriber/asr/factory.py tests/test_mimo_asr.py tests/test_cli.py
git commit -m "docs: document local asr default"
```

---

## Final Verification

- [ ] Run full tests:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] Check no direct worker state leaks remain:

```bash
rg -n 'store\.save|reporter\.' src/mimo_transcriber/pipeline.py
```

Expected: no matches inside `_slice_worker` or `_transcribe_worker`; matches outside workers are acceptable only for stage start/final finish during transition.

- [ ] Check CLI help:

```bash
uv run mimo-transcriber --help
```

Expected: output includes `--asr {mlx,mimo}` and `--stt-model MODEL`.

- [ ] Check git status:

```bash
git status --short
```

Expected: clean after final commit.
