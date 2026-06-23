# Experimental MPS Diarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit experimental `--device mps` path that diagnoses Apple GPU support, preflights the real pyannote pipeline on 10 seconds of audio, reuses a successful pipeline, and safely falls back to CPU without repeating normalization.

**Architecture:** Add a focused `devices.py` module for immutable capability and decision data. Keep audio sample creation in `audio.py`, and refactor `diarization.py` into three separable operations: construct/select a pipeline, apply a supplied pipeline, and recover a failed full MPS run on CPU. `pipeline.py` remains the workflow owner and receives a `DiarizationResult`, while existing CPU/CUDA behavior and all MiMo stages remain unchanged.

**Tech Stack:** Python 3.11, PyTorch 2.2+, pyannote.audio 3.3.2+, FFmpeg, pytest, pytest-asyncio.

## Global Constraints

- MPS is opt-in only through `--device mps`; `--device auto` must continue to choose CUDA on supported Linux systems and CPU elsewhere.
- MPS environment failures and pyannote compatibility failures automatically fall back to CPU.
- The preflight uses the real `pyannote/speaker-diarization-community-1` pipeline against the first 10 seconds of the normalized WAV.
- A successful preflight pipeline is reused for complete diarization.
- A failed complete MPS run retries only diarization on CPU; probing, normalization, slicing, and MiMo transcription are not repeated.
- CPU and CUDA behavior must remain backward compatible.
- The application diagnoses environment problems but never installs or replaces PyTorch or changes macOS.
- Normal logs and decision objects must never expose Hugging Face or MiMo credentials.
- Automated tests must not require a physical GPU, network access, Hugging Face access, or model downloads.
- MPS is considered recommended on a tested configuration only when median diarization time is at least 20 percent lower than CPU and repeated runs are stable.

---

## File Map

- Create `src/mimo_transcriber/devices.py`: immutable accelerator facts, selected-device decision, fallback categories, and capability collection.
- Modify `src/mimo_transcriber/config.py`: admit `mps` as a requested device while retaining ordinary CPU/CUDA resolution semantics.
- Modify `src/mimo_transcriber/audio.py`: create the bounded 10-second WAV used by real-pipeline preflight.
- Modify `src/mimo_transcriber/diarization.py`: construct pipelines, classify MPS failures, preflight MPS, apply supplied pipelines, clean up MPS, and recover complete-run failures on CPU.
- Modify `src/mimo_transcriber/pipeline.py`: create one preflight sample and consume `DiarizationResult` without repeating normalization.
- Modify `src/mimo_transcriber/cli.py`: expose `mps` in CLI choices.
- Modify `README.md`: document experimental semantics, fallback, diagnostics, and manual performance acceptance.
- Create `tests/test_devices.py`: capability and decision behavior.
- Modify `tests/test_audio.py`: preflight sample duration and format.
- Modify `tests/test_config.py`: ordinary device compatibility.
- Modify `tests/test_cli.py`: MPS parser acceptance.
- Rewrite `tests/test_diarization.py`: pipeline selection, preflight, classification, cleanup, reuse, and full-run recovery.
- Modify `tests/test_pipeline.py`: prove normalization occurs once and downstream stages wait for successful diarization.

---

### Task 1: Device Facts and Explicit MPS Request

**Files:**
- Create: `src/mimo_transcriber/devices.py`
- Modify: `src/mimo_transcriber/config.py`
- Modify: `src/mimo_transcriber/cli.py`
- Create: `tests/test_devices.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `SelectedDevice = Literal["cpu", "cuda", "mps"]`
- Produces: `FallbackCategory = Literal["not_built", "runtime_unavailable", "unsupported_operator", "out_of_memory", "preflight_failed", "full_run_failed"]`
- Produces: `DeviceCapabilities(cuda_available: bool, mps_built: bool, mps_available: bool, platform: str, machine: str)`
- Produces: `DeviceDecision(requested_device: Device, selected_device: SelectedDevice, mps_built: bool | None = None, mps_available: bool | None = None, preflight_elapsed_seconds: float | None = None, fallback_category: FallbackCategory | None = None, fallback_reason: str | None = None)`
- Produces: `collect_device_capabilities() -> DeviceCapabilities`
- Preserves: `resolve_device(requested: Device, cuda_available: Callable[[], bool] | None = None) -> Literal["cpu", "cuda"]` for `auto`, `cpu`, and `cuda`; explicit `mps` is handled by diarization selection.

- [ ] **Step 1: Write failing device model and CLI tests**

Create `tests/test_devices.py`:

```python
from mimo_transcriber.devices import (
    DeviceCapabilities,
    DeviceDecision,
)


def test_device_decision_records_sanitized_fallback_facts() -> None:
    capabilities = DeviceCapabilities(
        cuda_available=False,
        mps_built=True,
        mps_available=False,
        platform="Darwin",
        machine="arm64",
    )
    decision = DeviceDecision(
        requested_device="mps",
        selected_device="cpu",
        mps_built=capabilities.mps_built,
        mps_available=capabilities.mps_available,
        fallback_category="runtime_unavailable",
        fallback_reason="当前 PyTorch 运行时无法使用 MPS",
    )
    assert decision.requested_device == "mps"
    assert decision.selected_device == "cpu"
    assert decision.fallback_category == "runtime_unavailable"
```

Append to `tests/test_cli.py`:

```python
def test_parser_accepts_experimental_mps() -> None:
    args = build_parser().parse_args(["meeting.m4a", "--device", "mps"])
    assert args.device == "mps"
```

Replace the existing macOS test in `tests/test_config.py` and add explicit compatibility cases:

```python
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
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
uv run pytest tests/test_devices.py tests/test_config.py tests/test_cli.py -v
```

Expected: collection fails because `mimo_transcriber.devices` does not exist, and CLI parsing rejects `mps`.

- [ ] **Step 3: Add immutable device facts and decisions**

Create `src/mimo_transcriber/devices.py`:

```python
from __future__ import annotations

import platform as platform_module
from dataclasses import dataclass
from typing import Literal

from mimo_transcriber.config import Device

SelectedDevice = Literal["cpu", "cuda", "mps"]
FallbackCategory = Literal[
    "not_built",
    "runtime_unavailable",
    "unsupported_operator",
    "out_of_memory",
    "preflight_failed",
    "full_run_failed",
]


@dataclass(frozen=True)
class DeviceCapabilities:
    cuda_available: bool
    mps_built: bool
    mps_available: bool
    platform: str
    machine: str


@dataclass(frozen=True)
class DeviceDecision:
    requested_device: Device
    selected_device: SelectedDevice
    mps_built: bool | None = None
    mps_available: bool | None = None
    preflight_elapsed_seconds: float | None = None
    fallback_category: FallbackCategory | None = None
    fallback_reason: str | None = None


def collect_device_capabilities() -> DeviceCapabilities:
    import torch

    return DeviceCapabilities(
        cuda_available=torch.cuda.is_available(),
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        platform=platform_module.system(),
        machine=platform_module.machine(),
    )
```

In `src/mimo_transcriber/config.py`, extend the request type and reject direct standard resolution of MPS:

```python
Device = Literal["auto", "cpu", "cuda", "mps"]
```

Add this branch at the beginning of `resolve_device`, before checking CUDA:

```python
    if requested == "mps":
        raise ConfigError("MPS 必须通过实验性 diarization 预检选择")
```

In `src/mimo_transcriber/cli.py`, change the device argument:

```python
parser.add_argument(
    "--device",
    choices=("auto", "cpu", "cuda", "mps"),
    default="auto",
)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
uv run pytest tests/test_devices.py tests/test_config.py tests/test_cli.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the complete suite for compatibility**

Run:

```bash
uv run pytest -q
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit the device boundary**

```bash
git add src/mimo_transcriber/devices.py src/mimo_transcriber/config.py src/mimo_transcriber/cli.py tests/test_devices.py tests/test_config.py tests/test_cli.py
git commit -m "feat: model experimental MPS device selection"
```

---

### Task 2: Ten-Second Preflight Audio Sample

**Files:**
- Modify: `src/mimo_transcriber/audio.py`
- Modify: `tests/test_audio.py`

**Interfaces:**
- Produces: `create_preflight_sample(source: Path, target: Path, duration_seconds: float = 10.0) -> None`
- Consumes: a normalized mono 16 kHz WAV.
- Produces: a mono 16 kHz PCM WAV containing at most the first 10 seconds.

- [ ] **Step 1: Add a failing FFmpeg integration test**

Update the import in `tests/test_audio.py`:

```python
from mimo_transcriber.audio import (
    create_preflight_sample,
    encoded_audio_data,
    normalize_audio,
    probe_audio,
)
```

Append:

```python
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
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
uv run pytest tests/test_audio.py::test_create_preflight_sample_caps_normalized_audio_at_ten_seconds -v
```

Expected: collection fails because `create_preflight_sample` is missing.

- [ ] **Step 3: Implement bounded WAV creation**

Add to `src/mimo_transcriber/audio.py` after `normalize_audio`:

```python
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
```

This command naturally produces the entire file when the source is shorter than 10 seconds.

- [ ] **Step 4: Run audio tests**

Run:

```bash
uv run pytest tests/test_audio.py -v
uv run pytest -q
```

Expected: all audio tests and the complete suite pass.

- [ ] **Step 5: Commit the sample helper**

```bash
git add src/mimo_transcriber/audio.py tests/test_audio.py
git commit -m "feat: create diarization preflight sample"
```

---

### Task 3: Refactor Diarization Around a Supplied Pipeline

**Files:**
- Modify: `src/mimo_transcriber/diarization.py`
- Modify: `tests/test_diarization.py`

**Interfaces:**
- Produces: `speaker_kwargs(num_speakers: int | None, min_speakers: int, max_speakers: int) -> dict[str, int]`
- Produces: `create_pipeline(token: str, device: SelectedDevice) -> Any`
- Produces: `apply_diarization_pipeline(path: Path, pipeline: Any, num_speakers: int | None, min_speakers: int, max_speakers: int) -> list[SpeakerSegment]`
- Temporarily preserves the existing `diarize_audio(path, token, device, ...)` wrapper so this commit keeps the complete suite green. Task 6 removes the bridge after the workflow adopts `run_diarization`.

- [ ] **Step 1: Rewrite the adapter test for an injected pipeline**

Replace `tests/test_diarization.py` with:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_transcriber.diarization import (
    DiarizationError,
    apply_diarization_pipeline,
)


class Annotation:
    def itertracks(self, yield_label: bool):
        assert yield_label is True
        yield SimpleNamespace(start=0.2, end=1.8), None, "SPEAKER_07"


class Pipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, int]]] = []

    def __call__(self, path: str, **kwargs):
        self.calls.append((path, kwargs))
        return SimpleNamespace(speaker_diarization=Annotation())


def test_adapter_uses_supplied_pipeline_and_exact_speaker_count() -> None:
    pipeline = Pipeline()
    result = apply_diarization_pipeline(
        Path("normalized.wav"),
        pipeline,
        num_speakers=2,
        min_speakers=1,
        max_speakers=6,
    )
    assert [(item.start, item.end, item.raw_speaker) for item in result] == [
        (0.2, 1.8, "SPEAKER_07")
    ]
    assert pipeline.calls == [("normalized.wav", {"num_speakers": 2})]


def test_adapter_uses_minimum_and_maximum_when_count_is_unknown() -> None:
    pipeline = Pipeline()
    apply_diarization_pipeline(
        Path("normalized.wav"),
        pipeline,
        num_speakers=None,
        min_speakers=2,
        max_speakers=4,
    )
    assert pipeline.calls == [
        ("normalized.wav", {"min_speakers": 2, "max_speakers": 4})
    ]


def test_adapter_wraps_pipeline_failure() -> None:
    def fail(path: str, **kwargs):
        raise RuntimeError("backend failed")

    with pytest.raises(DiarizationError, match="说话人分离失败"):
        apply_diarization_pipeline(Path("normalized.wav"), fail, 2, 1, 6)
```

- [ ] **Step 2: Run the adapter tests and verify failure**

Run:

```bash
uv run pytest tests/test_diarization.py -v
```

Expected: tests fail because `apply_diarization_pipeline` does not exist.

- [ ] **Step 3: Separate construction from inference**

Replace `src/mimo_transcriber/diarization.py` with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from mimo_transcriber.devices import SelectedDevice
from mimo_transcriber.models import SpeakerSegment

MODEL_ID = "pyannote/speaker-diarization-community-1"


class DiarizationError(RuntimeError):
    pass


def speaker_kwargs(
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
) -> dict[str, int]:
    if num_speakers is not None:
        return {"num_speakers": num_speakers}
    return {"min_speakers": min_speakers, "max_speakers": max_speakers}


def create_pipeline(token: str, device: SelectedDevice) -> Any:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline.to(torch.device(device))
    return pipeline


def apply_diarization_pipeline(
    path: Path,
    pipeline: Any,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
) -> list[SpeakerSegment]:
    try:
        output = pipeline(
            str(path),
            **speaker_kwargs(num_speakers, min_speakers, max_speakers),
        )
        annotation = getattr(output, "speaker_diarization", output)
        return [
            SpeakerSegment(-1, float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    except Exception as exc:
        raise DiarizationError(f"说话人分离失败: {exc}") from exc


def diarize_audio(
    path: Path,
    token: str,
    device: SelectedDevice,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
) -> list[SpeakerSegment]:
    pipeline = create_pipeline(token, device)
    return apply_diarization_pipeline(
        path,
        pipeline,
        num_speakers,
        min_speakers,
        max_speakers,
    )
```

- [ ] **Step 4: Run the adapter tests**

Run:

```bash
uv run pytest tests/test_diarization.py -v
uv run pytest -q
```

Expected: all three focused tests and the complete suite pass.

- [ ] **Step 5: Commit the refactor**

```bash
git add src/mimo_transcriber/diarization.py tests/test_diarization.py
git commit -m "refactor: separate diarization pipeline construction"
```

---

### Task 4: MPS Preflight, Failure Classification, and Pipeline Reuse

**Files:**
- Modify: `src/mimo_transcriber/diarization.py`
- Modify: `tests/test_diarization.py`

**Interfaces:**
- Produces: `PipelineSelection(pipeline: Any, decision: DeviceDecision)`
- Produces: `classify_mps_failure(exc: BaseException, phase: Literal["preflight", "full"]) -> FallbackCategory`
- Produces: `clear_mps_cache() -> None`
- Produces: `select_diarization_pipeline(preflight_path: Path, token: str, requested_device: Device, num_speakers: int | None, min_speakers: int, max_speakers: int, *, capabilities: DeviceCapabilities | None = None, pipeline_factory: Callable[[str, SelectedDevice], Any] = create_pipeline, cache_clearer: Callable[[], None] = clear_mps_cache, clock: Callable[[], float] = time.monotonic) -> PipelineSelection`

- [ ] **Step 1: Add fake-pipeline selection tests**

Append to `tests/test_diarization.py`:

```python
from mimo_transcriber.devices import DeviceCapabilities
from mimo_transcriber.diarization import (
    classify_mps_failure,
    select_diarization_pipeline,
)


def capabilities(*, built: bool, available: bool) -> DeviceCapabilities:
    return DeviceCapabilities(
        cuda_available=False,
        mps_built=built,
        mps_available=available,
        platform="Darwin",
        machine="arm64",
    )


def test_unbuilt_mps_falls_back_without_constructing_mps() -> None:
    calls: list[str] = []
    cpu = Pipeline()

    def factory(token: str, device: str):
        calls.append(device)
        assert token == "secret"
        return cpu

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=False, available=False),
        pipeline_factory=factory,
    )

    assert calls == ["cpu"]
    assert selection.pipeline is cpu
    assert selection.decision.selected_device == "cpu"
    assert selection.decision.fallback_category == "not_built"


def test_unavailable_mps_runtime_falls_back_to_cpu() -> None:
    calls: list[str] = []

    def factory(token: str, device: str):
        calls.append(device)
        return Pipeline()

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=False),
        pipeline_factory=factory,
    )

    assert calls == ["cpu"]
    assert selection.decision.fallback_category == "runtime_unavailable"


def test_successful_preflight_returns_same_mps_pipeline() -> None:
    mps = Pipeline()
    times = iter([10.0, 12.5])

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=lambda token, device: mps,
        clock=lambda: next(times),
    )

    assert selection.pipeline is mps
    assert mps.calls == [("preflight.wav", {"num_speakers": 2})]
    assert selection.decision.selected_device == "mps"
    assert selection.decision.preflight_elapsed_seconds == 2.5


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("not implemented for the MPS device"), "unsupported_operator"),
        (RuntimeError("MPS backend out of memory"), "out_of_memory"),
        (RuntimeError("unexpected"), "preflight_failed"),
    ],
)
def test_preflight_failure_is_classified(error: RuntimeError, expected: str) -> None:
    assert classify_mps_failure(error, "preflight") == expected


def test_failed_mps_preflight_clears_cache_and_constructs_cpu() -> None:
    calls: list[str] = []
    cleared: list[bool] = []

    def factory(token: str, device: str):
        calls.append(device)
        if device == "mps":
            def fail(path: str, **kwargs):
                raise RuntimeError("not implemented for the MPS device")
            return fail
        return Pipeline()

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=factory,
        cache_clearer=lambda: cleared.append(True),
    )

    assert calls == ["mps", "cpu"]
    assert cleared == [True]
    assert selection.decision.selected_device == "cpu"
    assert selection.decision.fallback_category == "unsupported_operator"


def test_cache_cleanup_failure_does_not_block_cpu_fallback() -> None:
    def factory(token: str, device: str):
        if device == "mps":
            def fail(path: str, **kwargs):
                raise RuntimeError("unexpected")
            return fail
        return Pipeline()

    def cleanup_failure() -> None:
        raise RuntimeError("cleanup failed")

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=factory,
        cache_clearer=cleanup_failure,
    )

    assert selection.decision.selected_device == "cpu"
```

- [ ] **Step 2: Run selection tests and verify failure**

Run:

```bash
uv run pytest tests/test_diarization.py -v
```

Expected: import failures for `classify_mps_failure` and `select_diarization_pipeline`.

- [ ] **Step 3: Implement selection, classification, cleanup, and sanitized reasons**

Add these imports to `src/mimo_transcriber/diarization.py`:

```python
import logging
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal

from mimo_transcriber.config import Device, resolve_device
from mimo_transcriber.devices import (
    DeviceCapabilities,
    DeviceDecision,
    FallbackCategory,
    SelectedDevice,
    collect_device_capabilities,
)
```

Add:

```python
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSelection:
    pipeline: Any
    decision: DeviceDecision


def classify_mps_failure(
    exc: BaseException,
    phase: Literal["preflight", "full"],
) -> FallbackCategory:
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current).lower())
        current = current.__cause__
    message = " ".join(messages)
    if "out of memory" in message or "allocation" in message:
        return "out_of_memory"
    if "not implemented for" in message and "mps" in message:
        return "unsupported_operator"
    return "full_run_failed" if phase == "full" else "preflight_failed"


def fallback_reason(category: FallbackCategory) -> str:
    return {
        "not_built": "当前 PyTorch 未构建 MPS 支持",
        "runtime_unavailable": "当前 PyTorch 运行时无法使用 MPS",
        "unsupported_operator": "pyannote 需要的算子尚不支持 MPS",
        "out_of_memory": "MPS 可用内存不足",
        "preflight_failed": "MPS 预检未能完成",
        "full_run_failed": "完整 MPS 说话人分离未能完成",
    }[category]


def clear_mps_cache() -> None:
    try:
        import torch

        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception as exc:
        logger.debug("清理 MPS 缓存失败: %s", type(exc).__name__)


def _cpu_selection(
    requested_device: Device,
    token: str,
    capabilities: DeviceCapabilities,
    pipeline_factory: Callable[[str, SelectedDevice], Any],
    category: FallbackCategory | None = None,
) -> PipelineSelection:
    return PipelineSelection(
        pipeline=_build_pipeline_safely(token, "cpu", pipeline_factory),
        decision=DeviceDecision(
            requested_device=requested_device,
            selected_device="cpu",
            mps_built=capabilities.mps_built if requested_device == "mps" else None,
            mps_available=(
                capabilities.mps_available if requested_device == "mps" else None
            ),
            fallback_category=category,
            fallback_reason=fallback_reason(category) if category is not None else None,
        ),
    )


def _build_pipeline_safely(
    token: str,
    device: SelectedDevice,
    pipeline_factory: Callable[[str, SelectedDevice], Any],
) -> Any:
    try:
        return pipeline_factory(token, device)
    except Exception as exc:
        raise DiarizationError(
            f"{device.upper()} pipeline 加载失败: {type(exc).__name__}"
        ) from None


def select_diarization_pipeline(
    preflight_path: Path,
    token: str,
    requested_device: Device,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    *,
    capabilities: DeviceCapabilities | None = None,
    pipeline_factory: Callable[[str, SelectedDevice], Any] = create_pipeline,
    cache_clearer: Callable[[], None] = clear_mps_cache,
    clock: Callable[[], float] = time.monotonic,
) -> PipelineSelection:
    facts = capabilities or collect_device_capabilities()
    if requested_device != "mps":
        selected = resolve_device(
            requested_device,
            cuda_available=lambda: facts.cuda_available,
        )
        return PipelineSelection(
            pipeline=_build_pipeline_safely(token, selected, pipeline_factory),
            decision=DeviceDecision(requested_device, selected),
        )

    logger.info("正在检查 MPS 环境")
    if not facts.mps_built:
        return _cpu_selection(
            requested_device, token, facts, pipeline_factory, "not_built"
        )
    if not facts.mps_available:
        return _cpu_selection(
            requested_device,
            token,
            facts,
            pipeline_factory,
            "runtime_unavailable",
        )

    logger.info("正在使用 10 秒样本预检 pyannote")
    started = clock()
    pipeline: Any | None = None
    try:
        pipeline = pipeline_factory(token, "mps")
        segments = apply_diarization_pipeline(
            preflight_path,
            pipeline,
            num_speakers,
            min_speakers,
            max_speakers,
        )
        if not segments:
            raise DiarizationError("预检样本未检测到可用语音")
    except Exception as exc:
        category = classify_mps_failure(exc, "preflight")
        if pipeline is not None:
            del pipeline
        try:
            cache_clearer()
        except Exception as cleanup_exc:
            logger.debug("清理 MPS 缓存失败: %s", type(cleanup_exc).__name__)
        return _cpu_selection(
            requested_device,
            token,
            facts,
            pipeline_factory,
            category,
        )

    elapsed = clock() - started
    logger.info("MPS 预检通过，耗时 %.2f 秒", elapsed)
    return PipelineSelection(
        pipeline=pipeline,
        decision=DeviceDecision(
            requested_device="mps",
            selected_device="mps",
            mps_built=True,
            mps_available=True,
            preflight_elapsed_seconds=elapsed,
        ),
    )
```

Do not log `token`, `repr(exc)`, or raw exception messages in normal logs. The stable `fallback_reason` strings are deliberately independent of secret-bearing exceptions. `_build_pipeline_safely` suppresses the original exception chain so a provider error containing a credential cannot be printed later by the CLI's verbose traceback.

- [ ] **Step 4: Run diarization tests**

Run:

```bash
uv run pytest tests/test_diarization.py -v
uv run pytest -q
```

Expected: all adapter and selection tests and the complete suite pass.

- [ ] **Step 5: Add and run a credential non-leak test**

Append to `tests/test_diarization.py`:

```python
def test_fallback_decision_does_not_contain_token() -> None:
    token = "hf_secret_value"

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        token,
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=False),
        pipeline_factory=lambda supplied, device: Pipeline(),
    )

    assert token not in str(selection.decision)
    assert token not in (selection.decision.fallback_reason or "")


def test_cpu_pipeline_load_error_does_not_expose_token() -> None:
    token = "hf_secret_value"

    def fail(supplied: str, device: str):
        raise RuntimeError(f"provider rejected {supplied}")

    with pytest.raises(DiarizationError) as captured:
        select_diarization_pipeline(
            Path("preflight.wav"),
            token,
            "mps",
            2,
            1,
            6,
            capabilities=capabilities(built=False, available=False),
            pipeline_factory=fail,
        )

    assert token not in str(captured.value)
    assert captured.value.__cause__ is None
```

Run:

```bash
uv run pytest tests/test_diarization.py::test_fallback_decision_does_not_contain_token tests/test_diarization.py::test_cpu_pipeline_load_error_does_not_expose_token -v
```

Expected: PASS.

- [ ] **Step 6: Commit MPS preflight selection**

```bash
git add src/mimo_transcriber/diarization.py tests/test_diarization.py
git commit -m "feat: preflight pyannote on MPS"
```

---

### Task 5: Complete-Run CPU Recovery

**Files:**
- Modify: `src/mimo_transcriber/diarization.py`
- Modify: `tests/test_diarization.py`

**Interfaces:**
- Produces: `DiarizationResult(segments: list[SpeakerSegment], decision: DeviceDecision)`
- Produces: `run_diarization(normalized_path: Path, preflight_path: Path, token: str, requested_device: Device, num_speakers: int | None, min_speakers: int, max_speakers: int, *, capabilities: DeviceCapabilities | None = None, pipeline_factory: Callable[[str, SelectedDevice], Any] = create_pipeline, cache_clearer: Callable[[], None] = clear_mps_cache, clock: Callable[[], float] = time.monotonic) -> DiarizationResult`
- Guarantees: one MPS preflight at most, one complete MPS run at most, and one complete CPU retry at most.

- [ ] **Step 1: Add complete-run success and recovery tests**

Append to `tests/test_diarization.py`:

```python
from mimo_transcriber.diarization import run_diarization


def test_full_run_reuses_successfully_preflighted_mps_pipeline() -> None:
    mps = Pipeline()

    result = run_diarization(
        Path("normalized.wav"),
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=lambda token, device: mps,
    )

    assert mps.calls == [
        ("preflight.wav", {"num_speakers": 2}),
        ("normalized.wav", {"num_speakers": 2}),
    ]
    assert result.decision.selected_device == "mps"
    assert len(result.segments) == 1


def test_complete_mps_failure_retries_complete_diarization_once_on_cpu() -> None:
    calls: list[tuple[str, str]] = []

    class MpsPipeline(Pipeline):
        def __call__(self, path: str, **kwargs):
            calls.append(("mps", path))
            if path == "normalized.wav":
                raise RuntimeError("MPS backend out of memory")
            return SimpleNamespace(speaker_diarization=Annotation())

    class CpuPipeline(Pipeline):
        def __call__(self, path: str, **kwargs):
            calls.append(("cpu", path))
            return SimpleNamespace(speaker_diarization=Annotation())

    def factory(token: str, device: str):
        return MpsPipeline() if device == "mps" else CpuPipeline()

    result = run_diarization(
        Path("normalized.wav"),
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=factory,
    )

    assert calls == [
        ("mps", "preflight.wav"),
        ("mps", "normalized.wav"),
        ("cpu", "normalized.wav"),
    ]
    assert result.decision.selected_device == "cpu"
    assert result.decision.fallback_category == "full_run_failed"


def test_cpu_failure_after_mps_failure_is_fatal() -> None:
    class MpsPipeline(Pipeline):
        def __call__(self, path: str, **kwargs):
            if path == "normalized.wav":
                raise RuntimeError("MPS backend out of memory")
            return SimpleNamespace(speaker_diarization=Annotation())

    def factory(token: str, device: str):
        if device == "mps":
            return MpsPipeline()
        def fail(path: str, **kwargs):
            raise RuntimeError("cpu failed")
        return fail

    with pytest.raises(DiarizationError, match="CPU 回退也失败"):
        run_diarization(
            Path("normalized.wav"),
            Path("preflight.wav"),
            "secret",
            "mps",
            2,
            1,
            6,
            capabilities=capabilities(built=True, available=True),
            pipeline_factory=factory,
        )
```

- [ ] **Step 2: Run recovery tests and verify failure**

Run:

```bash
uv run pytest tests/test_diarization.py -v
```

Expected: import failure because `run_diarization` is missing.

- [ ] **Step 3: Implement bounded full-run recovery**

Add to `src/mimo_transcriber/diarization.py`:

```python
@dataclass(frozen=True)
class DiarizationResult:
    segments: list[SpeakerSegment]
    decision: DeviceDecision


def run_diarization(
    normalized_path: Path,
    preflight_path: Path,
    token: str,
    requested_device: Device,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    *,
    capabilities: DeviceCapabilities | None = None,
    pipeline_factory: Callable[[str, SelectedDevice], Any] = create_pipeline,
    cache_clearer: Callable[[], None] = clear_mps_cache,
    clock: Callable[[], float] = time.monotonic,
) -> DiarizationResult:
    selection = select_diarization_pipeline(
        preflight_path,
        token,
        requested_device,
        num_speakers,
        min_speakers,
        max_speakers,
        capabilities=capabilities,
        pipeline_factory=pipeline_factory,
        cache_clearer=cache_clearer,
        clock=clock,
    )
    logger.info(
        "正在使用 %s 处理完整音频",
        selection.decision.selected_device.upper(),
    )
    try:
        segments = apply_diarization_pipeline(
            normalized_path,
            selection.pipeline,
            num_speakers,
            min_speakers,
            max_speakers,
        )
        return DiarizationResult(segments, selection.decision)
    except DiarizationError:
        if selection.decision.selected_device != "mps":
            raise

        mps_decision = selection.decision
        del selection
        try:
            cache_clearer()
        except Exception as cleanup_exc:
            logger.debug("清理 MPS 缓存失败: %s", type(cleanup_exc).__name__)

        logger.warning("完整 MPS 说话人分离失败，已安全回退 CPU")
        cpu_pipeline = _build_pipeline_safely(token, "cpu", pipeline_factory)
        try:
            segments = apply_diarization_pipeline(
                normalized_path,
                cpu_pipeline,
                num_speakers,
                min_speakers,
                max_speakers,
            )
        except DiarizationError as cpu_exc:
            raise DiarizationError(
                "MPS 完整运行失败，CPU 回退也失败"
            ) from cpu_exc

        return DiarizationResult(
            segments,
            DeviceDecision(
                requested_device="mps",
                selected_device="cpu",
                mps_built=mps_decision.mps_built,
                mps_available=mps_decision.mps_available,
                preflight_elapsed_seconds=mps_decision.preflight_elapsed_seconds,
                fallback_category="full_run_failed",
                fallback_reason=fallback_reason("full_run_failed"),
            ),
        )
```

The caught MPS exception is intentionally not interpolated into normal log messages. Its category is stable and the CPU failure remains the raised cause if both devices fail.

- [ ] **Step 4: Run diarization tests**

Run:

```bash
uv run pytest tests/test_diarization.py -v
uv run pytest -q
```

Expected: all diarization tests and the complete suite pass.

- [ ] **Step 5: Commit complete-run recovery**

```bash
git add src/mimo_transcriber/diarization.py tests/test_diarization.py
git commit -m "feat: recover failed MPS diarization on CPU"
```

---

### Task 6: Integrate MPS Selection Into the Workflow Once

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `create_preflight_sample(source: Path, target: Path, duration_seconds: float = 10.0) -> None`
- Consumes: `run_diarization(...) -> DiarizationResult`
- Changes `PipelineDependencies`:
  - `create_preflight: Callable[[Path, Path], None] = create_preflight_sample`
  - `diarize: Callable[..., DiarizationResult] = run_diarization`
- Guarantees: normalization and preflight sample creation happen once before diarization; downstream slicing starts only after successful diarization.

- [ ] **Step 1: Update existing pipeline fakes to return `DiarizationResult`**

Add imports in `tests/test_pipeline.py`:

```python
from mimo_transcriber.devices import DeviceDecision
from mimo_transcriber.diarization import DiarizationResult, DiarizationError
```

Add this helper:

```python
def diarization_result(segments: list[SpeakerSegment]) -> DiarizationResult:
    return DiarizationResult(
        segments=segments,
        decision=DeviceDecision(
            requested_device="cpu",
            selected_device="cpu",
        ),
    )
```

Change each existing `diarize=lambda ...: [...]` fake to:

```python
diarize=lambda *args, **kwargs: diarization_result([
    SpeakerSegment(-1, 0, 1, "A"),
    SpeakerSegment(-1, 1, 2, "B"),
]),
```

Add this dependency to every existing `PipelineDependencies(...)` fixture so unit tests do not invoke FFmpeg on their fake WAV bytes:

```python
create_preflight=lambda source, target: target.write_bytes(b"sample"),
```

For the one-segment fail-fast test, use:

```python
diarize=lambda *args, **kwargs: diarization_result([
    SpeakerSegment(-1, 0, 1, "A"),
]),
```

- [ ] **Step 2: Add workflow-count and fatal-fallback tests**

Append to `tests/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_pipeline_normalizes_and_creates_preflight_only_once(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48_000, 2, None)
    calls: list[str] = []

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "完成"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    def normalize(source_path: Path, target: Path) -> None:
        calls.append("normalize")
        target.write_bytes(b"wav")

    def preflight(source_path: Path, target: Path) -> None:
        calls.append("preflight")
        target.write_bytes(b"sample")

    def diarize(*args, **kwargs):
        calls.append("diarize")
        return diarization_result([SpeakerSegment(-1, 0, 1, "A")])

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=normalize,
        create_preflight=preflight,
        diarize=diarize,
        slice_audio=lambda source, segment, target: target.write_bytes(b"mp3"),
        payload_fits=lambda path, segment: True,
        transcribe=transcribe,
    )

    await run_pipeline(
        AppConfig(
            input_path=source,
            output_path=output,
            num_speakers=1,
            device="mps",
        ),
        "mimo",
        "hf",
        dependencies,
    )

    assert calls == ["normalize", "preflight", "diarize"]


@pytest.mark.asyncio
async def test_fatal_cpu_fallback_failure_stops_before_slicing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    metadata = AudioMetadata(source, 1, "aac", 48_000, 2, None)
    sliced: list[bool] = []

    def fail_diarization(*args, **kwargs):
        raise DiarizationError("MPS 完整运行失败，CPU 回退也失败")

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=fail_diarization,
        slice_audio=lambda source, segment, target: sliced.append(True),
    )

    with pytest.raises(DiarizationError, match="CPU 回退也失败"):
        await run_pipeline(
            AppConfig(input_path=source, device="mps"),
            "mimo",
            "hf",
            dependencies,
        )

    assert sliced == []
```

- [ ] **Step 3: Run pipeline tests and verify failure**

Run:

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: tests fail because `PipelineDependencies` has no `create_preflight`, and the workflow expects a list instead of `DiarizationResult`.

- [ ] **Step 4: Wire the new flow into `pipeline.py`**

First finish the adapter refactor in `src/mimo_transcriber/diarization.py`:

- Rename `apply_diarization_pipeline(...)` to `diarize_audio(...)`.
- Delete the temporary compatibility wrapper that accepted `token` and `device`.
- Replace all internal `apply_diarization_pipeline(...)` calls in `select_diarization_pipeline` and `run_diarization` with `diarize_audio(...)`.

In `tests/test_diarization.py`, replace the imported name and all calls:

```python
from mimo_transcriber.diarization import DiarizationError, diarize_audio
```

The adapter calls retain the supplied-pipeline signature:

```python
result = diarize_audio(
    Path("normalized.wav"),
    pipeline,
    num_speakers=2,
    min_speakers=1,
    max_speakers=6,
)
```

Change imports:

```python
from mimo_transcriber.audio import (
    create_preflight_sample,
    normalize_audio,
    payload_fits,
    probe_audio,
    slice_mp3,
    workspace,
)
from mimo_transcriber.diarization import DiarizationResult, run_diarization
```

Remove the `resolve_device` import and change `PipelineDependencies`:

```python
@dataclass(frozen=True)
class PipelineDependencies:
    probe: Callable[[Path], Any] = probe_audio
    normalize: Callable[[Path, Path], None] = normalize_audio
    create_preflight: Callable[[Path, Path], None] = create_preflight_sample
    diarize: Callable[..., DiarizationResult] = run_diarization
    slice_audio: Callable[[Path, SpeakerSegment, Path], None] = slice_mp3
    payload_fits: Callable[[Path, SpeakerSegment], bool] = payload_fits
    transcribe: Callable[
        [list[tuple[SpeakerSegment, Path]], bool],
        Awaitable[list[SpeakerSegment]],
    ] | None = None
```

Replace the normalization/diarization block in `run_pipeline`:

```python
        normalized = temp / "normalized.wav"
        dependencies.normalize(config.input_path, normalized)
        preflight = temp / "preflight.wav"
        dependencies.create_preflight(normalized, preflight)
        diarization = dependencies.diarize(
            normalized,
            preflight,
            hf_token,
            config.device,
            config.num_speakers,
            config.min_speakers,
            config.max_speakers,
        )
        raw = diarization.segments
```

No other workflow stage should inspect the selected device.

- [ ] **Step 5: Run pipeline and complete tests**

Run:

```bash
uv run pytest tests/test_pipeline.py -v
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit workflow integration**

```bash
git add src/mimo_transcriber/pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate safe MPS diarization fallback"
```

---

### Task 7: User-Facing Progress, Diagnostics, and Documentation

**Files:**
- Modify: `src/mimo_transcriber/diarization.py`
- Modify: `tests/test_diarization.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `log_device_decision(decision: DeviceDecision) -> None`
- Normal logs expose environment facts, preflight progress, selected device, and fallback reason without raw exceptions or credentials.
- README records exact experimental semantics and the 20 percent manual acceptance rule.

- [ ] **Step 1: Add log-content tests**

Append to `tests/test_diarization.py`:

```python
import logging

from mimo_transcriber.devices import DeviceDecision
from mimo_transcriber.diarization import log_device_decision


def test_runtime_fallback_log_is_actionable_and_secret_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    decision = DeviceDecision(
        requested_device="mps",
        selected_device="cpu",
        mps_built=True,
        mps_available=False,
        fallback_category="runtime_unavailable",
        fallback_reason="当前 PyTorch 运行时无法使用 MPS",
    )

    with caplog.at_level(logging.INFO):
        log_device_decision(decision)

    rendered = caplog.text
    assert "请求设备: MPS" in rendered
    assert "MPS 构建支持: 是" in rendered
    assert "MPS 运行时可用: 否" in rendered
    assert "已回退 CPU" in rendered
    assert "检查 PyTorch、macOS 版本以及当前 Python 架构" in rendered
```

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
uv run pytest tests/test_diarization.py::test_runtime_fallback_log_is_actionable_and_secret_free -v
```

Expected: import failure because `log_device_decision` does not exist.

- [ ] **Step 3: Implement stable decision logging**

Add to `src/mimo_transcriber/diarization.py`:

```python
def _yes_no(value: bool | None) -> str:
    if value is None:
        return "未知"
    return "是" if value else "否"


def log_device_decision(decision: DeviceDecision) -> None:
    if decision.requested_device != "mps":
        return
    logger.info("请求设备: MPS")
    logger.info("MPS 构建支持: %s", _yes_no(decision.mps_built))
    logger.info("MPS 运行时可用: %s", _yes_no(decision.mps_available))
    if decision.selected_device == "cpu":
        logger.warning("MPS 未启用: %s", decision.fallback_reason)
        logger.warning("已回退 CPU")
        logger.info("建议检查 PyTorch、macOS 版本以及当前 Python 架构")
```

Call `log_device_decision(selection.decision)` in `run_diarization` immediately after selection. For complete-run fallback, create the final CPU `DeviceDecision`, pass it to `log_device_decision`, and then return it.

Replace the complete-run fallback return block with:

```python
        final_decision = DeviceDecision(
            requested_device="mps",
            selected_device="cpu",
            mps_built=mps_decision.mps_built,
            mps_available=mps_decision.mps_available,
            preflight_elapsed_seconds=mps_decision.preflight_elapsed_seconds,
            fallback_category="full_run_failed",
            fallback_reason=fallback_reason("full_run_failed"),
        )
        log_device_decision(final_decision)
        return DiarizationResult(segments, final_decision)
```

- [ ] **Step 4: Update README**

Change the environment section to:

```markdown
- Python 3.11
- uv
- FFmpeg 与 ffprobe
- macOS Apple Silicon 默认使用 CPU，可显式试用实验性 MPS
- Linux 可选 NVIDIA CUDA
```

Add an example:

```bash
uv run python -m mimo_transcriber meeting.m4a --device mps --verbose
```

Change the CLI table device row to:

```markdown
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或实验性 `mps` |
```

Add this subsection before “常见问题”:

```markdown
## 实验性 Apple MPS

`--device mps` 会先检查当前 PyTorch 的 MPS 构建与运行时状态，再使用标准化录音的前 10 秒运行真实 pyannote Community-1 预检。预检成功后会复用同一个 MPS pipeline 处理完整录音。

MPS 不可用、预检失败或完整运行失败时，程序会自动回退 CPU。完整运行回退不会重复 FFmpeg 标准化，也不会重复已经完成的 MiMo 请求。默认的 `--device auto` 暂时不会选择 MPS。

应用只提供诊断，不会自动更换 PyTorch 或修改 macOS。兼容性取决于 Apple 芯片、macOS、Python、PyTorch 与 pyannote 的具体版本组合。

评估一台机器是否值得使用 MPS 时，使用同一段 1～3 分钟、至少两位说话人的录音分别运行 CPU 和 MPS 各三次。比较 diarization 中位耗时；只有 MPS 稳定完成且至少快 20% 时才建议日常使用。
```

Add an MPS troubleshooting bullet:

```markdown
- MPS 回退 CPU：使用 `--verbose` 查看 `MPS 构建支持` 和 `MPS 运行时可用`；确认当前 Python 为 arm64，并检查 macOS、PyTorch 与 pyannote 版本兼容性。
```

- [ ] **Step 5: Run tests and CLI documentation check**

Run:

```bash
uv run pytest tests/test_diarization.py tests/test_cli.py -v
uv run python -m mimo_transcriber --help
```

Expected: tests pass, and help lists `--device {auto,cpu,cuda,mps}` without loading pyannote or downloading models.

- [ ] **Step 6: Commit diagnostics and docs**

```bash
git add src/mimo_transcriber/diarization.py tests/test_diarization.py README.md
git commit -m "docs: explain experimental MPS diarization"
```

---

### Task 8: Final Regression and Manual Compatibility Gate

**Files:**
- Modify only if verification exposes a defect in the files already listed above.
- Do not commit the user's recording or generated transcript/debug artifacts.

**Interfaces:**
- Verifies the complete feature against the approved design.
- Produces a recorded compatibility note in the final handoff, including exact machine, macOS, Python, PyTorch, and pyannote versions.

- [ ] **Step 1: Run formatting-independent repository checks**

Run:

```bash
git diff --check
uv run pytest -q
uv run python -m mimo_transcriber --help
```

Expected:

- `git diff --check` emits no output.
- The full pytest suite passes.
- Help lists `auto`, `cpu`, `cuda`, and `mps`.

- [ ] **Step 2: Verify current-machine diagnostics without model download**

Run:

```bash
.venv/bin/python -c 'import platform, torch, pyannote.audio; print(platform.platform()); print(platform.machine()); print(torch.__version__); print(pyannote.audio.__version__); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())'
```

Expected: prints the exact compatibility tuple. On the currently observed environment, `mps.is_built()` is true and `mps.is_available()` is false, so a formal `--device mps` run is expected to explain `runtime_unavailable` and select CPU.

- [ ] **Step 3: Verify the unavailable-runtime fallback with a short disposable audio file**

Create a 30-second disposable sample from the current local recording, then run with valid environment credentials:

```bash
ffmpeg -y -t 30 -i "/Users/yuancheng/Documents/Code/CScribe/新录音 14.m4a" -c:a aac /tmp/cscribe-mps-smoke.m4a
uv run python -m mimo_transcriber /tmp/cscribe-mps-smoke.m4a --num-speakers 2 --device mps --verbose
```

Expected on a runtime where MPS is unavailable:

- Logs show MPS was requested.
- Logs show build support and runtime availability.
- Logs state that CPU was selected.
- The command completes through CPU when credentials and source audio are valid.

Do not use the 52-minute recording for this diagnostic gate.

- [ ] **Step 4: Run the performance acceptance only on an MPS-available runtime**

On an MPS-available runtime, create a three-minute sample and first listen briefly to confirm it contains both speakers:

```bash
ffmpeg -y -t 180 -i "/Users/yuancheng/Documents/Code/CScribe/新录音 14.m4a" -c:a aac /tmp/cscribe-mps-benchmark.m4a
```

Run each command three times:

```bash
uv run python -m mimo_transcriber /tmp/cscribe-mps-benchmark.m4a --num-speakers 2 --device cpu --verbose
uv run python -m mimo_transcriber /tmp/cscribe-mps-benchmark.m4a --num-speakers 2 --device mps --verbose
```

Record only diarization elapsed time for each run. Compute:

```text
speedup_percent = (cpu_median - mps_median) / cpu_median * 100
```

Expected for a recommended configuration:

- All three MPS runs complete without crashes or unbounded memory growth.
- Speaker count and timestamp boundaries remain reasonably consistent with CPU.
- `speedup_percent >= 20`.

If MPS is stable but below 20 percent, report it as compatible but not recommended. If MPS is unavailable on the current machine, report the diagnostic result and leave performance acceptance pending for a compatible runtime rather than weakening the threshold.

- [ ] **Step 5: Confirm the commit boundary and working tree**

Run:

```bash
git status --short
git log --oneline -8
```

Expected: only pre-existing user files such as `uv.lock` or local recordings may remain modified/untracked; implementation commits are present and no recording, token file, transcript, or temporary audio is staged.
