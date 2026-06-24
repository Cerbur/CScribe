# Diarization Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize 2-person speaker attribution by adding explicit two-person mode, local deterministic speaker post-processing, overlap deduplication, and short-island smoothing.

**Architecture:** Add `speaker_stability.py` as a pure post-processing layer after `process_segments()` and before slicing. Extend config/CLI with conversation and stabilizer settings. Keep pyannote model execution local and make diarization model id configurable for later A/B tests.

**Tech Stack:** Python 3.11, dataclasses, argparse, pytest, existing pyannote pipeline wrapper, existing `SpeakerSegment` model.

## Global Constraints

- Default diarization model remains `pyannote/speaker-diarization-community-1`.
- All diarization stays local.
- Stabilizer must not call ASR, pyannote, ffmpeg, or external services.
- Stabilizer changes speaker labels and removes duplicate overlap candidates only; it does not edit ASR text.
- Explicit `--num-speakers` always overrides `--conversation-mode two-person`.
- If stabilizer would remove every segment, pipeline must fall back to the pre-stabilized segment list.
- Unit tests must not download models.

---

## File Structure

- Create `src/mimo_transcriber/speaker_stability.py`: stabilizer config, diagnostics, overlap dedupe, island smoothing.
- Modify `src/mimo_transcriber/config.py`: add conversation/stabilizer/model config and cache identity.
- Modify `src/mimo_transcriber/cli.py`: expose new flags.
- Modify `src/mimo_transcriber/diarization.py`: make model id configurable and resolve speaker kwargs with two-person mode.
- Modify `src/mimo_transcriber/pipeline.py`: run stabilizer after `process_segments()`.
- Modify `src/mimo_transcriber/formatter.py` or manifest/debug code: include stabilizer diagnostics when debug JSON is enabled.
- Create `tests/test_speaker_stability.py`.
- Modify `tests/test_config.py`, `tests/test_cli.py`, `tests/test_diarization.py`, `tests/test_pipeline.py`, `tests/test_cache.py`.

---

### Task 1: Speaker Stability Pure Functions

**Files:**
- Create: `src/mimo_transcriber/speaker_stability.py`
- Test: `tests/test_speaker_stability.py`

**Interfaces:**
- Produces: `SpeakerStabilityConfig(enabled: bool = True, mode: Literal["conservative", "balanced", "aggressive"] = "balanced")`
- Produces: `SpeakerStabilityDiagnostics(enabled: bool, mode: str, dropped_overlaps: int = 0, relabeled_islands: int = 0)`
- Produces: `StabilizedSegments(segments: list[SpeakerSegment], diagnostics: SpeakerStabilityDiagnostics)`
- Produces: `stabilize_speakers(segments: list[SpeakerSegment], config: SpeakerStabilityConfig) -> StabilizedSegments`

- [ ] **Step 1: Write failing stability tests**

Create `tests/test_speaker_stability.py`:

```python
from mimo_transcriber.models import SpeakerSegment
from mimo_transcriber.speaker_stability import SpeakerStabilityConfig, stabilize_speakers


def seg(start: float, end: float, speaker: str, segment_id: str = "") -> SpeakerSegment:
    return SpeakerSegment(
        index=0,
        start=start,
        end=end,
        raw_speaker=speaker,
        display_speaker="说话人 1" if speaker == "A" else "说话人 2",
        segment_id=segment_id,
    )


def test_disabled_stabilizer_returns_segments_unchanged() -> None:
    segments = [seg(0, 1, "A", "s0000")]

    result = stabilize_speakers(segments, SpeakerStabilityConfig(enabled=False))

    assert result.segments == segments
    assert result.diagnostics.enabled is False


def test_drops_highly_overlapped_duplicate_by_context() -> None:
    result = stabilize_speakers([
        seg(0, 2, "A", "s0000"),
        seg(2.1, 4.1, "A", "s0001"),
        seg(2.2, 4.0, "B", "s0002"),
        seg(4.2, 6, "A", "s0003"),
    ], SpeakerStabilityConfig())

    assert [item.raw_speaker for item in result.segments] == ["A", "A", "A"]
    assert result.diagnostics.dropped_overlaps == 1


def test_relabels_short_speaker_island_between_same_speaker() -> None:
    result = stabilize_speakers([
        seg(0, 3, "A", "s0000"),
        seg(3.2, 4.0, "B", "s0001"),
        seg(4.2, 7, "A", "s0002"),
    ], SpeakerStabilityConfig(mode="balanced"))

    assert [item.raw_speaker for item in result.segments] == ["A", "A", "A"]
    assert result.diagnostics.relabeled_islands == 1


def test_does_not_relabel_long_turn() -> None:
    result = stabilize_speakers([
        seg(0, 3, "A", "s0000"),
        seg(3.2, 7.0, "B", "s0001"),
        seg(7.2, 9, "A", "s0002"),
    ], SpeakerStabilityConfig(mode="balanced"))

    assert [item.raw_speaker for item in result.segments] == ["A", "B", "A"]


def test_reassigns_indexes_and_segment_ids_after_drop() -> None:
    result = stabilize_speakers([
        seg(0, 2, "A", "old0"),
        seg(0.1, 1.9, "B", "old1"),
    ], SpeakerStabilityConfig())

    assert [(item.index, item.segment_id) for item in result.segments] == [(0, "s0000")]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_speaker_stability.py -q
```

Expected: FAIL with missing module.

- [ ] **Step 3: Implement stability module**

Create `src/mimo_transcriber/speaker_stability.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mimo_transcriber.models import SpeakerSegment

StabilityMode = Literal["conservative", "balanced", "aggressive"]


@dataclass(frozen=True)
class SpeakerStabilityConfig:
    enabled: bool = True
    mode: StabilityMode = "balanced"


@dataclass(frozen=True)
class SpeakerStabilityDiagnostics:
    enabled: bool
    mode: str
    dropped_overlaps: int = 0
    relabeled_islands: int = 0


@dataclass(frozen=True)
class StabilizedSegments:
    segments: list[SpeakerSegment]
    diagnostics: SpeakerStabilityDiagnostics


def stabilize_speakers(
    segments: list[SpeakerSegment],
    config: SpeakerStabilityConfig,
) -> StabilizedSegments:
    if not config.enabled:
        return StabilizedSegments(
            list(segments),
            SpeakerStabilityDiagnostics(False, config.mode),
        )
    ordered = sorted(segments, key=lambda item: item.sort_key())
    deduped, dropped = _drop_duplicate_overlaps(ordered)
    smoothed, relabeled = _smooth_islands(deduped, config.mode)
    if not smoothed and ordered:
        smoothed = ordered
    _renumber(smoothed)
    return StabilizedSegments(
        smoothed,
        SpeakerStabilityDiagnostics(True, config.mode, dropped, relabeled),
    )


def _drop_duplicate_overlaps(segments: list[SpeakerSegment]) -> tuple[list[SpeakerSegment], int]:
    result: list[SpeakerSegment] = []
    dropped = 0
    for item in segments:
        if result and item.raw_speaker != result[-1].raw_speaker and _overlap_ratio(result[-1], item) >= 0.8:
            keep_existing = _context_score(result, result[-1].raw_speaker) >= _context_score(result, item.raw_speaker)
            if keep_existing or result[-1].duration >= item.duration:
                dropped += 1
                continue
            result[-1] = item
            dropped += 1
            continue
        result.append(item)
    return result, dropped


def _smooth_islands(segments: list[SpeakerSegment], mode: StabilityMode) -> tuple[list[SpeakerSegment], int]:
    max_duration, max_gap = {
        "conservative": (1.2, 0.5),
        "balanced": (2.0, 1.0),
        "aggressive": (3.0, 1.5),
    }[mode]
    result = list(segments)
    relabeled = 0
    for index in range(1, len(result) - 1):
        prev = result[index - 1]
        cur = result[index]
        nxt = result[index + 1]
        if (
            prev.raw_speaker == nxt.raw_speaker
            and cur.raw_speaker != prev.raw_speaker
            and cur.duration <= max_duration
            and cur.start - prev.end <= max_gap
            and nxt.start - cur.end <= max_gap
        ):
            cur.raw_speaker = prev.raw_speaker
            cur.display_speaker = prev.display_speaker
            relabeled += 1
    return result, relabeled


def _overlap_ratio(left: SpeakerSegment, right: SpeakerSegment) -> float:
    overlap = min(left.end, right.end) - max(left.start, right.start)
    if overlap <= 0:
        return 0.0
    return overlap / min(left.duration, right.duration)


def _context_score(result: list[SpeakerSegment], speaker: str) -> int:
    return sum(1 for item in result[-2:] if item.raw_speaker == speaker)


def _renumber(segments: list[SpeakerSegment]) -> None:
    names: dict[str, str] = {}
    for index, item in enumerate(segments):
        item.index = index
        item.segment_id = f"s{index:04d}"
        if item.raw_speaker not in names:
            names[item.raw_speaker] = f"说话人 {len(names) + 1}"
        item.display_speaker = names[item.raw_speaker]
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_speaker_stability.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/speaker_stability.py tests/test_speaker_stability.py
git commit -m "feat: add speaker stability post-processing"
```

---

### Task 2: Conversation Mode and Stabilizer Config

**Files:**
- Modify: `src/mimo_transcriber/config.py`
- Modify: `src/mimo_transcriber/cli.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `AppConfig.conversation_mode: Literal["auto", "two-person", "multi"]`
- Produces: `AppConfig.diarization_stabilizer: Literal["off", "conservative", "balanced", "aggressive"]`
- Produces: `AppConfig.speaker_stability_config() -> SpeakerStabilityConfig`
- Produces: `AppConfig.resolved_num_speakers() -> int | None`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
from mimo_transcriber.config import AppConfig


def test_two_person_mode_resolves_num_speakers(tmp_path):
    config = AppConfig(input_path=tmp_path / "in.m4a", conversation_mode="two-person")

    assert config.resolved_num_speakers() == 2


def test_explicit_num_speakers_overrides_two_person_mode(tmp_path):
    config = AppConfig(
        input_path=tmp_path / "in.m4a",
        conversation_mode="two-person",
        num_speakers=3,
    )

    assert config.resolved_num_speakers() == 3


def test_stabilizer_off_disables_config(tmp_path):
    config = AppConfig(input_path=tmp_path / "in.m4a", diarization_stabilizer="off")

    stability = config.speaker_stability_config()

    assert stability.enabled is False
```

Append to `tests/test_cli.py`:

```python
from mimo_transcriber.cli import build_parser


def test_cli_parses_diarization_stability_options() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--conversation-mode", "two-person",
        "--diarization-stabilizer", "aggressive",
    ])

    assert args.conversation_mode == "two-person"
    assert args.diarization_stabilizer == "aggressive"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_config.py tests/test_cli.py -q
```

Expected: FAIL because config and CLI fields are missing.

- [ ] **Step 3: Implement config**

Modify `src/mimo_transcriber/config.py`:

```python
from mimo_transcriber.speaker_stability import SpeakerStabilityConfig

ConversationMode = Literal["auto", "two-person", "multi"]
DiarizationStabilizer = Literal["off", "conservative", "balanced", "aggressive"]

@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    output_path: Path | None = None
    num_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 6
    language: Language = "auto"
    device: Device = "auto"
    conversation_mode: ConversationMode = "auto"
    diarization_stabilizer: DiarizationStabilizer = "balanced"

    def resolved_num_speakers(self) -> int | None:
        if self.num_speakers is not None:
            return self.num_speakers
        if self.conversation_mode == "two-person":
            return 2
        return None

    def speaker_stability_config(self) -> SpeakerStabilityConfig:
        mode = "balanced" if self.diarization_stabilizer == "off" else self.diarization_stabilizer
        return SpeakerStabilityConfig(
            enabled=self.diarization_stabilizer != "off",
            mode=mode,
        )
```

Keep `cache_parameters()` using `resolved_num_speakers()` instead of raw `num_speakers`.

- [ ] **Step 4: Implement CLI**

Modify parser in `src/mimo_transcriber/cli.py`:

```python
    parser.add_argument(
        "--conversation-mode",
        choices=("auto", "two-person", "multi"),
        default="auto",
    )
    parser.add_argument(
        "--diarization-stabilizer",
        choices=("off", "conservative", "balanced", "aggressive"),
        default="balanced",
    )
```

Pass into `AppConfig`:

```python
        conversation_mode=args.conversation_mode,
        diarization_stabilizer=args.diarization_stabilizer,
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_config.py tests/test_cli.py tests/test_speaker_stability.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/config.py src/mimo_transcriber/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: add diarization stability configuration"
```

---

### Task 3: Configurable Diarization Model and Speaker Count Resolution

**Files:**
- Modify: `src/mimo_transcriber/diarization.py`
- Modify: `src/mimo_transcriber/cache.py`
- Modify: `src/mimo_transcriber/pipeline.py`
- Test: `tests/test_diarization.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `AppConfig.resolved_num_speakers()`
- Produces: `create_pipeline(token: str, device: SelectedDevice, model_id: str = MODEL_ID) -> Any`
- Produces: `run_diarization(normalized_path, preflight_path, token, requested_device, num_speakers, min_speakers, max_speakers, model_id: str = MODEL_ID)`
- Produces: `AppConfig.diarization_model: str = "pyannote/speaker-diarization-community-1"`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_diarization.py`:

```python
from mimo_transcriber.diarization import speaker_kwargs


def test_speaker_kwargs_prefers_resolved_num_speakers() -> None:
    assert speaker_kwargs(2, 1, 6) == {"num_speakers": 2}
```

Append to `tests/test_cache.py`:

```python
from pathlib import Path

from mimo_transcriber.cache import TaskPaths, fingerprint_input
from mimo_transcriber.config import AppConfig


def test_task_hash_changes_with_diarization_model(tmp_path: Path) -> None:
    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)

    first = TaskPaths.for_run(AppConfig(input_path=source), fingerprint, tmp_path)
    second = TaskPaths.for_run(
        AppConfig(input_path=source, diarization_model="local/other-model"),
        fingerprint,
        tmp_path,
    )

    assert first.task_hash != second.task_hash
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_diarization.py tests/test_cache.py -q
```

Expected: FAIL for missing `diarization_model`.

- [ ] **Step 3: Add diarization model config and cache identity**

Modify `AppConfig`:

```python
    diarization_model: str = "pyannote/speaker-diarization-community-1"
```

Modify `cache_parameters()`:

```python
            "num_speakers": self.resolved_num_speakers(),
            "diarization_model": self.diarization_model,
```

- [ ] **Step 4: Wire resolved speaker count and model into pipeline**

Modify diarization call in `src/mimo_transcriber/pipeline.py`:

```python
                diarization = dependencies.diarize(
                    normalized,
                    preflight,
                    runtime.hf_token,
                    config.device,
                    config.resolved_num_speakers(),
                    config.min_speakers,
                    config.max_speakers,
                    model_id=config.diarization_model,
                )
```

If the dependency callable does not accept `model_id`, extend `PipelineDependencies.diarize` signature accordingly and update tests.

- [ ] **Step 5: Make diarization model configurable**

Modify `src/mimo_transcriber/diarization.py`:

```python
def create_pipeline(token: str, device: SelectedDevice, model_id: str = MODEL_ID) -> Any:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_id, token=token)
    pipeline.to(torch.device(device))
    return pipeline
```

Add `model_id` through `select_diarization_pipeline()` and `run_diarization()` signatures, defaulting to `MODEL_ID`.

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_diarization.py tests/test_cache.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mimo_transcriber/config.py src/mimo_transcriber/cache.py src/mimo_transcriber/diarization.py src/mimo_transcriber/pipeline.py tests/test_diarization.py tests/test_cache.py tests/test_pipeline.py
git commit -m "feat: configure diarization speaker count and model"
```

---

### Task 4: Pipeline Applies Stabilizer and Debug Diagnostics

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `src/mimo_transcriber/models.py`
- Modify: `src/mimo_transcriber/formatter.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_formatter.py`

**Interfaces:**
- Consumes: `stabilize_speakers(segments, config.speaker_stability_config())`
- Produces: `TranscriptionOutcome.speaker_stability: SpeakerStabilityDiagnostics | None`
- Produces: debug JSON `speaker_stability`

- [ ] **Step 1: Write failing model/formatter test**

Append to `tests/test_formatter.py`:

```python
from mimo_transcriber.speaker_stability import SpeakerStabilityDiagnostics


def test_debug_json_can_include_speaker_stability(tmp_path):
    metadata = AudioMetadata(tmp_path / "input.m4a", 3, "aac", 48000, 2, None)
    outcome = TranscriptionOutcome(
        metadata=metadata,
        segments=[
            SpeakerSegment(0, 0, 1, "A", "说话人 1", "你好", SegmentStatus.SUCCESS),
        ],
    )
    outcome.speaker_stability = SpeakerStabilityDiagnostics(
        enabled=True,
        mode="balanced",
        dropped_overlaps=1,
        relabeled_islands=2,
    )
    output = tmp_path / "out.txt"

    write_outputs(outcome, datetime(2026, 6, 24, 10, 0), output, debug_json=True)

    debug = output.with_suffix(".segments.json").read_text(encoding="utf-8")
    assert '"speaker_stability"' in debug
    assert '"dropped_overlaps": 1' in debug
```

- [ ] **Step 2: Write failing pipeline test**

Add to `tests/test_pipeline.py` using existing fake dependency style:

```python
@pytest.mark.asyncio
async def test_pipeline_applies_speaker_stabilizer_before_slicing(tmp_path):
    sliced = []

    def fake_slice(source, segment, target):
        sliced.append(segment.raw_speaker)
        target.write_bytes(b"mp3")

    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 7, "aac", 48000, 2, datetime(2026, 1, 1, 9))

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "ok"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    dependencies = PipelineDependencies(
        probe=lambda path: metadata,
        normalize=lambda source, target: target.write_bytes(b"wav"),
        create_preflight=lambda source, target: target.write_bytes(b"sample"),
        diarize=lambda *args, **kwargs: diarization_result([
            SpeakerSegment(-1, 0, 3, "A"),
            SpeakerSegment(-1, 3.2, 4.0, "B"),
            SpeakerSegment(-1, 4.2, 7, "A"),
        ]),
        slice_audio=fake_slice,
        payload_fits=lambda path, segment: True,
        transcribe=transcribe,
    )

    await run_pipeline(
        AppConfig(
            input_path=source,
            output_path=output,
            num_speakers=2,
            diarization_stabilizer="balanced",
        ),
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
        dependencies,
        cache_root=tmp_path,
    )

    assert sliced == ["A", "A", "A"]
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_pipeline.py tests/test_formatter.py -q
```

Expected: FAIL because diagnostics field and stabilizer call are missing.

- [ ] **Step 4: Add diagnostics to model and formatter**

Modify `src/mimo_transcriber/models.py`:

```python
from typing import Any

@dataclass
class TranscriptionOutcome:
    metadata: AudioMetadata
    segments: list[SpeakerSegment]
    keywords: list[str] = field(default_factory=list)
    summary: RunSummary = field(default_factory=RunSummary)
    speaker_stability: Any | None = None
```

Modify debug JSON payload in `formatter.py`:

```python
        if outcome.speaker_stability is not None:
            payload["speaker_stability"] = asdict(outcome.speaker_stability)
```

- [ ] **Step 5: Apply stabilizer in pipeline**

After:

```python
                raw = diarization.segments
                segments = process_segments(raw, metadata.duration_seconds)
```

Add:

```python
                stability = stabilize_speakers(
                    segments,
                    config.speaker_stability_config(),
                )
                segments = stability.segments
                speaker_stability = stability.diagnostics
```

When constructing `TranscriptionOutcome`, pass:

```python
speaker_stability=speaker_stability
```

Initialize `speaker_stability = None` before the cache branch. If diarization segments are loaded from manifest, set diagnostics to `None` unless the manifest stores them in a later enhancement.

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_speaker_stability.py tests/test_pipeline.py tests/test_formatter.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mimo_transcriber/pipeline.py src/mimo_transcriber/models.py src/mimo_transcriber/formatter.py tests/test_pipeline.py tests/test_formatter.py
git commit -m "feat: apply speaker stabilizer in pipeline"
```

---

### Task 5: CLI Docs and Final Verification

**Files:**
- Modify: `src/mimo_transcriber/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Documents: `--conversation-mode`, `--diarization-stabilizer`, `--diarization-model`

- [ ] **Step 1: Add CLI model option**

Modify parser:

```python
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
```

Pass to `AppConfig`:

```python
        diarization_model=args.diarization_model,
```

Add CLI test:

```python
def test_cli_parses_diarization_model() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--diarization-model", "local/model",
    ])

    assert args.diarization_model == "local/model"
```

- [ ] **Step 2: Update README**

Add:

```markdown
## 2 人对话说话人稳定

2 人录音建议固定说话人数：

```bash
uv run mimo-transcriber meeting.m4a --conversation-mode two-person
```

这会在未显式传 `--num-speakers` 时按 `--num-speakers 2` 运行 pyannote，并启用默认 `balanced` 后处理稳定器。对照调试：

```bash
uv run mimo-transcriber meeting.m4a --diarization-stabilizer off --debug-json
uv run mimo-transcriber meeting.m4a --diarization-stabilizer aggressive --debug-json
```

如需 A/B 本地模型：

```bash
uv run mimo-transcriber meeting.m4a --diarization-model pyannote/speaker-diarization-community-1
```
```

- [ ] **Step 3: Run relevant tests**

Run:

```bash
uv run pytest tests/test_speaker_stability.py tests/test_config.py tests/test_cli.py tests/test_diarization.py tests/test_cache.py tests/test_pipeline.py tests/test_formatter.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/cli.py README.md tests/test_cli.py
git commit -m "docs: document diarization stability controls"
```

---

## Self-Review

- Spec coverage: two-person mode, overlap dedupe, island smoothing, model id config, pipeline integration, debug diagnostics, tests, and docs are covered.
- Placeholder scan: No deferred placeholders remain; formatter and pipeline tests include inline fixtures.
- Type consistency: `SpeakerStabilityConfig`, `SpeakerStabilityDiagnostics`, `StabilizedSegments`, and `stabilize_speakers` names match across tasks.
