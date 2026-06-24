# Paragraph Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a display-only paragraph merging layer so CScribe renders continuous same-speaker speech as larger transcript blocks while keeping internal ASR segments unchanged.

**Architecture:** Create `paragraphs.py` to convert completed `SpeakerSegment` objects into `TranscriptBlock` objects using deterministic merge rules. Extend config/CLI with paragraph settings and update `formatter.py` to render blocks. Preserve debug JSON segments and add rendered block metadata.

**Tech Stack:** Python 3.11, dataclasses, argparse, pytest, existing `SpeakerSegment` and `TranscriptionOutcome`.

## Global Constraints

- ASR slicing and transcription must keep using internal `SpeakerSegment` objects.
- Paragraph settings affect final rendering only and must not change task cache identity.
- `paragraph_mode=off` and `--no-paragraph-merge` must preserve old one-segment-per-block rendering.
- Failed segment text `[该片段识别失败]` must not merge with normal transcript text.
- Different `raw_speaker` values must never merge.
- Unit tests must not call pyannote, ffmpeg, MiMo, or MLX.

---

## File Structure

- Create `src/mimo_transcriber/paragraphs.py`: paragraph config, block dataclass, merge predicates, and block builder.
- Modify `src/mimo_transcriber/config.py`: add paragraph config fields and validation.
- Modify `src/mimo_transcriber/cli.py`: expose paragraph CLI flags.
- Modify `src/mimo_transcriber/formatter.py`: render `TranscriptBlock` output and debug JSON `blocks`.
- Create `tests/test_paragraphs.py`: focused paragraph merge tests.
- Modify `tests/test_config.py`: paragraph config validation tests.
- Modify `tests/test_cli.py`: CLI parsing tests.
- Modify `tests/test_formatter.py`: rendered block and debug JSON tests.

---

### Task 1: Paragraph Block Builder

**Files:**
- Create: `src/mimo_transcriber/paragraphs.py`
- Test: `tests/test_paragraphs.py`

**Interfaces:**
- Produces: `TranscriptBlock(index: int, start: float, end: float, raw_speaker: str, display_speaker: str | None, text: str, source_segment_ids: list[str])`
- Produces: `ParagraphConfig(enabled: bool = True, mode: ParagraphMode = "balanced", gap: float | None = None, max_duration: float | None = None, max_chars: int = 900)`
- Produces: `build_transcript_blocks(segments: list[SpeakerSegment], config: ParagraphConfig) -> list[TranscriptBlock]`

- [ ] **Step 1: Write failing paragraph builder tests**

Create `tests/test_paragraphs.py`:

```python
from mimo_transcriber.models import SegmentStatus, SpeakerSegment
from mimo_transcriber.paragraphs import ParagraphConfig, build_transcript_blocks


def seg(
    start: float,
    end: float,
    speaker: str = "SPEAKER_00",
    text: str = "hello",
    segment_id: str = "",
) -> SpeakerSegment:
    return SpeakerSegment(
        index=0,
        start=start,
        end=end,
        raw_speaker=speaker,
        display_speaker="说话人 1" if speaker == "SPEAKER_00" else "说话人 2",
        text=text,
        status=SegmentStatus.SUCCESS,
        segment_id=segment_id,
    )


def test_balanced_merges_same_speaker_short_gap() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, text="我们先聊 Facebook。", segment_id="s0000"),
        seg(4, 7, text="然后看 Grab 的例子。", segment_id="s0001"),
    ], ParagraphConfig())

    assert len(blocks) == 1
    assert blocks[0].start == 0
    assert blocks[0].end == 7
    assert blocks[0].text == "我们先聊 Facebook。然后看 Grab 的例子。"
    assert blocks[0].source_segment_ids == ["s0000", "s0001"]


def test_different_speakers_do_not_merge() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, "SPEAKER_00", "你怎么看？", "s0000"),
        seg(3.2, 5, "SPEAKER_01", "我同意。", "s0001"),
    ], ParagraphConfig())

    assert len(blocks) == 2


def test_long_gap_does_not_merge() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, text="第一段。", segment_id="s0000"),
        seg(6, 8, text="第二段。", segment_id="s0001"),
    ], ParagraphConfig())

    assert len(blocks) == 2


def test_failed_segment_stays_separate() -> None:
    failed = seg(3.5, 5, text="[该片段识别失败]", segment_id="s0001")
    failed.status = SegmentStatus.FAILED

    blocks = build_transcript_blocks([
        seg(0, 3, text="正常文本。", segment_id="s0000"),
        failed,
    ], ParagraphConfig())

    assert len(blocks) == 2


def test_off_mode_keeps_one_block_per_segment() -> None:
    blocks = build_transcript_blocks([
        seg(0, 3, text="第一段", segment_id="s0000"),
        seg(3.1, 4, text="第二段", segment_id="s0001"),
    ], ParagraphConfig(enabled=False, mode="balanced"))

    assert [block.source_segment_ids for block in blocks] == [["s0000"], ["s0001"]]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_paragraphs.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mimo_transcriber.paragraphs'`.

- [ ] **Step 3: Implement paragraph builder**

Create `src/mimo_transcriber/paragraphs.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mimo_transcriber.models import SegmentStatus, SpeakerSegment

ParagraphMode = Literal["conservative", "balanced", "aggressive"]
FAILED_TEXT = "[该片段识别失败]"
SENTENCE_ENDINGS = ("。", "！", "？", ".", "!", "?")
CONTINUATIONS = (
    "然后", "所以", "但是", "而且", "就是", "那", "这个", "因为",
    "OK", "ok", "and", "so", "but", "then",
)


@dataclass
class TranscriptBlock:
    index: int
    start: float
    end: float
    raw_speaker: str
    display_speaker: str | None
    text: str
    source_segment_ids: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class ParagraphConfig:
    enabled: bool = True
    mode: ParagraphMode = "balanced"
    gap: float | None = None
    max_duration: float | None = None
    max_chars: int = 900


def build_transcript_blocks(
    segments: list[SpeakerSegment],
    config: ParagraphConfig,
) -> list[TranscriptBlock]:
    ordered = sorted(segments, key=lambda item: item.sort_key())
    blocks: list[TranscriptBlock] = []
    for segment in ordered:
        text = " ".join((segment.text or "").split())
        block = TranscriptBlock(
            index=-1,
            start=segment.start,
            end=segment.end,
            raw_speaker=segment.raw_speaker,
            display_speaker=segment.display_speaker,
            text=text,
            source_segment_ids=[segment.segment_id],
        )
        if config.enabled and blocks and _can_merge(blocks[-1], block, segment, config):
            _merge_into(blocks[-1], block)
        else:
            blocks.append(block)
    for index, block in enumerate(blocks):
        block.index = index
    return blocks


def _can_merge(
    left: TranscriptBlock,
    right: TranscriptBlock,
    right_segment: SpeakerSegment,
    config: ParagraphConfig,
) -> bool:
    if left.raw_speaker != right.raw_speaker:
        return False
    if not left.text or not right.text:
        return False
    if left.text == FAILED_TEXT or right.text == FAILED_TEXT:
        return False
    if right_segment.status is SegmentStatus.FAILED:
        return False
    gap = right.start - left.end
    if gap < 0:
        gap = 0
    if gap > _gap(config):
        return False
    if right.end - left.start > _max_duration(config):
        return False
    if len(left.text) + len(right.text) > config.max_chars:
        return False
    if config.mode == "aggressive":
        return not (left.text.endswith(("?", "？")) and len(right.text) <= 12)
    return _soft_boundary(left.text, right.text) or gap <= (_gap(config) / 2)


def _gap(config: ParagraphConfig) -> float:
    if config.gap is not None:
        return config.gap
    return {"conservative": 1.0, "balanced": 2.0, "aggressive": 3.0}[config.mode]


def _max_duration(config: ParagraphConfig) -> float:
    if config.max_duration is not None:
        return config.max_duration
    return {"conservative": 75.0, "balanced": 120.0, "aggressive": 180.0}[config.mode]


def _soft_boundary(left: str, right: str) -> bool:
    return (not left.endswith(SENTENCE_ENDINGS)) or right.startswith(CONTINUATIONS)


def _merge_into(left: TranscriptBlock, right: TranscriptBlock) -> None:
    left.end = right.end
    left.text = _join_text(left.text, right.text)
    left.source_segment_ids.extend(right.source_segment_ids)


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left[-1].isascii() and right[0].isascii():
        return f"{left} {right}"
    return f"{left}{right}"
```

- [ ] **Step 4: Run paragraph tests**

Run:

```bash
uv run pytest tests/test_paragraphs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/paragraphs.py tests/test_paragraphs.py
git commit -m "feat: add transcript paragraph blocks"
```

---

### Task 2: Paragraph Configuration and CLI

**Files:**
- Modify: `src/mimo_transcriber/config.py`
- Modify: `src/mimo_transcriber/cli.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ParagraphConfig` from Task 1
- Produces: `AppConfig.paragraph_config() -> ParagraphConfig`
- Produces CLI flags: `--paragraph-mode`, `--paragraph-gap`, `--paragraph-max-duration`, `--paragraph-max-chars`, `--no-paragraph-merge`

- [ ] **Step 1: Write failing config tests**

Append to `tests/test_config.py`:

```python
import pytest

from mimo_transcriber.config import AppConfig, ConfigError


def test_paragraph_mode_off_disables_paragraph_config(tmp_path):
    config = AppConfig(input_path=tmp_path / "input.m4a", paragraph_mode="off")

    paragraph = config.paragraph_config()

    assert paragraph.enabled is False


def test_paragraph_validation_rejects_negative_gap(tmp_path):
    config = AppConfig(input_path=tmp_path / "input.m4a", paragraph_gap=-1)

    with pytest.raises(ConfigError, match="--paragraph-gap"):
        config.validate_arguments()


def test_paragraph_validation_rejects_non_positive_max_chars(tmp_path):
    config = AppConfig(input_path=tmp_path / "input.m4a", paragraph_max_chars=0)

    with pytest.raises(ConfigError, match="--paragraph-max-chars"):
        config.validate_arguments()
```

- [ ] **Step 2: Write failing CLI test**

Append to `tests/test_cli.py`:

```python
from mimo_transcriber.cli import build_parser


def test_cli_parses_paragraph_options() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--paragraph-mode", "aggressive",
        "--paragraph-gap", "2.5",
        "--paragraph-max-duration", "180",
        "--paragraph-max-chars", "1200",
    ])

    assert args.paragraph_mode == "aggressive"
    assert args.paragraph_gap == 2.5
    assert args.paragraph_max_duration == 180
    assert args.paragraph_max_chars == 1200


def test_cli_no_paragraph_merge_sets_mode_off() -> None:
    args = build_parser().parse_args(["meeting.m4a", "--no-paragraph-merge"])

    assert args.no_paragraph_merge is True
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_config.py tests/test_cli.py -q
```

Expected: FAIL because paragraph fields and parser args do not exist.

- [ ] **Step 4: Implement config fields**

Modify `src/mimo_transcriber/config.py`:

```python
from mimo_transcriber.paragraphs import ParagraphConfig

ParagraphMode = Literal["off", "conservative", "balanced", "aggressive"]

@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    output_path: Path | None = None
    num_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 6
    language: Language = "auto"
    device: Device = "auto"
    paragraph_mode: ParagraphMode = "balanced"
    paragraph_gap: float | None = None
    paragraph_max_duration: float | None = None
    paragraph_max_chars: int = 900

    def paragraph_config(self) -> ParagraphConfig:
        mode = "balanced" if self.paragraph_mode == "off" else self.paragraph_mode
        return ParagraphConfig(
            enabled=self.paragraph_mode != "off",
            mode=mode,
            gap=self.paragraph_gap,
            max_duration=self.paragraph_max_duration,
            max_chars=self.paragraph_max_chars,
        )
```

Extend `validate_arguments()`:

```python
        if self.paragraph_gap is not None and self.paragraph_gap < 0:
            raise ConfigError("--paragraph-gap 不能为负数")
        if self.paragraph_max_duration is not None and self.paragraph_max_duration <= 0:
            raise ConfigError("--paragraph-max-duration 必须大于 0")
        if self.paragraph_max_chars <= 0:
            raise ConfigError("--paragraph-max-chars 必须大于 0")
```

Do not add paragraph fields to `cache_parameters()`.

- [ ] **Step 5: Implement CLI fields**

Modify `src/mimo_transcriber/cli.py` parser:

```python
    parser.add_argument(
        "--paragraph-mode",
        choices=("off", "conservative", "balanced", "aggressive"),
        default="balanced",
    )
    parser.add_argument("--paragraph-gap", type=float)
    parser.add_argument("--paragraph-max-duration", type=float)
    parser.add_argument("--paragraph-max-chars", type=int, default=900)
    parser.add_argument("--no-paragraph-merge", action="store_true")
```

Before constructing `AppConfig`:

```python
    paragraph_mode = "off" if args.no_paragraph_merge else args.paragraph_mode
```

Pass into `AppConfig`:

```python
        paragraph_mode=paragraph_mode,
        paragraph_gap=args.paragraph_gap,
        paragraph_max_duration=args.paragraph_max_duration,
        paragraph_max_chars=args.paragraph_max_chars,
```

- [ ] **Step 6: Run tests**

Run:

```bash
uv run pytest tests/test_config.py tests/test_cli.py tests/test_paragraphs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mimo_transcriber/config.py src/mimo_transcriber/cli.py tests/test_config.py tests/test_cli.py
git commit -m "feat: add paragraph merge configuration"
```

---

### Task 3: Formatter Uses Display Blocks

**Files:**
- Modify: `src/mimo_transcriber/formatter.py`
- Test: `tests/test_formatter.py`

**Interfaces:**
- Consumes: `build_transcript_blocks(segments, config)`
- Consumes: `AppConfig.paragraph_config()`
- Produces: `render_transcript(outcome, recording_time, paragraph_config: ParagraphConfig | None = None) -> str`
- Produces: debug JSON `blocks` list when `write_outputs(outcome, recording_time, output_path, debug_json, paragraph_config=ParagraphConfig())` is called

- [ ] **Step 1: Write failing formatter tests**

Append to `tests/test_formatter.py`:

```python
from datetime import datetime
from pathlib import Path

from mimo_transcriber.formatter import render_transcript, write_outputs
from mimo_transcriber.models import AudioMetadata, SegmentStatus, SpeakerSegment, TranscriptionOutcome
from mimo_transcriber.paragraphs import ParagraphConfig


def outcome_with_segments(tmp_path: Path) -> TranscriptionOutcome:
    return TranscriptionOutcome(
        metadata=AudioMetadata(
            source_path=tmp_path / "meeting.m4a",
            duration_seconds=10,
            codec="aac",
            sample_rate=44100,
            channels=1,
            creation_time=None,
        ),
        segments=[
            SpeakerSegment(0, 0, 3, "A", "说话人 1", "第一句。", SegmentStatus.SUCCESS, segment_id="s0000"),
            SpeakerSegment(1, 3.5, 5, "A", "说话人 1", "然后第二句。", SegmentStatus.SUCCESS, segment_id="s0001"),
        ],
        keywords=[],
    )


def test_render_transcript_uses_paragraph_blocks(tmp_path: Path) -> None:
    text = render_transcript(
        outcome_with_segments(tmp_path),
        datetime(2026, 6, 24, 10, 0),
        ParagraphConfig(),
    )

    assert text.count("说话人 1 00:00") == 1
    assert "第一句。然后第二句。" in text
    assert "说话人 1 00:03" not in text


def test_write_outputs_debug_json_includes_blocks(tmp_path: Path) -> None:
    output = tmp_path / "out.txt"

    write_outputs(
        outcome_with_segments(tmp_path),
        datetime(2026, 6, 24, 10, 0),
        output,
        debug_json=True,
        paragraph_config=ParagraphConfig(),
    )

    debug = output.with_suffix(".segments.json").read_text(encoding="utf-8")
    assert '"segments"' in debug
    assert '"blocks"' in debug
    assert '"source_segment_ids": [' in debug
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_formatter.py -q
```

Expected: FAIL because formatter signatures do not accept `paragraph_config`.

- [ ] **Step 3: Update formatter signatures and rendering**

Modify `src/mimo_transcriber/formatter.py`:

```python
from dataclasses import asdict
from mimo_transcriber.paragraphs import ParagraphConfig, build_transcript_blocks


def render_transcript(
    outcome: TranscriptionOutcome,
    recording_time: datetime,
    paragraph_config: ParagraphConfig | None = None,
) -> str:
    first = (
        f"{format_recording_time(recording_time)}|"
        f"{format_duration(outcome.metadata.duration_seconds)}"
    )
    config = paragraph_config or ParagraphConfig(enabled=False)
    ordered = sorted(outcome.segments, key=lambda s: s.sort_key())
    blocks = build_transcript_blocks(ordered, config)
    rendered = [
        f"{block.display_speaker} {format_timestamp(block.start)}\n{block.text}"
        for block in blocks
    ]
    transcript = "\n\n".join(rendered)
    return (
        f"{first}\n\n关键词:\n{'、'.join(outcome.keywords)}\n\n"
        f"文字记录:\n{transcript}\n"
    )
```

Modify `write_outputs()` signature:

```python
def write_outputs(
    outcome: TranscriptionOutcome,
    recording_time: datetime,
    output_path: Path,
    debug_json: bool,
    paragraph_config: ParagraphConfig | None = None,
) -> None:
```

Use the same `blocks` in debug JSON:

```python
        payload = {
            "source": outcome.metadata.source_path.name,
            "duration_seconds": outcome.metadata.duration_seconds,
            "speakers": len({item.raw_speaker for item in outcome.segments}),
            "segments": [asdict(item) for item in outcome.segments],
            "blocks": [asdict(item) for item in build_transcript_blocks(outcome.segments, paragraph_config or ParagraphConfig(enabled=False))],
        }
```

- [ ] **Step 4: Run formatter tests**

Run:

```bash
uv run pytest tests/test_formatter.py tests/test_paragraphs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/formatter.py tests/test_formatter.py
git commit -m "feat: render transcript paragraph blocks"
```

---

### Task 4: Pipeline Passes Paragraph Config to Output

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `AppConfig.paragraph_config()`
- Consumes: `write_outputs(outcome, recording_time, output_path, debug_json, paragraph_config=ParagraphConfig())`
- Produces: pipeline final output using paragraph settings without changing ASR cache identity

- [ ] **Step 1: Write failing pipeline test**

Append to `tests/test_pipeline.py`:

```python
@pytest.mark.asyncio
async def test_pipeline_passes_paragraph_config_to_formatter(tmp_path, monkeypatch):
    captured = {}

    def fake_write_outputs(outcome, recording_time, output_path, debug_json, paragraph_config=None):
        captured["paragraph_config"] = paragraph_config
        output_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("mimo_transcriber.pipeline.write_outputs", fake_write_outputs)

    source = tmp_path / "input.m4a"
    source.write_bytes(b"audio")
    output = tmp_path / "output.txt"
    metadata = AudioMetadata(source, 1, "aac", 48000, 2, datetime(2026, 1, 1, 9))

    async def transcribe(items, fail_fast):
        segment = items[0][0]
        segment.text = "完成"
        segment.status = SegmentStatus.SUCCESS
        return [segment]

    dependencies = PipelineDependencies(
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

    await run_pipeline(
        AppConfig(
            input_path=source,
            output_path=output,
            num_speakers=1,
            paragraph_mode="aggressive",
        ),
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
        dependencies,
        cache_root=tmp_path,
    )

    assert captured["paragraph_config"].enabled is True
    assert captured["paragraph_config"].mode == "aggressive"
```

- [ ] **Step 2: Run pipeline test to verify failure**

Run:

```bash
uv run pytest tests/test_pipeline.py::test_pipeline_passes_paragraph_config_to_formatter -q
```

Expected: FAIL because `write_outputs` is called without paragraph config.

- [ ] **Step 3: Pass paragraph config in pipeline**

Modify the final output call in `src/mimo_transcriber/pipeline.py`:

```python
            write_outputs(
                outcome,
                recording_time,
                config.resolved_output_path,
                config.debug_json,
                paragraph_config=config.paragraph_config(),
            )
```

- [ ] **Step 4: Run focused and full relevant tests**

Run:

```bash
uv run pytest tests/test_pipeline.py tests/test_formatter.py tests/test_paragraphs.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/pipeline.py tests/test_pipeline.py
git commit -m "feat: apply paragraph config in pipeline output"
```

---

### Task 5: Documentation and Final Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: paragraph merge defaults, flags, and debug behavior

- [ ] **Step 1: Update README**

Add a section near CLI options:

```markdown
## 段落合并输出

CScribe 默认会在输出 TXT 前合并同一说话人的连续短片段。ASR 内部切片不会被改变，`--debug-json` 仍会保留内部片段，并额外输出最终展示 blocks。

常用选项：

```bash
uv run mimo-transcriber meeting.m4a --paragraph-mode conservative
uv run mimo-transcriber meeting.m4a --paragraph-mode aggressive --paragraph-gap 2.5
uv run mimo-transcriber meeting.m4a --no-paragraph-merge
```
```

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document paragraph merge output"
```

---

## Self-Review

- Spec coverage: paragraph block model, config/CLI, formatter, debug JSON, cache non-impact, and tests are covered.
- Placeholder scan: No deferred placeholders remain; pipeline tests include concrete fake dependencies and assertions.
- Type consistency: `ParagraphConfig`, `TranscriptBlock`, and `build_transcript_blocks` names match across tasks.
