# Transcript Quality Worktrees Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve CScribe transcript readability, technical term accuracy, and two-speaker attribution stability without changing the ASR model.

**Architecture:** Use three isolated worktrees so each improvement can be tested independently: display-only paragraph merging, MiMo prompt terminology injection, and diarization post-processing stabilization. Merge the smallest passing slices back to `main` in that order.

**Tech Stack:** Python 3.11, pytest, pyannote.audio, OpenAI-compatible MiMo chat completions, existing CScribe segment models.

## Global Constraints

- Keep audio slicing and ASR provider boundaries compatible with existing cache and manifest behavior.
- Add tests before production changes and verify each test fails before implementing.
- Avoid heavy external dependencies unless the benefit is proven by a failing test.
- Preserve local diarization; do not require a hosted diarization service.

---

### Task 1: Display Paragraph Blocks

**Files:**
- Create: `src/mimo_transcriber/paragraphs.py`
- Modify: `src/mimo_transcriber/formatter.py`
- Test: `tests/test_paragraphs.py`

**Interfaces:**
- Consumes: `SpeakerSegment`
- Produces: `TranscriptBlock` and `build_transcript_blocks(segments: list[SpeakerSegment]) -> list[TranscriptBlock]`

- [ ] **Step 1: Write failing tests for same-speaker continuation merging and speaker-change separation.**
- [ ] **Step 2: Run `uv run pytest tests/test_paragraphs.py -q` and confirm failures are due to missing module/functions.**
- [ ] **Step 3: Implement paragraph block creation and use it in `render_transcript`.**
- [ ] **Step 4: Run `uv run pytest tests/test_paragraphs.py tests/test_formatter.py -q`.**
- [ ] **Step 5: Commit on `codex/paragraph-blocks`.**

### Task 2: MiMo Prompt Terms

**Files:**
- Modify: `src/mimo_transcriber/asr/base.py`
- Modify: `src/mimo_transcriber/asr/mimo.py`
- Modify: `src/mimo_transcriber/asr/factory.py`
- Modify: `src/mimo_transcriber/config.py`
- Modify: `src/mimo_transcriber/cli.py`
- Test: `tests/test_mimo_asr.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: optional CLI `--asr-prompt` and `--terms-file`
- Produces: MiMo request `extra_body["asr_options"]["prompt"]` when prompt text is available

- [ ] **Step 1: Write failing tests proving prompt text reaches MiMo request body and CLI config.**
- [ ] **Step 2: Run targeted tests and confirm prompt support is missing.**
- [ ] **Step 3: Add prompt fields, terms-file loading, cache identity inclusion, and request wiring.**
- [ ] **Step 4: Run `uv run pytest tests/test_mimo_asr.py tests/test_cli.py tests/test_config.py -q`.**
- [ ] **Step 5: Commit on `codex/mimo-prompt-terms`.**

### Task 3: Diarization Stabilizer

**Files:**
- Create: `src/mimo_transcriber/diarization_stabilizer.py`
- Modify: `src/mimo_transcriber/pipeline.py`
- Test: `tests/test_diarization_stabilizer.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: diarization `SpeakerSegment` list after `process_segments`
- Produces: stabilized `SpeakerSegment` list before slicing and transcription

- [ ] **Step 1: Write failing tests for short speaker islands and highly overlapping duplicate speaker turns.**
- [ ] **Step 2: Run `uv run pytest tests/test_diarization_stabilizer.py -q` and confirm failures are due to missing module/functions.**
- [ ] **Step 3: Implement conservative same-context relabeling and overlap trimming/deduplication.**
- [ ] **Step 4: Run `uv run pytest tests/test_diarization_stabilizer.py tests/test_segments.py tests/test_pipeline.py -q`.**
- [ ] **Step 5: Commit on `codex/diarization-stabilizer`.**

### Task 4: Merge Back

**Files:**
- Merge outputs from the three branches into `main`

**Interfaces:**
- Produces: one integrated `main` with all passing tests

- [ ] **Step 1: Merge `codex/paragraph-blocks` into `main` and run paragraph/formatter tests.**
- [ ] **Step 2: Merge `codex/mimo-prompt-terms` into `main` and run MiMo/CLI/config tests.**
- [ ] **Step 3: Merge `codex/diarization-stabilizer` into `main` and run stabilizer/pipeline tests.**
- [ ] **Step 4: Run full `uv run pytest -q`.**
