# Resumable Transcription Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TTY-aware progress display and a resumable, bounded pipeline that overlaps two FFmpeg slicers with concurrent MiMo transcription, persists verified work under `/tmp/cscribe`, and removes the cache after a fully successful output.

**Architecture:** Add stable segment identities, a versioned manifest store, deterministic task-cache paths, and a process lock. `run_pipeline` will recover completed stages, then connect two asynchronous slice workers to `--concurrency` transcription workers through a bounded queue; every durable transition is atomically persisted. A separate reporter translates pipeline events into Rich live output or ordinary non-TTY logs.

**Tech Stack:** Python 3.11, asyncio, dataclasses, pathlib, hashlib, JSON, Rich, pytest, pytest-asyncio, FFmpeg/ffprobe, existing OpenAI-compatible MiMo client.

## Global Constraints

- Python remains `>=3.11,<3.12`.
- Work directories live at `/tmp/cscribe/<task-hash>/`; recovery is best-effort because the OS may clear `/tmp`.
- Input fingerprints use absolute path, size, nanosecond mtime, and SHA-256 of the first and last 1 MiB; files smaller than 2 MiB are hashed in full.
- Cache identity includes speaker constraints, language, requested diarization device, keyword count, model IDs, audio constants, and processing-rule versions.
- Cache identity excludes concurrency, rate limit, retry count, verbose, fail-fast, and debug-json.
- Exactly two FFmpeg slice workers are allowed.
- The transcription queue capacity is `max(2, 2 * config.concurrency)`.
- MiMo performs one initial request plus at most three retries for timeout, connection, HTTP 429, and HTTP 5xx failures.
- Successful transcript segments are never requested again during recovery.
- Output ordering is `(start, end, segment_id)`, independent of completion order.
- Manifest, transcript cache files, and formal outputs use temporary-file-plus-`os.replace` atomic writes.
- A fully successful run removes its task directory and target index; partial failure or interruption preserves them.
- Existing unrelated worktree changes must not be staged or overwritten.

---

## File Structure

- Create `src/mimo_transcriber/progress.py`: progress protocol, no-op reporter, Rich TTY reporter, and logging fallback.
- Create `src/mimo_transcriber/cache.py`: input fingerprinting, deterministic task paths, target index, process lock, and safe cleanup.
- Create `src/mimo_transcriber/manifest.py`: versioned manifest dataclasses, JSON serialization, atomic persistence, artifact validation, and recovery mutations.
- Create `tests/test_progress.py`: TTY/non-TTY rendering and reporter fault-isolation tests.
- Create `tests/test_cache.py`: fingerprint, task identity, index cleanup, and live/stale lock tests.
- Create `tests/test_manifest.py`: round trips, atomic recovery, artifact validation, and status-transition tests.
- Modify `src/mimo_transcriber/models.py`: immutable `segment_id`, stable ordering, and JSON-safe segment conversion.
- Modify `src/mimo_transcriber/segments.py`: assign stable base IDs and derive child IDs during oversize splitting.
- Modify `src/mimo_transcriber/mimo_asr.py`: per-item transcription API plus retry/completion callbacks.
- Modify `src/mimo_transcriber/pipeline.py`: resumable stages, bounded slice/transcription workers, interruption behavior, output, and cleanup.
- Modify `src/mimo_transcriber/cli.py`: reporter lifecycle and task-in-progress error presentation.
- Modify `src/mimo_transcriber/config.py`: cache-identity snapshot and removal of the obsolete `keep_temp` behavior.
- Modify `src/mimo_transcriber/audio.py`: atomic slice targets and removal of random temporary-workspace ownership.
- Modify `src/mimo_transcriber/formatter.py`: stable ordering before output.
- Modify `pyproject.toml` and `uv.lock`: add Rich.
- Modify `README.md`: explain progress, automatic recovery, `/tmp` lifetime, concurrency, partial failure, and cleanup.
- Modify existing tests in `tests/test_models.py`, `tests/test_segments.py`, `tests/test_mimo_asr.py`, `tests/test_pipeline.py`, `tests/test_cli.py`, `tests/test_config.py`, and `tests/test_formatter.py`.

---

### Task 1: Stable Segment Identity and Ordering

**Files:**
- Modify: `src/mimo_transcriber/models.py`
- Modify: `src/mimo_transcriber/segments.py`
- Modify: `src/mimo_transcriber/formatter.py`
- Test: `tests/test_models.py`
- Test: `tests/test_segments.py`
- Test: `tests/test_formatter.py`

**Interfaces:**
- Produces: `SpeakerSegment.segment_id: str`
- Produces: `SpeakerSegment.sort_key() -> tuple[float, float, str]`
- Produces: `split_segment(segment: SpeakerSegment) -> list[SpeakerSegment]` with child IDs `<parent>.0` and `<parent>.1`
- Consumes: `process_segments(raw: list[SpeakerSegment], duration: float, min_duration: float = 0.4, merge_gap: float = 0.8, max_duration: float = 45.0) -> list[SpeakerSegment]` and `render_transcript(outcome: TranscriptionOutcome, recording_time: datetime) -> str`.

- [ ] **Step 1: Write failing identity and ordering tests**

```python
def test_process_segments_assigns_stable_ids() -> None:
    result = process_segments([seg(0, 1, "A"), seg(2, 3, "B")], 10)
    assert [item.segment_id for item in result] == ["s0000", "s0001"]


def test_split_segment_derives_child_ids() -> None:
    parent = SpeakerSegment(0, 0, 10, "A", segment_id="s0007")
    children = split_segment(parent)
    assert [item.segment_id for item in children] == ["s0007.0", "s0007.1"]


def test_render_transcript_uses_time_then_segment_id_order() -> None:
    metadata = AudioMetadata(Path("input.m4a"), 3, "aac", 48_000, 2, None)
    outcome = TranscriptionOutcome(metadata=metadata)
    outcome.segments = [
        SpeakerSegment(1, 2, 3, "B", "说话人 2", "后", segment_id="s0001"),
        SpeakerSegment(0, 0, 1, "A", "说话人 1", "先", segment_id="s0000"),
    ]
    rendered = render_transcript(outcome, datetime(2026, 6, 15, 10, 0))
    assert rendered.index("先") < rendered.index("后")
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_models.py tests/test_segments.py tests/test_formatter.py -q`

Expected: FAIL because `SpeakerSegment` has no `segment_id` and splitting does not derive IDs.

- [ ] **Step 3: Add stable IDs and ordering**

```python
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
    segment_id: str = ""

    def sort_key(self) -> tuple[float, float, str]:
        return (self.start, self.end, self.segment_id)
```

In `process_segments`, assign `item.segment_id = f"s{index:04d}"` together with the compatibility `index`. In `split_segment`, preserve all speaker fields and assign `.0` and `.1`. In `render_transcript`, iterate over `sorted(outcome.segments, key=SpeakerSegment.sort_key)`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_models.py tests/test_segments.py tests/test_formatter.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/models.py src/mimo_transcriber/segments.py src/mimo_transcriber/formatter.py tests/test_models.py tests/test_segments.py tests/test_formatter.py
git commit -m "refactor: add stable segment identities"
```

### Task 2: Deterministic Cache Identity and Process Lock

**Files:**
- Create: `src/mimo_transcriber/cache.py`
- Modify: `src/mimo_transcriber/config.py`
- Test: `tests/test_cache.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `InputFingerprint(path: str, size: int, mtime_ns: int, content_sha256: str)`
- Produces: `fingerprint_input(path: Path) -> InputFingerprint`
- Produces: `AppConfig.cache_parameters() -> dict[str, object]`
- Produces: `TaskPaths.for_run(config: AppConfig, fingerprint: InputFingerprint, root: Path = Path("/tmp/cscribe")) -> TaskPaths`
- Produces: `TaskLock.acquire()`, `TaskLock.release()`, and `TaskAlreadyRunningError`
- Consumes: stable constants `CACHE_SCHEMA_VERSION`, `PROCESSING_RULES_VERSION`, diarization model ID, and MiMo model ID.

- [ ] **Step 1: Write failing cache tests**

```python
def test_fingerprint_changes_when_edge_content_changes(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"a" * (2 * 1024 * 1024 + 8))
    before = fingerprint_input(source)
    source.write_bytes(b"b" + source.read_bytes()[1:])
    after = fingerprint_input(source)
    assert before.content_sha256 != after.content_sha256


def test_runtime_tuning_does_not_change_task_hash(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"
    source.write_bytes(b"audio")
    fingerprint = fingerprint_input(source)
    first = AppConfig(input_path=source, concurrency=2, max_retries=1)
    second = AppConfig(input_path=source, concurrency=8, max_retries=3)
    assert TaskPaths.for_run(first, fingerprint, tmp_path).task_hash == (
        TaskPaths.for_run(second, fingerprint, tmp_path).task_hash
    )


def test_second_live_lock_is_rejected(tmp_path: Path) -> None:
    first = TaskLock(tmp_path / "task.lock")
    second = TaskLock(tmp_path / "task.lock")
    first.acquire()
    with pytest.raises(TaskAlreadyRunningError):
        second.acquire()
    first.release()
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_cache.py tests/test_config.py -q`

Expected: FAIL because cache and lock APIs do not exist.

- [ ] **Step 3: Implement cache identity and task paths**

```python
@dataclass(frozen=True)
class TaskPaths:
    root: Path
    task_hash: str
    work_dir: Path
    manifest: Path
    lock: Path
    normalized: Path
    preflight: Path
    audio_dir: Path
    transcripts_dir: Path
    target_index: Path
```

Serialize identity JSON with `sort_keys=True` and compact separators, then SHA-256 the UTF-8 bytes. Normalize input/output with `Path.resolve()`. Make `cache_parameters()` include only the exact global-constraint fields; define fixed audio constants and processing version strings in one place.

- [ ] **Step 4: Implement live/stale process locking**

Use atomic `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)`. Store:

```json
{"pid": 1234, "process_started": 1719123456.25, "run_id": "uuid"}
```

Read process start time on macOS/Linux through an injected `process_probe(pid) -> float | None`; default implementation may use `ps -o lstart= -p PID` parsed to a timestamp. Treat a missing PID, unparsable record, or mismatched start time as stale, unlink it, and retry acquisition once.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/test_cache.py tests/test_config.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/cache.py src/mimo_transcriber/config.py tests/test_cache.py tests/test_config.py
git commit -m "feat: add deterministic task cache and lock"
```

### Task 3: Versioned Atomic Manifest Store

**Files:**
- Create: `src/mimo_transcriber/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Consumes: `InputFingerprint`, `TaskPaths`, `SpeakerSegment`, and `SegmentStatus`
- Produces: `ArtifactStatus = Literal["pending", "ready", "failed"]`
- Produces: `SegmentRecord(segment: SpeakerSegment, slice_status, slice_bytes, transcript_status, text, error)`
- Produces: `TaskManifest.new(identity: TaskIdentity, metadata: AudioMetadata) -> TaskManifest`
- Produces: `ManifestStore.load() -> TaskManifest`, `save(manifest: TaskManifest) -> None`, `reconcile_artifacts(manifest: TaskManifest, paths: TaskPaths) -> None`, and `reset_retryable_work(manifest: TaskManifest) -> None`

- [ ] **Step 1: Write failing manifest tests**

```python
def test_manifest_round_trip_preserves_segment_state(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    identity = TaskIdentity.for_test(task_hash="abc")
    metadata = AudioMetadata(Path("meeting.m4a"), 10, "aac", 48_000, 2, None)
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    manifest = TaskManifest.new(identity, metadata)
    manifest.segments = [SegmentRecord.from_segment(segment)]
    store.save(manifest)
    assert store.load().segments[0].segment.segment_id == "s0000"


def test_invalid_ready_slice_is_reset_to_pending(tmp_path: Path) -> None:
    paths = TaskPaths.for_test(tmp_path)
    store = ManifestStore(paths.manifest)
    manifest = TaskManifest.new(
        TaskIdentity.for_test(task_hash="abc"),
        AudioMetadata(Path("meeting.m4a"), 10, "aac", 48_000, 2, None),
    )
    record = SegmentRecord.from_segment(
        SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    )
    manifest.segments = [record]
    audio_path = paths.audio_dir / "s0000.mp3"
    audio_path.parent.mkdir(parents=True)
    record.slice_status = "ready"
    record.slice_bytes = 5
    audio_path.write_bytes(b"")
    store.reconcile_artifacts(manifest, paths)
    assert record.slice_status == "pending"


def test_failed_transcript_becomes_pending_for_new_run() -> None:
    manifest = TaskManifest.new(
        TaskIdentity.for_test(task_hash="abc"),
        AudioMetadata(Path("meeting.m4a"), 10, "aac", 48_000, 2, None),
    )
    record = SegmentRecord.from_segment(
        SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    )
    manifest.segments = [record]
    record.transcript_status = "failed"
    ManifestStore.reset_retryable_work(manifest)
    assert record.transcript_status == "pending"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_manifest.py -q`

Expected: FAIL because manifest types do not exist.

- [ ] **Step 3: Implement JSON-safe records and atomic save**

Use explicit `to_dict()`/`from_dict()` methods; do not serialize dataclasses with paths or enums implicitly. Atomic save:

```python
temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
with temporary.open("rb") as stream:
    os.fsync(stream.fileno())
os.replace(temporary, path)
```

Persist a standalone transcript JSON after each final segment result before marking the manifest record complete. Reconcile `ready` slices against non-zero size and recorded byte count; reconcile successful transcripts against a valid JSON file containing the same `segment_id` and non-empty text.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_manifest.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/manifest.py tests/test_manifest.py
git commit -m "feat: persist resumable task manifests"
```

### Task 4: TTY-Aware Progress Reporter

**Files:**
- Create: `src/mimo_transcriber/progress.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/test_progress.py`

**Interfaces:**
- Produces: `ProgressReporter` protocol with `start_stage`, `set_segment_total`, `segment_sliced`, `segment_completed`, `segment_retrying`, `finish`, and `close`
- Produces: `NullProgressReporter`
- Produces: `TerminalProgressReporter(stderr: TextIO, is_tty: bool | None = None)`
- Consumes: Rich `Progress`, `SpinnerColumn`, `TextColumn`, and `TimeElapsedColumn`.

- [ ] **Step 1: Write failing reporter tests**

```python
def test_non_tty_emits_plain_stage_and_retry_logs(caplog) -> None:
    reporter = TerminalProgressReporter(io.StringIO(), is_tty=False)
    reporter.start_stage("说话人分离")
    reporter.segment_retrying("s0007", 2, 3)
    assert "说话人分离" in caplog.text
    assert "s0007 重试 2/3" in caplog.text


def test_tty_renders_slice_and_transcript_counts() -> None:
    stream = io.StringIO()
    reporter = TerminalProgressReporter(stream, is_tty=True)
    reporter.set_segment_total(3)
    reporter.segment_sliced()
    reporter.segment_completed(True)
    reporter.close()
    assert "切片 1/3" in stream.getvalue()
    assert "转写 1/3" in stream.getvalue()
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_progress.py -q`

Expected: FAIL because `progress.py` does not exist.

- [ ] **Step 3: Add Rich and implement reporter**

Add `"rich>=14.0"` to project dependencies and run `uv lock`. Before staging `uv.lock`, compare it with the pre-task worktree version and preserve every pre-existing dependency change. Keep all mutable counts behind a `threading.Lock`, because synchronous slice callbacks may arrive through `asyncio.to_thread`. Wrap rendering calls in `_safe(action: Callable[[], None]) -> None`; on Rich failure, stop live rendering and switch to logging without raising into the pipeline.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_progress.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/mimo_transcriber/progress.py tests/test_progress.py
git commit -m "feat: add terminal progress reporting"
```

### Task 5: Observable Per-Segment MiMo Retries

**Files:**
- Modify: `src/mimo_transcriber/mimo_asr.py`
- Test: `tests/test_mimo_asr.py`

**Interfaces:**
- Consumes: `ProgressReporter`
- Produces: `MiMoTranscriber.transcribe_one(segment: SpeakerSegment, path: Path) -> SpeakerSegment`
- Retains: `transcribe_all(items: list[tuple[SpeakerSegment, Path]], fail_fast: bool) -> list[SpeakerSegment]` as a compatibility wrapper during migration.

- [ ] **Step 1: Write failing retry-event tests**

```python
@pytest.mark.asyncio
async def test_retry_events_report_one_through_three(tmp_path: Path) -> None:
    reporter = RecordingReporter()
    class HttpError(RuntimeError):
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def completion(text: str) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    async def no_sleep(seconds: float) -> None:
        return None

    request = AsyncMock(side_effect=[
        TimeoutError("1"),
        ConnectionError("2"),
        HttpError(500),
        completion("ok"),
    ])
    audio_path = tmp_path / "s0000.mp3"
    audio_path.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    transcriber = MiMoTranscriber(
        request=request,
        language="auto",
        concurrency=1,
        requests_per_minute=1000,
        max_retries=3,
        reporter=reporter,
        sleep=no_sleep,
    )
    result = await transcriber.transcribe_one(segment, audio_path)
    assert reporter.retries == [
        ("s0000", 1, 3),
        ("s0000", 2, 3),
        ("s0000", 3, 3),
    ]
    assert result.status is SegmentStatus.SUCCESS


@pytest.mark.asyncio
async def test_empty_response_is_not_retried(tmp_path: Path) -> None:
    request = AsyncMock(return_value=completion(" "))
    audio_path = tmp_path / "s0000.mp3"
    audio_path.write_bytes(b"audio")
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    transcriber = MiMoTranscriber(
        request=request,
        language="auto",
        concurrency=1,
        requests_per_minute=1000,
        max_retries=3,
        reporter=RecordingReporter(),
        sleep=no_sleep,
    )
    result = await transcriber.transcribe_one(segment, audio_path)
    assert request.await_count == 1
    assert result.status is SegmentStatus.FAILED
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_mimo_asr.py -q`

Expected: FAIL because reporter events and public `transcribe_one` do not exist.

- [ ] **Step 3: Implement per-segment API and events**

Move `_one` behavior to `transcribe_one`. Before each retry sleep call:

```python
retry_number = attempt + 1
self.reporter.segment_retrying(
    segment.segment_id,
    retry_number,
    self.max_retries,
)
```

Call `segment_completed(True/False)` exactly once after the final result. Keep retry classification unchanged: timeout, connection, 429, and integer status `>=500`. Sort the compatibility `transcribe_all` result by `SpeakerSegment.sort_key`.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_mimo_asr.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/mimo_asr.py tests/test_mimo_asr.py
git commit -m "feat: report segment retries and completion"
```

### Task 6: Recoverable Probe, Normalize, and Diarization Stages

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `src/mimo_transcriber/audio.py`
- Modify: `src/mimo_transcriber/manifest.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `TaskPaths`, `TaskLock`, `ManifestStore`, and `ProgressReporter`
- Produces: `load_or_probe_metadata(config, manifest, store, dependencies, reporter) -> AudioMetadata`
- Produces: `load_or_normalize_audio(config, paths, manifest, store, dependencies, reporter) -> Path`
- Produces: `load_or_diarize_segments(config, paths, manifest, store, dependencies, reporter, hf_token) -> list[SpeakerSegment]`
- Removes: random `workspace(keep)` ownership from the main pipeline.

- [ ] **Step 1: Write failing recovery tests**

```python
@pytest.mark.asyncio
async def test_recovery_skips_ready_normalization_and_diarization(tmp_path: Path) -> None:
    with pytest.raises(asyncio.CancelledError):
        await run_pipeline(config, "mimo", "hf", dependencies, cache_root=tmp_path)
    dependencies.normalize.reset_mock()
    dependencies.diarize.reset_mock()
    await run_pipeline(config, "mimo", "hf", recovering_dependencies, cache_root=tmp_path)
    dependencies.normalize.assert_not_called()
    dependencies.diarize.assert_not_called()


@pytest.mark.asyncio
async def test_changed_input_creates_fresh_task_and_cleans_old_cache(tmp_path: Path) -> None:
    with pytest.raises(asyncio.CancelledError):
        await run_pipeline(config, "mimo", "hf", dependencies, cache_root=tmp_path)
    old_work_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    source.write_bytes(b"changed audio")
    await run_pipeline(config, "mimo", "hf", successful_dependencies, cache_root=tmp_path)
    assert not old_work_dir.exists()
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_pipeline.py -k "recovery or changed_input" -q`

Expected: FAIL because the pipeline always creates a fresh temporary workspace.

- [ ] **Step 3: Implement staged recovery**

At startup:

```python
fingerprint = fingerprint_input(config.input_path)
paths = TaskPaths.for_run(config, fingerprint, cache_root)
with TaskLock(paths.lock):
    store = ManifestStore(paths.manifest)
    manifest = store.load_valid(config, fingerprint) or store.initialize(
        config,
        fingerprint,
    )
```

For each stage, validate artifact and manifest state before reuse. Mark a stage `ready` only after its output is atomically installed and then persisted. Serialize processed diarization segments, not third-party pyannote objects. Reporter stages must say whether work is being recovered or performed.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_pipeline.py -k "recovery or changed_input or normalizes" -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/pipeline.py src/mimo_transcriber/audio.py src/mimo_transcriber/manifest.py tests/test_pipeline.py
git commit -m "feat: recover preprocessing stages"
```

### Task 7: Bounded Slice-to-Transcription Pipeline

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `src/mimo_transcriber/audio.py`
- Modify: `src/mimo_transcriber/manifest.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `run_segment_workers(config, normalized, manifest, store, paths, dependencies, transcriber, reporter) -> list[SpeakerSegment]`
- Consumes: `MiMoTranscriber.transcribe_one`, `ManifestStore`, and reporter events.
- Guarantees: two slice workers, `max(2, 2 * concurrency)` queue capacity, stable child IDs, and immediate durable transcript writes.

- [ ] **Step 1: Write failing concurrency and overlap tests**

```python
@pytest.mark.asyncio
async def test_slicing_is_capped_at_two_and_overlaps_transcription(
    tmp_path: Path,
) -> None:
    active_slices = 0
    max_active_slices = 0
    first_transcription_started = asyncio.Event()
    release_slices = asyncio.Event()
    async def transcribe_one(segment: SpeakerSegment, path: Path) -> SpeakerSegment:
        first_transcription_started.set()
        segment.text = "ok"
        segment.status = SegmentStatus.SUCCESS
        return segment

    await asyncio.wait_for(first_transcription_started.wait(), timeout=1)
    assert first_transcription_started.is_set()
    assert max_active_slices == 2


@pytest.mark.asyncio
async def test_successful_cached_transcript_is_not_requested_again(
    tmp_path: Path,
) -> None:
    with pytest.raises(asyncio.CancelledError):
        await run_pipeline(config, "mimo", "hf", interrupting_dependencies, cache_root=tmp_path)
    await run_pipeline(config, "mimo", "hf", recovering_dependencies, cache_root=tmp_path)
    assert request_counts["s0000"] == 1


@pytest.mark.asyncio
async def test_oversize_slice_expands_stable_child_ids(tmp_path: Path) -> None:
    result = await run_segment_workers(
        config,
        normalized,
        manifest,
        store,
        paths,
        oversize_dependencies,
        transcriber,
        NullProgressReporter(),
    )
    assert [item.segment_id for item in result] == ["s0000.0", "s0000.1", "s0001"]
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_pipeline.py -k "slicing_is_capped or cached_transcript or oversize_slice" -q`

Expected: FAIL because slicing is synchronous and all slicing precedes transcription.

- [ ] **Step 3: Implement bounded workers**

Create:

```python
slice_queue: asyncio.Queue[SegmentRecord | None] = asyncio.Queue()
transcribe_queue: asyncio.Queue[tuple[SegmentRecord, Path] | None] = asyncio.Queue(
    maxsize=max(2, 2 * config.concurrency)
)
```

Start exactly two slice workers and `config.concurrency` transcription workers. Run blocking FFmpeg calls with `asyncio.to_thread`. A slice worker atomically writes `audio/<segment_id>.mp3`, validates payload size, persists ready status, reports `segment_sliced`, then awaits `transcribe_queue.put(...)`. A transcription worker persists `transcripts/<segment_id>.json` before marking the manifest record successful.

For oversize slices, delete the candidate, split to child records, atomically replace the parent record under an `asyncio.Lock`, update reporter total, and enqueue children back to `slice_queue`. Never publish the parent to transcription.

- [ ] **Step 4: Implement per-segment failure continuation**

On FFmpeg failure, set the record slice status to `failed`, transcript status to `failed`, text to `[该片段识别失败]`, persist, and report one failed completion. On MiMo final failure, persist the same placeholder while retaining the valid slice. If `fail_fast` is true, set a shared cancellation event, stop producing new work, drain/cancel workers, and raise after durable state is saved.

- [ ] **Step 5: Run focused and full pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/pipeline.py src/mimo_transcriber/audio.py src/mimo_transcriber/manifest.py tests/test_pipeline.py
git commit -m "feat: pipeline slicing into concurrent transcription"
```

### Task 8: Output, Cleanup, Interruption, and CLI Lifecycle

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `src/mimo_transcriber/cli.py`
- Modify: `src/mimo_transcriber/config.py`
- Modify: `src/mimo_transcriber/formatter.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_formatter.py`

**Interfaces:**
- Consumes: `TerminalProgressReporter`, `TaskAlreadyRunningError`, task cleanup APIs.
- Produces: final exit codes `0`, `1`, and `2` with existing meanings.
- Removes: `--keep-temp` and `AppConfig.keep_temp`, because successful cache cleanup is now mandatory and partial cache retention is automatic.

- [ ] **Step 1: Write failing lifecycle tests**

```python
@pytest.mark.asyncio
async def test_success_removes_task_cache_but_keeps_debug_json_when_requested(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        input_path=source,
        output_path=output,
        debug_json=True,
    )
    task_paths = TaskPaths.for_run(config, fingerprint_input(source), tmp_path)
    result = await run_pipeline(
        config,
        "mimo",
        "hf",
        successful_dependencies,
        cache_root=tmp_path,
    )
    assert result.exit_code == 0
    assert output.exists()
    assert output.with_suffix(".segments.json").exists()
    assert not task_paths.work_dir.exists()
    assert not task_paths.target_index.exists()


@pytest.mark.asyncio
async def test_partial_failure_keeps_cache_and_returns_two(tmp_path: Path) -> None:
    task_paths = TaskPaths.for_run(config, fingerprint_input(source), tmp_path)
    result = await run_pipeline(
        config,
        "mimo",
        "hf",
        partial_failure_dependencies,
        cache_root=tmp_path,
    )
    assert result.exit_code == 2
    assert task_paths.work_dir.exists()
    assert "[该片段识别失败]" in output.read_text()


def test_cli_rejects_second_active_task(caplog) -> None:
    assert asyncio.run(async_main(["meeting.m4a"])) == 1
    assert "任务正在运行" in caplog.text
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py tests/test_formatter.py -k "removes_task_cache or keeps_cache or active_task" -q`

Expected: FAIL because cleanup and reporter lifecycle are not integrated.

- [ ] **Step 3: Implement final output and cleanup ordering**

Build the outcome from manifest records, atomically write TXT and optional debug JSON, then:

```python
if outcome.has_failures:
    store.mark_output_written(partial=True)
else:
    store.mark_output_written(partial=False)
    cleanup_task(paths)
```

Never remove cache before both requested formal outputs are durable. Remove obsolete `--keep-temp` parser/config/README behavior.

- [ ] **Step 4: Integrate reporter and interruption handling**

Construct `TerminalProgressReporter(sys.stderr)` in `async_main`, pass it into `run_pipeline`, and close it in `finally`. Catch `TaskAlreadyRunningError` with a concise message. Catch `KeyboardInterrupt`/`asyncio.CancelledError` at the pipeline boundary only after worker cancellation, manifest flush, and lock release; return exit code `130` for user interruption without deleting cache.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py tests/test_formatter.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/pipeline.py src/mimo_transcriber/cli.py src/mimo_transcriber/config.py src/mimo_transcriber/formatter.py tests/test_pipeline.py tests/test_cli.py tests/test_formatter.py
git commit -m "feat: finalize resumable CLI lifecycle"
```

### Task 9: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: completed public CLI behavior.
- Produces: user documentation and final regression evidence.

- [ ] **Step 1: Add an end-to-end fake-dependency recovery test**

The test must execute these exact phases against a temporary cache root:

```python
with pytest.raises(asyncio.CancelledError):
    await run_pipeline(
        config,
        "mimo",
        "hf",
        dependencies_that_interrupt_after_one_success,
        cache_root=tmp_path,
    )
second = await run_pipeline(
    config,
    "mimo",
    "hf",
    successful_dependencies,
    cache_root=tmp_path,
)

assert normalize.call_count == 1
assert diarize.call_count == 1
assert request_counts == {"s0000": 1, "s0001": 2}
assert second.exit_code == 0
assert output.exists()
assert not task_paths.work_dir.exists()
```

- [ ] **Step 2: Document the final behavior**

Update README with:

- TTY dynamic display and non-TTY logging.
- Two concurrent FFmpeg slice workers and `--concurrency` MiMo workers.
- Automatic recovery by rerunning the same command.
- `/tmp/cscribe` best-effort lifetime and OS cleanup caveat.
- Successful segment reuse and fresh retry budget for failed segments.
- Partial TXT with exit code `2`, retained cache, and next-run retry.
- Successful cache cleanup and optional retained `.segments.json`.
- Same-task lock rejection.
- Removal of `--keep-temp`.

- [ ] **Step 3: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests pass; FFmpeg-dependent tests may be skipped only when FFmpeg/ffprobe are unavailable.

- [ ] **Step 4: Run static repository checks**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `uv run python -m mimo_transcriber --help`

Expected: exit code `0`; help includes current CLI options and does not include `--keep-temp`.

- [ ] **Step 5: Perform a local fake/no-network smoke test**

Run the end-to-end pipeline test with verbose output:

`uv run pytest tests/test_pipeline.py -k "end_to_end_resume" -vv -s`

Expected: first run preserves cache after interruption; second run reports recovered stages, skips the successful segment, writes output, and deletes cache.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_pipeline.py tests/test_cli.py
git commit -m "docs: explain resumable transcription workflow"
```

### Task 10: Final Regression and Manual Acceptance

**Files:**
- No code changes expected.

**Interfaces:**
- Verifies: all requirements from the approved design.

- [ ] **Step 1: Run the complete suite once more from a clean process**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Verify only intended files changed**

Run: `git status --short`

Expected: only pre-existing unrelated user changes remain; all plan implementation commits are clean.

- [ ] **Step 3: Run a real recording acceptance test when credentials are available**

Run:

```bash
uv run python -m mimo_transcriber "新录音 14.m4a" --num-speakers 2 --verbose
```

Expected:

- A single live TTY line shows diarization elapsed time, then slice/transcription counts.
- No more than two FFmpeg slice processes run concurrently.
- Output TXT is generated.
- A fully successful run leaves no matching task directory under `/tmp/cscribe`.
- If one segment exhausts retries, exit code is `2`, the TXT contains `[该片段识别失败]`, and rerunning the same command retries only failed work.

- [ ] **Step 4: Record acceptance evidence**

Append the exact command, exit code, output path, success/failure counts, and whether `/tmp/cscribe` was cleaned to the implementation handoff message. Do not commit credentials, audio, generated transcript, or cache artifacts.
