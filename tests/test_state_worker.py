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
