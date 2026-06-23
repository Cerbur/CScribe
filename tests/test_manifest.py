from pathlib import Path

from mimo_transcriber.manifest import (
    ManifestStore,
    SegmentRecord,
    TaskManifest,
    TaskIdentity,
)
from mimo_transcriber.models import AudioMetadata, SpeakerSegment


def test_manifest_round_trip_preserves_segment_state(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    identity = TaskIdentity(task_hash="abc", fingerprint_size=100, fingerprint_mtime_ns=0, fingerprint_sha256="ff")
    metadata = AudioMetadata(Path("meeting.m4a"), 10, "aac", 48_000, 2, None)
    segment = SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    manifest = TaskManifest.new(identity, metadata)
    manifest.segments = [SegmentRecord.from_segment(segment)]
    store.save(manifest)
    assert store.load().segments[0].segment.segment_id == "s0000"


def test_invalid_ready_slice_is_reset_to_pending(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path / "manifest.json")
    manifest = TaskManifest.new(
        TaskIdentity(task_hash="abc", fingerprint_size=100, fingerprint_mtime_ns=0, fingerprint_sha256="ff"),
        AudioMetadata(Path("meeting.m4a"), 10, "aac", 48_000, 2, None),
    )
    record = SegmentRecord.from_segment(
        SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    )
    manifest.segments = [record]
    audio_path = tmp_path / "audio" / "s0000.mp3"
    audio_path.parent.mkdir(parents=True)
    record.slice_status = "ready"
    record.slice_bytes = 5
    audio_path.write_bytes(b"")
    store.reconcile_artifacts(manifest, tmp_path)
    assert record.slice_status == "pending"


def test_failed_transcript_becomes_pending_for_new_run() -> None:
    manifest = TaskManifest.new(
        TaskIdentity(task_hash="abc", fingerprint_size=100, fingerprint_mtime_ns=0, fingerprint_sha256="ff"),
        AudioMetadata(Path("meeting.m4a"), 10, "aac", 48_000, 2, None),
    )
    record = SegmentRecord.from_segment(
        SpeakerSegment(0, 0, 1, "A", segment_id="s0000")
    )
    manifest.segments = [record]
    record.transcript_status = "failed"
    ManifestStore.reset_retryable_work(manifest)
    assert record.transcript_status == "pending"
