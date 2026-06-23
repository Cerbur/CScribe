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
