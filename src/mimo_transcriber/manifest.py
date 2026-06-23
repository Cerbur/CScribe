from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mimo_transcriber.models import AudioMetadata, SegmentStatus, SpeakerSegment

logger = logging.getLogger(__name__)

ArtifactStatus = Literal["pending", "ready", "failed"]


def _mtime_or_now(source_path: Path) -> str:
    """Return the file's mtime as an ISO string, or now() if unavailable."""
    try:
        mtime = source_path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


@dataclass
class TaskIdentity:
    task_hash: str
    fingerprint_size: int
    fingerprint_mtime_ns: int
    fingerprint_sha256: str
    schema_version: int = 1
    processing_version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "task_hash": self.task_hash,
            "fingerprint_size": self.fingerprint_size,
            "fingerprint_mtime_ns": self.fingerprint_mtime_ns,
            "fingerprint_sha256": self.fingerprint_sha256,
            "schema_version": self.schema_version,
            "processing_version": self.processing_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskIdentity:
        return cls(
            task_hash=data["task_hash"],
            fingerprint_size=data["fingerprint_size"],
            fingerprint_mtime_ns=data["fingerprint_mtime_ns"],
            fingerprint_sha256=data["fingerprint_sha256"],
            schema_version=data.get("schema_version", 1),
            processing_version=data.get("processing_version", 1),
        )


@dataclass
class SegmentRecord:
    segment: SpeakerSegment
    slice_status: ArtifactStatus = "pending"
    slice_bytes: int = 0
    transcript_status: ArtifactStatus = "pending"
    text: str | None = None
    error: str | None = None

    @classmethod
    def from_segment(cls, segment: SpeakerSegment) -> SegmentRecord:
        return cls(
            segment=segment,
            slice_status="pending",
            slice_bytes=0,
            transcript_status="pending",
            text=None,
            error=None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "segment": {
                "index": self.segment.index,
                "start": self.segment.start,
                "end": self.segment.end,
                "raw_speaker": self.segment.raw_speaker,
                "display_speaker": self.segment.display_speaker,
                "text": self.segment.text,
                "status": str(self.segment.status),
                "error": self.segment.error,
                "segment_id": self.segment.segment_id,
            },
            "slice_status": self.slice_status,
            "slice_bytes": self.slice_bytes,
            "transcript_status": self.transcript_status,
            "text": self.text,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentRecord:
        seg_data = data["segment"]
        segment = SpeakerSegment(
            index=seg_data["index"],
            start=seg_data["start"],
            end=seg_data["end"],
            raw_speaker=seg_data["raw_speaker"],
            display_speaker=seg_data.get("display_speaker"),
            text=seg_data.get("text"),
            status=SegmentStatus(seg_data.get("status", "pending")),
            error=seg_data.get("error"),
            segment_id=seg_data.get("segment_id", ""),
        )
        return cls(
            segment=segment,
            slice_status=data.get("slice_status", "pending"),
            slice_bytes=data.get("slice_bytes", 0),
            transcript_status=data.get("transcript_status", "pending"),
            text=data.get("text"),
            error=data.get("error"),
        )


@dataclass
class TaskManifest:
    identity: TaskIdentity
    metadata_source_path: str
    metadata_duration: float
    metadata_codec: str
    metadata_sample_rate: int
    metadata_channels: int
    metadata_creation_time: str | None
    normalize_status: ArtifactStatus = "pending"
    diarization_status: ArtifactStatus = "pending"
    diarization_device: str | None = None
    output_written: bool = False
    segments: list[SegmentRecord] = field(default_factory=list)
    created_at: str = ""

    @classmethod
    def new(cls, identity: TaskIdentity, metadata: AudioMetadata) -> TaskManifest:
        return cls(
            identity=identity,
            metadata_source_path=str(metadata.source_path),
            metadata_duration=metadata.duration_seconds,
            metadata_codec=metadata.codec,
            metadata_sample_rate=metadata.sample_rate,
            metadata_channels=metadata.channels,
            metadata_creation_time=(
                metadata.creation_time.isoformat() if metadata.creation_time else None
            ),
            created_at=_mtime_or_now(metadata.source_path),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "metadata": {
                "source_path": self.metadata_source_path,
                "duration": self.metadata_duration,
                "codec": self.metadata_codec,
                "sample_rate": self.metadata_sample_rate,
                "channels": self.metadata_channels,
                "creation_time": self.metadata_creation_time,
            },
            "normalize_status": self.normalize_status,
            "diarization_status": self.diarization_status,
            "diarization_device": self.diarization_device,
            "output_written": self.output_written,
            "segments": [item.to_dict() for item in self.segments],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskManifest:
        meta = data["metadata"]
        return cls(
            identity=TaskIdentity.from_dict(data["identity"]),
            metadata_source_path=meta["source_path"],
            metadata_duration=meta["duration"],
            metadata_codec=meta["codec"],
            metadata_sample_rate=meta["sample_rate"],
            metadata_channels=meta["channels"],
            metadata_creation_time=meta.get("creation_time"),
            normalize_status=data.get("normalize_status", "pending"),
            diarization_status=data.get("diarization_status", "pending"),
            diarization_device=data.get("diarization_device"),
            output_written=data.get("output_written", False),
            segments=[SegmentRecord.from_dict(item) for item in data.get("segments", [])],
            created_at=data.get("created_at", ""),
        )


class ManifestStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> TaskManifest:
        with open(self.path, "r", encoding="utf-8") as stream:
            return TaskManifest.from_dict(json.load(stream))

    def save(self, manifest: TaskManifest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)
        tmp.write_text(payload + "\n", encoding="utf-8")
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)

    @staticmethod
    def reconcile_artifacts(manifest: TaskManifest, paths: Path) -> None:
        audio_dir = Path(paths) / "audio"
        for record in manifest.segments:
            if record.slice_status == "ready":
                slice_path = audio_dir / f"{record.segment.segment_id}.mp3"
                if not slice_path.exists() or slice_path.stat().st_size == 0:
                    logger.debug("切片 %s 缺失或为空，重置为 pending", record.segment.segment_id)
                    record.slice_status = "pending"
                    record.slice_bytes = 0
                elif record.slice_bytes > 0 and slice_path.stat().st_size != record.slice_bytes:
                    logger.debug("切片 %s 大小不匹配，重置为 pending", record.segment.segment_id)
                    record.slice_status = "pending"
                    record.slice_bytes = 0

    @staticmethod
    def reset_retryable_work(manifest: TaskManifest) -> None:
        for record in manifest.segments:
            if record.transcript_status == "failed":
                record.transcript_status = "pending"
                record.error = None
            if record.slice_status == "failed":
                record.slice_status = "pending"
                record.slice_bytes = 0
                record.error = None
