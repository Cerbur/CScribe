from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mimo_transcriber.models import TranscriptionOutcome


def format_timestamp(seconds: float) -> str:
    total = math.floor(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{secs:02d}"
        if hours
        else f"{minutes:02d}:{secs:02d}"
    )


def format_duration(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    prefix = f"{hours}小时 " if hours else ""
    return f"{prefix}{minutes}分钟 {secs}秒"


def format_recording_time(value: datetime) -> str:
    local = value.astimezone() if value.tzinfo else value
    period = "上午" if local.hour < 12 else "下午"
    hour = local.hour % 12 or 12
    return f"{local.year}年{local.month}月{local.day}日 {period} {hour}:{local.minute:02d}"


def render_transcript(outcome: TranscriptionOutcome, recording_time: datetime) -> str:
    first = (
        f"{format_recording_time(recording_time)}|"
        f"{format_duration(outcome.metadata.duration_seconds)}"
    )
    ordered = sorted(outcome.segments, key=lambda s: s.sort_key())
    blocks = [
        f"{segment.display_speaker} {format_timestamp(segment.start)}\n{segment.text or ''}"
        for segment in ordered
    ]
    transcript = "\n\n".join(blocks)
    return (
        f"{first}\n\n关键词:\n{'、'.join(outcome.keywords)}\n\n"
        f"文字记录:\n{transcript}\n"
    )


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_outputs(
    outcome: TranscriptionOutcome,
    recording_time: datetime,
    output_path: Path,
    debug_json: bool,
) -> None:
    _atomic_write(output_path, render_transcript(outcome, recording_time))
    if debug_json:
        payload = {
            "source": outcome.metadata.source_path.name,
            "duration_seconds": outcome.metadata.duration_seconds,
            "speakers": len({item.raw_speaker for item in outcome.segments}),
            "segments": [asdict(item) for item in outcome.segments],
        }
        if outcome.speaker_stability is not None:
            payload["speaker_stability"] = asdict(outcome.speaker_stability)
        json_path = output_path.with_suffix(".segments.json")
        _atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
