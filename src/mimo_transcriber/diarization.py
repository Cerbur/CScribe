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
