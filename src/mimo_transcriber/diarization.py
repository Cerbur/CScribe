from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mimo_transcriber.models import SpeakerSegment

MODEL_ID = "pyannote/speaker-diarization-community-1"


class DiarizationError(RuntimeError):
    pass


def _default_pipeline(token: str, device: str) -> Any:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    pipeline.to(torch.device(device))
    return pipeline


def diarize_audio(
    path: Path,
    token: str,
    device: str,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    pipeline_factory: Callable[[str, str], Any] = _default_pipeline,
) -> list[SpeakerSegment]:
    pipeline = pipeline_factory(token, device)
    kwargs = (
        {"num_speakers": num_speakers}
        if num_speakers is not None
        else {"min_speakers": min_speakers, "max_speakers": max_speakers}
    )
    try:
        output = pipeline(str(path), **kwargs)
        annotation = getattr(output, "speaker_diarization", output)
        return [
            SpeakerSegment(-1, float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
    except Exception as exc:
        raise DiarizationError(f"说话人分离失败: {exc}") from exc
