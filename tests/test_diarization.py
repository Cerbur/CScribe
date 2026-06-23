from pathlib import Path
from types import SimpleNamespace

from mimo_transcriber.diarization import diarize_audio


class Annotation:
    def itertracks(self, yield_label: bool):
        assert yield_label is True
        yield SimpleNamespace(start=0.2, end=1.8), None, "SPEAKER_07"


class Pipeline:
    def __init__(self) -> None:
        self.kwargs = {}

    def __call__(self, path: str, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(speaker_diarization=Annotation())


def test_adapter_extracts_current_community_output() -> None:
    pipeline = Pipeline()
    result = diarize_audio(
        Path("normalized.wav"),
        token="secret",
        device="cpu",
        num_speakers=2,
        min_speakers=1,
        max_speakers=6,
        pipeline_factory=lambda token, device: pipeline,
    )
    assert [(item.start, item.end, item.raw_speaker) for item in result] == [
        (0.2, 1.8, "SPEAKER_07")
    ]
    assert pipeline.kwargs == {"num_speakers": 2}
