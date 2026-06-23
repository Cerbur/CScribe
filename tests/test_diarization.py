from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_transcriber.diarization import (
    DiarizationError,
    apply_diarization_pipeline,
)


class Annotation:
    def itertracks(self, yield_label: bool):
        assert yield_label is True
        yield SimpleNamespace(start=0.2, end=1.8), None, "SPEAKER_07"


class Pipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, int]]] = []

    def __call__(self, path: str, **kwargs):
        self.calls.append((path, kwargs))
        return SimpleNamespace(speaker_diarization=Annotation())


def test_adapter_uses_supplied_pipeline_and_exact_speaker_count() -> None:
    pipeline = Pipeline()
    result = apply_diarization_pipeline(
        Path("normalized.wav"),
        pipeline,
        num_speakers=2,
        min_speakers=1,
        max_speakers=6,
    )
    assert [(item.start, item.end, item.raw_speaker) for item in result] == [
        (0.2, 1.8, "SPEAKER_07")
    ]
    assert pipeline.calls == [("normalized.wav", {"num_speakers": 2})]


def test_adapter_uses_minimum_and_maximum_when_count_is_unknown() -> None:
    pipeline = Pipeline()
    apply_diarization_pipeline(
        Path("normalized.wav"),
        pipeline,
        num_speakers=None,
        min_speakers=2,
        max_speakers=4,
    )
    assert pipeline.calls == [
        ("normalized.wav", {"min_speakers": 2, "max_speakers": 4})
    ]


def test_adapter_wraps_pipeline_failure() -> None:
    def fail(path: str, **kwargs):
        raise RuntimeError("backend failed")

    with pytest.raises(DiarizationError, match="说话人分离失败"):
        apply_diarization_pipeline(Path("normalized.wav"), fail, 2, 1, 6)
