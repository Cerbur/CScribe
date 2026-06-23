from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_transcriber.devices import DeviceCapabilities
from mimo_transcriber.diarization import (
    DiarizationError,
    apply_diarization_pipeline,
    classify_mps_failure,
    select_diarization_pipeline,
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


def capabilities(*, built: bool, available: bool) -> DeviceCapabilities:
    return DeviceCapabilities(
        cuda_available=False,
        mps_built=built,
        mps_available=available,
        platform="Darwin",
        machine="arm64",
    )


def test_unbuilt_mps_falls_back_without_constructing_mps() -> None:
    calls: list[str] = []
    cpu = Pipeline()

    def factory(token: str, device: str):
        calls.append(device)
        assert token == "secret"
        return cpu

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=False, available=False),
        pipeline_factory=factory,
    )

    assert calls == ["cpu"]
    assert selection.pipeline is cpu
    assert selection.decision.selected_device == "cpu"
    assert selection.decision.fallback_category == "not_built"


def test_unavailable_mps_runtime_falls_back_to_cpu() -> None:
    calls: list[str] = []

    def factory(token: str, device: str):
        calls.append(device)
        return Pipeline()

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=False),
        pipeline_factory=factory,
    )

    assert calls == ["cpu"]
    assert selection.decision.fallback_category == "runtime_unavailable"


def test_successful_preflight_returns_same_mps_pipeline() -> None:
    mps = Pipeline()
    times = iter([10.0, 12.5])

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=lambda token, device: mps,
        clock=lambda: next(times),
    )

    assert selection.pipeline is mps
    assert mps.calls == [("preflight.wav", {"num_speakers": 2})]
    assert selection.decision.selected_device == "mps"
    assert selection.decision.preflight_elapsed_seconds == 2.5


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("not implemented for the MPS device"), "unsupported_operator"),
        (RuntimeError("MPS backend out of memory"), "out_of_memory"),
        (RuntimeError("unexpected"), "preflight_failed"),
    ],
)
def test_preflight_failure_is_classified(error: RuntimeError, expected: str) -> None:
    assert classify_mps_failure(error, "preflight") == expected


def test_failed_mps_preflight_clears_cache_and_constructs_cpu() -> None:
    calls: list[str] = []
    cleared: list[bool] = []

    def factory(token: str, device: str):
        calls.append(device)
        if device == "mps":
            def fail(path: str, **kwargs):
                raise RuntimeError("not implemented for the MPS device")
            return fail
        return Pipeline()

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=factory,
        cache_clearer=lambda: cleared.append(True),
    )

    assert calls == ["mps", "cpu"]
    assert cleared == [True]
    assert selection.decision.selected_device == "cpu"
    assert selection.decision.fallback_category == "unsupported_operator"


def test_cache_cleanup_failure_does_not_block_cpu_fallback() -> None:
    def factory(token: str, device: str):
        if device == "mps":
            def fail(path: str, **kwargs):
                raise RuntimeError("unexpected")
            return fail
        return Pipeline()

    def cleanup_failure() -> None:
        raise RuntimeError("cleanup failed")

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        "secret",
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=True),
        pipeline_factory=factory,
        cache_clearer=cleanup_failure,
    )

    assert selection.decision.selected_device == "cpu"


def test_fallback_decision_does_not_contain_token() -> None:
    token = "hf_secret_value"

    selection = select_diarization_pipeline(
        Path("preflight.wav"),
        token,
        "mps",
        2,
        1,
        6,
        capabilities=capabilities(built=True, available=False),
        pipeline_factory=lambda supplied, device: Pipeline(),
    )

    assert token not in str(selection.decision)
    assert token not in (selection.decision.fallback_reason or "")


def test_cpu_pipeline_load_error_does_not_expose_token() -> None:
    token = "hf_secret_value"

    def fail(supplied: str, device: str):
        raise RuntimeError(f"provider rejected {supplied}")

    with pytest.raises(DiarizationError) as captured:
        select_diarization_pipeline(
            Path("preflight.wav"),
            token,
            "mps",
            2,
            1,
            6,
            capabilities=capabilities(built=False, available=False),
            pipeline_factory=fail,
        )

    assert token not in str(captured.value)
    assert captured.value.__cause__ is None
