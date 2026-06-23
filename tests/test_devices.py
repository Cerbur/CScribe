from mimo_transcriber.devices import (
    DeviceCapabilities,
    DeviceDecision,
)


def test_device_decision_records_sanitized_fallback_facts() -> None:
    capabilities = DeviceCapabilities(
        cuda_available=False,
        mps_built=True,
        mps_available=False,
        platform="Darwin",
        machine="arm64",
    )
    decision = DeviceDecision(
        requested_device="mps",
        selected_device="cpu",
        mps_built=capabilities.mps_built,
        mps_available=capabilities.mps_available,
        fallback_category="runtime_unavailable",
        fallback_reason="当前 PyTorch 运行时无法使用 MPS",
    )
    assert decision.requested_device == "mps"
    assert decision.selected_device == "cpu"
    assert decision.fallback_category == "runtime_unavailable"
