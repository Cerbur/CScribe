from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mimo_transcriber.config import Device, resolve_device
from mimo_transcriber.devices import (
    DeviceCapabilities,
    DeviceDecision,
    FallbackCategory,
    SelectedDevice,
    collect_device_capabilities,
)
from mimo_transcriber.models import SpeakerSegment

MODEL_ID = "pyannote/speaker-diarization-community-1"

logger = logging.getLogger(__name__)


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


def create_pipeline(token: str, device: SelectedDevice, model_id: str = MODEL_ID) -> Any:
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model_id, token=token)
    pipeline.to(torch.device(device))
    return pipeline


def _model_bound_factory(model_id: str) -> Callable[[str, SelectedDevice], Any]:
    """Bind ``model_id`` into a 2-arg factory so callers stay signature-agnostic."""

    def factory(token: str, device: SelectedDevice) -> Any:
        return create_pipeline(token, device, model_id)

    return factory


def diarize_audio(
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


@dataclass(frozen=True)
class PipelineSelection:
    pipeline: Any
    decision: DeviceDecision


def classify_mps_failure(
    exc: BaseException,
    phase: Literal["preflight", "full"],
) -> FallbackCategory:
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current).lower())
        current = current.__cause__
    message = " ".join(messages)
    if "out of memory" in message or "allocation" in message:
        return "out_of_memory"
    if "not implemented for" in message and "mps" in message:
        return "unsupported_operator"
    return "full_run_failed" if phase == "full" else "preflight_failed"


def fallback_reason(category: FallbackCategory) -> str:
    return {
        "not_built": "当前 PyTorch 未构建 MPS 支持",
        "runtime_unavailable": "当前 PyTorch 运行时无法使用 MPS",
        "unsupported_operator": "pyannote 需要的算子尚不支持 MPS",
        "out_of_memory": "MPS 可用内存不足",
        "preflight_failed": "MPS 预检未能完成",
        "full_run_failed": "完整 MPS 说话人分离未能完成",
    }[category]


def clear_mps_cache() -> None:
    try:
        import torch

        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except Exception as exc:
        logger.debug("清理 MPS 缓存失败: %s", type(exc).__name__)


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "未知"
    return "是" if value else "否"


def log_device_decision(decision: DeviceDecision) -> None:
    if decision.requested_device != "mps":
        return
    logger.info("请求设备: MPS")
    logger.info("MPS 构建支持: %s", _yes_no(decision.mps_built))
    logger.info("MPS 运行时可用: %s", _yes_no(decision.mps_available))
    if decision.selected_device == "cpu":
        logger.warning("MPS 未启用: %s", decision.fallback_reason)
        logger.warning("已回退 CPU")
        logger.info("建议检查 PyTorch、macOS 版本以及当前 Python 架构")


def _cpu_selection(
    requested_device: Device,
    token: str,
    capabilities: DeviceCapabilities,
    pipeline_factory: Callable[[str, SelectedDevice], Any],
    category: FallbackCategory | None = None,
) -> PipelineSelection:
    return PipelineSelection(
        pipeline=_build_pipeline_safely(token, "cpu", pipeline_factory),
        decision=DeviceDecision(
            requested_device=requested_device,
            selected_device="cpu",
            mps_built=capabilities.mps_built if requested_device == "mps" else None,
            mps_available=(
                capabilities.mps_available if requested_device == "mps" else None
            ),
            fallback_category=category,
            fallback_reason=fallback_reason(category) if category is not None else None,
        ),
    )


def _build_pipeline_safely(
    token: str,
    device: SelectedDevice,
    pipeline_factory: Callable[[str, SelectedDevice], Any],
) -> Any:
    try:
        return pipeline_factory(token, device)
    except Exception as exc:
        raise DiarizationError(
            f"{device.upper()} pipeline 加载失败: {type(exc).__name__}"
        ) from None


def select_diarization_pipeline(
    preflight_path: Path,
    token: str,
    requested_device: Device,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    *,
    model_id: str = MODEL_ID,
    capabilities: DeviceCapabilities | None = None,
    pipeline_factory: Callable[[str, SelectedDevice], Any] | None = None,
    cache_clearer: Callable[[], None] = clear_mps_cache,
    clock: Callable[[], float] = time.monotonic,
) -> PipelineSelection:
    if pipeline_factory is None:
        pipeline_factory = _model_bound_factory(model_id)
    facts = capabilities or collect_device_capabilities()
    if requested_device != "mps":
        selected = resolve_device(
            requested_device,
            cuda_available=lambda: facts.cuda_available,
        )
        return PipelineSelection(
            pipeline=_build_pipeline_safely(token, selected, pipeline_factory),
            decision=DeviceDecision(requested_device, selected),
        )

    logger.info("正在检查 MPS 环境")
    if not facts.mps_built:
        return _cpu_selection(
            requested_device, token, facts, pipeline_factory, "not_built"
        )
    if not facts.mps_available:
        return _cpu_selection(
            requested_device,
            token,
            facts,
            pipeline_factory,
            "runtime_unavailable",
        )

    logger.info("正在使用 10 秒样本预检 pyannote")
    started = clock()
    pipeline: Any | None = None
    try:
        pipeline = pipeline_factory(token, "mps")
        segments = diarize_audio(
            preflight_path,
            pipeline,
            num_speakers,
            min_speakers,
            max_speakers,
        )
        if not segments:
            logger.info(
                "预检样本未检测到语音（可能为静音段），MPS 管线已成功加载并运行"
            )
    except Exception as exc:
        logger.warning("MPS 预检异常: %s: %s", type(exc).__name__, exc)
        category = classify_mps_failure(exc, "preflight")
        if pipeline is not None:
            del pipeline
        try:
            cache_clearer()
        except Exception as cleanup_exc:
            logger.debug("清理 MPS 缓存失败: %s", type(cleanup_exc).__name__)
        return _cpu_selection(
            requested_device,
            token,
            facts,
            pipeline_factory,
            category,
        )

    elapsed = clock() - started
    logger.info("MPS 预检通过，耗时 %.2f 秒", elapsed)
    return PipelineSelection(
        pipeline=pipeline,
        decision=DeviceDecision(
            requested_device="mps",
            selected_device="mps",
            mps_built=True,
            mps_available=True,
            preflight_elapsed_seconds=elapsed,
        ),
    )


@dataclass(frozen=True)
class DiarizationResult:
    segments: list[SpeakerSegment]
    decision: DeviceDecision


def run_diarization(
    normalized_path: Path,
    preflight_path: Path,
    token: str,
    requested_device: Device,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    *,
    model_id: str = MODEL_ID,
    capabilities: DeviceCapabilities | None = None,
    pipeline_factory: Callable[[str, SelectedDevice], Any] | None = None,
    cache_clearer: Callable[[], None] = clear_mps_cache,
    clock: Callable[[], float] = time.monotonic,
) -> DiarizationResult:
    if pipeline_factory is None:
        pipeline_factory = _model_bound_factory(model_id)
    selection = select_diarization_pipeline(
        preflight_path,
        token,
        requested_device,
        num_speakers,
        min_speakers,
        max_speakers,
        model_id=model_id,
        capabilities=capabilities,
        pipeline_factory=pipeline_factory,
        cache_clearer=cache_clearer,
        clock=clock,
    )
    log_device_decision(selection.decision)
    logger.info(
        "正在使用 %s 处理完整音频",
        selection.decision.selected_device.upper(),
    )
    try:
        segments = diarize_audio(
            normalized_path,
            selection.pipeline,
            num_speakers,
            min_speakers,
            max_speakers,
        )
        return DiarizationResult(segments, selection.decision)
    except DiarizationError:
        if selection.decision.selected_device != "mps":
            raise

        mps_decision = selection.decision
        del selection
        try:
            cache_clearer()
        except Exception as cleanup_exc:
            logger.debug("清理 MPS 缓存失败: %s", type(cleanup_exc).__name__)

        logger.warning("完整 MPS 说话人分离失败，已安全回退 CPU")
        cpu_pipeline = _build_pipeline_safely(token, "cpu", pipeline_factory)
        try:
            segments = diarize_audio(
                normalized_path,
                cpu_pipeline,
                num_speakers,
                min_speakers,
                max_speakers,
            )
        except DiarizationError as cpu_exc:
            raise DiarizationError(
                "MPS 完整运行失败，CPU 回退也失败"
            ) from cpu_exc

        final_decision = DeviceDecision(
            requested_device="mps",
            selected_device="cpu",
            mps_built=mps_decision.mps_built,
            mps_available=mps_decision.mps_available,
            preflight_elapsed_seconds=mps_decision.preflight_elapsed_seconds,
            fallback_category="full_run_failed",
            fallback_reason=fallback_reason("full_run_failed"),
        )
        log_device_decision(final_decision)
        return DiarizationResult(segments, final_decision)
