from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Sequence

from mimo_transcriber.cache import TaskAlreadyRunningError
from mimo_transcriber.config import AppConfig, ConfigError, validate_runtime
from mimo_transcriber.pipeline import run_pipeline
from mimo_transcriber.progress import TerminalProgressReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimo-transcriber",
        description="将多人 M4A 录音按说话人转写为带时间戳的 TXT。",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=6)
    parser.add_argument(
        "--conversation-mode",
        choices=("auto", "two-person", "multi"),
        default="auto",
    )
    parser.add_argument(
        "--diarization-stabilizer",
        choices=("off", "conservative", "balanced", "aggressive"),
        default="balanced",
    )
    parser.add_argument(
        "--diarization-model",
        default="pyannote/speaker-diarization-community-1",
    )
    parser.add_argument("--language", choices=("auto", "zh", "en"), default="auto")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--requests-per-minute", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--asr", choices=("mlx", "mimo"), default="mlx")
    parser.add_argument("--stt-model")
    parser.add_argument("--keyword-count", type=int, default=20)
    parser.add_argument("--debug-json", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _setup_logging(debug: bool) -> None:
    """配置 Python logging，仅 --debug 时输出应用日志。

    第三方库（httpcore / huggingface_hub / urllib3 / matplotlib 等）
    始终抑制到 WARNING 以上，避免正常运行时产生大量噪音。
    """
    # 第三方库：始终抑制
    for noisy in (
        "httpcore",
        "httpx",
        "huggingface_hub",
        "urllib3",
        "matplotlib",
        "openai",
        "PIL",
        "asyncio",
        "fsspec",
        "filelock",
        "huggingface",
        "transformers",
        "torch",
        "pyannote",
        "aiohttp",
        "botocore",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if debug:
        # --debug：打开自身 DEBUG 日志
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            force=True,
        )
    else:
        # 默认：抑制所有日志
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(message)s",
            force=True,
        )


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 日志策略：
    #   - 默认：根 logger 设为 WARNING，抑制所有第三方库和自身的日志输出
    #   - --debug：仅打开 mimotranscriber 自身的 DEBUG 日志
    #   - 进度信息由 TerminalProgressReporter 直接输出，不经过 logging 模块
    _setup_logging(debug=args.debug)

    config = AppConfig(
        input_path=args.input,
        output_path=args.output,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=args.language,
        device=args.device,
        conversation_mode=args.conversation_mode,
        diarization_stabilizer=args.diarization_stabilizer,
        diarization_model=args.diarization_model,
        concurrency=args.concurrency,
        requests_per_minute=args.requests_per_minute,
        max_retries=args.max_retries,
        asr=args.asr,
        stt_model=args.stt_model,
        keyword_count=args.keyword_count,
        debug_json=args.debug_json,
        fail_fast=args.fail_fast,
        debug=args.debug,
        verbose=args.verbose,
    )

    reporter = TerminalProgressReporter(sys.stderr)
    try:
        runtime = validate_runtime(config)
        result = await run_pipeline(config, runtime, reporter=reporter)
        print(f"输出文件: {result.outcome.summary.output_path}", file=sys.stderr)
        return result.exit_code
    except TaskAlreadyRunningError:
        print("错误: 相同任务正在运行，请等待前一次任务完成", file=sys.stderr)
        return 1
    except (ConfigError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        if args.debug:
            logging.exception("详细错误")
        return 1
    except asyncio.CancelledError:
        print("用户中断，已保留工作目录供后续恢复", file=sys.stderr)
        return 130
    finally:
        reporter.close()


def main() -> None:
    # HF / pyannote 缓存重定向由 mimo_transcriber 包导入时无条件完成
    # （见 mimo_transcriber/__init__.py），CLI 入口无需再处理。
    raise SystemExit(asyncio.run(async_main()))
