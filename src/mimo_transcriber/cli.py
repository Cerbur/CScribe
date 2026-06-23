from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Sequence

from mimo_transcriber.config import AppConfig, ConfigError, validate_runtime
from mimo_transcriber.pipeline import run_pipeline


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
    parser.add_argument("--language", choices=("auto", "zh", "en"), default="auto")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=int, default=80)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--keyword-count", type=int, default=20)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--debug-json", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = AppConfig(
        input_path=args.input,
        output_path=args.output,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        language=args.language,
        device=args.device,
        concurrency=args.concurrency,
        requests_per_minute=args.requests_per_minute,
        max_retries=args.max_retries,
        keyword_count=args.keyword_count,
        keep_temp=args.keep_temp,
        debug_json=args.debug_json,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
    )
    try:
        mimo_key, hf_token = validate_runtime(config)
        result = await run_pipeline(config, mimo_key, hf_token)
        logging.info("输出文件: %s", result.outcome.summary.output_path)
        logging.info(
            "片段: %d 成功 / %d 失败；耗时 %.2f 秒",
            result.outcome.summary.succeeded,
            result.outcome.summary.failed,
            result.outcome.summary.elapsed_seconds,
        )
        return result.exit_code
    except (ConfigError, RuntimeError) as exc:
        logging.error("%s", exc)
        if args.verbose:
            logging.exception("详细错误")
        return 1


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))
