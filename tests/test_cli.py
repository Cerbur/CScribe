from pathlib import Path

import pytest

from mimo_transcriber.cli import async_main, build_parser


def test_parser_exposes_required_defaults() -> None:
    args = build_parser().parse_args(["meeting.m4a"])
    assert args.language == "auto"
    assert args.device == "auto"
    assert args.concurrency == 4
    assert args.requests_per_minute == 80
    assert args.max_retries == 3
    assert args.keyword_count == 20


def test_parser_accepts_experimental_mps() -> None:
    args = build_parser().parse_args(["meeting.m4a", "--device", "mps"])
    assert args.device == "mps"


def test_parser_does_not_have_keep_temp() -> None:
    parser = build_parser()
    actions = [action.dest for action in parser._actions]
    assert "keep_temp" not in actions
