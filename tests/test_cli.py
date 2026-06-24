from pathlib import Path

import pytest

from mimo_transcriber.cli import async_main, build_parser


def test_parser_exposes_required_defaults() -> None:
    args = build_parser().parse_args(["meeting.m4a"])
    assert args.language == "auto"
    assert args.device == "auto"
    assert args.concurrency == 2
    assert args.requests_per_minute == 20
    assert args.max_retries == 3
    assert args.keyword_count == 20


def test_parser_accepts_experimental_mps() -> None:
    args = build_parser().parse_args(["meeting.m4a", "--device", "mps"])
    assert args.device == "mps"


def test_parser_accepts_debug_flag() -> None:
    args = build_parser().parse_args(["meeting.m4a", "--debug"])
    assert args.debug is True


def test_parser_does_not_have_keep_temp() -> None:
    parser = build_parser()
    actions = [action.dest for action in parser._actions]
    assert "keep_temp" not in actions


def test_parser_defaults_to_mlx_asr() -> None:
    args = build_parser().parse_args(["meeting.m4a"])

    assert args.asr == "mlx"
    assert args.stt_model is None


def test_parser_accepts_mimo_asr_and_model() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--asr",
        "mimo",
        "--stt-model",
        "mimo-v2.5-asr",
    ])

    assert args.asr == "mimo"
    assert args.stt_model == "mimo-v2.5-asr"


def test_cli_parses_diarization_stability_options() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--conversation-mode", "two-person",
        "--diarization-stabilizer", "aggressive",
    ])

    assert args.conversation_mode == "two-person"
    assert args.diarization_stabilizer == "aggressive"


def test_cli_parses_diarization_model() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--diarization-model", "local/model",
    ])

    assert args.diarization_model == "local/model"
