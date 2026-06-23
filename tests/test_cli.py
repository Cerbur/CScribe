from mimo_transcriber.cli import build_parser


def test_parser_exposes_required_defaults() -> None:
    args = build_parser().parse_args(["meeting.m4a"])
    assert args.language == "auto"
    assert args.device == "auto"
    assert args.concurrency == 4
    assert args.requests_per_minute == 80
    assert args.max_retries == 3
    assert args.keyword_count == 20
