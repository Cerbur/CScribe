from pathlib import Path

from mimo_transcriber.terms import (
    build_terms_prompt,
    correct_terms,
    parse_terms_file,
    parse_terms_text,
    prompt_digest,
)


def test_parse_terms_text_supports_terms_comments_and_replacements() -> None:
    config = parse_terms_text("""
    # meeting terms
    Facebook
    Grab
    飞书 => Facebook
    格拉布 => Grab
    Facebook
    """)

    assert config.terms == ("Facebook", "Grab")
    assert config.replacements == {"飞书": "Facebook", "格拉布": "Grab"}


def test_parse_terms_file_reads_utf8(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("Gleap\nGleep => Gleap\n", encoding="utf-8")

    config = parse_terms_file(path)

    assert config.terms == ("Gleap",)
    assert config.replacements == {"Gleep": "Gleap"}


def test_build_terms_prompt_combines_user_prompt_and_terms() -> None:
    prompt = build_terms_prompt(
        "这是投资会议。",
        ["Facebook", "Grab"],
    )

    assert prompt is not None
    assert prompt.startswith("这是投资会议。")
    assert "Facebook, Grab" in prompt
    assert "保留英文原文" in prompt


def test_prompt_digest_hides_raw_prompt() -> None:
    digest = prompt_digest("secret prompt")

    assert digest is not None
    assert digest.startswith("sha256:")
    assert "secret" not in digest


def test_correct_terms_applies_only_explicit_replacements() -> None:
    text = correct_terms("飞书 和 格拉布", {"飞书": "Facebook", "格拉布": "Grab"})

    assert text == "Facebook 和 Grab"
