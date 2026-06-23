from mimo_transcriber.keywords import extract_keywords


def test_keywords_keep_meaningful_english_and_drop_noise() -> None:
    result = extract_keywords(
        ["我们使用 Java Spring Redis Agent RAG 构建系统。", "123 了 的 和"],
        10,
    )
    assert "Java" in result
    assert "Agent" in result
    assert "123" not in result
