from __future__ import annotations

import re

import jieba.analyse

STOPWORDS = {"的", "了", "和", "是", "在", "我", "你", "他", "她", "它"}


def _valid(value: str) -> bool:
    token = value.strip()
    if not token or token in STOPWORDS or token.isdigit():
        return False
    if len(token) == 1 and (not token.isascii() or re.fullmatch(r"\W", token)):
        return False
    return True


def extract_keywords(texts: list[str], count: int) -> list[str]:
    if count == 0:
        return []
    candidates = jieba.analyse.extract_tags(
        "\n".join(texts), topK=max(count * 3, count)
    )
    result: list[str] = []
    for candidate in candidates:
        if _valid(candidate) and candidate not in result:
            result.append(candidate)
        if len(result) == count:
            break
    return result
