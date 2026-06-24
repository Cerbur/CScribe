from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TermConfig:
    terms: tuple[str, ...] = ()
    replacements: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.replacements is None:
            object.__setattr__(self, "replacements", {})


def parse_terms_text(text: str) -> TermConfig:
    terms: list[str] = []
    replacements: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            wrong, right = [part.strip() for part in line.split("=>", 1)]
            if wrong and right:
                replacements[wrong] = right
                _append_unique(terms, right)
            continue
        _append_unique(terms, line)
    return TermConfig(tuple(terms), replacements)


def parse_terms_file(path: Path) -> TermConfig:
    return parse_terms_text(path.read_text(encoding="utf-8"))


def build_terms_prompt(
    user_prompt: str | None,
    terms: Sequence[str],
    limit: int = 100,
) -> str | None:
    parts: list[str] = []
    if user_prompt and user_prompt.strip():
        parts.append(user_prompt.strip())
    selected = [term for term in terms if term.strip()][:limit]
    if selected:
        joined = ", ".join(selected)
        parts.append(
            "音频是中英混杂的技术讨论。请优先按以下专有名词转写，"
            f"保留英文原文，不要翻译成中文或相近同音词：{joined}。"
        )
    return "\n\n".join(parts) or None


def prompt_digest(prompt: str | None) -> str | None:
    if not prompt:
        return None
    value = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"sha256:{value}"


def correct_terms(text: str, replacements: Mapping[str, str]) -> str:
    result = text
    for wrong, right in replacements.items():
        result = result.replace(wrong, right)
    return result


def _append_unique(values: list[str], value: str) -> None:
    cleaned = value.strip()
    if cleaned and cleaned not in values:
        values.append(cleaned)
