# MiMo Prompt Terms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve MiMo ASR recognition of mixed Chinese/English technical terms by adding prompt injection, a lightweight local terms file, and explicit post-ASR term correction.

**Architecture:** Add `terms.py` to parse text term files, build compact ASR prompts, and apply explicit replacement corrections. Extend ASR config/cache identity with prompt digest and term metadata. Wire prompt into `MimoAsrEngine` request body while keeping MLX behavior unchanged.

**Tech Stack:** Python 3.11, dataclasses, hashlib, pathlib, argparse, pytest, OpenAI-compatible MiMo chat completions.

## Global Constraints

- No external terminology service or database.
- Terms file format is plain text with comments and optional `wrong => right` mappings.
- Prompt and terms must affect ASR cache identity so stale ASR results are not reused.
- Cache identity must include a digest, not the full prompt text.
- Post-ASR correction only applies explicit mappings; no fuzzy replacement in this version.
- Unit tests must not call the real MiMo API.
- MLX ASR may ignore prompt and terms safely.

---

## File Structure

- Create `src/mimo_transcriber/terms.py`: term parsing, prompt construction, prompt digest, explicit corrections.
- Modify `src/mimo_transcriber/asr/base.py`: add prompt-related ASR config fields and cache identity.
- Modify `src/mimo_transcriber/asr/factory.py`: load terms and pass prompt/corrections into engines.
- Modify `src/mimo_transcriber/asr/mimo.py`: send prompt in MiMo request and apply corrections after text extraction.
- Modify `src/mimo_transcriber/asr/mlx.py`: accept prompt-related config without behavior change.
- Modify `src/mimo_transcriber/config.py`: add `asr_prompt`, `terms_file`, `term_correction`; validate files.
- Modify `src/mimo_transcriber/cli.py`: add prompt/terms flags.
- Modify `README.md`: document terms file usage.
- Create `tests/test_terms.py`.
- Modify `tests/test_asr_config.py`, `tests/test_mimo_asr.py`, `tests/test_config.py`, `tests/test_cli.py`, `tests/test_asr_factory.py`.

---

### Task 1: Terms Parser, Prompt Builder, and Corrections

**Files:**
- Create: `src/mimo_transcriber/terms.py`
- Test: `tests/test_terms.py`

**Interfaces:**
- Produces: `TermConfig(terms: tuple[str, ...], replacements: Mapping[str, str])`
- Produces: `parse_terms_text(text: str) -> TermConfig`
- Produces: `parse_terms_file(path: Path) -> TermConfig`
- Produces: `build_terms_prompt(user_prompt: str | None, terms: Sequence[str], limit: int = 100) -> str | None`
- Produces: `prompt_digest(prompt: str | None) -> str | None`
- Produces: `correct_terms(text: str, replacements: Mapping[str, str]) -> str`

- [ ] **Step 1: Write failing tests**

Create `tests/test_terms.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_terms.py -q
```

Expected: FAIL with missing `mimo_transcriber.terms`.

- [ ] **Step 3: Implement terms module**

Create `src/mimo_transcriber/terms.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run pytest tests/test_terms.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/terms.py tests/test_terms.py
git commit -m "feat: add ASR terms prompt helpers"
```

---

### Task 2: Config, CLI, and Cache Identity

**Files:**
- Modify: `src/mimo_transcriber/config.py`
- Modify: `src/mimo_transcriber/cli.py`
- Modify: `src/mimo_transcriber/asr/base.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_asr_config.py`

**Interfaces:**
- Consumes: `prompt_digest()`
- Produces: `AppConfig.asr_prompt: str | None`
- Produces: `AppConfig.terms_file: Path | None`
- Produces: `AppConfig.term_correction: bool`
- Produces: `AsrConfig(prompt: str | None = None, term_count: int = 0, term_correction: bool = True)`

- [ ] **Step 1: Write failing config tests**

Append to `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from mimo_transcriber.config import AppConfig, ConfigError


def test_terms_file_must_exist(tmp_path: Path) -> None:
    config = AppConfig(input_path=tmp_path / "in.m4a", terms_file=tmp_path / "missing.txt")

    with pytest.raises(ConfigError, match="--terms-file"):
        config.validate_arguments()


def test_asr_prompt_blank_is_allowed(tmp_path: Path) -> None:
    config = AppConfig(input_path=tmp_path / "in.m4a", asr_prompt="   ")

    config.validate_arguments()
```

Append to `tests/test_cli.py`:

```python
from mimo_transcriber.cli import build_parser


def test_cli_parses_terms_options() -> None:
    args = build_parser().parse_args([
        "meeting.m4a",
        "--asr-prompt", "技术会议",
        "--terms-file", "terms.txt",
        "--no-term-correction",
    ])

    assert args.asr_prompt == "技术会议"
    assert str(args.terms_file) == "terms.txt"
    assert args.no_term_correction is True
```

Append to `tests/test_asr_config.py`:

```python
from mimo_transcriber.asr.base import AsrConfig


def test_prompt_digest_changes_asr_identity() -> None:
    base = AsrConfig(provider="mimo", language="zh")
    prompted = AsrConfig(provider="mimo", language="zh", prompt="Facebook", term_count=1)

    assert base.cache_identity() != prompted.cache_identity()
    assert prompted.cache_identity()["settings"]["prompt_digest"].startswith("sha256:")
    assert prompted.cache_identity()["settings"]["term_count"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_config.py tests/test_cli.py tests/test_asr_config.py -q
```

Expected: FAIL because fields and parser options are missing.

- [ ] **Step 3: Implement config and CLI**

Modify `src/mimo_transcriber/config.py`:

```python
    asr_prompt: str | None = None
    terms_file: Path | None = None
    term_correction: bool = True
```

In `validate_arguments()`:

```python
        if self.terms_file is not None:
            if not self.terms_file.is_file() or not os.access(self.terms_file, os.R_OK):
                raise ConfigError(f"--terms-file 不存在或不可读: {self.terms_file}")
            if len(self.terms_file.read_text(encoding="utf-8").splitlines()) > 1000:
                raise ConfigError("--terms-file 行数不能超过 1000")
```

Modify `src/mimo_transcriber/cli.py`:

```python
    parser.add_argument("--asr-prompt")
    parser.add_argument("--terms-file", type=Path)
    parser.add_argument("--no-term-correction", action="store_true")
```

Pass:

```python
        asr_prompt=args.asr_prompt,
        terms_file=args.terms_file,
        term_correction=not args.no_term_correction,
```

- [ ] **Step 4: Implement ASR cache identity**

Modify `src/mimo_transcriber/asr/base.py`:

```python
from mimo_transcriber.terms import prompt_digest

@dataclass(frozen=True)
class AsrConfig:
    provider: AsrProvider = "mlx"
    stt_model: str | None = None
    language: Language = "auto"
    prompt: str | None = None
    term_count: int = 0
    term_correction: bool = True
    terms_file: Path | None = None

    def cache_identity(self) -> dict[str, object]:
        settings = {
            "model": self.resolved_model(),
            "language": self.language,
        }
        digest = prompt_digest(self.prompt)
        if digest is not None:
            settings["prompt_digest"] = digest
            settings["term_count"] = self.term_count
            settings["term_correction"] = self.term_correction
        return {
            "kind": "asr-engine",
            "engine": "mimo" if self.provider == "mimo" else "mlx-whisper",
            "identity_version": 1,
            "settings": settings,
        }
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_terms.py tests/test_config.py tests/test_cli.py tests/test_asr_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/config.py src/mimo_transcriber/cli.py src/mimo_transcriber/asr/base.py tests/test_config.py tests/test_cli.py tests/test_asr_config.py
git commit -m "feat: add ASR prompt and terms config"
```

---

### Task 3: Wire Terms Through ASR Factory

**Files:**
- Modify: `src/mimo_transcriber/asr/factory.py`
- Test: `tests/test_asr_factory.py`

**Interfaces:**
- Consumes: `parse_terms_file()`, `build_terms_prompt()`
- Produces: concrete ASR engines receiving `prompt`, `term_replacements`, and `term_correction`

- [ ] **Step 1: Write failing factory tests**

Append to `tests/test_asr_factory.py`:

```python
from pathlib import Path

from mimo_transcriber.asr.base import AsrConfig, RuntimeConfig
from mimo_transcriber.asr.factory import create_asr_engine


def test_factory_builds_mimo_engine_with_terms_prompt(tmp_path: Path) -> None:
    terms = tmp_path / "terms.txt"
    terms.write_text("Facebook\n飞书 => Facebook\n", encoding="utf-8")

    engine = create_asr_engine(
        AsrConfig(provider="mimo", language="zh", terms_file=terms, prompt="技术会议"),
        RuntimeConfig(hf_token="hf", mimo_api_key="mimo"),
    )

    assert engine.prompt is not None
    assert "Facebook" in engine.prompt
    assert engine.term_replacements == {"飞书": "Facebook"}
```

If `AsrConfig` intentionally should not carry `terms_file`, put `terms_file` on `AppConfig` and add a small adapter in `pipeline.py`. Keep the test target as “engine receives prompt and replacements”.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_asr_factory.py::test_factory_builds_mimo_engine_with_terms_prompt -q
```

Expected: FAIL because factory does not load terms or engine does not expose prompt.

- [ ] **Step 3: Implement factory wiring**

Modify `src/mimo_transcriber/asr/factory.py` so MiMo engine construction includes:

```python
term_config = parse_terms_file(config.terms_file) if config.terms_file else TermConfig()
prompt = build_terms_prompt(config.prompt, term_config.terms)

return MimoAsrEngine(
    request=openai_request(runtime.mimo_api_key, model=config.resolved_model(), prompt=prompt),
    model=config.resolved_model(),
    language=config.language,
    concurrency=concurrency,
    requests_per_minute=requests_per_minute,
    max_retries=max_retries,
    event_sink=event_sink,
    prompt=prompt,
    term_replacements=term_config.replacements if config.term_correction else {},
)
```

Add imports at the top of `factory.py`:

```python
from mimo_transcriber.terms import TermConfig, build_terms_prompt, parse_terms_file
```

- [ ] **Step 4: Run factory tests**

Run:

```bash
uv run pytest tests/test_asr_factory.py tests/test_terms.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mimo_transcriber/asr/factory.py tests/test_asr_factory.py
git commit -m "feat: pass terms prompt into ASR engines"
```

---

### Task 4: MiMo Request Prompt and Explicit Corrections

**Files:**
- Modify: `src/mimo_transcriber/asr/mimo.py`
- Test: `tests/test_mimo_asr.py`

**Interfaces:**
- Consumes: `correct_terms(text, replacements)`
- Produces: `openai_request(api_key: str, model: str = "mimo-v2.5-asr", timeout: float = 120.0, prompt: str | None = None) -> Request`
- Produces: `MimoAsrEngine(request, model, language, concurrency, requests_per_minute, max_retries, sleep=asyncio.sleep, reporter=None, event_sink=None, prompt: str | None = None, term_replacements: Mapping[str, str] | None = None)`

- [ ] **Step 1: Write failing MiMo request tests**

Append to `tests/test_mimo_asr.py`:

```python
import pytest

from mimo_transcriber.asr.mimo import MimoAsrEngine, openai_request
from mimo_transcriber.models import SpeakerSegment


def test_openai_request_sends_prompt(monkeypatch):
    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return object()

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {
                "completions": FakeCompletions(),
            })()

    monkeypatch.setattr("openai.AsyncOpenAI", FakeClient)

    request = openai_request("key", prompt="Facebook prompt")
    import asyncio
    asyncio.run(request("data:audio/mp3;base64,abc", "zh"))

    assert captured["extra_body"]["asr_options"]["prompt"] == "Facebook prompt"


@pytest.mark.asyncio
async def test_mimo_engine_applies_explicit_term_corrections(tmp_path):
    audio = tmp_path / "s.mp3"
    audio.write_bytes(b"audio")

    class Message:
        content = "飞书 和 格拉布"

    class Choice:
        message = Message()

    class Completion:
        choices = [Choice()]

    async def request(data_url, language):
        return Completion()

    engine = MimoAsrEngine(
        request=request,
        model="mimo-v2.5-asr",
        language="zh",
        concurrency=1,
        requests_per_minute=60,
        max_retries=0,
        term_replacements={"飞书": "Facebook", "格拉布": "Grab"},
    )

    result = await engine.transcribe_one(SpeakerSegment(0, 0, 1, "A", segment_id="s0000"), audio)

    assert result.text == "Facebook 和 Grab"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_mimo_asr.py -q
```

Expected: FAIL because prompt and term replacements are unsupported.

- [ ] **Step 3: Implement MiMo prompt request**

Modify `openai_request()`:

```python
def openai_request(
    api_key: str,
    model: str = "mimo-v2.5-asr",
    timeout: float = 120.0,
    prompt: str | None = None,
) -> Request:
    async def request(data_url: str, language: str) -> Any:
        asr_options: dict[str, object] = {"language": language}
        if prompt:
            asr_options["prompt"] = prompt
        return await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": data_url},
                }],
            }],
            extra_body={"asr_options": asr_options},
        )
```

- [ ] **Step 4: Implement MiMo correction**

Modify `MimoAsrEngine.__init__()`:

```python
        prompt: str | None = None,
        term_replacements: Mapping[str, str] | None = None,
```

Store:

```python
        self.prompt = prompt
        self.term_replacements = dict(term_replacements or {})
```

After text extraction:

```python
                text = extract_content(completion.choices[0].message.content)
                if self.term_replacements:
                    text = correct_terms(text, self.term_replacements)
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run pytest tests/test_mimo_asr.py tests/test_terms.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/asr/mimo.py tests/test_mimo_asr.py
git commit -m "feat: send MiMo ASR prompt and correct terms"
```

---

### Task 5: Pipeline Integration, Docs, and Verification

**Files:**
- Modify: `src/mimo_transcriber/pipeline.py`
- Modify: `README.md`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `AppConfig.asr_prompt`, `AppConfig.terms_file`, `AppConfig.term_correction`
- Produces: ASR engine config carrying prompt and term settings

- [ ] **Step 1: Write or update pipeline test**

Add to `tests/test_pipeline.py` using the existing fake ASR factory or dependency pattern:

```python
def test_pipeline_includes_terms_in_asr_cache_identity(tmp_path):
    terms = tmp_path / "terms.txt"
    terms.write_text("Facebook\n", encoding="utf-8")
    config = AppConfig(
        input_path=tmp_path / "in.m4a",
        asr="mimo",
        asr_prompt="技术会议",
        terms_file=terms,
    )

    identity = config.asr_cache_identity()

    assert identity["settings"]["prompt_digest"].startswith("sha256:")
    assert identity["settings"]["term_count"] == 1
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run pytest tests/test_pipeline.py::test_pipeline_includes_terms_in_asr_cache_identity -q
```

Expected: FAIL until `AppConfig.asr_cache_identity()` loads terms and builds prompt.

- [ ] **Step 3: Update AppConfig ASR cache identity**

Modify `AppConfig.asr_cache_identity()` in `src/mimo_transcriber/config.py`:

```python
        term_config = parse_terms_file(self.terms_file) if self.terms_file else TermConfig()
        prompt = build_terms_prompt(self.asr_prompt, term_config.terms)
        return AsrConfig(
            provider=self.asr,
            stt_model=self.stt_model,
            language=self.language,
            prompt=prompt,
            term_count=len(term_config.terms),
            term_correction=self.term_correction,
            terms_file=self.terms_file,
        ).cache_identity()
```

Keep the implementation aligned with the actual `AsrConfig` fields chosen in Task 2.

- [ ] **Step 4: Update README**

Add:

```markdown
## MiMo 术语提示

MiMo ASR 支持通过 prompt 提醒模型保留技术和品牌专有名词。创建 `terms.txt`：

```text
Facebook
Grab
Gleap
飞书 => Facebook
格拉布 => Grab
```

运行：

```bash
uv run mimo-transcriber meeting.m4a --asr mimo --terms-file terms.txt
uv run mimo-transcriber meeting.m4a --asr mimo --asr-prompt "中英混杂技术讨论，请保留英文品牌词。"
```

术语文件变化会改变 ASR 缓存身份，因此会重新转写。
```

- [ ] **Step 5: Run full relevant tests**

Run:

```bash
uv run pytest tests/test_terms.py tests/test_asr_config.py tests/test_asr_factory.py tests/test_mimo_asr.py tests/test_config.py tests/test_cli.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mimo_transcriber/config.py src/mimo_transcriber/pipeline.py README.md tests/test_pipeline.py
git commit -m "docs: document MiMo terms prompt"
```

---

## Self-Review

- Spec coverage: parser, prompt, cache digest, CLI, MiMo request, explicit correction, docs, and tests are covered.
- Placeholder scan: The factory and pipeline tasks acknowledge existing signature differences and define the required behavior and concrete assertions.
- Type consistency: `TermConfig`, `build_terms_prompt`, `prompt_digest`, and `correct_terms` names match across tasks.
