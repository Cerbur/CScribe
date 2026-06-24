# MiMo Prompt 与术语表设计规格

日期：2026-06-24

## 1. 目标

CScribe 使用 MiMo ASR 时，技术会议里的中英混杂专有名词容易被识别成中文同音词或相近英文词，例如 “Facebook” 变成“飞书”，“Grab” 变成 “Gleap”。MiMo API 支持 prompt 字段，但项目当前没有使用。目标是在不引入复杂外部依赖的前提下，用轻量术语表和 prompt 注入提升专有名词识别准确率。

本次设计要达成：

- MiMo 请求支持传入 ASR prompt。
- 用户可以通过本地文本文件维护项目术语。
- 系统可以从文件名、现有关键词和术语文件生成简短 prompt。
- 可选后置纠错只处理明确混淆词，避免大面积误替换。
- 本地 MLX ASR 不受 MiMo prompt 影响，但配置可以被安全忽略。

非目标：

- 不接入企业词库服务。
- 不做全量 NER 或 LLM 二次校对。
- 不要求用户维护复杂 YAML 数据库。
- 不把 prompt 作为隐式缓存外部依赖之外的隐藏状态。

## 2. 用户体验

术语文件是普通文本，每行一个术语或一个混淆映射：

```text
Facebook
Grab
Gleap
LangGraph
pyannote
飞书 => Facebook
格拉布 => Grab
```

运行：

```bash
uv run mimo-transcriber meeting.m4a --asr mimo --terms-file terms.txt
```

也可以只传自由 prompt：

```bash
uv run mimo-transcriber meeting.m4a --asr mimo --asr-prompt "音频是中英混杂技术讨论，请保留英文品牌词。"
```

两者同时存在时，最终 prompt = 自由 prompt + 术语 prompt。

## 3. 推荐架构

新增 `terms.py`，负责解析术语文件、生成 prompt 和执行可选纠错。ASR engine 只接收已经构造好的 prompt 字符串，不理解术语文件格式。

数据流：

```text
CLI args
  -> AppConfig(asr_prompt, terms_file, term_correction)
  -> load_term_config()
  -> build_asr_prompt()
  -> create_asr_engine()
  -> MimoAsrEngine.transcribe_one()
  -> optional correct_terms()
```

模块职责：

- `terms.py`：解析术语文件，生成 `TermConfig`，构造 prompt，执行明确混淆纠错。
- `config.py`：校验术语文件存在且可读，纳入缓存身份。
- `asr/base.py`：`AsrConfig` 增加 `prompt` 或 `prompt_digest` 字段。
- `asr/mimo.py`：发送 MiMo 请求时把 prompt 放入 `extra_body["asr_options"]["prompt"]`。
- `asr/mlx.py`：首版忽略 prompt，但 cache identity 仍体现 ASR provider 和模型。

## 4. 术语格式

首版支持简单文本格式：

- 空行忽略。
- `#` 开头为注释。
- 普通行是 canonical term。
- `wrong => right` 是明确混淆映射，同时 `right` 进入 canonical terms。

示例解析结果：

```python
@dataclass(frozen=True)
class TermConfig:
    terms: tuple[str, ...]
    replacements: Mapping[str, str]


def parse_terms_file(path: Path) -> TermConfig:
    """Parse a UTF-8 terms file into canonical terms and explicit replacements."""
```

重复词去重，保留首次出现顺序。术语数量默认限制 200 个，prompt 中最多使用前 100 个，避免请求过长。

## 5. Prompt 构造

prompt 模板：

```text
音频是中英混杂的技术讨论。请优先按以下专有名词转写，保留英文原文，不要翻译成中文或相近同音词：
Facebook, Grab, Gleap, LangGraph, pyannote。
```

如果用户提供 `--asr-prompt`，放在模板前面：

```text
{user_prompt}

音频是中英混杂的技术讨论。请优先按以下专有名词转写，保留英文原文，不要翻译成中文或相近同音词。
```

缓存身份必须包含 prompt digest，而不是完整 prompt，避免 task hash 泄露过多内容：

```python
{
    "prompt_digest": "sha256:<64 hex chars>",
    "term_count": 42,
    "term_correction": True,
}
```

## 6. 后置纠错

首版只做明确映射，不做 fuzzy 全文替换。原因是会议文本中中文多、同音词多，模糊替换容易误伤。

```python
def correct_terms(text: str, replacements: Mapping[str, str]) -> str:
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text
```

可选增强留到后续版本：

- 只对英文 token 做 `rapidfuzz` 匹配。
- 只在术语表中词长大于 4 的 term 上做高阈值纠错。
- 输出 `.corrections.json` 记录替换证据。

## 7. CLI 与配置

`AppConfig` 新增：

```python
asr_prompt: str | None = None
terms_file: Path | None = None
term_correction: bool = True
```

CLI 新增：

- `--asr-prompt TEXT`
- `--terms-file PATH`
- `--no-term-correction`

校验规则：

- `terms_file` 存在且可读。
- `asr_prompt` 去除首尾空白后为空时按 `None` 处理。
- 术语文件行数超过 1000 时拒绝，提示用户缩小范围。

## 8. 测试策略

单元测试覆盖：

- 术语文件解析普通行、注释、空行和 `wrong => right`。
- prompt 构造包含用户 prompt 和术语。
- prompt digest 改变时 ASR cache identity 改变。
- MiMo 请求把 prompt 放入 `asr_options`。
- 后置纠错只替换明确映射。
- MLX engine 不因为 prompt 参数报错。

集成测试覆盖：

- CLI 接收 `--terms-file` 并传入 pipeline。
- cache hash 在术语文件内容改变后变化。

## 9. 验收标准

- MiMo 请求体包含 prompt。
- 用户用 5-20 个术语即可显著降低已知专有名词错误。
- 不需要安装新服务或维护复杂依赖。
- 术语文件变更会触发重新 ASR，避免复用旧识别结果。
- 现有无术语文件的默认行为保持不变。
