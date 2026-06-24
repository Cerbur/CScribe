# 段落合并输出设计规格

日期：2026-06-24

## 1. 目标

CScribe 当前把 diarization 后的 `SpeakerSegment` 直接交给 ASR，再按片段逐条输出。即使同一说话人的连续发言被 pyannote 切成多个短片段，最终 TXT 也会显示成 5-10 个短时间块。目标是在不更换 ASR 模型的前提下，让最终 TXT 更接近飞书的“整段发言在一个时间块里”。

本次设计要达成：

- ASR 内部仍可以使用较短、较安全的音频切片。
- 最终 TXT 输出新增展示层段落合并，不再直接暴露所有内部切片。
- `.segments.json` 保留内部片段，便于调试和回归定位。
- 合并规则必须可配置，默认偏保守，避免把短问短答错误合并。
- 不改变 diarization 模型、不改变 ASR provider、不改变缓存目录结构。

非目标：

- 不重写 pyannote diarization。
- 不做 LLM 摘要、改写或重分段。
- 不改变关键词提取算法。
- 不要求 TXT 输出保留每个内部片段的时间戳。

## 2. 用户体验

默认输出启用平衡段落合并：

```bash
uv run mimo-transcriber meeting.m4a --asr mimo --num-speakers 2
```

用户可以按需要调整：

```bash
uv run mimo-transcriber meeting.m4a --paragraph-mode conservative
uv run mimo-transcriber meeting.m4a --paragraph-mode aggressive --paragraph-gap 2.5
uv run mimo-transcriber meeting.m4a --no-paragraph-merge
```

输出示例从：

```text
说话人 1 03:10
我们接下来聊一下 Facebook 的策略。

说话人 1 03:14
然后 Grab 这边也有一个类似场景。
```

变成：

```text
说话人 1 03:10
我们接下来聊一下 Facebook 的策略。然后 Grab 这边也有一个类似场景。
```

## 3. 推荐架构

新增一个纯展示层模块 `paragraphs.py`，负责把已转写成功的 `SpeakerSegment` 转为 `TranscriptBlock`。`SpeakerSegment` 继续代表内部处理片段；`TranscriptBlock` 只代表输出块。

数据流：

```text
diarization raw turns
  -> process_segments()
  -> slice/audio/asr workers
  -> SpeakerSegment(text, status)
  -> build_transcript_blocks()
  -> render_transcript()
```

核心边界：

- `segments.py` 仍负责内部切片清洗、同说话人短间隔初步合并、长片段拆分。
- `paragraphs.py` 只读已完成的片段，输出展示块，不写 manifest，不影响 ASR。
- `formatter.py` 渲染 `TranscriptBlock`，而不是直接渲染 `SpeakerSegment`。
- `debug_json` 继续输出内部 `segments`，并可额外输出 `blocks` 方便对照。

## 4. 段落合并规则

首版采用确定性规则，不引入机器学习依赖。

默认 `balanced` 模式：

- 只合并相同 `raw_speaker` 的相邻片段。
- 片段间隔 `gap <= 2.0s` 才允许合并。
- 合并后总时长 `<= 120s`。
- 当前块文本不以强句末标点结束时，优先合并。
- 下一段以承接词开头时，优先合并。
- 如果当前块已达到 `max_chars=900`，停止合并。
- 失败片段文本为 `[该片段识别失败]` 时，不和正常片段合并。

`conservative` 模式：

- `gap <= 1.0s`
- `max_duration=75s`
- 只有当前文本没有句末标点，或下一段明显承接时合并。

`aggressive` 模式：

- `gap <= 3.0s`
- `max_duration=180s`
- 同说话人短停顿默认合并，除非上一段以问号结尾且下一段很短。

关键函数：

```python
@dataclass
class TranscriptBlock:
    index: int
    start: float
    end: float
    raw_speaker: str
    display_speaker: str | None
    text: str
    source_segment_ids: list[str]


@dataclass(frozen=True)
class ParagraphConfig:
    enabled: bool = True
    mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
    gap: float | None = None
    max_duration: float | None = None
    max_chars: int = 900


def build_transcript_blocks(
    segments: list[SpeakerSegment],
    config: ParagraphConfig,
) -> list[TranscriptBlock]:
    """Return display blocks derived from completed internal segments."""
```

## 5. CLI 与配置

`AppConfig` 新增：

```python
paragraph_mode: Literal["off", "conservative", "balanced", "aggressive"] = "balanced"
paragraph_gap: float | None = None
paragraph_max_duration: float | None = None
paragraph_max_chars: int = 900
```

CLI 新增：

- `--paragraph-mode {off,conservative,balanced,aggressive}`
- `--paragraph-gap SECONDS`
- `--paragraph-max-duration SECONDS`
- `--paragraph-max-chars N`
- `--no-paragraph-merge` 作为 `--paragraph-mode off` 的易用别名

缓存身份不包含这些字段，因为它们只影响最终渲染，不影响 diarization、切片或 ASR 结果。重新运行同一音频时可以复用 manifest 后重新写输出。

## 6. 错误处理

- 空文本片段不参与合并；如果整段为空，渲染为空字符串。
- 失败片段单独成块，避免把错误提示混入正常发言。
- 配置校验拒绝负数 gap、负数 max duration、非正 `paragraph_max_chars`。
- 当 `paragraph_mode=off` 时，每个 `SpeakerSegment` 生成一个 `TranscriptBlock`，保持旧行为。

## 7. 测试策略

单元测试覆盖：

- 同说话人短间隔合并。
- 不同说话人不合并。
- 长停顿不合并。
- 失败片段不合并。
- `off` 模式保持一段一块。
- `conservative/balanced/aggressive` 的阈值差异。
- `formatter.render_transcript()` 使用 block 输出。
- `debug_json` 同时包含内部 `segments` 和展示 `blocks`。

集成测试覆盖：

- pipeline 复用缓存后仍能按照新的 paragraph 配置重新写 TXT。
- 输出块数量少于内部片段数量，但 debug JSON 内部片段数量不变。

## 8. 验收标准

- 对同一说话人连续发言，输出块数量明显少于内部片段数量。
- 2 人短问短答不会被合并到同一个块。
- 修改 paragraph 参数不会触发重新 ASR。
- 现有测试继续通过。
- 新增测试覆盖主要合并规则。
