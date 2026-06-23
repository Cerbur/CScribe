# 本地默认 ASR Engine 设计规格

日期：2026-06-24

## 1. 目标

CScribe 默认使用本地 Apple Silicon 友好的 `mlx-whisper` 执行语音识别，MiMo ASR 改为可选远端引擎。上层流水线不理解具体模型、provider、限流或重试策略，只依赖一个稳定的 ASR Engine 边界。

本次设计要达成：

- 默认命令不上传音频片段，优先本地转写。
- 保留 MiMo 作为显式可选后端。
- 把模型选择、重试、限流、并发策略和缓存身份收进 ASR 层。
- 为后续接入 WhisperKit、faster-whisper、HTTP STT 服务或自定义 provider 留出清晰扩展点。
- 不重写说话人分离、切片、输出格式或关键词提取。

非目标：

- 不在本次实现公开第三方插件协议。
- 不同时实现 WhisperKit。
- 不为本地 MLX 首版做复杂自动调参。
- 不改变现有 TXT 和 `.segments.json` 输出结构。

## 2. 用户体验

默认调用：

```bash
uv run mimo-transcriber meeting.m4a
```

默认等价于使用本地 MLX Whisper engine。MiMo 改为显式选择：

```bash
uv run mimo-transcriber meeting.m4a --asr mimo
```

用户可以自定义模型：

```bash
uv run mimo-transcriber meeting.m4a --asr mlx --stt-model mlx-community/whisper-small
uv run mimo-transcriber meeting.m4a --asr mimo --stt-model mimo-v2.5-asr
```

CLI 名称暂时保持 `mimo-transcriber`，避免扩大迁移范围。README 要说明项目默认已经不依赖 MiMo；未来可单独评估是否新增 `cscribe` 命令别名。

## 3. 推荐架构

新增 `src/mimo_transcriber/asr/` 包，作为唯一转写边界。

```text
pipeline
  -> asr factory
       -> mlx engine
       -> mimo engine
```

模块职责：

- `asr/base.py`：定义 `AsrEngine` 协议、`AsrConfig`、缓存身份类型和通用错误语义。
- `asr/factory.py`：根据 `AsrConfig`、环境密钥和 reporter 创建具体 engine。
- `asr/mlx.py`：封装 `mlx_whisper`，处理模型加载、语言参数、本地串行推理、文本抽取和片段失败。
- `asr/mimo.py`：迁移现有 MiMo 请求、限流、429 全局退避、重试和响应解析。
- `pipeline.py`：只持有 `AsrEngine`，不接触 provider、模型、API 请求、MLX 导入或重试细节。

现有 `mimo_asr.py` 可以迁移到 `asr/mimo.py`，或保留为兼容 shim 再逐步删除。新代码不再从 pipeline 直接导入 `openai_request` 或 `MiMoTranscriber`。

## 4. ASR Engine 边界

上层只使用一个统一接口：

```python
class AsrEngine(Protocol):
    @property
    def cache_identity(self) -> Mapping[str, object]: ...

    async def transcribe_one(self, segment: SpeakerSegment, path: Path) -> SpeakerSegment: ...

    async def transcribe_all(
        self,
        items: list[tuple[SpeakerSegment, Path]],
        fail_fast: bool,
    ) -> list[SpeakerSegment]: ...
```

`pipeline.py` 不解释 `cache_identity` 的内部字段，只把它纳入任务 hash。`pipeline.py` 不判断当前 engine 是本地还是远端，也不根据模型名改变行为。

ASR 层内部负责：

- provider 选择。
- 默认模型和用户模型。
- 模型名或本地路径的 provider 专属解释。
- 网络重试、本地推理失败策略和限流。
- 单模型实例锁、并发控制或远端并发请求。
- provider 专属依赖和运行时校验。

`SpeakerSegment` 仍是跨层数据对象。成功时写入 `text` 和 `SegmentStatus.SUCCESS`；最终失败时写入 `[该片段识别失败]`、`SegmentStatus.FAILED` 和简洁 `error`。

## 5. 配置设计

`AppConfig` 新增 ASR 相关配置：

- `asr`: `Literal["mlx", "mimo"]`，默认 `"mlx"`。
- `stt_model`: `str | None`，默认由 engine 决定。

保留既有参数：

- `--language auto|zh|en`：由 ASR engine 自己映射。
- `--concurrency`：pipeline 仍用于 worker 数量，但 engine 可在内部串行化真实推理。
- `--requests-per-minute`：只对 MiMo engine 有效。
- `--max-retries`：由 engine 自己解释；MiMo 沿用网络重试，本地 MLX 首版明确不重试推理，因此该参数在 `asr == "mlx"` 时不影响行为。

CLI 新增：

```text
--asr {mlx,mimo}
--stt-model MODEL
```

默认模型：

- `mlx`: 首选 `mlx-community/whisper-large-v3-turbo`。实现计划阶段需要用当前 `mlx-whisper` 文档或包行为验证模型名可直接使用；若包推荐模型名变化，更新默认值和 README。
- `mimo`: `mimo-v2.5-asr`。

## 6. 运行时校验

总是校验：

- 输入文件存在且可读。
- `ffmpeg` 与 `ffprobe` 可执行。
- `HF_TOKEN` 已配置，因为说话人分离仍依赖 pyannote。
- 输出父目录可创建。

仅 `asr == "mimo"` 时校验：

- `MIMO_API_KEY` 已配置。
- MiMo 相关 OpenAI SDK 请求配置可创建。

仅 `asr == "mlx"` 时校验：

- 可以导入 `mlx_whisper`。
- 当前平台和 Python 环境满足 MLX 运行要求。

MLX 模型下载和缓存由 `mlx-whisper` 生态处理。首次运行可能下载模型，README 要提前说明耗时、磁盘占用和网络需求；已下载后可离线运行。

## 7. 本地 MLX Engine

`MlxAsrEngine` 负责延迟加载模型并转写片段。

行为规则：

- 首次实际转写前加载 `mlx_whisper`，避免 CLI help 或配置错误路径产生重依赖成本。
- 使用 `asyncio.Lock` 或内部单 worker 串行调用 MLX 推理，避免多个片段同时争抢同一 GPU/统一内存。
- `--language auto` 不传固定语言，让模型自动识别。
- `--language zh|en` 映射为 MLX Whisper 接受的语言参数。
- 从返回对象中抽取 `text`，去除多余空白。
- 空文本视为片段失败。
- 单片段推理异常默认只标记该片段失败，继续处理其他片段。
- 模型加载失败或依赖缺失是启动/关键失败，返回退出码 `1`。
- `--fail-fast` 打开时，首个最终失败片段终止任务并不写正式 TXT。

首版不为 MLX 推理做自动重试。若后续发现特定可恢复错误，可以在 engine 内部增加本地 retry，而不影响 pipeline。

## 8. MiMo Engine

`MimoAsrEngine` 迁移现有行为：

- 使用 MiMo OpenAI 风格 API。
- 把音频片段编码为 `input_audio` data URL。
- 使用 `mimo-v2.5-asr` 默认模型。
- 保留并发 semaphore。
- 保留全局 requests-per-minute 限流。
- 保留 429 全局 cooldown、超时/连接错误/5xx 重试和空响应不重试规则。
- 通过 reporter 报告重试和最终完成。

MiMo engine 的模型 ID 来自 `stt_model` 或默认值，但该字段只在 MiMo engine 内部解释。

## 9. 缓存身份

缓存身份由 ASR Engine 或 `AsrConfig` 生成，上层只把它作为不透明 JSON 纳入 task hash。

示例：

```json
{
  "kind": "asr-engine",
  "engine": "mlx-whisper",
  "identity_version": 1,
  "settings": {
    "model": "mlx-community/whisper-large-v3-turbo",
    "language": "auto"
  }
}
```

MiMo 示例：

```json
{
  "kind": "asr-engine",
  "engine": "mimo",
  "identity_version": 1,
  "settings": {
    "model": "mimo-v2.5-asr",
    "language": "auto"
  }
}
```

`cache.py` 不再包含 `MIMO_MODEL_ID` 或 provider 专属字段。它只调用配置提供的 `asr_cache_identity()` 或读取已经规范化的 `config.asr_identity`。

缓存 hash 会因此变化一次。这是预期迁移：旧 `/tmp/cscribe` 缓存不会被错误复用，新运行会重新生成工作目录。

## 10. Pipeline 改动

`run_pipeline` 的密钥参数从 `mimo_key, hf_token` 调整为通用 runtime 配置：

```python
@dataclass(frozen=True)
class RuntimeConfig:
    hf_token: str
    mimo_api_key: str | None = None
```

运行时校验函数返回 `RuntimeConfig`。`asr/factory.py` 只读取当前 engine 需要的字段；pipeline 不解释这些 secrets。

`_run_segment_workers` 不再创建 `MiMoTranscriber`。它接收已经创建好的 `AsrEngine`，或通过 `PipelineDependencies` 注入测试 engine。

切片、manifest 恢复、失败统计、关键词和输出写入保持现有流程。

## 11. 错误处理与退出语义

关键失败返回退出码 `1`：

- 输入或依赖校验失败。
- FFmpeg/ffprobe 不可用。
- HF token 缺失。
- 选择 MiMo 但缺少 MiMo API key。
- 选择 MLX 但缺少 `mlx_whisper` 或模型加载失败。
- 说话人分离失败。
- 输出写入失败。

片段失败默认返回退出码 `2` 并写正式 TXT：

- MiMo 单片段用尽重试。
- MLX 单片段推理异常。
- 任一 engine 返回空文本。
- 切片失败。

`--fail-fast` 对所有 engine 一致：首个片段最终失败时终止任务，不写正式 TXT。

## 12. 测试计划

新增或调整测试：

- `test_asr_config.py`：默认 engine、模型默认值、cache identity 稳定性、CLI 参数映射。
- `test_asr_factory.py`：`mlx` 和 `mimo` engine 创建路径，provider 错误提示。
- `test_mlx_asr.py`：mock `mlx_whisper.transcribe`，验证语言映射、空文本失败、异常失败、内部串行锁和返回排序。
- `test_mimo_asr.py`：迁移现有 MiMo retry、429、空响应和排序测试。
- `test_config.py`：MiMo API key 只在 `--asr mimo` 时必填；默认本地不要求。
- `test_cache.py`：task hash 纳入不透明 ASR identity；改变 engine 或模型会改变 hash；改变并发、重试、debug 不改变 hash。
- `test_pipeline.py`：pipeline 使用 fake engine，证明上层不依赖 provider/model。
- `test_cli.py`：默认 `--asr mlx`，显式 `--asr mimo` 可解析。

不在单元测试中下载真实模型或调用真实 MiMo API。真实本地模型验证作为手动冒烟测试记录在实现计划中。

## 13. 文档更新

README 需要更新：

- 项目介绍改为默认本地 MLX Whisper，可选 MiMo。
- Quick Start 说明首次本地模型下载。
- `.env` 中 `MIMO_API_KEY` 改为仅 MiMo 模式需要。
- CLI 参数表新增 `--asr` 和 `--stt-model`。
- FAQ 增加 Apple Silicon、本地模型下载、MiMo 可选模式和缓存变化说明。

原“怎么换 ASR 模型”章节改为“怎么选择 ASR Engine 和 STT 模型”，并说明模型对 pipeline 透明，由 engine 层解释。

## 14. 迁移策略

实现按小步迁移：

1. 引入 ASR 配置和缓存身份，不改变当前行为。
2. 将 MiMo 迁移进 `asr/mimo.py`，让 pipeline 通过 engine 调用。
3. 添加 MLX engine，并把默认 ASR 改为 `mlx`。
4. 更新运行时校验、README 和测试。

每一步都保持测试可运行，避免同时改变 pipeline、缓存和真实模型接入导致定位困难。
