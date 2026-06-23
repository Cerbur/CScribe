# 本地默认 ASR Engine 设计规格

日期：2026-06-24

## 1. 目标

CScribe 默认使用本地 Apple Silicon 友好的 `mlx-whisper` 执行语音识别，MiMo ASR 改为可选远端引擎。上层流水线不理解具体模型、provider、限流或重试策略，只依赖一个稳定的 ASR Engine 边界。

本次设计要达成：

- 默认命令不上传音频片段，优先本地转写。
- 保留 MiMo 作为显式可选后端。
- 把模型选择、重试、限流、并发策略和缓存身份收进 ASR 层。
- 把终端显示、manifest 持久化、切片 worker 和 ASR worker 分成明确边界。
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

## 3. 推荐架构与分层

整体分层：

```text
CLI / Terminal UI
  -> Pipeline Orchestrator
       -> MediaPreparation
            -> ffprobe
            -> normalize
            -> preflight
       -> Segmentation
            -> diarization
            -> segment cleanup and split planning
       -> Worker Runtime
            -> slice workers
            -> ASR workers
            -> state worker
       -> Output Assembly
            -> keywords
            -> formatter
```

核心规则：

- Worker 产出事件和下游产物。
- `StateWorker` 投影事件，串行更新 manifest、内存运行状态和终端进度。
- `Pipeline Orchestrator` 管阶段、队列、worker 生命周期和最终汇总。
- `ProgressReporter` 不被业务 worker 直接调用。
- `ManifestStore.save()` 不在切片 worker 或 ASR worker 中直接调用。
- ASR engine 只负责片段转写和自身内部策略，不负责持久化、调度或显示。

模块职责：

- `asr/base.py`：定义 `AsrEngine` 协议、`AsrConfig`、缓存身份类型和通用错误语义。
- `asr/factory.py`：根据 `AsrConfig`、环境密钥和事件 sink 创建具体 engine。
- `asr/mlx.py`：封装 `mlx_whisper`，处理模型加载、语言参数、本地串行推理、文本抽取和片段失败。
- `asr/mimo.py`：迁移现有 MiMo 请求、限流、429 全局退避、重试和响应解析。
- `events.py` 或 `pipeline_events.py`：定义阶段、切片、转写和状态投影事件。
- `state_worker.py` 或 `run_state.py`：实现轻量事件队列消费者，投影 manifest、completed map、segment index 和进度。
- `pipeline.py`：作为 Orchestrator，持有 `AsrEngine`，创建队列和 worker，不接触 provider、模型、API 请求、MLX 导入或重试细节。

现有 `mimo_asr.py` 可以迁移到 `asr/mimo.py`，或保留为兼容 shim 再逐步删除。新代码不再从 pipeline 直接导入 `openai_request` 或 `MiMoTranscriber`。

### 3.1 显示层

显示层由 CLI 创建 `ProgressReporter`，交给 `StateWorker` 使用。它只知道阶段名、计数、重试事件和最终汇总，不理解 ffmpeg、pyannote、ASR provider 或 manifest 结构。

未来如果要增加 JSON logs、WebSocket UI 或交互式 TUI，可以替换或扩展 reporter，而不修改切片 worker 和 ASR worker。

### 3.2 媒体准备层

媒体准备层封装：

- `ffprobe` 元数据读取。
- 输入音频标准化为工作 WAV。
- MPS 预检样本创建。

这一层失败是关键失败，直接中止任务。它可以复用现有 `audio.py` 函数，但 Orchestrator 要把阶段开始和结果事件交给 state/progress 边界处理。

### 3.3 片段拆分层

片段拆分层负责把 diarization 输出变成稳定的 `SpeakerSegment` 清单：

- 清洗重叠或异常区间。
- 映射说话人显示名。
- 合并或处理短段。
- 为每个片段分配稳定 `segment_id`。
- 对过大切片执行拆分规划。

这一层尽量不写磁盘。若切片时才发现 payload 过大，切片 worker 产生 `SegmentsExpanded` 事件，由 `StateWorker` 更新 manifest 和总数。

### 3.4 Worker Runtime

Worker runtime 使用三个队列：

- `slice_queue`: 待切片片段。
- `asr_queue`: 已切好的音频片段。
- `state_queue`: 所有阶段、切片、转写和恢复事件。

切片 worker 消费 `slice_queue`，产出 `SliceReady` 或 `SliceFailed` 事件；成功时同时把 `AudioSlice` 放入 `asr_queue`。

ASR worker 消费 `asr_queue`，调用 `AsrEngine`，产出 `TranscriptSucceeded` 或 `TranscriptFailed` 事件。重试事件由 ASR engine 通过事件 sink 或 callback 产生，最终仍进入 `state_queue`。

`StateWorker` 消费 `state_queue`，串行更新 manifest、内存 completed map、segments_by_id 和 `ProgressReporter`。

### 3.5 StateWorker / RunStateProjector

`StateWorker` 是运行状态投影层，不是业务调度器。它负责：

- 持有 `TaskManifest`。
- 持有 `segments_by_id`。
- 持有 completed map。
- 调用 `ManifestStore.save()`。
- 调用 `ProgressReporter`。
- 提供最终 snapshot 给 Orchestrator。

`StateWorker` 不把 `AudioSlice` 放入 ASR 队列，不决定切片或 ASR 调度，也不调用 ASR engine。

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

ASR engine 不直接调用 `ProgressReporter` 或 `ManifestStore`。需要报告重试时，engine 通过事件 sink 发送 `TranscriptRetrying`：

```python
AsrEventSink = Callable[[object], Awaitable[None]]
```

实现时可收紧为具体事件 union。MiMo engine 用它报告重试；MLX 首版通常只产出最终成功或失败。

## 4.1 事件协议

事件是 worker 和状态投影层之间的唯一事实边界。首版事件包括：

```python
@dataclass(frozen=True)
class StageStarted:
    name: str

@dataclass(frozen=True)
class SegmentTotalChanged:
    total: int

@dataclass(frozen=True)
class SliceReady:
    segment_id: str
    path: Path
    bytes: int

@dataclass(frozen=True)
class SliceFailed:
    segment_id: str
    error: str

@dataclass(frozen=True)
class SegmentsExpanded:
    parent_id: str
    children: list[SpeakerSegment]

@dataclass(frozen=True)
class TranscriptRetrying:
    segment_id: str
    retry_number: int
    max_retries: int

@dataclass(frozen=True)
class TranscriptSucceeded:
    segment_id: str
    text: str

@dataclass(frozen=True)
class TranscriptFailed:
    segment_id: str
    error: str
```

事件投影必须按 `segment_id` 幂等 upsert。系统不要求不同 worker 发出的事件在全局上完全有序；`StateWorker` 要能处理 `TranscriptSucceeded` 先于对应 `SliceReady` 被消费的情况。

`SegmentsExpanded` 替换 manifest 中的父片段，并更新总片段数。已经发布到下游队列的其他片段身份不变。

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
- 通过事件 sink 报告重试。最终成功或失败由 ASR worker 投影为 transcript 事件。

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

## 10. Pipeline 和 worker 改动

`run_pipeline` 的密钥参数从 `mimo_key, hf_token` 调整为通用 runtime 配置：

```python
@dataclass(frozen=True)
class RuntimeConfig:
    hf_token: str
    mimo_api_key: str | None = None
```

运行时校验函数返回 `RuntimeConfig`。`asr/factory.py` 只读取当前 engine 需要的字段；pipeline 不解释这些 secrets。

`_run_segment_workers` 不再创建 `MiMoTranscriber`。它接收已经创建好的 `AsrEngine`，或通过 `PipelineDependencies` 注入测试 engine。

Orchestrator 负责：

- 创建 `slice_queue`、`asr_queue`、`state_queue`。
- 从 manifest 恢复已完成片段，并通过 state event 初始化进度。
- 启动切片 workers、ASR workers 和一个 `StateWorker`。
- 等待切片和 ASR 队列 drain。
- 关闭 state queue，并从 `StateWorker` 获取最终 snapshot。
- 根据 snapshot 生成关键词、输出 TXT/JSON 和退出码。

切片 worker 负责：

- 消费待切片 `SegmentRecord` 或 `SpeakerSegment`。
- 调用 `dependencies.slice_audio`。
- 校验 payload。
- 成功时发送 `SliceReady` 并把 `AudioSlice` 放入 `asr_queue`。
- 失败时发送 `SliceFailed`。
- 若片段过大，发送 `SegmentsExpanded`，并把子片段重新放回 `slice_queue`。

ASR worker 负责：

- 消费 `AudioSlice`。
- 调用 `AsrEngine.transcribe_one()`。
- 成功时发送 `TranscriptSucceeded`。
- 失败时发送 `TranscriptFailed`。
- 在 `config.fail_fast` 打开且发生最终失败时通知 Orchestrator 取消剩余工作。

`StateWorker` 负责：

- 串行处理所有事件。
- 更新 manifest。
- 更新 in-memory `completed` map 和 `segments_by_id`。
- 调用 `ProgressReporter`。
- 根据策略批量或逐事件保存 manifest。首版可以逐事件保存，后续再优化写频率。
- 暴露 `snapshot_completed_segments()` 给 Orchestrator。

首版保存策略以恢复正确性优先：`SliceReady`、`SliceFailed`、`SegmentsExpanded`、`TranscriptSucceeded` 和 `TranscriptFailed` 必须在事件投影后尽快持久化。`TranscriptRetrying` 可以只更新进度，不必写入 manifest。后续如果批量保存，必须保证进程中断后最多重复少量可重试工作，不能丢失已确认成功的转写文本。

切片、manifest 恢复、失败统计、关键词和输出写入的业务语义保持现有流程，但持久化和 progress 调用从 worker 内部迁移到 `StateWorker`。

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
- `test_pipeline_events.py`：验证事件 dataclass、segment_id 幂等投影和 `SegmentsExpanded` 语义。
- `test_state_worker.py`：验证 manifest 更新、completed snapshot、progress 调用、事件乱序 upsert 和 failed/success 状态。
- `test_config.py`：MiMo API key 只在 `--asr mimo` 时必填；默认本地不要求。
- `test_cache.py`：task hash 纳入不透明 ASR identity；改变 engine 或模型会改变 hash；改变并发、重试、debug 不改变 hash。
- `test_pipeline.py`：pipeline 使用 fake engine 和 fake state events，证明上层不依赖 provider/model，worker 不直接调用 manifest save 或 reporter。
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
2. 引入事件类型和 `StateWorker`，先让现有 MiMo 路径通过事件更新 manifest/progress。
3. 将 MiMo 迁移进 `asr/mimo.py`，让 pipeline 通过 engine 调用。
4. 添加 MLX engine，并把默认 ASR 改为 `mlx`。
5. 更新运行时校验、README 和测试。

每一步都保持测试可运行，避免同时改变 pipeline、缓存和真实模型接入导致定位困难。
