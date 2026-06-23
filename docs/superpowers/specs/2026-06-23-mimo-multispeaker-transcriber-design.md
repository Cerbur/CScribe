# M4A 多说话人录音转写 CLI 设计规格

日期：2026-06-23

## 1. 项目目标

构建一个可靠、可测试的本地 Python CLI，将多人 M4A 录音转换为带说话人标签、时间戳和关键词的 UTF-8 文本。

处理链路为：

1. 校验运行环境与输入。
2. 探测音频元数据。
3. 将输入标准化为适合说话人分离的 WAV。
4. 使用 pyannote 按音色执行说话人分离。
5. 清洗、合并和拆分说话人区间。
6. 将每个区间转换为 MP3。
7. 并发调用小米 MiMo-V2.5-ASR。
8. 按原始时间顺序合并结果。
9. 本地提取关键词。
10. 原子写入 TXT，并可选写入调试 JSON。

首版不提供 Web 页面、数据库、服务端、离线 ASR、跨录音身份识别或断点续跑。

## 2. 平台与验收范围

支持以下运行环境：

- macOS Apple Silicon，包括 M1 Pro。
- Linux x86_64，可选 NVIDIA CUDA。
- Python 3.11。
- `uv` 管理项目、虚拟环境和依赖。
- FFmpeg 与 ffprobe 作为外部程序。

设备选择规则：

- `--device cpu`：强制 CPU。
- `--device cuda`：要求 CUDA 可用，否则启动失败并给出明确提示。
- `--device auto`：Linux 上 CUDA 可用时选择 CUDA，否则选择 CPU。
- macOS 的 `auto` 使用 CPU；首版不依赖 MPS。

M1 Pro 的首版验收标准是正确、稳定完成，不设置硬性耗时上限，但记录总耗时和各阶段耗时。首轮真实冒烟样本为：

- 文件：`/Users/yuancheng/Downloads/新录音 15.m4a`
- 时长：约 37.7 秒
- 编码：ALAC
- 采样率：48 kHz
- 声道：双声道
- 已知说话人数：2

长录音性能与稳定性验证属于后续里程碑，不阻塞首版交付。

## 3. 推荐架构

采用模块化单流程架构。一次 CLI 调用串联完整处理链路，各阶段通过明确数据对象交接。

```text
参数与环境校验
  -> 音频探测及 WAV 标准化
  -> pyannote 说话人分离
  -> 区间清洗、映射、合并与拆分
  -> MP3 切片
  -> MiMo 并发转写
  -> 关键词提取
  -> TXT 与可选 JSON 原子写入
```

首版不保存可恢复任务状态，但阶段接口不得依赖临时文件的隐式命名或全局状态，以便未来增加 manifest 和断点续跑。

## 4. 项目结构

```text
.
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── src/
│   └── mimo_transcriber/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── pipeline.py
│       ├── audio.py
│       ├── diarization.py
│       ├── segments.py
│       ├── mimo_asr.py
│       ├── keywords.py
│       └── formatter.py
└── tests/
    ├── test_audio.py
    ├── test_formatter.py
    ├── test_segments.py
    ├── test_speaker_mapping.py
    ├── test_keywords.py
    ├── test_mimo_asr.py
    └── test_pipeline.py
```

模块职责：

- `cli.py`：解析参数、配置日志、调用 pipeline、映射退出码。
- `config.py`：加载 `.env` 与环境变量，完成参数及运行环境校验。
- `models.py`：定义跨模块数据结构和状态枚举。
- `pipeline.py`：编排阶段、统计结果、处理关键失败和部分失败。
- `audio.py`：封装 ffprobe、标准化转码、MP3 切片和媒体校验。
- `diarization.py`：延迟加载 pyannote、选择设备、兼容模型返回结构。
- `segments.py`：规范化区间、映射说话人、合并、短段处理和拆分。
- `mimo_asr.py`：请求构造、Base64、并发、限流、超时、重试和响应解析。
- `keywords.py`：本地关键词提取及过滤。
- `formatter.py`：时间格式、TXT、调试 JSON 和原子写入。

`cli.py` 不承载音频处理或网络请求细节。`audio.py` 不感知 MiMo。`mimo_asr.py` 不负责切片或最终输出。

## 5. 核心数据模型

使用带类型注解的 dataclass 或等价清晰结构。

### 5.1 AudioMetadata

至少包括：

- `source_path`
- `duration_seconds`
- `codec`
- `sample_rate`
- `channels`
- `creation_time`

录音时间优先使用 metadata 中的 `creation_time`；缺失时使用输入文件修改时间。最终按运行机器当前本地时区格式化。

### 5.2 SpeakerSegment

至少包括：

- `index`
- `start`
- `end`
- `raw_speaker`
- `display_speaker`
- `text`
- `status`
- `error`

`index` 在完成全部区间后处理后统一分配，并作为异步转写结果恢复时间顺序的稳定键。

### 5.3 RunSummary

至少包括：

- 总耗时和阶段耗时
- 检测到的说话人数
- 最终片段数
- 转写成功数
- 转写失败数
- 输出文件路径
- 临时目录路径（仅保留时）

## 6. CLI 设计

调用形式：

```bash
uv run python -m mimo_transcriber INPUT [OPTIONS]
```

至少支持：

- `input`：必填 M4A 文件。
- `-o, --output`：输出 TXT；默认与输入同目录、同文件名 stem。
- `--num-speakers`：已知准确说话人数。
- `--min-speakers`：自动估计下限，默认 1。
- `--max-speakers`：自动估计上限，默认 6。
- `--language auto|zh|en`：默认 `auto`。
- `--device auto|cpu|cuda`：默认 `auto`。
- `--concurrency`：默认 4。
- `--requests-per-minute`：默认 80。
- `--max-retries`：默认 3。
- `--keyword-count`：默认 20。
- `--keep-temp`
- `--debug-json`
- `--fail-fast`
- `-v, --verbose`

参数校验规则：

- `num-speakers > 0`。
- `min-speakers > 0`。
- `max-speakers >= min-speakers`。
- `concurrency > 0`。
- `requests-per-minute > 0`。
- `max-retries >= 0`。
- `keyword-count >= 0`。
- 指定 `--num-speakers` 时，不向 pyannote 传递 min/max；CLI 可以接受保留默认值，但不得同时让两组约束生效。

## 7. 启动校验与配置

启动时检查：

- 输入存在、是普通文件且可读。
- `ffmpeg` 和 `ffprobe` 均可执行。
- `MIMO_API_KEY` 已配置。
- `HF_TOKEN` 已配置。
- 输出父目录存在或可安全创建。
- CUDA 请求与实际能力一致。

使用 `python-dotenv` 读取当前工作目录的 `.env`，现有环境变量优先。日志和异常中禁止打印 Token。

缺失依赖或配置时，输出具体修复命令或配置方法，不向普通用户暴露无上下文的 Python 堆栈。详细堆栈只在 verbose/debug 场景保留。

## 8. 音频处理

### 8.1 元数据探测

使用 ffprobe 的 JSON 输出读取总时长、编码、采样率、声道和 `creation_time`。所有外部命令使用参数数组、`shell=False`、检查退出码并捕获 stderr。

### 8.2 标准化 WAV

转换规格：

- 16 kHz
- 单声道
- PCM S16LE
- 无视频流

临时文件名固定语义为 `normalized.wav`，位于本次运行专属临时目录。

### 8.3 MP3 切片

每个最终区间生成单独 MP3：

- 16 kHz
- 单声道
- 48 kbps
- 无视频流
- 文件名包含四位 segment index

发送前检查 Base64 字符串长度不超过 10 MB。若超过限制，继续将对应区间等分为更短的连续子区间，重新编号后再切片和检查；不得直接放弃整个任务。正常的 45 秒上限应使超限极少发生，但大小检查仍是最终防线。

### 8.4 临时目录

默认使用 `TemporaryDirectory`，成功或异常均清理。`--keep-temp` 使用可保留的临时目录，并在日志和运行摘要中输出路径。

## 9. 说话人分离

模型固定为：

```text
pyannote/speaker-diarization-community-1
```

使用 `HF_TOKEN` 初始化。模型与重量级依赖延迟加载，使 `--help` 和纯单元测试无需下载模型。

已知说话人数时向 pipeline 传准确人数。未指定时传：

- `min_speakers=1`，或用户值。
- `max_speakers=6`，或用户值。

适配当前 pyannote 返回对象，最终只向下游暴露 `start`、`end` 和 `speaker`，隔离第三方返回结构变化。

不进行语音源分离，不根据文本猜测人物姓名，不保证重叠讲话完全准确。

## 10. 区间后处理与说话人映射

处理顺序固定如下：

1. 按 `start` 升序，并使用原始顺序作为相同起点的稳定次序。
2. 将边界限制在 `[0, audio_duration]`。
3. 丢弃 `end <= start` 的非法区间。
4. 为原始 speaker 按首次出现顺序分配 `说话人 1`、`说话人 2` 等名称。
5. 处理小于 0.4 秒的短区间。
6. 合并同说话人的相邻区间。
7. 将超过 45 秒的区间拆成连续子区间。
8. 分配最终稳定 index。

短区间规则：

- 若前一个区间与短区间属于同一说话人，且间隔不超过 0.8 秒，合并到前一个区间。
- 否则若后一个区间属于同一说话人，且间隔不超过 0.8 秒，合并到后一个区间。
- 两边均可合并时优先前一个区间。
- 无法合并时跳过，并记录 debug 日志。

普通合并规则：

- 原始 speaker 相同。
- 后段开始时间减前段结束时间不超过 0.8 秒。
- 合并后的连续区间可以超过 45 秒，随后由拆分步骤处理。

拆分不添加重叠或额外上下文，保持连续、无空洞的子区间。

相同 `raw_speaker` 在单个文件内始终映射到同一展示名称；展示编号不依赖 `SPEAKER_00` 中的数字。

## 11. MiMo ASR

使用 OpenAI 兼容客户端：

- Base URL：`https://api.xiaomimimo.com/v1`
- Model：`mimo-v2.5-asr`
- Key：`MIMO_API_KEY`
- 默认单次请求超时：120 秒

消息包含 Data URL 形式的 MP3 Base64，并在 `extra_body.asr_options.language` 中传入语言。

响应解析兼容：

- `message.content` 为字符串。
- `message.content` 为空。
- SDK 返回内容对象或内容列表。

无法提取有效文本视为该片段失败。成功文本只执行：

- 去除首尾空白。
- 将连续空白压缩为一个空格。
- 保留原有中英文标点。

不得润色、改写或补全。

## 12. 并发、限流与重试

- `asyncio.Semaphore` 控制最大并发。
- 共享异步限流器控制全局每分钟请求数。
- 429、连接异常、超时和 5xx 可重试。
- 4xx 中除 429 外默认不重试。
- 指数退避加入小幅随机抖动，避免同步重试。
- `max-retries` 表示首次请求失败后允许的额外重试次数。
- 所有结果最终按 segment index 排序。

普通日志不得打印 API Key 或完整 Base64。

## 13. 失败语义与退出码

错误分三类：

### 13.1 启动错误

输入、依赖、Token、参数或设备不合法时立即停止，不创建正式输出文件。

### 13.2 关键阶段错误

ffprobe、标准化转码、pyannote 初始化或 diarization 整体失败时停止，不生成可能误导用户的正式 TXT。

### 13.3 片段错误

默认行为：

- 记录 index、start、end 和最终错误。
- 该片段文本写为 `[该片段识别失败]`。
- 继续处理其余片段。
- 仍生成 TXT 和可选 JSON。
- 进程返回非零退出码。

`--fail-fast` 在首个片段最终失败后取消尚未开始的请求并终止，不写正式 TXT。已经生成的临时文件按 `--keep-temp` 规则处理。

退出码：

- `0`：全部成功。
- `1`：启动或关键阶段失败。
- `2`：输出已生成，但存在片段识别失败。

## 14. 关键词提取

使用 `jieba.analyse.extract_tags` 从所有成功识别文本提取关键词，默认最多 20 个。

过滤：

- 空字符串。
- 纯数字。
- 单个标点。
- 常见停用词。
- 单个无意义汉字。

保留有意义的英文技术术语及原始大小写；输出去重并保持提取器给出的相关性顺序。没有有效关键词时保留空关键词行，不使任务失败。失败占位文本不参与关键词提取。

## 15. 输出设计

### 15.1 TXT

默认输出路径为输入文件同目录下的 `<stem>.txt`。编码为 UTF-8，文件末尾保留换行。

格式：

```text
2026年6月15日 下午 11:32|1小时 8分钟 33秒

关键词:
微服务、缓存、Agent、RAG

文字记录:
说话人 1 00:02
喂，哈喽。

说话人 2 00:03
你好，听得到的。
```

规则：

- 时长始终显示秒。
- 小于一小时的时间戳为 `MM:SS`。
- 大于等于一小时为 `HH:MM:SS`。
- 时间戳使用 segment 开始时间向下取整到秒。
- 不输出原始 speaker 标签。
- 每个片段之间空一行。
- 使用同目录临时文件写入，flush/close 成功后通过 `os.replace` 原子替换。

### 15.2 调试 JSON

`--debug-json` 额外生成 `<output-stem>.segments.json`，包括：

- source
- duration_seconds
- speakers
- segments

每个 segment 至少包括 index、start、end、raw_speaker、speaker、text、status 和 error。JSON 也使用 UTF-8 和原子写入。

## 16. 日志与可观测性

默认日志显示：

- 当前阶段。
- 已处理/总片段数。
- 重试摘要。
- 最终成功、失败和耗时汇总。
- 输出路径。

`--verbose` 增加：

- ffprobe 解析摘要。
- 区间被裁剪、合并、跳过或拆分的原因。
- 每个片段的开始、结束、耗时和重试次数。
- 设备选择依据。

日志不包含密钥、完整 Base64 或完整隐私音频内容。

## 17. README

README 必须以项目介绍开头，并提供醒目的 `Quick Start` 章节。内容至少包括：

- 项目能力、适用场景与数据流。
- Python、uv、FFmpeg 和可选 CUDA 要求。
- macOS Homebrew 安装命令。
- Linux 安装提示。
- Hugging Face 注册、接受模型条款及创建 Read token 的步骤。
- `MIMO_API_KEY`、`HF_TOKEN` 和 `.env` 配置。
- `uv sync` 安装。
- 已知人数、自动估计、调试和保留临时文件示例。
- 全部 CLI 参数。
- M1 Pro 使用 CPU、Linux 可使用 CUDA 的说明。
- 首次运行下载模型的说明。
- 仅切片发送给 MiMo、完整原始音频不直接上传的隐私边界。
- 说话人编号只在当前录音有效。
- 重叠讲话、短插话和音质对准确率的影响。
- 常见错误排查。

Quick Start 应能让已安装 FFmpeg、已配置两个 Token 的用户复制以下流程运行：

```bash
uv sync
uv run python -m mimo_transcriber meeting.m4a --num-speakers 2
```

## 18. 测试策略

### 18.1 单元测试

无需真实 Token、网络或模型下载，覆盖：

- `MM:SS` 与 `HH:MM:SS`。
- 中文时长格式。
- 录音时间格式。
- 同说话人相邻区间合并。
- 不同说话人不合并。
- 短区间向前、向后合并及跳过。
- 超长区间拆分。
- 首次出现顺序映射。
- TXT 和 JSON 格式。
- 原子写入。
- 关键词提取与过滤。
- MiMo 请求结构。
- 字符串、空值和对象响应解析。
- 可重试与不可重试错误。
- 指数退避与最大重试次数。
- 并发和结果排序。
- 部分失败仍生成结果并返回退出码 2。
- fail-fast 不生成正式输出。

### 18.2 本地集成测试

测试运行时生成一个很短的 WAV fixture，并在本机真实调用 FFmpeg/ffprobe 验证：

- 元数据读取。
- M4A/WAV 到标准 WAV 的转码。
- WAV 到 MP3 的切片。
- formatter 输出。

缺少 FFmpeg 时该组测试应明确 skip 并说明原因，单元测试仍可运行；正式验收环境必须安装 FFmpeg。

### 18.3 真实冒烟测试

使用真实 Token、模型和短录音：

```bash
uv run python -m mimo_transcriber \
  "/Users/yuancheng/Downloads/新录音 15.m4a" \
  --num-speakers 2 \
  --language auto \
  --verbose
```

检查：

- 成功读取和标准化 ALAC M4A。
- pyannote 输出并稳定映射两个说话人。
- MiMo 收到切片并返回文本。
- TXT 格式、排序、时间戳和结尾换行正确。
- 输出中没有原始 speaker 标签。
- 日志及文件不泄漏 Token。
- 汇总包含耗时与片段统计。

自动检查结构和失败语义；人工抽查说话人分离与文字准确性。

## 19. 交付验收

完成实现后执行：

```bash
uv sync
uv run pytest
uv run python -m mimo_transcriber --help
```

随后执行真实冒烟测试，并汇报：

1. 创建的文件。
2. 核心实现与模块边界。
3. 单元及集成测试结果。
4. 真实冒烟测试结果。
5. 本地安装和实际运行命令。
6. M1 Pro 的实际耗时。
7. 仍存在的准确率边界。

## 20. 后续但不属于首版

- 长录音性能与稳定性验收。
- manifest、缓存和断点续跑。
- MPS 专项支持。
- 本地离线 ASR。
- 跨录音真实人物识别。
- Web 或服务端形态。
