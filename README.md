# CScribe

本地运行的多人录音转写 CLI。CScribe 使用 pyannote 区分说话人，默认通过本地 MLX Whisper 转写切分后的音频片段，并生成带时间戳、说话人和关键词的 TXT。

默认模式不会上传音频片段。需要使用 MiMo ASR 时，可以通过 `--asr mimo` 显式选择远端后端。

## 功能

- 多说话人分离，支持指定或自动估计说话人数
- 中文、英文和自动语言识别
- 并发转写、限流、失败重试和中断续跑
- 输出带时间戳、说话人和关键词的 UTF-8 TXT
- 支持 CPU、CUDA，以及实验性的 Apple MPS
- 可选输出片段级调试 JSON

## Quick Start

需要 Python 3.11、[uv](https://docs.astral.sh/uv/) 和 FFmpeg。

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
# sudo apt-get update && sudo apt-get install -y ffmpeg

uv sync
cp .env.example .env
```

默认本地转写只需要：

```dotenv
HF_TOKEN=你的_Hugging_Face_Token
```

使用 MiMo 时额外填写：

```dotenv
MIMO_API_KEY=你的_MiMo_API_Key
```

首次运行前，需要在 Hugging Face 接受 [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) 的使用条款，并创建 Read 权限 Token。

```bash
uv run mimo-transcriber meeting.m4a --num-speakers 2
```

默认在录音旁生成同名 `.txt` 文件。相同任务中断后，再次执行相同命令会自动复用 `/tmp/cscribe/` 中的有效缓存。

## CLI 参数

```text
uv run mimo-transcriber INPUT [参数]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `INPUT` | 必填 | 本地 M4A 文件 |
| `-o, --output PATH` | 输入文件同名 TXT | 输出路径 |
| `--num-speakers N` | 自动估计 | 准确的说话人数；显式指定时优先于 `--conversation-mode` |
| `--min-speakers N` | `1` | 自动估计人数下限 |
| `--max-speakers N` | `6` | 自动估计人数上限 |
| `--conversation-mode {auto,two-person,multi}` | `auto` | 对话模式；`two-person` 在未显式指定人数时按 2 人运行 pyannote |
| `--diarization-stabilizer {off,conservative,balanced,aggressive}` | `balanced` | 说话人归属后处理强度；`off` 关闭稳定器 |
| `--diarization-model MODEL` | `pyannote/speaker-diarization-community-1` | pyannote 说话人分离模型 ID |
| `--language {auto,zh,en}` | `auto` | 转写语言 |
| `--device {auto,cpu,cuda,mps}` | `auto` | 说话人分离设备；MPS 为实验性支持 |
| `--paragraph-mode {off,conservative,balanced,aggressive}` | `balanced` | 同说话人连续片段合并强度；`off` 关闭合并 |
| `--paragraph-gap 秒` | 按模式 | 段落合并的间隔阈值 |
| `--paragraph-max-duration 秒` | 按模式 | 合并后单块最大时长 |
| `--paragraph-max-chars N` | `900` | 合并后单块最大字符数 |
| `--no-paragraph-merge` | 关闭 | `--paragraph-mode off` 的易用别名 |
| `--asr {mlx,mimo}` | `mlx` | ASR 引擎；默认本地 MLX Whisper，`mimo` 为远端 MiMo |
| `--stt-model MODEL` | 引擎默认值 | STT 模型；由所选 ASR 引擎解释 |
| `--asr-prompt TEXT` | 关闭 | MiMo ASR 提示，引导保留专有名词；本地 MLX 忽略 |
| `--terms-file PATH` | 关闭 | 术语表，影响 MiMo 转写与 ASR 缓存身份 |
| `--no-term-correction` | 关闭 | 禁用转写后的显式术语映射纠错 |
| `--concurrency N` | `2` | ASR worker 数量；MiMo 可并发请求，MLX 首版内部串行推理 |
| `--requests-per-minute N` | `20` | MiMo 每分钟请求上限；本地 MLX 忽略 |
| `--max-retries N` | `3` | 每个片段首次失败后的最大重试次数 |
| `--keyword-count N` | `20` | 输出的关键词数量 |
| `--debug-json` | 关闭 | 额外生成 `.segments.json` |
| `--fail-fast` | 关闭 | 任一片段最终失败时立即停止，不生成正式 TXT |
| `--debug` | 关闭 | 输出应用调试日志 |
| `-v, --verbose` | 关闭 | 显示更详细的进度和阶段耗时 |

退出码：`0` 表示全部成功；`1` 表示启动或关键阶段失败；`2` 表示 TXT 已生成，但部分片段转写失败。

## 段落合并输出

CScribe 默认会在输出 TXT 前合并同一说话人的连续短片段。ASR 内部切片不会被改变，`--debug-json` 仍会保留内部片段，并额外输出最终展示 blocks。

常用选项：

```bash
uv run mimo-transcriber meeting.m4a --paragraph-mode conservative
uv run mimo-transcriber meeting.m4a --paragraph-mode aggressive --paragraph-gap 2.5
uv run mimo-transcriber meeting.m4a --no-paragraph-merge
```

## 怎么选择 ASR 引擎和 STT 模型

默认使用本地 MLX Whisper：

```bash
uv run mimo-transcriber meeting.m4a
```

指定本地模型：

```bash
uv run mimo-transcriber meeting.m4a --asr mlx --stt-model mlx-community/whisper-small
```

使用 MiMo：

```bash
uv run mimo-transcriber meeting.m4a --asr mimo --stt-model mimo-v2.5-asr
```

`--stt-model` 对上层流水线透明，由所选 ASR 引擎解释。切换 ASR 引擎或模型会改变缓存身份，避免复用旧模型的转写结果。

## 2 人对话说话人稳定

2 人录音建议固定说话人数：

```bash
uv run mimo-transcriber meeting.m4a --conversation-mode two-person
```

这会在未显式传 `--num-speakers` 时按 `--num-speakers 2` 运行 pyannote，并启用默认 `balanced` 后处理稳定器。显式 `--num-speakers` 始终优先于 `--conversation-mode`。稳定器只调整说话人归属与重叠去重，不会修改转写文本。对照调试：

```bash
uv run mimo-transcriber meeting.m4a --diarization-stabilizer off --debug-json
uv run mimo-transcriber meeting.m4a --diarization-stabilizer aggressive --debug-json
```

开启 `--debug-json` 时，调试 JSON 会包含 `speaker_stability` 诊断（`dropped_overlaps`、`relabeled_islands` 等）。如需 A/B 本地模型：

```bash
uv run mimo-transcriber meeting.m4a --diarization-model pyannote/speaker-diarization-community-1
```

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

## 常见问题

- **提示找不到 `ffmpeg` 或 `ffprobe`**：macOS 执行 `brew install ffmpeg`；Ubuntu / Debian 安装 `ffmpeg` 包。

- **Hugging Face 返回 401 / 403**：确认已经接受 pyannote Community-1 的模型条款，并使用具有 Read 权限的 `HF_TOKEN`。

- **MiMo 返回 429、5xx 或请求超时**：程序会自动退避重试。仍频繁失败时，降低 `--concurrency` 或 `--requests-per-minute`。

- **结果中出现 `[该片段识别失败]`**：使用 `--debug` 查看日志。程序默认保留缓存并返回退出码 `2`，再次执行相同命令只重试失败片段。

- **CUDA 不可用或 MPS 自动回退 CPU**：先改用 `--device cpu`。MPS 兼容性取决于 macOS、Python、PyTorch 和 pyannote 的版本组合，可配合 `--debug` 查看原因。

- **首次本地 MLX 转写卡住或非常慢**：首次运行需要下载约 1.5 GB 的 Whisper 权重到项目本地的 `<repo>/.models/`（可用 `CSCRIBE_MODEL_CACHE` 环境变量改路径）。下载走直连以绕过系统代理与 `hf_xet`，并支持断点续传与自动重试；网络较慢时会持续较长时间，可随时中断后再次执行相同命令继续下载。需要更小的模型可加 `--stt-model mlx-community/whisper-small`。

- **模型缓存都放在哪里**：说话人分离（pyannote）和 MLX Whisper 权重都缓存在项目目录的 `<repo>/.models/` 下（已 gitignore，不进版本库）。其中 pyannote 走 `huggingface_hub`，缓存根目录是 `<repo>/.models/hf`（即 `HF_HOME`），可在启动前用 `HF_HOME` 环境变量覆盖；MLX 权重用自定义直连下载器，单独存放在 `<repo>/.models/` 平铺目录，可用 `CSCRIBE_MODEL_CACHE` 覆盖。

- **代理环境下下载停滞在 0 字节**：本机的系统代理会拖住大文件传输。CScribe 的模型下载默认直连，不受系统 `*_PROXY` 与 macOS 系统代理影响。

- **第二个相同任务无法启动**：同一任务有进程锁，防止重复发送 API 请求。等待前一个任务结束后再运行。

## 开源协议

[MIT License](LICENSE)
