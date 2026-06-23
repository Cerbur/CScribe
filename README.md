# CScribe

本地运行的多人录音转写 CLI。CScribe 使用 pyannote 区分说话人，将切分后的音频片段发送给 MiMo ASR，并生成带时间戳、说话人和关键词的 TXT。

完整原始录音不会直接上传，只有说话人分离后的短音频片段会发送给 MiMo API。

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

在 `.env` 中填写：

```dotenv
MIMO_API_KEY=你的_MiMo_API_Key
HF_TOKEN=你的_Hugging_Face_Token
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
| `--num-speakers N` | 自动估计 | 准确的说话人数 |
| `--min-speakers N` | `1` | 自动估计人数下限 |
| `--max-speakers N` | `6` | 自动估计人数上限 |
| `--language {auto,zh,en}` | `auto` | 转写语言 |
| `--device {auto,cpu,cuda,mps}` | `auto` | 说话人分离设备；MPS 为实验性支持 |
| `--concurrency N` | `2` | MiMo 最大并发请求数 |
| `--requests-per-minute N` | `20` | 全局每分钟请求上限 |
| `--max-retries N` | `3` | 每个片段首次失败后的最大重试次数 |
| `--keyword-count N` | `20` | 输出的关键词数量 |
| `--debug-json` | 关闭 | 额外生成 `.segments.json` |
| `--fail-fast` | 关闭 | 任一片段最终失败时立即停止，不生成正式 TXT |
| `--debug` | 关闭 | 输出应用调试日志 |
| `-v, --verbose` | 关闭 | 显示更详细的进度和阶段耗时 |

退出码：`0` 表示全部成功；`1` 表示启动或关键阶段失败；`2` 表示 TXT 已生成，但部分片段转写失败。

## 怎么换 ASR 模型

当前 ASR 模型为 `mimo-v2.5-asr`，尚未提供 CLI 或环境变量配置。更换模型时需要同步修改：

1. `src/mimo_transcriber/mimo_asr.py` 中请求 MiMo API 的 `model`
2. `src/mimo_transcriber/cache.py` 中的 `MIMO_MODEL_ID`

两处必须保持一致。缓存身份包含模型 ID，修改后旧模型的续跑缓存不会被错误复用。新模型还需要兼容当前 MiMo OpenAI 风格接口、`input_audio` 消息格式和 `asr_options.language` 参数。

## 常见问题

- **提示找不到 `ffmpeg` 或 `ffprobe`**：macOS 执行 `brew install ffmpeg`；Ubuntu / Debian 安装 `ffmpeg` 包。

- **Hugging Face 返回 401 / 403**：确认已经接受 pyannote Community-1 的模型条款，并使用具有 Read 权限的 `HF_TOKEN`。

- **MiMo 返回 429、5xx 或请求超时**：程序会自动退避重试。仍频繁失败时，降低 `--concurrency` 或 `--requests-per-minute`。

- **结果中出现 `[该片段识别失败]`**：使用 `--debug` 查看日志。程序默认保留缓存并返回退出码 `2`，再次执行相同命令只重试失败片段。

- **CUDA 不可用或 MPS 自动回退 CPU**：先改用 `--device cpu`。MPS 兼容性取决于 macOS、Python、PyTorch 和 pyannote 的版本组合，可配合 `--debug` 查看原因。

- **第二个相同任务无法启动**：同一任务有进程锁，防止重复发送 API 请求。等待前一个任务结束后再运行。

## 开源协议

[MIT License](LICENSE)
