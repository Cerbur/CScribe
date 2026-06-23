# CScribe

一个本地运行的 Python CLI：使用 FFmpeg 标准化 M4A，使用 pyannote 按音色区分说话人，将切割后的短 MP3 片段发送给 MiMo-V2.5-ASR，最后生成带说话人、时间戳和关键词的 UTF-8 TXT。

完整原始录音不会直接发送给 MiMo；只有 diarization 后的音频片段会上传。

## Quick Start

```bash
brew install ffmpeg
uv sync
export MIMO_API_KEY="..."
export HF_TOKEN="..."
uv run python -m mimo_transcriber meeting.m4a --num-speakers 2
```

## 环境要求

- Python 3.11
- uv
- FFmpeg 与 ffprobe
- macOS Apple Silicon 默认使用 CPU，可显式试用实验性 MPS
- Linux 可选 NVIDIA CUDA

## Hugging Face 授权

登录 Hugging Face，接受 `pyannote/speaker-diarization-community-1` 的模型条款，然后创建 Read 权限 Token 并保存为 `HF_TOKEN`。首次运行会下载模型。

## 使用示例

```bash
uv run python -m mimo_transcriber meeting.m4a --num-speakers 3 --language zh
uv run python -m mimo_transcriber meeting.m4a --language auto
uv run python -m mimo_transcriber meeting.m4a --debug-json --keep-temp --verbose
uv run python -m mimo_transcriber meeting.m4a --device mps --verbose
```

## 准确率与隐私边界

- 说话人编号只在当前录音内有效，不能跨录音识别真实人物。
- 重叠讲话、极短插话、背景噪声和低质量音频会降低准确率。
- CPU 可以运行，长音频的 diarization 可能较慢。
- 音频片段会发送给 MiMo API；完整原始文件不会直接上传。

## Linux 安装

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

安装 uv 后运行 `uv sync`。带 NVIDIA GPU 的 Linux 可以使用 `--device cuda`；否则使用默认的 `--device auto`。

## 环境变量

复制 `.env.example` 为 `.env`，或在 shell 中设置：

```bash
export MIMO_API_KEY="..."
export HF_TOKEN="..."
```

已有环境变量优先于 `.env`。不要把 `.env` 或真实 Token 提交到 Git。

## CLI 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | 本地 M4A 输入文件 |
| `-o, --output` | 输入同名 TXT | 输出路径 |
| `--num-speakers` | 自动估计 | 已知准确说话人数 |
| `--min-speakers` | `1` | 自动估计下限 |
| `--max-speakers` | `6` | 自动估计上限 |
| `--language` | `auto` | `auto`、`zh` 或 `en` |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或实验性 `mps` |
| `--concurrency` | `4` | MiMo 最大并发 |
| `--requests-per-minute` | `80` | 全局每分钟请求上限 |
| `--max-retries` | `3` | 首次失败后的最大重试次数 |
| `--keyword-count` | `20` | 关键词数量 |
| `--keep-temp` | 关闭 | 保留 WAV/MP3 临时文件 |
| `--debug-json` | 关闭 | 额外生成 `.segments.json` |
| `--fail-fast` | 关闭 | 首段最终失败即停止且不写正式 TXT |
| `-v, --verbose` | 关闭 | 输出调试日志和阶段耗时 |

## 输出与退出码

TXT 第一行是录音时间和时长，随后是关键词与按时间排序的说话人片段。退出码 `0` 表示全部成功，`1` 表示启动或关键阶段失败，`2` 表示 TXT 已生成但存在失败片段。

## 实验性 Apple MPS

`--device mps` 会先检查当前 PyTorch 的 MPS 构建与运行时状态，再使用标准化录音的前 10 秒运行真实 pyannote Community-1 预检。预检成功后会复用同一个 MPS pipeline 处理完整录音。

MPS 不可用、预检失败或完整运行失败时，程序会自动回退 CPU。完整运行回退不会重复 FFmpeg 标准化，也不会重复已经完成的 MiMo 请求。默认的 `--device auto` 暂时不会选择 MPS。

应用只提供诊断，不会自动更换 PyTorch 或修改 macOS。兼容性取决于 Apple 芯片、macOS、Python、PyTorch 与 pyannote 的具体版本组合。

评估一台机器是否值得使用 MPS 时，使用同一段 1～3 分钟、至少两位说话人的录音分别运行 CPU 和 MPS 各三次。比较 diarization 中位耗时；只有 MPS 稳定完成且至少快 20% 时才建议日常使用。

## 常见问题

- `ffmpeg/ffprobe not found`：macOS 运行 `brew install ffmpeg`；Ubuntu/Debian 安装 `ffmpeg` 包。
- Hugging Face 401/403：确认已接受 Community-1 模型条款，并使用 Read 权限 `HF_TOKEN`。
- CUDA 不可用：改用 `--device cpu`，或检查 NVIDIA 驱动和 PyTorch CUDA 环境。
- MiMo 429/5xx/超时：程序会按指数退避重试；可降低 `--concurrency` 或 `--requests-per-minute`。
- 某些片段显示 `[该片段识别失败]`：查看 verbose 日志；默认仍会生成结果并返回退出码 2。
- MPS 回退 CPU：使用 `--verbose` 查看 `MPS 构建支持` 和 `MPS 运行时可用`；确认当前 Python 为 arm64，并检查 macOS、PyTorch 与 pyannote 版本兼容性。
