# 说话人归属稳定性设计规格

日期：2026-06-24

## 1. 目标

CScribe 当前依赖 `pyannote/speaker-diarization-community-1` 产生说话人片段，再做基础清洗和合并。2 人对话末段偶发出现同一句话归给两个说话人、短片段漂移、说话人标签不稳定。目标是在保持本地运行的前提下，通过 2 人场景约束和后处理稳定器，让归属准确率更接近飞书。

本次设计要达成：

- 2 人对话显式或自动使用 `num_speakers=2`。
- 后处理能消解高度重叠的重复片段。
- 后处理能平滑短时 speaker island。
- 稳定器只改变 speaker 归属和重叠边界，不改 ASR 文本。
- 模型升级或替换保留为可选配置，不阻塞第一轮改进。

非目标：

- 不训练 diarization 模型。
- 不接入云端 diarization。
- 不做声纹注册或人工标注 UI。
- 首版不实现 embedding 级重聚类。

## 2. 用户体验

2 人对话推荐命令：

```bash
uv run mimo-transcriber meeting.m4a --num-speakers 2 --diarization-stabilizer balanced
```

如果用户没有传 `--num-speakers`，但传了 `--conversation-mode two-person`，系统等价于 `num_speakers=2`：

```bash
uv run mimo-transcriber meeting.m4a --conversation-mode two-person
```

用户可以关闭稳定器对照：

```bash
uv run mimo-transcriber meeting.m4a --diarization-stabilizer off --debug-json
```

## 3. 推荐架构

新增 `speaker_stability.py`，在 `process_segments()` 之后、切片之前运行。它接收 pyannote 清洗后的内部片段，输出稳定后的内部片段。

数据流：

```text
run_diarization()
  -> process_segments(raw)
  -> stabilize_speakers()
  -> prepare_audio_segments()
  -> ASR
```

模块职责：

- `diarization.py`：继续负责模型加载和 raw turns。
- `segments.py`：继续负责 clipping、短段合并、长段拆分和 `segment_id`。
- `speaker_stability.py`：负责重叠消解、短孤岛平滑、2 人模式约束。
- `config.py` / `cli.py`：暴露稳定器配置和 2 人模式。

稳定器不读取音频、不调用 pyannote、不调用 ASR。

## 4. 稳定规则

### 4.1 2 人模式约束

配置：

```python
conversation_mode: Literal["auto", "two-person", "multi"] = "auto"
diarization_stabilizer: Literal["off", "conservative", "balanced", "aggressive"] = "balanced"
```

规则：

- 如果 `conversation_mode="two-person"` 且 `num_speakers is None`，运行时传给 pyannote 的参数为 `num_speakers=2`。
- 如果用户显式传 `--num-speakers`，尊重用户参数。
- 如果 `conversation_mode="multi"`，不自动固定人数。

### 4.2 重叠消解

对相邻或重叠片段，计算 overlap：

```python
overlap = min(a.end, b.end) - max(a.start, b.start)
ratio = overlap / min(a.duration, b.duration)
```

默认规则：

- `ratio >= 0.8` 且 speaker 不同：认为是重复归属候选。
- 保留与前后上下文 speaker 更一致的片段。
- 如果上下文无法判断，保留时长更长的片段。
- 被丢弃片段记录 debug reason。

### 4.3 短孤岛平滑

模式阈值：

- `conservative`: `max_island_duration=1.2s`, `max_gap=0.5s`
- `balanced`: `max_island_duration=2.0s`, `max_gap=1.0s`
- `aggressive`: `max_island_duration=3.0s`, `max_gap=1.5s`

规则：

```text
A turn, then B_short, then A turn  =>  A turn, then A_short, then A turn
```

仅当：

- 中间片段 speaker 与前后都不同。
- 前后片段 speaker 相同。
- 中间片段 duration 小于阈值。
- 两侧 gap 都小于阈值。

### 4.4 输出诊断

当 `--debug-json` 开启时，额外输出：

```json
"speaker_stability": {
  "enabled": true,
  "mode": "balanced",
  "dropped_overlaps": 2,
  "relabeled_islands": 3
}
```

## 5. 模型选择

首版不默认换模型，但把模型 id 配置化：

```python
diarization_model: str = "pyannote/speaker-diarization-community-1"
```

CLI：

```bash
--diarization-model pyannote/speaker-diarization-community-1
```

缓存身份包含 model id，避免不同模型复用 diarization 结果。

后续可独立 A/B：

- `pyannote/speaker-diarization-community-1`
- pyannote 其他本地可用 diarization pipeline
- DiariZen 本地 pipeline

## 6. 错误处理

- 稳定器输入为空时返回空列表。
- 稳定器不会产生负时长片段。
- 丢弃重叠片段后重新分配连续 `index` 和 `segment_id`。
- 如果稳定器内部发现片段顺序异常，先按 `sort_key()` 排序。
- 如果所有片段被消解为空，回退到原始 `process_segments()` 输出并记录 warning。

## 7. 测试策略

单元测试覆盖：

- `conversation_mode="two-person"` 自动固定 `num_speakers=2`。
- 显式 `--num-speakers` 优先级高于 conversation mode。
- 高度重叠不同 speaker 片段消解。
- 短孤岛重标。
- 真实抢话式部分重叠不被错误删除。
- 稳定器关闭时保持原片段。
- `segment_id` 在稳定后连续。

集成测试覆盖：

- pipeline 使用稳定后的片段进入切片队列。
- debug JSON 输出稳定器统计。
- diarization model id 改变时 cache hash 改变。

## 8. 验收标准

- 2 人对话默认推荐路径能稳定传 `num_speakers=2`。
- 同一句话被两个说话人重复输出的概率明显下降。
- 短时 speaker drift 被平滑，且不会合并不同人的完整发言。
- 不需要联网服务或额外人工标注。
- 现有 diarization 和 pipeline 测试继续通过。
