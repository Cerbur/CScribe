# Experimental MPS Diarization Design

## Summary

CScribe will add an experimental Apple Metal Performance Shaders (MPS) execution path for local pyannote speaker diarization. Users must opt in explicitly with `--device mps`. The application will diagnose the local PyTorch MPS runtime, run the real Community-1 pipeline against a short audio sample, and use MPS for the complete recording only after that preflight succeeds.

Any MPS failure safely falls back to CPU for diarization. Existing CPU and CUDA behavior remains unchanged, and `--device auto` continues to choose CUDA on supported Linux systems and CPU elsewhere.

MPS is considered recommended on a tested machine only when it completes reliably and reduces median diarization time by at least 20 percent compared with CPU.

## Goals

- Allow Apple Silicon users to opt into MPS-backed pyannote diarization.
- Detect and explain PyTorch or operating-system conditions that make MPS unavailable.
- Exercise the real pyannote pipeline before committing a long recording to MPS.
- Reuse a successfully preflighted MPS pipeline for the full recording.
- Fall back to CPU without repeating audio normalization or downstream transcription.
- Preserve concise user-facing diagnostics and detailed verbose logs.
- Keep device selection, pipeline construction, and diarization independently testable.

## Non-Goals

- MPS will not become the default for `--device auto` in this iteration.
- The application will not install or replace PyTorch, alter macOS, or repair the Python environment automatically.
- The application will not split pyannote internals across MPS and CPU devices.
- The application will not run CPU and MPS benchmarks before every transcription.
- The application will not guarantee that every pyannote or PyTorch release supports MPS.
- This work will not change MiMo API transcription, audio segmentation, or output formatting.

## User Interface

The CLI device option becomes:

```text
--device auto|cpu|cuda|mps
```

Device semantics are:

- `auto`: choose CUDA when the existing Linux CUDA conditions are satisfied; otherwise choose CPU.
- `cpu`: require CPU execution.
- `cuda`: require CUDA availability and fail clearly when it is unavailable.
- `mps`: request the experimental MPS path, including diagnostics, real-pipeline preflight, and automatic CPU fallback.

Selecting `mps` does not promise that MPS will be used. It requests an attempt under the fallback policy defined below.

## Architecture

### DeviceCapabilities

`DeviceCapabilities` is an immutable snapshot of accelerator support. It contains the facts needed for device decisions without loading pyannote models:

- CUDA availability.
- MPS build support from `torch.backends.mps.is_built()`.
- MPS runtime availability from `torch.backends.mps.is_available()`.
- Platform and machine architecture when useful for diagnostics.

Capability collection must not mutate the environment or trigger package installation.

### DeviceDecision

`DeviceDecision` records both user intent and the actual execution result:

- `requested_device`
- `selected_device`
- `mps_built`
- `mps_available`
- `preflight_elapsed_seconds`
- `fallback_category`
- `fallback_reason`

Fields unrelated to a request may be absent. The decision object is suitable for logging and tests and must not contain tokens or secret-bearing exception representations.

### Pipeline Selection

`select_diarization_pipeline(...)` owns pipeline construction and the experimental MPS decision:

1. Collect capabilities.
2. Resolve ordinary CPU, CUDA, and automatic requests using existing semantics.
3. For an MPS request, reject unavailable runtime conditions into a CPU decision.
4. Build a Community-1 pipeline on MPS.
5. Run that pipeline on the preflight sample.
6. Return the reusable pipeline and an MPS decision when preflight succeeds.
7. Dispose of the failed MPS pipeline, release references, clear the MPS cache when supported, and construct a CPU pipeline when preflight fails.

The returned value contains both the selected pipeline and `DeviceDecision`. Pipeline selection does not perform full-file diarization.

### Full Diarization

`diarize_audio(...)` consumes an already-created pipeline and the speaker-count constraints. It applies the pipeline to the complete normalized recording and converts the Community-1 output into CScribe `SpeakerSegment` objects.

Keeping construction outside this function allows the preflighted pipeline to be reused and makes a full-run retry explicit.

## Data Flow

The pipeline runs in this order:

```text
Probe source metadata
  -> normalize the complete recording once
  -> create a short preflight sample
  -> collect device capabilities
  -> select and preflight the diarization pipeline
  -> diarize the complete normalized recording
  -> process speaker segments
  -> create MP3 segments
  -> send segments to MiMo
  -> write outputs
```

The preflight sample is derived from the already normalized WAV. It targets 10 seconds and is capped by the recording duration. It must contain decodable audio. If the first window cannot exercise the pipeline because it contains no usable speech, the preflight is treated as inconclusive and falls back to CPU rather than scanning arbitrary parts of the recording.

The first implementation deliberately favors predictable cost over sophisticated speech-window selection.

## MPS Preflight

Preflight must call the same Community-1 pipeline class and speaker constraints used by the complete run. A generic tensor operation is insufficient because it cannot identify unsupported operators inside segmentation, embedding, or related pyannote components.

The preflight result is used only as a compatibility gate. Its speaker labels and segments are discarded.

On success:

- Record preflight elapsed time.
- Retain and reuse the loaded MPS pipeline.
- Log that complete diarization is starting on MPS.

On failure:

- Categorize and log the failure.
- Dispose of the MPS pipeline before creating the CPU pipeline.
- Return a CPU selection and fallback decision.

## Full-Run Recovery

Preflight cannot guarantee that a long recording will fit memory or avoid every unsupported path. If complete MPS diarization fails:

1. Categorize the failure as `full_run_failed` while retaining a more specific internal cause for verbose logging.
2. Release the MPS pipeline and clear the MPS cache when supported.
3. Construct a CPU pipeline.
4. Repeat only complete diarization against the existing normalized WAV.
5. Continue with segment processing and MiMo transcription after CPU diarization succeeds.

Audio probing and normalization must not be repeated. MPS receives one preflight attempt and one complete-run attempt; there is no retry loop between devices.

If CPU diarization also fails, the application raises the existing fatal diarization error with both the CPU failure and a concise note that MPS previously fell back. It does not proceed to potentially misleading transcript output.

## Failure Categories

Stable categories support tests and concise diagnostics:

- `not_built`: the installed PyTorch package lacks MPS support.
- `runtime_unavailable`: PyTorch was built with MPS but cannot use it in the current runtime.
- `unsupported_operator`: an operation required by the real pipeline is not implemented for MPS.
- `out_of_memory`: MPS allocation or device-memory exhaustion.
- `preflight_failed`: another error occurred during the short real-pipeline check.
- `full_run_failed`: preflight succeeded but complete MPS diarization failed.

Classification must use exception types and stable message fragments conservatively. Unknown exceptions become the broader applicable category instead of pretending to be precisely diagnosed.

## Logging and Diagnostics

Normal logs should expose progress because diarization can otherwise appear frozen:

```text
正在检查 MPS 环境
正在使用 10 秒样本预检 pyannote
MPS 预检通过，耗时 4.2 秒
正在使用 MPS 处理 52:21 音频
```

Fallback logs should state the useful facts:

```text
请求设备: MPS
MPS 构建支持: 是
MPS 运行时可用: 否
已回退 CPU
建议检查 PyTorch、macOS 版本以及当前 Python 架构
```

Normal logs contain a sanitized, concise reason. `--verbose` may include the original traceback and underlying exception chain. Hugging Face tokens, MiMo keys, request headers, and other credentials must never be included in either form.

The diagnostics explain environment problems but do not attempt automatic remediation.

## Resource Cleanup

MPS failure recovery must:

- Remove strong references to the failed pipeline before constructing the CPU pipeline.
- Invoke `torch.mps.empty_cache()` only when the API exists and MPS was initialized.
- Avoid assuming that cache cleanup guarantees memory reclamation.
- Never terminate or restart the process solely to switch to CPU.

Cleanup failures are debug-level information and must not prevent CPU fallback.

## Testing

### Unit Tests

- The CLI accepts `mps`.
- `auto` on macOS still selects CPU.
- Explicit MPS with `is_built() == false` chooses CPU with `not_built`.
- Explicit MPS with built but unavailable runtime chooses CPU with `runtime_unavailable`.
- A successful preflight returns and reuses the same MPS pipeline.
- Unsupported operators, out-of-memory errors, and unknown preflight errors receive the correct categories.
- Failed cleanup does not block CPU fallback.
- Existing CPU and CUDA decisions retain their behavior.
- Decision and normal log formatting do not leak supplied tokens.

### Pipeline Tests

- Full MPS success performs one normalization, one preflight call, and one complete MPS diarization.
- Preflight failure performs one normalization and complete CPU diarization, with no complete MPS call.
- Complete MPS failure performs one normalization followed by one complete MPS call and one complete CPU call.
- CPU failure after MPS fallback aborts before slicing audio or calling MiMo.
- Speaker constraints are identical between preflight and full diarization.

Tests use fake pipelines and injected capability providers. They must not require a physical GPU, Hugging Face access, or model downloads.

### Manual Compatibility Check

Use the same 1-to-3-minute recording with at least two speakers:

1. Run CPU diarization three times and record the median duration.
2. Run explicit MPS diarization three times and record the median duration.
3. Compare detected speaker count, segment count, and timestamp boundaries for reasonable consistency.
4. Confirm that repeated MPS runs complete without crashes or unbounded memory growth.
5. Confirm that median MPS diarization time is at least 20 percent lower than CPU.

The benchmark measures diarization time, not MiMo network time or total command time.

If MPS is stable but does not meet the 20 percent threshold, the feature remains available as experimental but documentation marks it compatible and not recommended on that tested configuration.

## Documentation Changes

README documentation will:

- Describe `mps` as experimental and opt-in.
- Explain that `auto` does not select MPS.
- Explain preflight and CPU fallback.
- Provide a diagnostic command or verbose invocation.
- State that compatibility and speed depend on the exact macOS, PyTorch, pyannote, and hardware combination.
- Retain CUDA as the established GPU path.

## Compatibility and Rollout

This is an additive CLI change. Existing invocations retain their current device behavior. No output file format changes are required.

The feature should initially ship without claiming general MPS support. A tested compatibility note should identify the exact machine, operating system, Python, PyTorch, and pyannote versions used for manual validation.

Future work may promote MPS into `auto` only after multiple supported configurations demonstrate stable completion and useful speedups. That decision is outside this design.
