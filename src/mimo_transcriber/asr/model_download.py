"""Robust Hugging Face model download for the MLX ASR engine.

``mlx_whisper.transcribe`` downloads its model lazily through ``huggingface_hub``.
That path is unreliable on this host for two independent reasons:

1. ``hf_xet`` (the Xet CAS protocol) fails to fetch large blobs.
2. The macOS *system* proxy (read by ``urllib.request.getproxies`` even with no
   ``*_PROXY`` env vars) forwards request headers but stalls large response bodies.

A plain HTTPS connection to the ``resolve`` endpoint (which redirects to the CDN)
works reliably, just slowly. This module downloads model files directly with an
``httpx`` client whose ``trust_env=False`` ignores both the system proxy and env
proxies, with HTTP Range resume + retries so a multi-hour download survives the
flaky connection. The resulting directory is handed to ``mlx_whisper`` as a local
path, sidestepping its own download entirely.

Model weights are cached under ``<repo_root>/.models`` (the repo root is the
nearest ancestor of this file containing ``pyproject.toml``), overridable via the
``CSCRIBE_MODEL_CACHE`` environment variable which always wins when set.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable

import httpx

from mimo_transcriber.paths import project_root

logger = logging.getLogger(__name__)

HUGGINGFACE_BASE = "https://huggingface.co"
DEFAULT_REVISION = "main"
_CHUNK = 256 * 1024

ProgressFn = Callable[[str, int, int], None]


def _retry_backoff(attempt: int) -> float:
    """Seconds to wait before the next retry. Overridable in tests."""
    return min(2.0 ** attempt, 30.0)


class ModelDownloadError(RuntimeError):
    """A model could not be downloaded after exhausting retries."""


def default_model_cache() -> Path:
    return Path(
        os.environ.get(
            "CSCRIBE_MODEL_CACHE",
            str(project_root() / ".models"),
        )
    )


def _direct_client() -> httpx.Client:
    # trust_env=False is the crux of the fix: it stops httpx from picking up the
    # macOS SystemConfiguration proxy (and any *_PROXY env vars) that stall large
    # transfers. We then connect directly to the resolve -> CDN endpoint.
    return httpx.Client(
        follow_redirects=True,
        trust_env=False,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    )


def _list_repo_files(
    client: httpx.Client, repo_id: str, base_url: str
) -> list[str]:
    resp = client.get(f"{base_url}/api/models/{repo_id}")
    if resp.status_code != 200:
        raise ModelDownloadError(
            f"无法获取模型文件列表 {repo_id}: HTTP {resp.status_code}"
        )
    data = resp.json()
    return [item["rfilename"] for item in data.get("siblings", [])]


def _remote_size(
    client: httpx.Client, repo_id: str, filename: str, base_url: str
) -> int | None:
    url = f"{base_url}/{repo_id}/resolve/{DEFAULT_REVISION}/{filename}"
    resp = client.head(url)
    if resp.status_code not in (200, 206):
        return None
    length = resp.headers.get("content-length")
    return int(length) if length else None


def _download_one(
    client: httpx.Client,
    repo_id: str,
    filename: str,
    dest: Path,
    base_url: str,
    max_retries: int,
    on_progress: ProgressFn | None,
) -> None:
    url = f"{base_url}/{repo_id}/resolve/{DEFAULT_REVISION}/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if a previously completed file still matches the remote size.
    if dest.exists():
        expected = _remote_size(client, repo_id, filename, base_url)
        if expected is not None and dest.stat().st_size == expected:
            return
        dest.unlink(missing_ok=True)

    part = dest.with_name(f"{dest.name}.part")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 416:
                    break  # range not satisfiable: nothing left
                if resp.status_code not in (200, 206):
                    raise ModelDownloadError(
                        f"下载 {filename} 失败: HTTP {resp.status_code}"
                    )
                # Server ignored the Range header: restart from scratch.
                if have > 0 and resp.status_code == 200:
                    have = 0
                content_length = int(resp.headers.get("content-length", "0") or 0)
                total = have + content_length
                append = have > 0 and resp.status_code == 206
                written = have
                with open(part, "ab" if append else "wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=_CHUNK):
                        fh.write(chunk)
                        written += len(chunk)
                        if on_progress:
                            on_progress(filename, written, total)
            final_size = part.stat().st_size
            if total == 0 or final_size >= total:
                part.replace(dest)
                return
            logger.warning(
                "下载 %s 未完成 (%d/%d 字节)，尝试断点续传",
                filename, final_size, total,
            )
        except (httpx.HTTPError, OSError, ModelDownloadError) as exc:
            last_error = exc
            logger.warning(
                "下载 %s 第 %d/%d 次失败: %s: %s",
                filename, attempt, max_retries, type(exc).__name__, exc,
            )
            time.sleep(_retry_backoff(attempt))
    raise ModelDownloadError(
        f"下载 {filename} 失败（已重试 {max_retries} 次）: {last_error}"
    )


def ensure_model(
    repo_id: str,
    *,
    cache_dir: Path | None = None,
    base_url: str = HUGGINGFACE_BASE,
    max_retries: int = 5,
    on_progress: ProgressFn | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Ensure all files of ``repo_id`` exist locally and return their directory.

    Idempotent: a ``.complete`` marker records a fully-fetched model so repeated
    calls are free. An interrupted run resumes each file via HTTP Range.
    """
    cache_dir = cache_dir or default_model_cache()
    target = cache_dir / repo_id.replace("/", "__")
    complete_marker = target / ".complete"
    if complete_marker.exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    own_client = client is None
    http = client or _direct_client()
    try:
        files = _list_repo_files(http, repo_id, base_url)
        if not files:
            raise ModelDownloadError(f"模型仓库 {repo_id} 没有可下载的文件")
        for filename in files:
            _download_one(
                http, repo_id, filename, target / filename,
                base_url, max_retries, on_progress,
            )
    finally:
        if own_client:
            http.close()

    complete_marker.write_text("ok", encoding="utf-8")
    return target


def make_stderr_progress() -> ProgressFn:
    """Rate-limited per-file progress printer for stderr (≈1 update/sec)."""
    state: dict[str, object] = {"name": "", "last": 0.0}

    def _report(filename: str, downloaded: int, total: int) -> None:
        now = time.monotonic()
        if filename != state["name"]:
            state["name"] = filename
            state["last"] = 0.0
        if not total or now - float(state["last"]) < 1.0:
            return
        state["last"] = now
        print(
            f"  下载模型 {filename}: "
            f"{downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB "
            f"({downloaded * 100 / total:.0f}%)",
            file=sys.stderr,
            flush=True,
        )

    return _report
