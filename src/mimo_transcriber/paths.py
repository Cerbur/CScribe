"""项目内路径约定（仅依赖标准库）。

本模块故意只 import ``os`` 与 ``pathlib``，避免触发任何项目内或第三方
import，从而可以在任何入口点（CLI、测试、库）以最低代价被引用，也不会
产生循环依赖。所有"模型 / 任务缓存"默认都落在项目目录内，纳入
``.gitignore``，便于 ``rm -rf .models/ .run/`` 一键清理。
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """项目根目录：从本文件向上查找首个含 ``pyproject.toml`` 的祖先目录。

    找不到时回退到 ``parents[3]``（本文件位于
    ``<repo>/src/mimo_transcriber/paths.py``，向上 3 层即为 repo 根）。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


def task_cache_dir() -> Path:
    """任务工作目录根（normalized.wav / 切片 / manifest 等落盘位置）。

    默认 ``<repo>/.run/cscribe``，可用 ``CSCRIBE_TASK_CACHE`` 环境变量覆盖。
    """
    return Path(
        os.environ.get(
            "CSCRIBE_TASK_CACHE",
            str(project_root() / ".run" / "cscribe"),
        )
    )


def hf_cache_home() -> Path:
    """HuggingFace 缓存的 ``HF_HOME``（``<repo>/.models/hf``）。

    配合 :func:`configure_hf_cache` 在包导入时把 pyannote 等模型无条件
    重定向到项目目录内。
    """
    return project_root() / ".models" / "hf"
