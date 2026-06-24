"""M4A multi-speaker transcription."""

import os
import sys

from mimo_transcriber.paths import hf_cache_home

__version__ = "0.1.0"


# 无条件把 HF / pyannote 缓存重定向到项目内（<repo>/.models/hf）：
# 任何入口点（CLI、测试、库直接 import）只要 import 了本包，都会让
# huggingface_hub 走项目内缓存，避免污染 ~/.cache/huggingface。
# setdefault 保证用户显式设置的 HF_HOME / HF_HUB_CACHE 仍然胜出。
def configure_hf_cache() -> None:
    hf_home = str(hf_cache_home())
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_home() / "hub"))
    # 防御性兜底：若 huggingface_hub 已先于本包被 import（常量已固化），
    # 同步更新其常量，避免环境变量被忽略。
    if "huggingface_hub" in sys.modules:
        try:
            from huggingface_hub import constants as _hf_constants

            _hf_constants.HF_HOME = hf_home
            _hf_constants.HF_HUB_CACHE = str(hf_cache_home() / "hub")
        except Exception:
            pass


configure_hf_cache()
