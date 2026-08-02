"""通用的可复现写入与哈希工具。"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SEED


def set_seed(seed: int = SEED) -> None:
    """统一 Python 与 NumPy 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """以 UTF-8 编码稳定写出表格。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def save_json(payload: dict, path: Path) -> None:
    """以便于代码审阅的缩进格式写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    """计算文件的 SHA-256，用于确认模型版本。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

