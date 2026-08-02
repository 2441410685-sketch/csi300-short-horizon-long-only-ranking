"""排序标签与模型选轮测试。"""

import numpy as np
import pandas as pd

from csi300_ranker.config import DATE, LABEL, MODEL_PATH
from csi300_ranker.features import FEATURE_COLUMNS
from csi300_ranker.model import choose_iteration, relevance_labels


def test_feature_contract_has_186_unique_columns() -> None:
    """确认模型输入数量和列名均唯一。"""
    assert len(FEATURE_COLUMNS) == 186
    assert len(set(FEATURE_COLUMNS)) == 186


def test_frozen_model_uses_the_same_feature_order() -> None:
    """直接读取文本模型头，确认随仓库发布的模型可由当前因子推理。"""
    feature_line = next(
        line
        for line in MODEL_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("feature_names=")
    )
    assert feature_line.removeprefix("feature_names=").split() == FEATURE_COLUMNS


def test_relevance_labels_are_daily_and_ordered() -> None:
    """确认相关性等级只比较同日股票且保持收益顺序。"""
    frame = pd.DataFrame(
        {
            DATE: pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-02"] * 4),
            LABEL: [0.04, -0.02, 0.01, 0.00, 0.10, 0.20, -0.10, 0.00],
        }
    )
    labels = relevance_labels(frame)
    for start in (0, 4):
        order = np.argsort(frame[LABEL].to_numpy()[start : start + 4])
        assert np.all(np.diff(labels[start : start + 4][order]) >= 0)


def test_choose_iteration_uses_smoothed_top5() -> None:
    """确认选轮依据为五轮移动平均而非单轮尖峰。"""
    history = [0.0] * 19 + [0.10, -0.10, -0.10, -0.10, -0.10, 0.04, 0.04, 0.04, 0.04, 0.04]
    selected, smoothed = choose_iteration(history, minimum_round=20)
    assert selected == 29
    assert smoothed[selected - 1] == np.nanmax(smoothed[19:])
