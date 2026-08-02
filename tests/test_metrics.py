"""Top5 与横截面 Alpha 指标测试。"""

import numpy as np
import pandas as pd

from csi300_ranker.config import DATE, LABEL, STOCK
from csi300_ranker.metrics import daily_metrics, summarize_daily


def test_daily_metrics_selects_highest_scores() -> None:
    """确认 Top5 取最高分，Alpha 使用同日候选池等权收益。"""
    frame = pd.DataFrame(
        {
            DATE: pd.to_datetime(["2026-01-05"] * 6),
            STOCK: [f"{index:06d}" for index in range(6)],
            LABEL: [0.06, 0.05, 0.04, 0.03, 0.02, -0.10],
        }
    )
    daily = daily_metrics(frame, np.array([6, 5, 4, 3, 2, 1], dtype=float))
    expected_top5 = np.mean([0.06, 0.05, 0.04, 0.03, 0.02])
    assert np.isclose(daily.loc[0, "top5_return"], expected_top5)
    assert np.isclose(
        daily.loc[0, "alpha_return"], expected_top5 - frame[LABEL].mean()
    )


def test_summary_counts_signal_days() -> None:
    """确认汇总表保留日期范围和信号日数量。"""
    daily = pd.DataFrame(
        {
            DATE: pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "rank_ic": [0.1, 0.2],
            "top5_return": [0.02, -0.01],
            "market_return": [0.00, 0.00],
            "alpha_return": [0.02, -0.01],
            "top5_positive": [1, 0],
            "beats_market": [1, 0],
        }
    )
    summary = summarize_daily(daily)
    assert summary["n_signal_days"] == 2
    assert summary["top5_positive_ratio"] == 0.5

