"""LightGBM LambdaRank 模型、标签与选轮逻辑。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DATE,
    GAP_DAYS,
    LABEL,
    RELEVANCE_LEVELS,
    SEED,
    SMOOTH_ROUNDS,
    TOP_K,
)
from .metrics import query_sizes


def parameters(num_threads: int) -> dict:
    """返回与 F3 模型一致的确定性 LambdaRank 参数。"""
    return {
        "objective": "lambdarank",
        "metric": "None",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 0.10,
        "lambda_l2": 1.00,
        "label_gain": [(1 << level) - 1 for level in range(RELEVANCE_LEVELS)],
        "lambdarank_truncation_level": TOP_K,
        "seed": SEED,
        "data_random_seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": int(num_threads),
        "verbosity": -1,
    }


def relevance_labels(frame: pd.DataFrame) -> np.ndarray:
    """把每日未来收益百分位映射为 0 至 9 级排序相关性。"""
    percentile = frame.groupby(DATE, sort=False)[LABEL].rank(
        method="average", pct=True
    )
    labels = np.ceil(percentile.to_numpy() * RELEVANCE_LEVELS) - 1
    return np.clip(labels, 0, RELEVANCE_LEVELS - 1).astype("int32")


def ranking_dataset(
    lgb,
    values: np.ndarray,
    frame: pd.DataFrame,
    feature_names: list[str],
    reference=None,
):
    """创建以交易日为 query 的 LightGBM Dataset。"""
    return lgb.Dataset(
        np.ascontiguousarray(values, dtype="float32"),
        label=relevance_labels(frame),
        group=query_sizes(frame),
        feature_name=feature_names,
        reference=reference,
        free_raw_data=False,
    )


def inner_split(
    frame: pd.DataFrame,
    valid_days: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """留出尾部验证窗口，并用五个交易日隔离训练标签。"""
    dates = np.array(sorted(frame[DATE].unique()))
    valid_start = len(dates) - valid_days
    fit_end = valid_start - GAP_DAYS
    fit_mask = frame[DATE].isin(dates[:fit_end]).to_numpy()
    valid_mask = frame[DATE].isin(dates[valid_start:]).to_numpy()
    bounds = {
        "fit_start": pd.Timestamp(dates[0]).date().isoformat(),
        "fit_end": pd.Timestamp(dates[fit_end - 1]).date().isoformat(),
        "valid_start": pd.Timestamp(dates[valid_start]).date().isoformat(),
        "valid_end": pd.Timestamp(dates[-1]).date().isoformat(),
    }
    return fit_mask, valid_mask, bounds


def choose_iteration(
    top5_history: list[float],
    minimum_round: int,
) -> tuple[int, np.ndarray]:
    """仅按验证 Top5 收益的五轮移动平均选择树轮数。"""
    values = np.asarray(top5_history, dtype="float64")
    smoothed = pd.Series(values).rolling(
        SMOOTH_ROUNDS, min_periods=SMOOTH_ROUNDS
    ).mean().to_numpy()
    first = max(minimum_round, SMOOTH_ROUNDS) - 1
    selected = first + int(np.nanargmax(smoothed[first:]))
    return selected + 1, smoothed

