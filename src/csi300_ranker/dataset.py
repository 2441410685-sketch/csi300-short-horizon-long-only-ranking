"""把动态股票池、成熟标签与同日因子组装为模型矩阵。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLEAN_PRICE_PATH, DATE, FEATURE_PATH, LABEL, SAMPLE_PATH, STOCK
from .features import FEATURE_COLUMNS


def load_samples(include_label: bool) -> pd.DataFrame:
    """读取动态成分样本；预测阶段不加载未来标签。"""
    columns = [DATE, STOCK, "is_tradeable"]
    if include_label:
        columns.append(LABEL)
    frame = pd.read_csv(
        SAMPLE_PATH,
        usecols=columns,
        dtype={STOCK: "string"},
        low_memory=False,
    )
    frame[DATE] = pd.to_datetime(frame[DATE])
    frame[STOCK] = frame[STOCK].str.zfill(6)
    return frame


def load_features() -> pd.DataFrame:
    """按冻结列顺序读取 186 维 float32 因子。"""
    frame = pd.read_parquet(
        FEATURE_PATH,
        columns=[DATE, STOCK, *FEATURE_COLUMNS],
    )
    frame[DATE] = pd.to_datetime(frame[DATE])
    frame[STOCK] = frame[STOCK].astype("string").str.zfill(6)
    return frame


def training_matrix(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray]:
    """选取区间内可交易且标签成熟的样本并合并因子。"""
    samples = load_samples(include_label=True)
    samples[LABEL] = pd.to_numeric(samples[LABEL], errors="coerce")
    samples = samples.loc[
        samples[DATE].between(start_date, end_date)
        & samples["is_tradeable"].eq(1)
        & samples[LABEL].notna(),
        [DATE, STOCK, LABEL],
    ]
    frame = samples.merge(
        load_features(),
        on=[DATE, STOCK],
        how="inner",
        validate="one_to_one",
    ).sort_values([DATE, STOCK], kind="mergesort")
    values = frame[FEATURE_COLUMNS].to_numpy(dtype="float32", copy=True)
    return frame[[DATE, STOCK, LABEL]].reset_index(drop=True), values


def prediction_matrix(
    signal_date: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray]:
    """读取信号日可交易成分股及其同日因子，不接触未来字段。"""
    universe = load_samples(include_label=False)
    universe = universe.loc[
        universe[DATE].eq(signal_date) & universe["is_tradeable"].eq(1),
        [DATE, STOCK],
    ]
    features = load_features()
    features = features.loc[features[DATE].eq(signal_date)]
    frame = universe.merge(
        features,
        on=[DATE, STOCK],
        how="inner",
        validate="one_to_one",
    ).sort_values(STOCK, kind="mergesort")
    values = frame[FEATURE_COLUMNS].to_numpy(dtype="float32", copy=True)
    return frame[[DATE, STOCK]].reset_index(drop=True), values


def latest_signal_date() -> pd.Timestamp:
    """返回样本表中的最新可用交易日。"""
    return pd.Timestamp(load_samples(include_label=False)[DATE].max())


def mature_label_cutoff(signal_date: pd.Timestamp, gap_days: int) -> pd.Timestamp:
    """从信号日向前移动 gap 个交易日，得到标签成熟边界。"""
    calendar = pd.read_csv(CLEAN_PRICE_PATH, usecols=[DATE])[DATE]
    dates = np.sort(pd.to_datetime(calendar).unique())
    position = int(np.searchsorted(dates, np.datetime64(signal_date)))
    if position >= len(dates) or dates[position] != np.datetime64(signal_date):
        raise ValueError(f"{signal_date.date()} 不是数据中的交易日")
    return pd.Timestamp(dates[position - gap_days])

