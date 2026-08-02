"""清洗日线数据，并生成严格的 T+1 至 T+5 收益标签。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CLEAN_PRICE_PATH,
    DATE,
    LABEL,
    MEMBERSHIP_PATH,
    RAW_PRICE_PATH,
    SAMPLE_PATH,
    STOCK,
    ensure_directories,
)
from .utils import save_csv


PRICE_COLUMNS = ("open", "high", "low", "close", "preclose")
NUMERIC_COLUMNS = (
    *PRICE_COLUMNS,
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "isST",
)


def normalize_stock_id(values: pd.Series) -> pd.Series:
    """把交易所代码统一为保留前导零的六位字符串。"""
    return values.astype("string").str.split(".").str[-1].str.zfill(6)


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """转换字段类型，保留有效且唯一的日线记录。"""
    frame = raw.copy()
    frame[DATE] = pd.to_datetime(frame[DATE], errors="coerce")
    frame[STOCK] = normalize_stock_id(frame["code"])
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[DATE, STOCK, "open", "high", "low", "close"])
    frame = frame.loc[
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & frame["high"].ge(frame[["open", "close"]].max(axis=1))
        & frame["low"].le(frame[["open", "close"]].min(axis=1))
    ]
    frame = frame.drop_duplicates([DATE, STOCK], keep="last")
    return frame.sort_values([DATE, STOCK], kind="mergesort").reset_index(drop=True)


def future_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """在完整交易日历上计算 (open_T+5-open_T+1)/open_T+1。"""
    calendar = pd.DatetimeIndex(prices[DATE].unique()).sort_values()
    stocks = prices[STOCK].drop_duplicates().astype("string")
    grid_index = pd.MultiIndex.from_product(
        [stocks, calendar], names=[STOCK, DATE]
    )
    open_price = prices.set_index([STOCK, DATE])["open"].reindex(grid_index)
    grid = open_price.rename("open").reset_index()
    grouped = grid.groupby(STOCK, sort=False)["open"]
    open_t1 = grouped.shift(-1)
    open_t5 = grouped.shift(-5)
    grid[LABEL] = ((open_t5 - open_t1) / open_t1).astype("float32")
    return grid[[DATE, STOCK, LABEL]]


def build_samples(prices: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """合并历史成分、可交易状态和未来标签，形成建模样本。"""
    members = membership.copy()
    members[DATE] = pd.to_datetime(members[DATE])
    members[STOCK] = normalize_stock_id(members["stock_id"])
    members = members[[DATE, STOCK]].drop_duplicates()
    samples = prices.merge(members, on=[DATE, STOCK], how="inner")
    samples["is_tradeable"] = (
        samples["tradestatus"].eq(1)
        & samples["isST"].eq(0)
        & samples["volume"].gt(0)
    ).astype("int8")
    samples = samples.merge(
        future_labels(prices),
        on=[DATE, STOCK],
        how="left",
        validate="one_to_one",
    )
    return samples.sort_values([DATE, STOCK], kind="mergesort").reset_index(drop=True)


def main() -> None:
    """从原始行情生成清洗行情和训练样本。"""
    ensure_directories()
    raw = pd.read_csv(RAW_PRICE_PATH, dtype={"code": "string"}, low_memory=False)
    membership = pd.read_csv(
        MEMBERSHIP_PATH,
        dtype={"code": "string", "stock_id": "string"},
        low_memory=False,
    )
    prices = clean_prices(raw)
    samples = build_samples(prices, membership)
    save_csv(prices, CLEAN_PRICE_PATH)
    save_csv(samples, SAMPLE_PATH)
    print(f"数据准备完成：{len(prices):,} 行行情，{len(samples):,} 行样本。")


if __name__ == "__main__":
    main()

