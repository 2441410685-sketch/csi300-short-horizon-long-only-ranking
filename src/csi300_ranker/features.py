"""生成模型实际使用的 Alpha158 与 28 个技术因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLEAN_PRICE_PATH, DATE, FEATURE_PATH, STOCK, ensure_directories


WINDOWS = (5, 10, 20, 30, 60)

ALPHA158_COLUMNS = [
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    "OPEN0", "HIGH0", "LOW0", "VWAP0",
    *[f"ROC{w}" for w in WINDOWS],
    *[f"MA{w}" for w in WINDOWS],
    *[f"STD{w}" for w in WINDOWS],
    *[f"{p}{w}" for p in ("BETA", "RSQR", "RESI") for w in WINDOWS],
    *[f"MAX{w}" for w in WINDOWS],
    *[f"MIN{w}" for w in WINDOWS],
    *[f"QTLU{w}" for w in WINDOWS],
    *[f"QTLD{w}" for w in WINDOWS],
    *[f"RANK{w}" for w in WINDOWS],
    *[f"RSV{w}" for w in WINDOWS],
    *[f"{p}{w}" for p in ("IMAX", "IMIN", "IMXD") for w in WINDOWS],
    *[f"CORR{w}" for w in WINDOWS],
    *[f"CORD{w}" for w in WINDOWS],
    *[f"{p}{w}" for p in ("CNTP", "CNTN", "CNTD") for w in WINDOWS],
    *[f"{p}{w}" for p in ("SUMP", "SUMN", "SUMD") for w in WINDOWS],
    *[f"VMA{w}" for w in WINDOWS],
    *[f"VSTD{w}" for w in WINDOWS],
    *[f"WVMA{w}" for w in WINDOWS],
    *[f"{p}{w}" for p in ("VSUMP", "VSUMN", "VSUMD") for w in WINDOWS],
]

TECHNICAL_COLUMNS = [
    "sma_5", "sma_20", "ema_12", "ema_26", "rsi", "macd", "macd_signal",
    "volume_change", "obv", "volume_ma_5", "volume_ma_20", "volume_ratio",
    "kdj_k", "kdj_d", "kdj_j", "boll_mid", "boll_std", "atr_14", "ema_60",
    "volatility_10", "volatility_20", "return_1", "return_5", "return_10",
    "high_low_spread", "open_close_spread", "high_close_spread", "low_close_spread",
]

FEATURE_COLUMNS = [*ALPHA158_COLUMNS, *TECHNICAL_COLUMNS]


def rolling_slope(values: pd.Series, window: int) -> pd.Series:
    """计算固定窗口线性回归斜率。"""
    x = np.arange(window, dtype="float64")
    centered = x - x.mean()
    denominator = float(np.square(centered).sum())
    return values.rolling(window).apply(
        lambda y: float(np.dot(y - np.mean(y), centered) / denominator),
        raw=True,
    )


def baseline_rsquared(values: pd.Series, window: int) -> pd.Series:
    """复现冻结特征口径中的趋势拟合度对齐方式。"""
    output = np.zeros(len(values), dtype="float64")
    if len(values) >= window:
        sample = values.iloc[:window].to_numpy(dtype="float64")
        if np.std(sample) > 0:
            corr = np.corrcoef(sample, np.arange(window, dtype="float64"))[0, 1]
            output[window - 1] = corr * corr
    return pd.Series(output, index=values.index)


def exponential_average(values: pd.Series, window: int) -> pd.Series:
    """计算带完整暖启动窗口的指数移动平均。"""
    return values.ewm(span=window, adjust=False, min_periods=window).mean()


def relative_strength_index(values: pd.Series, window: int = 14) -> pd.Series:
    """计算 Wilder 风格 RSI。"""
    change = values.diff()
    gain = change.clip(lower=0).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()
    loss = (-change.clip(upper=0)).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()
    relative_strength = gain / (loss + 1e-12)
    return 100 - 100 / (1 + relative_strength)


def stochastic_kd(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """计算 9-3-3 随机指标 K、D。"""
    lowest = low.rolling(9).min()
    highest = high.rolling(9).max()
    fast_k = 100 * (close - lowest) / (highest - lowest + 1e-12)
    slow_k = fast_k.rolling(3).mean()
    return slow_k, slow_k.rolling(3).mean()


def average_true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """计算 Wilder 风格平均真实波幅。"""
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """按收盘涨跌方向累计成交量。"""
    contribution = np.sign(close.diff()).fillna(0) * volume
    if len(contribution):
        contribution.iloc[0] = volume.iloc[0]
    return contribution.cumsum()


def engineer_stock(group: pd.DataFrame) -> pd.DataFrame:
    """按单只股票的历史序列计算全部 186 个模型因子。"""
    frame = group.sort_values(DATE, kind="mergesort").reset_index(drop=True)
    open_price = frame["open"].astype("float64")
    high = frame["high"].astype("float64")
    low = frame["low"].astype("float64")
    close = frame["close"].astype("float64")
    volume = frame["volume"].astype("float64")
    amount = frame["amount"].astype("float64")
    vwap = amount / (volume + 1e-12)
    upper_body = pd.concat([open_price, close], axis=1).max(axis=1)
    lower_body = pd.concat([open_price, close], axis=1).min(axis=1)

    values: dict[str, pd.Series] = {
        "KMID": (close - open_price) / (open_price + 1e-12),
        "KLEN": (high - low) / (open_price + 1e-12),
        "KMID2": (close - open_price) / (high - low + 1e-12),
        "KUP": (high - upper_body) / (open_price + 1e-12),
        "KUP2": (high - upper_body) / (high - low + 1e-12),
        "KLOW": (lower_body - low) / (open_price + 1e-12),
        "KLOW2": (lower_body - low) / (high - low + 1e-12),
        "KSFT": (2 * close - high - low) / (open_price + 1e-12),
        "KSFT2": (2 * close - high - low) / (high - low + 1e-12),
        "OPEN0": open_price / (close + 1e-12),
        "HIGH0": high / (close + 1e-12),
        "LOW0": low / (close + 1e-12),
        "VWAP0": vwap / (close + 1e-12),
    }

    for window in WINDOWS:
        slope = rolling_slope(close, window)
        intercept = close.rolling(window).mean() - slope * ((window - 1) / 2)
        values[f"ROC{window}"] = close.shift(window) / (close + 1e-12)
        values[f"MA{window}"] = close.rolling(window).mean() / (close + 1e-12)
        values[f"STD{window}"] = close.rolling(window).std(ddof=0) / (close + 1e-12)
        values[f"BETA{window}"] = slope / (close + 1e-12)
        values[f"RSQR{window}"] = baseline_rsquared(close, window)
        values[f"RESI{window}"] = (
            close - (slope * (window - 1) + intercept)
        ) / (close + 1e-12)
        values[f"MAX{window}"] = high.rolling(window).max() / (close + 1e-12)
        values[f"MIN{window}"] = low.rolling(window).min() / (close + 1e-12)
        values[f"QTLU{window}"] = close.rolling(window).quantile(0.8) / (close + 1e-12)
        values[f"QTLD{window}"] = close.rolling(window).quantile(0.2) / (close + 1e-12)
        values[f"RANK{window}"] = close.rolling(window).rank(pct=True)
        rolling_low = low.rolling(window).min()
        rolling_high = high.rolling(window).max()
        values[f"RSV{window}"] = (close - rolling_low) / (
            rolling_high - rolling_low + 1e-12
        )
        imax = high.rolling(window).apply(np.argmax, raw=True)
        imin = low.rolling(window).apply(np.argmin, raw=True)
        values[f"IMAX{window}"] = imax / window
        values[f"IMIN{window}"] = imin / window
        values[f"IMXD{window}"] = (imax - imin) / window

    log_volume = np.log(volume + 1)
    close_ratio = close / close.shift(1)
    log_volume_ratio = np.log(volume / (volume.shift(1) + 1e-12) + 1)
    for window in WINDOWS:
        values[f"CORR{window}"] = close.rolling(window).corr(log_volume)
        values[f"CORD{window}"] = close_ratio.fillna(0).rolling(window).corr(
            log_volume_ratio.fillna(0)
        )

    price_change = close.diff()
    volume_change = volume.diff()
    weighted_return = close.pct_change().abs() * volume
    for window in WINDOWS:
        positive = close.gt(close.shift(1)).rolling(window).mean()
        negative = close.lt(close.shift(1)).rolling(window).mean()
        values[f"CNTP{window}"] = positive
        values[f"CNTN{window}"] = negative
        values[f"CNTD{window}"] = positive - negative

        price_total = price_change.abs().rolling(window).sum()
        price_up = price_change.clip(lower=0).rolling(window).sum()
        price_down = (-price_change.clip(upper=0)).rolling(window).sum()
        values[f"SUMP{window}"] = price_up / (price_total + 1e-12)
        values[f"SUMN{window}"] = price_down / (price_total + 1e-12)
        values[f"SUMD{window}"] = (price_up - price_down) / (price_total + 1e-12)

        values[f"VMA{window}"] = volume.rolling(window).mean() / (volume + 1e-12)
        values[f"VSTD{window}"] = volume.rolling(window).std(ddof=0) / (volume + 1e-12)
        values[f"WVMA{window}"] = weighted_return.rolling(window).std() / (
            weighted_return.rolling(window).mean() + 1e-12
        )
        volume_total = volume_change.abs().rolling(window).sum()
        volume_up = volume_change.clip(lower=0).rolling(window).sum()
        volume_down = (-volume_change.clip(upper=0)).rolling(window).sum()
        values[f"VSUMP{window}"] = volume_up / (volume_total + 1e-12)
        values[f"VSUMN{window}"] = volume_down / (volume_total + 1e-12)
        values[f"VSUMD{window}"] = (volume_up - volume_down) / (volume_total + 1e-12)

    ema12 = exponential_average(close, 12)
    ema26 = exponential_average(close, 26)
    macd = ema12 - ema26
    kdj_k, kdj_d = stochastic_kd(high, low, close)
    return_1 = close.pct_change()
    technical = {
        "sma_5": close.rolling(5).mean(),
        "sma_20": close.rolling(20).mean(),
        "ema_12": ema12,
        "ema_26": ema26,
        "rsi": relative_strength_index(close),
        "macd": macd,
        "macd_signal": macd.ewm(span=9, adjust=False, min_periods=9).mean(),
        "volume_change": volume.pct_change(),
        "obv": on_balance_volume(close, volume),
        "volume_ma_5": volume.rolling(5).mean(),
        "volume_ma_20": volume.rolling(20).mean(),
        "volume_ratio": volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-12),
        "kdj_k": kdj_k,
        "kdj_d": kdj_d,
        "kdj_j": 3 * kdj_k - 2 * kdj_d,
        "boll_mid": close.rolling(20).mean(),
        "boll_std": close.rolling(20).std(ddof=0),
        "atr_14": average_true_range(high, low, close),
        "ema_60": exponential_average(close, 60),
        "volatility_10": return_1.rolling(10).std(),
        "volatility_20": return_1.rolling(20).std(),
        "return_1": return_1,
        "return_5": close.pct_change(5),
        "return_10": close.pct_change(10),
        "high_low_spread": high - low,
        "open_close_spread": open_price - close,
        "high_close_spread": high - close,
        "low_close_spread": low - close,
    }
    factor_frame = pd.DataFrame({**values, **technical})[FEATURE_COLUMNS]
    factor_frame = factor_frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    factor_frame = factor_frame.astype("float32")
    return pd.concat([frame[[DATE, STOCK]], factor_frame], axis=1)


def build_feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    """逐股票生成因子，并恢复按日期、股票排序的模型表。"""
    blocks: list[pd.DataFrame] = []
    groups = prices.groupby(STOCK, sort=True)
    for index, (_, group) in enumerate(groups, start=1):
        blocks.append(engineer_stock(group))
        if index % 50 == 0:
            print(f"已计算 {index}/{prices[STOCK].nunique()} 只股票", flush=True)
    return pd.concat(blocks, ignore_index=True).sort_values(
        [DATE, STOCK], kind="mergesort"
    ).reset_index(drop=True)


def main() -> None:
    """从清洗行情构建可直接训练和预测的 float32 因子缓存。"""
    ensure_directories()
    prices = pd.read_csv(
        CLEAN_PRICE_PATH,
        dtype={STOCK: "string"},
        low_memory=False,
    )
    prices[DATE] = pd.to_datetime(prices[DATE])
    prices[STOCK] = prices[STOCK].str.zfill(6)
    factors = build_feature_table(prices)
    factors.to_parquet(FEATURE_PATH, index=False)
    print(f"因子生成完成：{len(factors):,} 行，{len(FEATURE_COLUMNS)} 个因子。")


if __name__ == "__main__":
    main()

