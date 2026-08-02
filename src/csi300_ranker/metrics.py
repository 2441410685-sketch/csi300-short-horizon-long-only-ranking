"""横截面排序与短线多头组合评价指标。"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import DATE, HORIZON_DAYS, LABEL, STOCK, TOP_K


def query_sizes(frame: pd.DataFrame) -> np.ndarray:
    """统计每天的股票数，定义 LightGBM ranking query。"""
    return frame.groupby(DATE, sort=False).size().to_numpy(dtype="int32")


def daily_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """逐日计算 Rank IC、Top5 收益、市场收益与横截面 Alpha。"""
    work = frame[[DATE, STOCK, LABEL]].copy()
    work["prediction"] = np.asarray(prediction, dtype="float64")
    rows: list[dict] = []
    for signal_date, group in work.groupby(DATE, sort=False):
        ordered = group.sort_values(
            ["prediction", STOCK],
            ascending=[False, True],
            kind="mergesort",
        )
        top5_return = float(ordered.head(TOP_K)[LABEL].mean())
        market_return = float(group[LABEL].mean())
        rows.append(
            {
                DATE: pd.Timestamp(signal_date),
                "rank_ic": float(group["prediction"].corr(group[LABEL], method="spearman")),
                "top5_return": top5_return,
                "market_return": market_return,
                "alpha_return": top5_return - market_return,
                "top5_positive": int(top5_return > 0),
                "beats_market": int(top5_return > market_return),
            }
        )
    return pd.DataFrame(rows)


def annualized_ratio(values: pd.Series) -> float:
    """按五日持有期把均值与样本标准差之比年化。"""
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation == 0:
        return 0.0
    return float(np.sqrt(252 / HORIZON_DAYS) * values.mean() / standard_deviation)


def summarize_daily(daily: pd.DataFrame) -> dict[str, float | int | str]:
    """汇总最近一年回测的核心收益、风险与排序指标。"""
    rank_ic = daily["rank_ic"].astype("float64")
    rank_ic_std = float(rank_ic.std(ddof=1))
    return {
        "start_date": pd.Timestamp(daily[DATE].min()).date().isoformat(),
        "end_date": pd.Timestamp(daily[DATE].max()).date().isoformat(),
        "n_signal_days": int(len(daily)),
        "top5_return_mean": float(daily["top5_return"].mean()),
        "top5_return_std": float(daily["top5_return"].std(ddof=1)),
        "top5_sharpe_annualized_5d": annualized_ratio(daily["top5_return"]),
        "top5_positive_ratio": float(daily["top5_positive"].mean()),
        "market_return_mean": float(daily["market_return"].mean()),
        "alpha_return_mean": float(daily["alpha_return"].mean()),
        "alpha_information_ratio_annualized_5d": annualized_ratio(daily["alpha_return"]),
        "top5_beats_market_ratio": float(daily["beats_market"].mean()),
        "rank_ic_mean": float(rank_ic.mean()),
        "rank_icir": float(rank_ic.mean() / rank_ic_std) if rank_ic_std > 0 else 0.0,
    }


def lightgbm_validation_metric(frame: pd.DataFrame) -> Callable:
    """构造按日 Top5 收益和 Rank IC 的 LightGBM 验证回调。"""
    labels = frame[LABEL].to_numpy(dtype="float64")
    stocks = frame[STOCK].astype(str).to_numpy()
    boundaries = np.concatenate(([0], np.cumsum(query_sizes(frame))))

    def evaluate(prediction: np.ndarray, _dataset):
        """按 query 边界切分预测，供每轮模型评估。"""
        top5_returns: list[float] = []
        rank_ics: list[float] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            score = prediction[start:end]
            target = labels[start:end]
            top = np.lexsort((stocks[start:end], -score))[:TOP_K]
            top5_returns.append(float(target[top].mean()))
            score_rank = pd.Series(score).rank(method="average")
            target_rank = pd.Series(target).rank(method="average")
            rank_ics.append(float(score_rank.corr(target_rank)))
        return [
            ("top5_return", float(np.nanmean(top5_returns)), True),
            ("rank_ic", float(np.nanmean(rank_ics)), True),
        ]

    return evaluate

