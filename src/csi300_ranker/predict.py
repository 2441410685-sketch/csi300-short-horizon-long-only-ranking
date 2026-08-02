"""加载最终模型，为最新交易日生成等权 Top5 多头名单。"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import LATEST_SELECTION_PATH, MODEL_PATH, STOCK, TOP_K, ensure_directories
from .dataset import latest_signal_date, prediction_matrix
from .features import FEATURE_COLUMNS
from .utils import save_csv, set_seed


def parse_args() -> argparse.Namespace:
    """读取可选信号日，默认使用数据中的最新交易日。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-date", default=None)
    return parser.parse_args()


def rank_selection(
    signal_date: pd.Timestamp,
    stock_ids: pd.Series,
    scores: np.ndarray,
) -> pd.DataFrame:
    """按分数降序选前五只；同分时用股票代码稳定排序。"""
    ranked = pd.DataFrame(
        {
            "signal_date": signal_date.date().isoformat(),
            STOCK: stock_ids.astype("string").str.zfill(6),
            "score": np.asarray(scores, dtype="float64"),
        }
    ).sort_values(
        ["score", STOCK], ascending=[False, True], kind="mergesort"
    ).head(TOP_K)
    ranked.insert(1, "rank", np.arange(1, len(ranked) + 1))
    ranked["weight"] = 1 / TOP_K
    return ranked.reset_index(drop=True)


def main() -> None:
    """执行无标签推理并写出最新股票名单。"""
    import lightgbm as lgb

    args = parse_args()
    ensure_directories()
    set_seed()
    signal_date = pd.Timestamp(args.signal_date) if args.signal_date else latest_signal_date()
    frame, values = prediction_matrix(signal_date)
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    if booster.feature_name() != FEATURE_COLUMNS:
        raise ValueError("模型特征顺序与当前因子定义不一致")
    scores = booster.predict(values, num_iteration=booster.current_iteration())
    selection = rank_selection(signal_date, frame[STOCK], scores)
    save_csv(selection, LATEST_SELECTION_PATH)
    print(selection.to_string(index=False))
    print(f"结果已保存：{LATEST_SELECTION_PATH}")


if __name__ == "__main__":
    main()

