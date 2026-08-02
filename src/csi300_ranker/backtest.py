"""执行最近一年 12 折扩展窗口样本外回测。"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .config import (
    DATE,
    GAP_DAYS,
    LABEL,
    MAX_BOOST_ROUNDS,
    REPORT_DIR,
    STOCK,
    TRAIN_START_DATE,
    ensure_directories,
)
from .dataset import latest_signal_date, mature_label_cutoff, training_matrix
from .metrics import daily_metrics, summarize_daily
from .train import fit_with_validation
from .utils import save_csv, save_json, set_seed


TEST_DAYS = 20
N_FOLDS = 12


def parse_args() -> argparse.Namespace:
    """读取回测资源参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "4")))
    parser.add_argument("--boost-rounds", type=int, default=MAX_BOOST_ROUNDS)
    return parser.parse_args()


def fold_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """逐折汇总收益、Alpha、胜率与年化比率。"""
    rows: list[dict] = []
    for fold, group in daily.groupby("fold", sort=True):
        rows.append({"fold": int(fold), **summarize_daily(group)})
    return pd.DataFrame(rows)


def main() -> None:
    """按日期滚动训练，在互不重叠的 12 个测试窗口生成 OOF 预测。"""
    import lightgbm as lgb

    args = parse_args()
    ensure_directories()
    set_seed()
    label_cutoff = mature_label_cutoff(latest_signal_date(), GAP_DAYS)
    frame, values = training_matrix(pd.Timestamp(TRAIN_START_DATE), label_cutoff)
    dates = np.array(sorted(frame[DATE].unique()))
    first_test = len(dates) - TEST_DAYS * N_FOLDS
    prediction_blocks: list[pd.DataFrame] = []
    history_blocks: list[pd.DataFrame] = []

    for fold in range(N_FOLDS):
        test_start = first_test + fold * TEST_DAYS
        test_dates = dates[test_start : test_start + TEST_DAYS]
        train_dates = dates[: test_start - GAP_DAYS]
        train_mask = frame[DATE].isin(train_dates).to_numpy()
        test_mask = frame[DATE].isin(test_dates).to_numpy()
        train_frame = frame.loc[train_mask].reset_index(drop=True)
        test_frame = frame.loc[test_mask].reset_index(drop=True)
        booster, history, audit = fit_with_validation(
            lgb,
            train_frame,
            values[train_mask],
            args.num_threads,
            valid_days=120,
            boost_rounds=args.boost_rounds,
        )
        prediction = booster.predict(
            values[test_mask], num_iteration=audit["selected_iteration"]
        )
        block = test_frame.copy()
        block["prediction"] = prediction
        block["fold"] = fold
        prediction_blocks.append(block)
        history["fold"] = fold
        history_blocks.append(history)
        print(
            f"fold={fold:02d} test={pd.Timestamp(test_dates[0]).date()}~"
            f"{pd.Timestamp(test_dates[-1]).date()} trees={audit['selected_iteration']}",
            flush=True,
        )

    oof = pd.concat(prediction_blocks, ignore_index=True)
    daily_blocks: list[pd.DataFrame] = []
    for fold, group in oof.groupby("fold", sort=True):
        daily = daily_metrics(group, group["prediction"].to_numpy())
        daily["fold"] = int(fold)
        daily_blocks.append(daily)
    daily = pd.concat(daily_blocks, ignore_index=True)
    summary = summarize_daily(daily)
    oof.to_csv(REPORT_DIR / "backtest_oof_predictions.csv.gz", index=False, compression="gzip")
    save_csv(daily, REPORT_DIR / "backtest_daily.csv")
    save_csv(fold_summary(daily), REPORT_DIR / "backtest_by_fold.csv")
    save_csv(pd.DataFrame([summary]), REPORT_DIR / "backtest_summary.csv")
    save_csv(pd.concat(history_blocks, ignore_index=True), REPORT_DIR / "backtest_training_history.csv")
    save_json(
        {
            "design": "expanding window, 12 folds, 20 test days per fold",
            "train_start": TRAIN_START_DATE,
            "gap_days": GAP_DAYS,
            "test_days_per_fold": TEST_DAYS,
            "folds": N_FOLDS,
            "selection_metric": "inner validation Top5 return only",
            **summary,
        },
        REPORT_DIR / "backtest_manifest.json",
    )
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()

