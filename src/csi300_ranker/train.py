"""用时间尾部验证选择树轮数，并在全部成熟样本上训练最终模型。"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .config import (
    DATE,
    FEATURE_IMPORTANCE_PATH,
    GAP_DAYS,
    INNER_VALID_DAYS,
    MAX_BOOST_ROUNDS,
    MIN_SELECT_ROUND,
    MODEL_MANIFEST_PATH,
    MODEL_PATH,
    SEED,
    TRAINING_HISTORY_PATH,
    TRAIN_START_DATE,
    ensure_directories,
)
from .dataset import latest_signal_date, mature_label_cutoff, training_matrix
from .features import FEATURE_COLUMNS
from .metrics import lightgbm_validation_metric
from .model import choose_iteration, inner_split, parameters, ranking_dataset
from .utils import file_sha256, save_csv, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """读取训练截止日和资源参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--train-start", default=TRAIN_START_DATE)
    parser.add_argument("--inner-valid-days", type=int, default=INNER_VALID_DAYS)
    parser.add_argument("--boost-rounds", type=int, default=MAX_BOOST_ROUNDS)
    parser.add_argument("--num-threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "4")))
    return parser.parse_args()


def fit_with_validation(
    lgb,
    frame: pd.DataFrame,
    values: np.ndarray,
    num_threads: int,
    valid_days: int,
    boost_rounds: int,
) -> tuple[object, pd.DataFrame, dict]:
    """按验证 Top5 选轮，再在整个外层训练段上重新拟合。"""
    fit_mask, valid_mask, bounds = inner_split(frame, valid_days)
    fit_frame = frame.loc[fit_mask].reset_index(drop=True)
    valid_frame = frame.loc[valid_mask].reset_index(drop=True)
    fit_set = ranking_dataset(
        lgb, values[fit_mask], fit_frame, FEATURE_COLUMNS
    )
    valid_set = ranking_dataset(
        lgb,
        values[valid_mask],
        valid_frame,
        FEATURE_COLUMNS,
        reference=fit_set,
    )
    evaluations: dict = {}
    lgb.train(
        parameters(num_threads),
        fit_set,
        num_boost_round=boost_rounds,
        valid_sets=[valid_set],
        valid_names=["inner_valid"],
        feval=lightgbm_validation_metric(valid_frame),
        callbacks=[lgb.record_evaluation(evaluations), lgb.log_evaluation(100)],
    )
    selected_iteration, smoothed = choose_iteration(
        evaluations["inner_valid"]["top5_return"], MIN_SELECT_ROUND
    )
    full_set = ranking_dataset(lgb, values, frame, FEATURE_COLUMNS)
    booster = lgb.train(
        parameters(num_threads),
        full_set,
        num_boost_round=selected_iteration,
    )
    history = pd.DataFrame(
        {
            "iteration": np.arange(1, boost_rounds + 1),
            "inner_top5_return": evaluations["inner_valid"]["top5_return"],
            "inner_rank_ic": evaluations["inner_valid"]["rank_ic"],
            "inner_top5_smoothed": smoothed,
        }
    )
    history["selected"] = history["iteration"].eq(selected_iteration)
    audit = {
        **bounds,
        "selected_iteration": int(selected_iteration),
        "selected_top5_return": float(
            history.loc[history["selected"], "inner_top5_return"].iloc[0]
        ),
        "selected_smoothed_top5": float(
            history.loc[history["selected"], "inner_top5_smoothed"].iloc[0]
        ),
    }
    return booster, history, audit


def feature_importance(booster) -> pd.DataFrame:
    """按总增益占比输出可解释的因子重要性。"""
    gain = booster.feature_importance(importance_type="gain").astype("float64")
    share = gain / gain.sum() if gain.sum() > 0 else gain
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "gain_share": share}).sort_values(
        ["gain_share", "feature"], ascending=[False, True], kind="mergesort"
    )


def main() -> None:
    """完成成熟标签截断、验证选轮、全量重训与模型落盘。"""
    import lightgbm as lgb

    args = parse_args()
    ensure_directories()
    set_seed()
    started = time.monotonic()
    signal_date = pd.Timestamp(args.signal_date) if args.signal_date else latest_signal_date()
    label_cutoff = mature_label_cutoff(signal_date, GAP_DAYS)
    frame, values = training_matrix(pd.Timestamp(args.train_start), label_cutoff)
    booster, history, audit = fit_with_validation(
        lgb,
        frame,
        values,
        args.num_threads,
        args.inner_valid_days,
        args.boost_rounds,
    )
    booster.save_model(str(MODEL_PATH))
    save_csv(history, TRAINING_HISTORY_PATH)
    save_csv(feature_importance(booster), FEATURE_IMPORTANCE_PATH)
    save_json(
        {
            "model": "LightGBM LambdaRank F3",
            "created_at": datetime.now().astimezone().isoformat(),
            "signal_date": signal_date.date().isoformat(),
            "label_available_through": label_cutoff.date().isoformat(),
            "train_start": pd.Timestamp(args.train_start).date().isoformat(),
            "train_end": pd.Timestamp(frame[DATE].max()).date().isoformat(),
            "training_rows": int(len(frame)),
            "training_dates": int(frame[DATE].nunique()),
            "feature_count": len(FEATURE_COLUMNS),
            "label": "(open_T+5-open_T+1)/open_T+1",
            "gap_days": GAP_DAYS,
            "random_seed": SEED,
            "selection_metric": "validation Top5 return only",
            "model_sha256": file_sha256(MODEL_PATH),
            "elapsed_seconds": float(time.monotonic() - started),
            **audit,
        },
        MODEL_MANIFEST_PATH,
    )
    print(f"训练完成：{len(frame):,} 行，最终 {audit['selected_iteration']} 轮。")
    print(f"模型已保存：{MODEL_PATH}")


if __name__ == "__main__":
    main()

