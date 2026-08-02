"""从 BaoStock 更新沪深300历史成分与日线行情。"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from .config import (
    DOWNLOAD_START_DATE,
    MEMBERSHIP_PATH,
    RAW_PRICE_PATH,
    ensure_directories,
)
from .utils import save_csv


PRICE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,isST"
)


def parse_args() -> argparse.Namespace:
    """读取下载区间；每次完整刷新可保持复权口径一致。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DOWNLOAD_START_DATE)
    parser.add_argument("--end-date", default=date.today().isoformat())
    return parser.parse_args()


def result_frame(result, operation: str) -> pd.DataFrame:
    """把 BaoStock 游标转换为 DataFrame。"""
    if result.error_code != "0":
        raise RuntimeError(f"{operation}: {result.error_msg}")
    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def load_trade_dates(bs, start_date: str, end_date: str) -> list[str]:
    """返回指定区间内的开市日期。"""
    calendar = result_frame(
        bs.query_trade_dates(start_date=start_date, end_date=end_date),
        "交易日历下载失败",
    )
    return calendar.loc[calendar["is_trading_day"].eq("1"), "calendar_date"].tolist()


def load_membership(bs, trading_dates: list[str]) -> pd.DataFrame:
    """逐交易日下载当时的沪深300成分，避免使用当前名单回填历史。"""
    blocks: list[pd.DataFrame] = []
    for trade_date in trading_dates:
        block = result_frame(
            bs.query_hs300_stocks(date=trade_date),
            f"{trade_date} 成分股下载失败",
        )
        if block.empty:
            continue
        block["date"] = trade_date
        block["stock_id"] = block["code"].str.split(".").str[-1].str.zfill(6)
        blocks.append(block[["date", "code", "stock_id"]])
    return pd.concat(blocks, ignore_index=True)


def load_prices(
    bs,
    codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """按股票下载前复权日线；完整刷新避免不同复权快照混用。"""
    blocks: list[pd.DataFrame] = []
    for index, code in enumerate(codes, start=1):
        result = bs.query_history_k_data_plus(
            code,
            PRICE_FIELDS,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        block = result_frame(result, f"{code} 行情下载失败")
        if not block.empty:
            blocks.append(block)
        if index % 50 == 0:
            print(f"已下载 {index}/{len(codes)} 只股票", flush=True)
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    """登录 BaoStock，刷新动态股票池和对应行情。"""
    import baostock as bs

    args = parse_args()
    ensure_directories()
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(login.error_msg)
    try:
        dates = load_trade_dates(bs, args.start_date, args.end_date)
        membership = load_membership(bs, dates)
        codes = sorted(membership["code"].unique())
        prices = load_prices(bs, codes, args.start_date, args.end_date)
    finally:
        bs.logout()
    save_csv(membership, MEMBERSHIP_PATH)
    save_csv(prices, RAW_PRICE_PATH)
    print(
        f"数据更新完成：{len(dates)} 个交易日，{len(codes)} 只历史成分股。",
        flush=True,
    )


if __name__ == "__main__":
    main()
