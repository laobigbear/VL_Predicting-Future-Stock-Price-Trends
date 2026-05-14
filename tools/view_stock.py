"""CLI tool to inspect downloaded stock parquet files."""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "prices"
STOCK_DIR = DATA_DIR / "stocks"
MACRO_DIR = DATA_DIR / "macro"


def list_available() -> None:
    stocks = sorted(p.stem for p in STOCK_DIR.glob("*.parquet"))
    macros = sorted(p.stem for p in MACRO_DIR.glob("*.parquet"))
    print("=== 股票 ===")
    print("  " + "  ".join(stocks))
    print("=== 總經 ===")
    print("  " + "  ".join(macros))


def load(ticker: str) -> pd.DataFrame:
    for directory in (STOCK_DIR, MACRO_DIR):
        path = directory / f"{ticker}.parquet"
        if path.exists():
            return pd.read_parquet(path)
    return None


def show(df: pd.DataFrame, ticker: str, rows: int, tail: bool, stats: bool) -> None:
    print(f"\n{'='*50}")
    print(f"  {ticker}　共 {len(df):,} 筆　{df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"{'='*50}")

    if stats:
        print("\n--- 統計摘要 ---")
        print(df.describe().to_string())
    else:
        sample = df.tail(rows) if tail else df.head(rows)
        label = f"最後 {rows} 筆" if tail else f"前 {rows} 筆"
        print(f"\n--- {label} ---")
        print(sample.to_string())

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="查看股票/總經 parquet 資料")
    parser.add_argument("ticker", nargs="?", help="股號或代碼（例如 2330、TAIEX）")
    parser.add_argument("-n", "--rows", type=int, default=10, help="顯示筆數（預設 10）")
    parser.add_argument("--tail", action="store_true", help="顯示最後 N 筆（預設顯示前 N 筆）")
    parser.add_argument("--stats", action="store_true", help="顯示統計摘要")
    parser.add_argument("--list", action="store_true", help="列出所有可用檔案")
    args = parser.parse_args()

    if args.list or args.ticker is None:
        list_available()
        if args.ticker is None:
            print("\n用法：uv run tools/view_stock.py <股號>")
        return

    df = load(args.ticker)
    if df is None:
        print(f"[ERROR] 找不到 {args.ticker}，請用 --list 查看可用清單")
        sys.exit(1)

    show(df, args.ticker, args.rows, args.tail, args.stats)


if __name__ == "__main__":
    main()
