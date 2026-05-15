"""
Yahoo Finance OHLCV 下載模組。
涵蓋台股個股、TAIEX 指數及總經變數（台幣匯率、VIX）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


@dataclass
class DownloadConfig:
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    output_dir: str = "data/raw/prices"
    sleep_sec: float = 0.3
    # TWSE tickers use ".TW" suffix on Yahoo Finance
    tickers: list[str] = field(default_factory=list)
    force_refresh: bool = False  # True 時忽略既有檔案，強制重新下載


# 總經 / 指數 tickers
MACRO_TICKERS: dict[str, str] = {
    "TAIEX": "^TWII",
    "VIX": "^VIX",
    "USDTWD": "TWD=X",
}


def _yf_ticker(stock_no: str) -> str:
    """Convert bare stock number to Yahoo Finance .TW format."""
    return f"{stock_no}.TW" if not stock_no.endswith(".TW") else stock_no


def _coverage_check(
    out_path: Path,
    cfg_start: str,
    cfg_end: str,
    force: bool = False,
) -> tuple[str | None, pd.DataFrame | None]:
    """判斷是否需要下載，回傳 (fetch_start, existing_df)。

    - fetch_start=None  → 已是最新，跳過
    - fetch_start 有值 + existing_df=None  → 全新下載
    - fetch_start 有值 + existing_df 有值  → 增量下載後 append
    """
    if force or not out_path.exists():
        return cfg_start, None
    try:
        existing = pd.read_parquet(out_path)
        if existing.empty:
            return cfg_start, None
        latest: pd.Timestamp = existing.index.max()
        end_dt = pd.Timestamp(cfg_end)
        if latest >= end_dt - pd.Timedelta(days=5):  # 5 天容差（含週末假日）
            return None, existing
        new_start = (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return new_start, existing
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Cannot read {out_path}: {exc}. Re-downloading.")
        return cfg_start, None


def _merge_and_save(existing: pd.DataFrame | None, new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Append new rows to existing data, deduplicate, and save."""
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df.sort_index()
    combined.to_parquet(out_path)
    return combined


def download_single(
    ticker: str,
    start: str,
    end: str,
    retries: int = 3,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """Download OHLCV for one ticker; return adjusted-close based DataFrame.

    Returns:
        DataFrame indexed by date with columns: open, high, low, close, volume, adj_close.
        Empty DataFrame on failure.
    """
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if raw.empty:
                return pd.DataFrame()
            # yfinance may return MultiIndex columns when auto_adjust=False
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "adj_close",
                }
            )
            df.index.name = "date"
            df.index = pd.to_datetime(df.index)
            return df[["open", "high", "low", "close", "volume", "adj_close"]].copy()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"[WARN] {ticker} download failed: {exc}")
                return pd.DataFrame()
            time.sleep(sleep * (attempt + 1))
    return pd.DataFrame()


def download_stock_universe(cfg: DownloadConfig) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all tickers in cfg.tickers（支援增量更新）。

    已是最新的檔案直接跳過；缺漏期間的資料增量補抓後 append。
    Saves each stock as parquet: output_dir/stocks/{ticker}.parquet
    Returns dict mapping ticker -> DataFrame.
    """
    out_dir = Path(cfg.output_dir) / "stocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    skipped = 0

    for stock_no in tqdm(cfg.tickers, desc="Stocks", unit="ticker"):
        out_path = out_dir / f"{stock_no}.parquet"
        fetch_start, existing = _coverage_check(
            out_path, cfg.start_date, cfg.end_date, force=cfg.force_refresh
        )

        if fetch_start is None:
            results[stock_no] = existing  # type: ignore[assignment]
            skipped += 1
            continue

        df = download_single(_yf_ticker(stock_no), fetch_start, cfg.end_date, sleep=cfg.sleep_sec)
        if not df.empty:
            results[stock_no] = _merge_and_save(existing, df, out_path)
        time.sleep(cfg.sleep_sec)

    action = "skipped as up-to-date"
    print(f"[INFO] Stocks done: {len(results)} total, {skipped} {action}")
    return results


def download_macro(cfg: DownloadConfig) -> dict[str, pd.DataFrame]:
    """Download TAIEX index, VIX, and USD/TWD exchange rate（支援增量更新）。

    已是最新的指標直接跳過；缺漏期間的資料增量補抓後 append。
    Saves each as parquet: output_dir/macro/{name}.parquet
    Returns dict mapping name -> DataFrame.
    """
    out_dir = Path(cfg.output_dir) / "macro"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for name, ticker in tqdm(MACRO_TICKERS.items(), desc="Macro", unit="series"):
        out_path = out_dir / f"{name}.parquet"
        fetch_start, existing = _coverage_check(
            out_path, cfg.start_date, cfg.end_date, force=cfg.force_refresh
        )

        if fetch_start is None:
            print(f"[SKIP] {name}: already up-to-date")
            results[name] = existing  # type: ignore[assignment]
            continue

        df = download_single(ticker, fetch_start, cfg.end_date, sleep=cfg.sleep_sec)
        if not df.empty:
            results[name] = _merge_and_save(existing, df, out_path)
            total = len(results[name])
            label = "appended" if existing is not None else "downloaded"
            print(f"[INFO] {name} ({ticker}): {label}, {total} rows total")
        else:
            print(f"[WARN] {name} ({ticker}): empty result")
        time.sleep(cfg.sleep_sec)

    return results


def load_taiex_components(path: str | None = None) -> list[str]:
    """Load TAIEX component stock numbers from a CSV or return a small default list.

    CSV should have a column 'ticker' with stock numbers (without .TW suffix).
    If path is None, returns a default sample of 20 stocks for smoke testing.
    """
    if path is not None:
        try:
            df = pd.read_csv(path, dtype=str)
            return df["ticker"].str.strip().tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Cannot load components from {path}: {exc}")

    # Fallback: representative TAIEX blue chips for testing
    return [
        "2330",
        "2317",
        "2454",
        "2308",
        "2382",
        "2303",
        "2412",
        "3711",
        "2881",
        "2882",
        "2891",
        "2884",
        "1301",
        "1303",
        "2886",
        "2002",
        "1216",
        "2207",
        "3008",
        "4938",
    ]


def smoke_test() -> None:
    """Download 3 days of 2330 (TSMC) and TAIEX to verify connectivity."""
    cfg = DownloadConfig(start_date="2023-01-02", end_date="2023-01-06")
    df = download_single("2330.TW", cfg.start_date, cfg.end_date)
    print(f"smoke_test 2330.TW: {len(df)} rows\n{df.to_string()}")
    macro = download_macro(cfg)
    for name, mdf in macro.items():
        print(f"  {name}: {len(mdf)} rows")


if __name__ == "__main__":
    smoke_test()
