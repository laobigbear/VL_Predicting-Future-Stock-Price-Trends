"""
TWSE 三大法人每日買賣超爬蟲。
資料來源：TWSE 公開資訊觀測站 https://www.twse.com.tw
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


@dataclass
class CrawlerConfig:
    start_date: str = "20100101"
    end_date: str = "20251231"
    output_dir: str = "data/raw/twse"
    sleep_sec: float = 0.5  # polite delay between requests
    max_retries: int = 3


_TWSE_INST_URL = "https://www.twse.com.tw/rwd/zh/fund/T86?date={date}&selectType=ALL&response=json"
_TWSE_STOCK_URL = (
    "https://www.twse.com.tw/rwd/zh/fund/TWT38U?date={date}&stockNo={ticker}&response=json"
)

_COLUMNS_INST = [
    "ticker",
    "name",
    "fini_buy",
    "fini_sell",
    "fini_net",
    "itf_buy",
    "itf_sell",
    "itf_net",
    "dealer_self_buy",
    "dealer_self_sell",
    "dealer_self_net",
    "dealer_hedge_buy",
    "dealer_hedge_sell",
    "dealer_hedge_net",
    "dealer_net",
]


def _parse_num(s: str) -> float:
    """Strip commas and convert to float; return NaN on failure."""
    try:
        return float(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return float("nan")


def fetch_daily_institutional(
    date_str: str, session: requests.Session, retries: int = 3
) -> pd.DataFrame:
    """Fetch all-stock institutional trading for one trading date.

    Args:
        date_str: Date in YYYYMMDD format.
        session: Reusable requests.Session.
        retries: Max retry attempts on transient network failures.

    Returns:
        DataFrame with columns defined by _COLUMNS_INST, or empty DataFrame if no data.
    """
    import json

    url = _TWSE_INST_URL.format(date=date_str)
    _empty = pd.DataFrame(columns=_COLUMNS_INST + ["date"])

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            # Non-2xx response — not recoverable by retrying
            return _empty
        except Exception as exc:  # noqa: BLE001 — network error, retriable
            if attempt == retries - 1:
                print(f"[WARN] {date_str} fetch failed after {retries} attempts: {exc}")
                return _empty
            time.sleep(2**attempt)
            continue

        text = resp.text.strip()
        if not text:
            # Empty body = no data for this date (non-retriable)
            return _empty

        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError):
            # Server returned non-JSON (e.g. HTML error page) — non-retriable
            return _empty

        break

    if payload.get("stat") != "OK" or "data" not in payload:
        return pd.DataFrame(columns=_COLUMNS_INST + ["date"])

    rows = []
    for row in payload["data"]:
        # row has 16+ fields; we extract the first 15 that match _COLUMNS_INST
        if len(row) < 15:
            continue
        parsed = [str(row[0]).strip(), str(row[1]).strip()] + [_parse_num(v) for v in row[2:15]]
        rows.append(parsed)

    df = pd.DataFrame(rows, columns=_COLUMNS_INST)
    df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
    return df


def _trading_dates(start: str, end: str) -> list[str]:
    """Generate YYYYMMDD strings for each business day in [start, end]."""
    idx = pd.bdate_range(start=start, end=end, freq="B")
    return [d.strftime("%Y%m%d") for d in idx]


def crawl_range(cfg: CrawlerConfig | None = None) -> pd.DataFrame:
    """Crawl institutional trading data for the full date range in cfg.

    Returns concatenated DataFrame sorted by (date, ticker).
    Saves parquet to cfg.output_dir / institutional_trading.parquet.
    """
    if cfg is None:
        cfg = CrawlerConfig()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "institutional_trading.parquet"

    # Resume: skip already-fetched dates
    existing: pd.DataFrame | None = None
    fetched_dates: set[str] = set()
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            fetched_dates = set(existing["date"].dt.strftime("%Y%m%d").unique())
            print(f"[INFO] Resuming: {len(fetched_dates)} dates already fetched.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not read existing parquet: {exc}")

    dates = _trading_dates(cfg.start_date, cfg.end_date)
    dates_to_fetch = [d for d in dates if d not in fetched_dates]
    print(f"[INFO] Fetching {len(dates_to_fetch)} dates...")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research-bot)"

    frames: list[pd.DataFrame] = []
    for date_str in tqdm(dates_to_fetch, desc="Crawling TWSE", unit="day"):
        df = fetch_daily_institutional(date_str, session, retries=cfg.max_retries)
        if not df.empty:
            frames.append(df)
        time.sleep(cfg.sleep_sec)

    if frames:
        new_data = pd.concat(frames, ignore_index=True)
        all_data = (
            pd.concat([existing, new_data], ignore_index=True) if existing is not None else new_data
        )
    else:
        all_data = (
            existing if existing is not None else pd.DataFrame(columns=_COLUMNS_INST + ["date"])
        )

    # Numeric columns cleanup
    num_cols = [c for c in _COLUMNS_INST if c not in ("ticker", "name")]
    for col in num_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")

    all_data.sort_values(["date", "ticker"], inplace=True)
    all_data.reset_index(drop=True, inplace=True)

    try:
        all_data.to_parquet(out_path, index=False)
        print(f"[INFO] Saved to {out_path} ({len(all_data)} rows)")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Could not save parquet: {exc}")

    return all_data


def smoke_test() -> None:
    """Fetch a single day to verify the crawler works."""
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research-bot)"
    df = fetch_daily_institutional("20230103", session)
    print(f"smoke_test: {len(df)} rows returned")
    if not df.empty:
        print(df.head(3).to_string())
    else:
        print("[WARN] Empty result — market may have been closed on that date.")


if __name__ == "__main__":
    smoke_test()
