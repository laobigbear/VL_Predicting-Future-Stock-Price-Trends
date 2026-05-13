"""
特徵工程模組：
  - 法人異質性指標（BSR、CNBS、Turnover、Concentration）
  - 技術指標（MA、RSI、MACD、Bollinger Bands、ATR）
  - 總經控制變數（VIX、匯率）
  - 滾動 Z-Score 正規化（防止前視偏差）
  - 目標變數生成（N=1/5/20 三分類）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    # 技術指標窗口
    ma_windows: tuple[int, ...] = (5, 20, 60)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_period: int = 14

    # 法人指標滾動窗口
    cnbs_windows: tuple[int, ...] = (5, 20)

    # 正規化
    zscore_window: int = 252
    zscore_min_periods: int = 63

    # 目標變數
    horizons: tuple[int, ...] = (1, 5, 20)
    threshold_pct: float = 1.5  # ±1.5% 為上漲/下跌門檻

    # 交易成本（回測用）
    tx_cost_rate: float = 0.001425  # 0.1425%
    tx_tax_rate: float = 0.003  # 賣出 0.3%


# ─── 法人異質性指標 ─────────────────────────────────────────────────────────────


def compute_bsr(buy: pd.Series, sell: pd.Series, eps: float = 1.0) -> pd.Series:
    """Buy-Sell Ratio：(buy - sell) / (buy + sell + eps)，範圍 (-1, +1)。"""
    return (buy - sell) / (buy + sell + eps)


def compute_cnbs(net: pd.Series, window: int) -> pd.Series:
    """Cumulative Net Buy-Sell over rolling window（張）。"""
    return net.rolling(window=window, min_periods=1).sum()


def compute_inst_features(
    inst_df: pd.DataFrame,
    float_shares: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute all institutional heterogeneity features for one stock.

    Args:
        inst_df: DataFrame indexed by date with columns:
                 fini_buy, fini_sell, fini_net,
                 itf_buy, itf_sell, itf_net,
                 dealer_net (= dealer_self_net + dealer_hedge_net)
        float_shares: Series indexed by date (float shares in 張).
                      If None, turnover features are skipped.

    Returns:
        DataFrame of institutional features indexed by date.
    """
    df = inst_df.copy()
    out = pd.DataFrame(index=df.index)

    for prefix in ("fini", "itf"):
        buy = df.get(f"{prefix}_buy", pd.Series(0.0, index=df.index))
        sell = df.get(f"{prefix}_sell", pd.Series(0.0, index=df.index))
        net = df.get(f"{prefix}_net", pd.Series(0.0, index=df.index))

        out[f"{prefix}_bsr"] = compute_bsr(buy, sell)
        for w in (5, 20):
            out[f"{prefix}_cnbs{w}"] = compute_cnbs(net, w)

        if float_shares is not None:
            fs = float_shares.reindex(df.index).ffill().replace(0, np.nan)
            out[f"{prefix}_turnover"] = (buy + sell) / fs * 100.0

    # dealer: only net available from aggregated TWSE T86
    dealer_net = df.get("dealer_net", pd.Series(0.0, index=df.index))
    out["dealer_bsr"] = dealer_net.apply(np.sign)  # simplified sign indicator
    for w in (5, 20):
        out[f"dealer_cnbs{w}"] = compute_cnbs(dealer_net, w)

    return out


# ─── 技術指標 ──────────────────────────────────────────────────────────────────


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_ma_features(close: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    """MA levels and cross-ratio features."""
    out = pd.DataFrame(index=close.index)
    mas = {}
    for w in cfg.ma_windows:
        mas[w] = close.rolling(w, min_periods=1).mean()
        out[f"ma{w}"] = mas[w]

    # ratio features (avoid division by zero)
    if 5 in mas and 20 in mas:
        out["ma5_20_ratio"] = (mas[5] - mas[20]) / mas[20].replace(0, np.nan)
    if 20 in mas and 60 in mas:
        out["ma20_60_ratio"] = (mas[20] - mas[60]) / mas[60].replace(0, np.nan)

    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_macd(close: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    macd_line = _ema(close, cfg.macd_fast) - _ema(close, cfg.macd_slow)
    signal = _ema(macd_line, cfg.macd_signal)
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal, "macd_hist": macd_line - signal},
        index=close.index,
    )


def compute_bollinger(close: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    """Upper band, lower band, %B (position within band)."""
    ma = close.rolling(cfg.bb_window, min_periods=1).mean()
    std = close.rolling(cfg.bb_window, min_periods=1).std()
    upper = ma + cfg.bb_std * std
    lower = ma - cfg.bb_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"bb_upper": upper, "bb_lower": lower, "bb_pct_b": pct_b}, index=close.index
    )


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_technical_features(
    ohlcv: pd.DataFrame,
    cfg: FeatureConfig,
) -> pd.DataFrame:
    """Compute all technical indicators from OHLCV DataFrame.

    Args:
        ohlcv: DataFrame with columns: open, high, low, close (or adj_close), volume.

    Returns:
        DataFrame of technical features indexed by date.
    """
    close = ohlcv.get("adj_close", ohlcv["close"])
    high = ohlcv["high"]
    low = ohlcv["low"]

    ma_df = compute_ma_features(close, cfg)
    macd_df = compute_macd(close, cfg)
    bb_df = compute_bollinger(close, cfg)

    rsi = compute_rsi(close, cfg.rsi_period).rename("rsi14")
    atr = compute_atr(high, low, close, cfg.atr_period).rename("atr14")

    # Log return
    log_ret = np.log(close / close.shift(1)).rename("log_ret")

    return pd.concat([ma_df, macd_df, bb_df, rsi, atr, log_ret], axis=1)


# ─── 總經特徵 ──────────────────────────────────────────────────────────────────


def compute_macro_features(macro_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute macro control variables from downloaded macro DataFrames.

    Args:
        macro_dict: {"VIX": df, "USDTWD": df, ...} each with adj_close or close column.

    Returns:
        DataFrame with daily macro features.
    """
    frames: list[pd.Series] = []

    if "VIX" in macro_dict:
        vix_close = macro_dict["VIX"].get("adj_close", macro_dict["VIX"]["close"])
        frames.append(vix_close.rename("vix"))
        frames.append(vix_close.pct_change().rename("vix_ret"))
        frames.append(vix_close.rolling(20, min_periods=1).mean().rename("vix_ma20"))

    if "USDTWD" in macro_dict:
        fx_close = macro_dict["USDTWD"].get("adj_close", macro_dict["USDTWD"]["close"])
        frames.append(fx_close.rename("usdtwd"))
        frames.append(fx_close.pct_change().rename("usdtwd_ret"))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index)
    return out


# ─── 正規化 ────────────────────────────────────────────────────────────────────


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Rolling Z-Score using only past data (no look-ahead)."""
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def apply_zscore_normalization(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    skip_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply rolling Z-Score to all numeric columns, except those in skip_cols.

    skip_cols: columns already bounded (e.g., rsi14, bb_pct_b, *_bsr).
    """
    bounded = skip_cols or ["rsi14", "bb_pct_b", "fini_bsr", "itf_bsr", "dealer_bsr"]
    out = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if col in bounded:
            continue
        out[col] = rolling_zscore(df[col], cfg.zscore_window, cfg.zscore_min_periods)
    return out


# ─── 目標變數 ──────────────────────────────────────────────────────────────────


def compute_target(close: pd.Series, horizon: int, threshold_pct: float = 1.5) -> pd.Series:
    """Three-class trend label for future N-day cumulative return.

    Labels: 1=Up, 0=Flat, -1=Down (stored as int8).
    Note: last `horizon` rows will be NaN (future unknown).
    """
    fwd_ret = (close.shift(-horizon) / close - 1.0) * 100.0
    label = pd.Series(0, index=close.index, dtype="int8")
    label[fwd_ret >= threshold_pct] = 1
    label[fwd_ret <= -threshold_pct] = -1
    label[close.shift(-horizon).isna()] = pd.NA
    return label.astype("Int8")


def build_all_targets(close: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    """Build target columns for all horizons."""
    return pd.DataFrame(
        {f"target_h{h}": compute_target(close, h, cfg.threshold_pct) for h in cfg.horizons},
        index=close.index,
    )


# ─── 主整合函數 ────────────────────────────────────────────────────────────────


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    inst_df: pd.DataFrame | None = None,
    macro_df: pd.DataFrame | None = None,
    float_shares: pd.Series | None = None,
    cfg: FeatureConfig | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Assemble full feature matrix for one stock.

    Args:
        ohlcv: OHLCV DataFrame indexed by date.
        inst_df: Institutional trading DataFrame indexed by date.
        macro_df: Macro features DataFrame indexed by date.
        float_shares: Series of float shares indexed by date.
        cfg: FeatureConfig instance.
        normalize: Whether to apply rolling Z-Score normalization.

    Returns:
        DataFrame with all features + target columns, indexed by date.
        Rows with all-NaN features are dropped.
    """
    if cfg is None:
        cfg = FeatureConfig()

    close = ohlcv.get("adj_close", ohlcv["close"])

    tech_df = compute_technical_features(ohlcv, cfg)

    parts: list[pd.DataFrame] = [tech_df]

    if inst_df is not None:
        inst_feats = compute_inst_features(inst_df, float_shares)
        parts.append(inst_feats)

    if macro_df is not None:
        parts.append(macro_df)

    feature_df = pd.concat(parts, axis=1)
    feature_df.index = pd.to_datetime(feature_df.index)

    if normalize:
        feature_df = apply_zscore_normalization(feature_df, cfg)

    targets = build_all_targets(close, cfg)
    out = pd.concat([feature_df, targets], axis=1)

    # Drop rows where ALL features are NaN (warm-up period)
    feat_cols = [c for c in out.columns if not c.startswith("target_")]
    out = out.dropna(how="all", subset=feat_cols)

    return out


def smoke_test() -> None:
    """Minimal end-to-end test with synthetic data."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    idx = pd.bdate_range("2022-01-01", periods=n)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    ohlcv = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1000, 10000, n).astype(float),
        },
        index=idx,
    )
    df = build_feature_matrix(ohlcv)
    print(f"smoke_test: shape={df.shape}")
    print(df.dtypes.to_string())
    print(df.tail(3).to_string())


if __name__ == "__main__":
    smoke_test()
